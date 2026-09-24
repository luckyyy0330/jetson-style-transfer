"""双槽帧队列 — 2 线程 2 槽调度器

移植自 jetson_media/src/pipeline.cpp 的语义：
- 恰好 2 个槽位，状态 Empty / Ready / Processing
- Latest 策略：只淘汰最旧的 Ready 槽，绝不碰 Processing 槽
- Block 策略：满槽时生产者等待（不丢帧）
- 锁纪律：释放资源绝不在队列锁内
"""

import threading
import time
from enum import Enum
from typing import Optional, Any, Callable


class SlotState(Enum):
    EMPTY = 0
    READY = 1
    PROCESSING = 2


class QueuePolicy(Enum):
    BLOCK = 0   # 不丢帧，满槽时生产者阻塞
    LATEST = 1  # 直播预览：淘汰最旧的 Ready，绝不碰 Processing


class FrameSlot:
    """单个帧槽位"""
    __slots__ = ("state", "payload", "seq", "timestamp")

    def __init__(self):
        self.state = SlotState.EMPTY
        self.payload: Any = None
        self.seq: int = 0
        self.timestamp: float = 0.0


class FrameQueue2:
    """2 槽帧队列

    用法（生产者，通常在 GStreamer 回调线程）:
        queue.push(payload, seq)

    用法（消费者，通常在推理线程）:
        slot = queue.acquire()          # 阻塞直到有 Ready 槽
        try:
            result = process(slot.payload)
        finally:
            queue.release(slot)         # 归还槽位（绝不在锁内释放 CUDA/EGL）
    """

    def __init__(self, policy: QueuePolicy = QueuePolicy.LATEST, name: str = "fq",
                 on_retire: Optional[Callable[[Any], None]] = None):
        self.policy = policy
        self.name = name
        self._slots = [FrameSlot(), FrameSlot()]
        self._cond = threading.Condition()
        self._capture_done = False
        self._stopped = False
        self._on_retire_cb = on_retire
        self.dropped = 0
        self.pushed = 0
        self.consumed = 0

    def push(self, payload: Any, seq: int = 0) -> bool:
        """投产一个帧。返回 True 表示入队成功，False 表示被丢弃或已停止。"""
        retired_payload = None
        with self._cond:
            if self._stopped or self._capture_done:
                return False

            # 找一个 Empty 槽
            target = None
            for s in self._slots:
                if s.state == SlotState.EMPTY:
                    target = s
                    break

            if target is None:
                if self.policy == QueuePolicy.LATEST:
                    # 淘汰最旧的 Ready（绝不碰 Processing）
                    ready_slots = [s for s in self._slots if s.state == SlotState.READY]
                    if ready_slots:
                        target = min(ready_slots, key=lambda s: s.seq)
                        retired_payload = target.payload
                        self.dropped += 1
                    else:
                        # 两个都在 Processing —— 丢弃本次
                        self.dropped += 1
                        return False
                else:
                    # Block 等待
                    while not self._stopped and not any(
                        s.state == SlotState.EMPTY for s in self._slots
                    ):
                        self._cond.wait(timeout=0.1)
                    if self._stopped:
                        return False
                    for s in self._slots:
                        if s.state == SlotState.EMPTY:
                            target = s
                            break
                    if target is None:
                        self.dropped += 1
                        return False

            target.payload = payload
            target.seq = seq
            target.timestamp = time.monotonic()
            target.state = SlotState.READY
            self.pushed += 1
            self._cond.notify_all()

        # 锁外释放被淘汰帧的资源（锁纪律：绝不在锁内释放 CUDA/EGL/V4L2）
        if retired_payload is not None:
            self._on_retire(retired_payload)
        return True

    def acquire(self, timeout: float = 1.0) -> Optional[FrameSlot]:
        """获取一个 Ready 槽并置为 Processing。超时返回 None。"""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                # 取最新 seq 的 Ready（Latest 策略下的语义）
                ready_slots = [s for s in self._slots if s.state == SlotState.READY]
                if ready_slots:
                    target = max(ready_slots, key=lambda s: s.seq)
                    target.state = SlotState.PROCESSING
                    self.consumed += 1
                    return target

                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._stopped:
                    return None
                self._cond.wait(timeout=min(remaining, 0.05))

    def release(self, slot: FrameSlot):
        """归还槽位。payload 的资源释放留给调用方在锁外做。"""
        with self._cond:
            slot.state = SlotState.EMPTY
            slot.payload = None
            self._cond.notify_all()

    def mark_capture_done(self):
        """通知消费者采集已结束"""
        with self._cond:
            self._capture_done = True
            self._cond.notify_all()

    def stop(self):
        """停止队列"""
        with self._cond:
            self._stopped = True
            self._cond.notify_all()

    def stats(self) -> dict:
        """队列计数器"""
        return {
            "pushed": self.pushed,
            "consumed": self.consumed,
            "dropped": self.dropped,
        }

    def _on_retire(self, payload: Any):
        """被淘汰帧的回调（锁外调用），释放资源用"""
        if self._on_retire_cb is not None:
            try:
                self._on_retire_cb(payload)
            except Exception:
                pass

    @property
    def is_idle(self) -> bool:
        """队列是否空闲（无 Ready/Processing）"""
        with self._cond:
            return all(s.state == SlotState.EMPTY for s in self._slots)
