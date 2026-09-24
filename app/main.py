"""Jetson 风格转换设备 - 主程序入口

功能：
- 初始化 CSI 摄像头（NV12 / BGRx，无 CPU videoconvert）
- 加载 GAN 风格迁移模型（ONNX / TensorRT + 可选 style_cuda 快路径）
- 实时风格迁移（三线程流水：采集 / 推理 / 显示）
- OpenCV 窗口显示原始 + 风格化画面
- 每个风格对应独立的模型，切换风格即切换模型

线程模型（Phase 1.4，照 jetson_media/src/pipeline.cpp）:
    GStreamer streaming 线程   ──push──>  FrameQueue2(Latest)
                                              │
                                    推理线程  │ acquire
                                              ▼
                                     Submit / ReleaseInput / Wait
                                              │
                                              │ push (orig, styled)
                                              ▼
    显示（主线程）  <──acquire──  FrameQueue2(Latest)
            │
            └──mailbox──> 推理线程（切风格/改强度；UI 线程绝不碰 CUDA）

30fps 闸门是 strength=1.0。strength>1.0 走两次 cudaGraphLaunch，
天然 2× GPU 耗时（约 10–14fps），属可接受。
"""

import os
import sys
import time
import signal
import logging
import argparse
import threading
import queue
from typing import Optional, Tuple

import cv2
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.camera.csi_camera import CSICamera, list_cameras, FrameView, FMT_NV12, FMT_BGRX8, FMT_BGR8
from app.style.style_manager import StyleManager, Style
from app.style.engine import MultiBackendEngine, BackendType
from app.export.storage import StorageManager
from app.pipeline.frame_queue import FrameQueue2, QueuePolicy

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GstRecorder:
    """硬件 H.264 录像（nvv4l2h264enc），失败回退 cv2.VideoWriter"""

    def __init__(self, path: str, width: int, height: int, fps: float = 30.0):
        self.path = path
        self.width = width
        self.height = height
        self.fps = max(fps, 1.0)
        self._pipe = None
        self._appsrc = None
        self._cv_writer = None
        self._use_gst = False
        self.frames = 0

    def start(self) -> bool:
        try:
            import gi
            gi.require_version('Gst', '1.0')
            from gi.repository import Gst, GLib
            Gst.init(None)

            pipeline_str = (
                f"appsrc name=src format=time do-timestamp=true ! "
                f"video/x-raw, format=BGR, width={self.width}, height={self.height}, "
                f"framerate={int(self.fps)}/1 ! "
                f"nvvidconv ! video/x-raw(memory:NVMM), format=NV12 ! "
                f"nvv4l2h264enc bitrate=8000000 iframeinterval=30 ! "
                f"h264parse ! qtmux ! filesink location={self.path}"
            )
            self._pipe = Gst.parse_launch(pipeline_str)
            self._appsrc = self._pipe.get_by_name("src")
            ret = self._pipe.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("nvv4l2h264enc 管道启动失败")
            self._use_gst = True
            logger.info(f"硬件录像 (nvv4l2h264enc): {self.path}")
            return True
        except Exception as e:
            logger.warning(f"硬件录像不可用 ({e})，回退 cv2.VideoWriter")
            self._teardown_gst()
            return self._start_cv()

    def _start_cv(self) -> bool:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if not self.path.endswith(".mp4"):
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self._cv_writer = cv2.VideoWriter(self.path, fourcc, self.fps,
                                          (self.width, self.height))
        if not self._cv_writer.isOpened():
            logger.error("无法创建视频写入器")
            self._cv_writer = None
            return False
        logger.info(f"软编码录像 (mp4v): {self.path}")
        return True

    def write(self, frame_bgr: np.ndarray):
        self.frames += 1
        if self._use_gst and self._appsrc is not None:
            try:
                from gi.repository import Gst, GLib
                data = frame_bgr.tobytes()
                buf = Gst.Buffer.new_allocate(None, len(data), None)
                buf.fill(0, data)
                self._appsrc.emit("push-buffer", buf)
                return
            except Exception as e:
                logger.warning(f"硬件编码推帧失败: {e}，回退软编码")
                self._teardown_gst()
                self._use_gst = False
                if self._cv_writer is None:
                    self._start_cv()
        if self._cv_writer is not None:
            self._cv_writer.write(frame_bgr)

    def stop(self) -> str:
        if self._use_gst:
            try:
                from gi.repository import Gst
                if self._appsrc is not None:
                    self._appsrc.emit("end-of-stream")
                if self._pipe is not None:
                    self._pipe.send_event(Gst.Event.new_eos())
                    self._pipe.get_state(int(2 * Gst.SECOND))
            except Exception:
                pass
            self._teardown_gst()
        if self._cv_writer is not None:
            self._cv_writer.release()
            self._cv_writer = None
        return self.path

    def _teardown_gst(self):
        if self._pipe is not None:
            try:
                from gi.repository import Gst
                self._pipe.set_state(Gst.State.NULL)
            except Exception:
                pass
        self._pipe = None
        self._appsrc = None
        self._use_gst = False


def frame_to_bgr(fv: FrameView) -> np.ndarray:
    """FrameView → BGR（显示用，采集线程内调用）"""
    if fv.fmt == FMT_NV12:
        yuv = np.ascontiguousarray(fv.data[: fv.h * 3 // 2, : fv.w])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    if fv.fmt == FMT_BGRX8:
        bgrx = fv.data[: fv.h, : fv.w * 4].reshape(fv.h, fv.w, 4)
        return cv2.cvtColor(bgrx, cv2.COLOR_BGRA2BGR)
    # BGR8
    return np.ascontiguousarray(fv.data[: fv.h, : fv.w * 3].reshape(fv.h, fv.w, 3))


class StyleTransferApp:
    """风格转换应用主类"""

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 800,
        height: int = 600,
        models_dir: str = "models",
        image_path: str = None,
        preferred_backend: str = "auto",
        style_id: Optional[int] = None,
        style_strength: float = 1.0,
    ):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.models_dir = models_dir
        self.image_path = image_path
        self.preferred_backend = preferred_backend
        self.initial_style_id = style_id
        self.style_strength = style_strength

        # 跳帧相关
        self.skip_frames = 0
        self._frame_counter = 0

        # 录像相关
        self.is_recording = False
        self._recorder: Optional[GstRecorder] = None
        self._recording_start_time = 0.0
        self._recording_frame_count = 0
        self._recording_compare_path = ""
        self._btn_rect = (0, 0, 0, 0)

        # 组件
        self.camera: Optional[CSICamera] = None
        self.style_manager: Optional[StyleManager] = None
        self.engine: Optional[MultiBackendEngine] = None
        self.storage: Optional[StorageManager] = None

        # ---- 流水线队列（Phase 1.4）----
        self.frame_queue = FrameQueue2(policy=QueuePolicy.LATEST, name="capture")
        self.result_queue = FrameQueue2(policy=QueuePolicy.LATEST, name="result")
        self.mailbox: "queue.Queue[Tuple]" = queue.Queue(maxsize=16)
        self._infer_thread: Optional[threading.Thread] = None

        # 状态
        self.is_running = False
        self.current_style: Optional[Style] = None
        self.last_frame: Optional[np.ndarray] = None
        self.last_styled_frame: Optional[np.ndarray] = None
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = 0.0

        # 性能 HUD
        self._steady_infer_ms = 0.0
        self._graph_state = 0
        self._infer_errors = 0

        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info("收到退出信号，正在关闭...")
        self.stop()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def initialize(self) -> bool:
        logger.info("初始化应用...")

        # 1. 存储管理器
        self.storage = StorageManager()

        # 2. 风格配置
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "styles.json"
        )
        self.style_manager = StyleManager(
            config_path=config_path,
            models_dir=self.models_dir
        )

        if self.initial_style_id:
            self.style_manager.set_current_style(self.initial_style_id)

        current_style = self.style_manager.get_current_style()
        if current_style is None:
            logger.error("没有可用的风格配置")
            return False
        self.current_style = current_style

        resolved_model_path = self.style_manager.resolve_model_path(current_style)
        logger.info(f"当前风格: {current_style.name} (模型: {resolved_model_path})")

        # 3. 推理引擎
        prefer_backend = None
        if self.preferred_backend == "tensorrt":
            prefer_backend = BackendType.TENSORRT
        elif self.preferred_backend == "onnx":
            prefer_backend = BackendType.ONNX_RUNTIME

        self.engine = MultiBackendEngine(
            model_dir=self.models_dir,
            input_size=current_style.input_size,
            prefer_backend=prefer_backend,
            strength=self.style_strength,
        )

        if not self.engine.load_model(style=current_style):
            logger.error("推理引擎加载失败")
            logger.error("请确保已下载模型: python scripts/download_models.py --all")
            return False

        logger.info(f"推理后端: {self.engine.current_backend.value}")
        logger.info(
            f"CUDA 快路径: {'启用 (graph=' + str(int(self.engine.graph_active)) + ')'}"
            if self.engine.use_gpu_path else "CUDA 快路径: 未启用（CPU 前后处理）"
        )
        if self.style_strength != 1.0:
            logger.info(f"风格强度: {self.style_strength}（注意 >1.0 走双前向，约 2× GPU 耗时）")
        if self.skip_frames > 0:
            logger.info(f"跳帧模式: 每 {self.skip_frames + 1} 帧处理 1 帧")

        # 4. 摄像头（图片模式跳过）
        if self.image_path:
            logger.info(f"图片模式: {self.image_path}")
            test_img = cv2.imread(self.image_path)
            if test_img is None:
                logger.error(f"无法读取图片: {self.image_path}")
                return False
            logger.info(f"图片尺寸: {test_img.shape[1]}x{test_img.shape[0]}")
        else:
            logger.info(f"打开摄像头 {self.camera_id}...")
            self.camera = CSICamera(
                camera_id=self.camera_id,
                width=self.width,
                height=self.height,
            )
            # nvvidconv 硬件缩放到模型输入尺寸（省掉 Python resize + CPU videoconvert）
            if current_style.input_size:
                self.camera.pipeline_size = tuple(current_style.input_size)
            # 采集线程直接投递到 frame_queue
            self.camera.sink = self._capture_sink
            if not self.camera.open():
                logger.error("摄像头打开失败")
                return False
            logger.info("摄像头打开成功")

        logger.info("初始化完成!")
        return True

    # ------------------------------------------------------------------
    # 采集线程（GStreamer streaming 线程）
    # ------------------------------------------------------------------
    def _capture_sink(self, fv: FrameView) -> bool:
        """接收一帧。fv.data 是借用视图，必须在返回前拷走。

        只做两次 CPU 拷贝（都不在 GPU 关键路径）：
          1. raw 像素 → 槽位（Gst 缓冲 unmap 后即被 Argus 池回收）
          2. BGR 转换 → 显示左路
        staging 拷贝在推理线程的 copy_source 里（Phase 3 可前移到这里）。
        """
        raw_owned = fv.data.copy()
        orig_bgr = frame_to_bgr(fv)

        meta = (fv.w, fv.h, fv.pitch, fv.fmt, fv.seq, fv.ts)
        # payload = (raw, meta, orig_bgr)
        return self.frame_queue.push((raw_owned, meta, orig_bgr), fv.seq)

    # ------------------------------------------------------------------
    # 推理线程
    # ------------------------------------------------------------------
    def _infer_loop(self):
        """推理线程主循环：Submit / ReleaseInput / Wait 三段式"""
        logger.info("推理线程已启动")
        while self.is_running:
            # 1. 先处理命令（切风格/改强度）
            self._drain_mailbox()
            if not self.is_running:
                break

            # 2. 取帧
            slot = self.frame_queue.acquire(timeout=0.1)
            if slot is None:
                continue

            raw, meta, orig_bgr = slot.payload
            src_w, src_h, src_pitch, src_fmt, seq, ts = meta

            try:
                # 跳帧：只更新计数，不推理（上一帧结果仍在 result_queue 里被显示）
                skip = (
                    self.skip_frames > 0
                    and self._frame_counter % (self.skip_frames + 1) != 0
                )
                self._frame_counter += 1

                if not skip:
                    if self.engine.use_gpu_path:
                        styled = self._infer_gpu(raw, orig_bgr, src_w, src_h, src_pitch, src_fmt)
                    else:
                        styled, _ = self.engine.transfer(orig_bgr, style=self.current_style)

                    # 3. 发布结果。orig_bgr 是采集线程已持有的独立数组，无需再拷。
                    #    styled 必须是 owned（wait() 返回持有副本，已脱离 mapped 内存）。
                    self.result_queue.push((orig_bgr, styled, seq), seq)

            except Exception as e:
                self._infer_errors += 1
                logger.error(f"推理线程异常: {e}")
                if self._infer_errors > 20:
                    logger.error("推理线程连续异常过多，退出")
                    self.is_running = False
            finally:
                # 槽位在此归还 —— 绝不在上面的分支里单独 release（会双释放）
                self.frame_queue.release(slot)

        self.frame_queue.stop()
        self.result_queue.stop()
        logger.info("推理线程已退出")

    def _infer_gpu(self, raw: np.ndarray, orig_bgr: np.ndarray,
                   src_w: int, src_h: int, src_pitch: int, src_fmt: int) -> Optional[np.ndarray]:
        """GPU 快路径：copy_source → submit → release_input → wait → strength 混合"""
        t0 = time.perf_counter()

        # Submit：staging 拷贝 + set_source + run（Graph 捕获或 stream 回退）
        self.engine.copy_source(raw)
        self.engine.submit(src_w, src_h, src_pitch, src_fmt, 0, False)
        self.engine.release_input()   # TRT 还在跑，staging 已可复用

        styled, _ = self.engine.wait()  # owned 数组，已脱离 mapped 内存

        strength = self.style_strength
        if strength > 1.0:
            # 第二前向：对第一轮结果再推理（2× GPU 耗时）
            self.engine.copy_source(styled)
            self.engine.submit(styled.shape[1], styled.shape[0], styled.strides[0], 0, 0, False)
            self.engine.release_input()
            styled2, _ = self.engine.wait()
            alpha = strength - 1.0
            styled = cv2.addWeighted(styled, 1.0 - alpha, styled2, alpha, 0)
        elif strength < 1.0:
            styled = cv2.addWeighted(orig_bgr, 1.0 - strength, styled, strength, 0)

        t1 = time.perf_counter()
        self._steady_infer_ms = self._steady_infer_ms * 0.9 + (t1 - t0) * 1000 * 0.1
        self._graph_state = 1 if self.engine.graph_active else 0
        return styled

    def _drain_mailbox(self):
        """处理 UI 线程投递的命令。切风格必须在这里做（quiesce + switch_style）。"""
        while True:
            try:
                cmd = self.mailbox.get_nowait()
            except queue.Empty:
                return
            try:
                kind = cmd[0]
                if kind == "switch_style":
                    self._do_switch_style(cmd[1])
                elif kind == "set_strength":
                    self.style_strength = float(cmd[1])
                    self.engine.strength = self.style_strength
                    logger.info(f"风格强度: {self.style_strength}")
                elif kind == "stop":
                    self.is_running = False
                    return
            except Exception as e:
                logger.error(f"命令执行失败 {cmd}: {e}")

    def _do_switch_style(self, style_id: int):
        """切换风格（推理线程内执行）"""
        new_style = self.style_manager.get_style(style_id)
        if new_style is None:
            logger.warning(f"风格 {style_id} 不存在")
            return

        resolved_path = self.style_manager.resolve_model_path(new_style)
        logger.info(f"切换风格: {new_style.name} (模型: {resolved_path})")

        # quiesce + switch_style（MultiBackendEngine.switch_style 内部会 quiesce）
        ok = self.engine.switch_style(new_style)
        if ok:
            self.current_style = new_style
            self.style_manager.set_current_style(style_id)
            # 相机缩放目标跟着变
            if self.camera is not None and new_style.input_size:
                self.camera.pipeline_size = tuple(new_style.input_size)
            logger.info(f"风格切换完成: {new_style.name}")
        else:
            logger.error(f"风格切换失败: {new_style.name}")

    # ------------------------------------------------------------------
    # 显示线程（主线程）
    # ------------------------------------------------------------------
    def run(self):
        if not self.initialize():
            logger.error("初始化失败，退出")
            return

        self.is_running = True
        self.start_time = time.time()

        # 图片模式：串行跑一次，不起线程
        if self.image_path:
            self._run_image_mode()
            self.stop()
            return

        logger.info("应用已启动，按 Ctrl+C 退出")
        logger.info("快捷键: r=录像 | 空格=拍照 | 1-5=切换风格 | +/-=强度 | q=退出")

        # 启动推理线程
        self._infer_thread = threading.Thread(
            target=self._infer_loop, name="infer", daemon=True
        )
        self._infer_thread.start()

        cv2.namedWindow("Style Transfer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Style Transfer", 800, 600)
        cv2.setMouseCallback("Style Transfer", self._on_mouse)

        try:
            self._display_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _run_image_mode(self):
        """图片模式：串行处理一张图并保存"""
        frame = cv2.imread(self.image_path)
        if frame is None:
            logger.error(f"无法读取图片: {self.image_path}")
            return

        if self.engine.use_gpu_path:
            raw = frame
            src_h, src_w = frame.shape[:2]
            src_pitch = frame.strides[0]
            self.engine.copy_source(raw)
            self.engine.submit(src_w, src_h, src_pitch, FMT_BGR8, 0, False)
            self.engine.release_input()
            styled, t = self.engine.wait()
        else:
            styled, t = self.engine.transfer(frame, style=self.current_style)

        output_dir = "assets/captures"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        styled_path = os.path.join(
            output_dir, f"styled_{self.current_style.id}_{timestamp}.jpg"
        )
        cv2.imwrite(styled_path, styled)
        logger.info(f"风格迁移完成 ({t*1000:.1f}ms)，已保存: {styled_path}")

    def _display_loop(self):
        """显示线程主循环：只做显示 + 输入，绝不碰 CUDA"""
        while self.is_running:
            slot = self.result_queue.acquire(timeout=0.2)
            if slot is None:
                # 无结果时仍要处理键盘
                self._handle_keys(cv2.waitKey(1) & 0xFF)
                continue

            orig, styled, seq = slot.payload

            # 拷一份到 last_*（拍照/录像用；result 槽下一帧就复用了）
            self.last_frame = orig
            self.last_styled_frame = styled

            # 录像：左右对比
            if self.is_recording and self._recorder is not None:
                display_w, display_h = 800, 600
                o = cv2.resize(orig, (display_w, display_h))
                s = cv2.resize(styled, (display_w, display_h))
                self._recorder.write(np.hstack([o, s]))
                self._recording_frame_count += 1

            # FPS
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed if elapsed > 0 else 0

            self._compose_and_show(orig, styled)

            self.result_queue.release(slot)

    def _compose_and_show(self, orig: np.ndarray, styled: np.ndarray):
        """拼显示画面并展示"""
        display_w, display_h = 800, 600
        orig_display = cv2.resize(orig, (display_w, display_h))
        styled_display = cv2.resize(styled, (display_w, display_h))
        display = np.hstack([orig_display, styled_display])

        backend_info = self.engine.current_backend.value if self.engine else "N/A"
        style_name = self.current_style.name if self.current_style else "N/A"
        graph = self._graph_state
        status_text = (
            f"FPS: {self.fps:.1f} | infer:{self._steady_infer_ms:.1f}ms "
            f"| graph:{graph} | {backend_info} | {style_name}"
        )
        if self.style_strength != 1.0:
            status_text += f" | x{self.style_strength:.1f}"
        if self.is_recording:
            rec_sec = time.time() - self._recording_start_time
            status_text += f" | REC {rec_sec:.0f}s"

        cv2.putText(
            display, status_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 0, 255) if self.is_recording else (0, 255, 0), 2,
        )
        self._draw_button(display)

        cv2.imshow("Style Transfer", display)
        self._handle_keys(cv2.waitKey(1) & 0xFF)

    def _handle_keys(self, key: int):
        """键盘快捷键（显示线程）。切风格只投递命令，不直接碰 CUDA。"""
        if key == ord('q'):
            self.stop()
        elif key == ord('1'):
            self._post_cmd(("switch_style", 1))
        elif key == ord('2'):
            self._post_cmd(("switch_style", 2))
        elif key == ord('3'):
            self._post_cmd(("switch_style", 3))
        elif key == ord('4'):
            self._post_cmd(("switch_style", 4))
        elif key == ord('5'):
            self._post_cmd(("switch_style", 5))
        elif key == ord('+') or key == ord('='):
            self._post_cmd(("set_strength", round(self.style_strength + 0.1, 2)))
        elif key == ord('-') or key == ord('_'):
            self._post_cmd(("set_strength", max(0.1, round(self.style_strength - 0.1, 2))))
        elif key == ord(' '):
            self._on_capture()
        elif key == ord('r'):
            if self.is_recording:
                self._stop_recording()
            else:
                self._start_recording()

    def _post_cmd(self, cmd: Tuple):
        """投递命令到推理线程（UI 线程绝不直接调用 switch_style）"""
        try:
            self.mailbox.put_nowait(cmd)
        except queue.Full:
            logger.warning(f"命令队列已满，丢弃: {cmd}")

    def _on_capture(self):
        """拍照回调（显示线程）"""
        if self.last_frame is None:
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        original_path = self.storage.save_image(
            self.last_frame, f"original_{timestamp}.jpg"
        )
        saved = [original_path]

        if self.last_styled_frame is not None:
            styled_path = self.storage.save_image(
                self.last_styled_frame, f"styled_{timestamp}.jpg"
            )
            comparison = np.hstack([self.last_frame, self.last_styled_frame])
            comp_path = self.storage.save_image(
                comparison, f"comparison_{timestamp}.jpg"
            )
            saved.extend([styled_path, comp_path])

        logger.info(f"已保存: {', '.join(saved)}")

    def _start_recording(self):
        """开始录像（左右对比：左原图，右风格化）"""
        if self.is_recording or self.last_styled_frame is None:
            return

        h, w = self.last_styled_frame.shape[:2]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs("assets/captures", exist_ok=True)

        # 显示尺寸的对比帧
        display_w, display_h = 800, 600
        actual_fps = max(self.fps, 1.0)
        compare_path = os.path.join(
            "assets/captures", f"compare_{timestamp}.mp4"
        )

        recorder = GstRecorder(compare_path, display_w * 2, display_h, fps=actual_fps)
        if not recorder.start():
            logger.error("无法创建录像器")
            return

        self._recorder = recorder
        self._recording_compare_path = compare_path
        self._recording_start_time = time.time()
        self._recording_frame_count = 0
        self.is_recording = True
        logger.info(
            f"开始录像 ({actual_fps:.1f}fps, {display_w * 2}x{display_h} 对比): {compare_path}"
        )

    def _stop_recording(self):
        """停止录像"""
        if not self.is_recording:
            return

        self.is_recording = False
        path = self._recording_compare_path
        if self._recorder is not None:
            self._recorder.stop()
            self._recorder = None

        duration = time.time() - self._recording_start_time
        logger.info(
            f"录像已保存 ({self._recording_frame_count} 帧, {duration:.1f}秒): {path}"
        )

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            bx, by, bw, bh = self._btn_rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                if self.is_recording:
                    self._stop_recording()
                else:
                    self._start_recording()

    def _draw_button(self, display: np.ndarray):
        h, w = display.shape[:2]
        btn_w, btn_h = 120, 40
        margin = 15
        bx = w - btn_w - margin
        by = margin
        self._btn_rect = (bx, by, btn_w, btn_h)

        if self.is_recording:
            cv2.rectangle(display, (bx, by), (bx + btn_w, by + btn_h), (0, 0, 220), -1)
            cv2.circle(display, (bx + 18, by + btn_h // 2), 8, (255, 255, 255), -1)
            cv2.putText(
                display, "STOP", (bx + 32, by + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
        else:
            cv2.rectangle(display, (bx, by), (bx + btn_w, by + btn_h), (60, 60, 60), -1)
            cv2.circle(display, (bx + 18, by + btn_h // 2), 8, (0, 0, 220), -1)
            cv2.putText(
                display, "REC", (bx + 32, by + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2
            )

    def stop(self):
        """停止应用（幂等，多次调用安全）"""
        if not self.is_running and self.engine is None:
            return

        self.is_running = False

        if self.is_recording:
            self._stop_recording()

        # 唤醒推理线程并等它退出
        try:
            self.mailbox.put_nowait(("stop",))
        except queue.Full:
            pass
        self.frame_queue.stop()
        self.result_queue.stop()
        if self._infer_thread is not None and self._infer_thread.is_alive():
            self._infer_thread.join(timeout=2.0)
            self._infer_thread = None

        if self.camera:
            self.camera.release()
            self.camera = None

        if self.engine:
            self.engine.unload()
            self.engine = None

        if not self.image_path:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        qs = self.frame_queue.stats()
        rs = self.result_queue.stats()
        logger.info(f"队列统计: 采集={qs} | 结果={rs} | 推理异常={self._infer_errors}")
        logger.info("应用已停止")


def main():
    parser = argparse.ArgumentParser(description="Jetson 风格转换设备")
    parser.add_argument("--camera", type=int, default=0, help="摄像头 ID")
    parser.add_argument("--usb", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--image", type=str, help="使用本地图片测试（无需摄像头）")
    parser.add_argument("--width", type=int, default=800, help="摄像头宽度")
    parser.add_argument("--height", type=int, default=600, help="摄像头高度")
    parser.add_argument("--models", default="models", help="模型目录")
    parser.add_argument("--backend", choices=["tensorrt", "onnx", "auto"], default="auto",
                       help="推理后端 (默认: auto，自动检测最优)")
    parser.add_argument("--style", type=int, default=None, help="初始风格 ID (默认: 配置文件中的 default_style_id)")
    parser.add_argument("--strength", type=float, default=1.0,
                       help="风格强度 (1.0=正常, 1.5=略强, 2.0=极强, <1.0=减弱。>1.0 走双前向，约 2× GPU 耗时)")
    parser.add_argument("--skip-frames", type=int, default=0, help="跳帧数 (0=每帧都处理，1=隔1帧，2=隔2帧...，默认: 0)")
    parser.add_argument("--list-cameras", action="store_true", help="列出可用摄像头")

    args = parser.parse_args()

    if args.list_cameras:
        cameras = list_cameras()
        print("可用摄像头:")
        for cam in cameras:
            print(f"  {cam['id']}: {cam['name']} ({cam['type']})")
        return

    app = StyleTransferApp(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        models_dir=args.models,
        image_path=args.image,
        preferred_backend=args.backend,
        style_id=args.style,
        style_strength=args.strength,
    )
    app.skip_frames = args.skip_frames

    app.run()


if __name__ == "__main__":
    main()
