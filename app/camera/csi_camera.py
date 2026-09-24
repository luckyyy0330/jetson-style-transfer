"""CSI 摄像头模块 - Jetson Orin Nano 专用

支持 NVIDIA Jetson 的 CSI 接口摄像头（如 IMX219）
使用 GStreamer Python 绑定进行视频采集（绕过 OpenCV 的 GStreamer 后端）

零拷贝/低税设计（Phase 1.3）:
- 管道里 **没有 CPU `videoconvert` 元素**（它强制 NVMM→system 下载 + CPU BGRx→BGR）
- `nvvidconv` 硬件缩放后直接出 **system 内存 NV12**（1.5 B/px）或 BGRx（4 B/px）回退
- 回调只做一次 `memcpy`（Gst 缓冲 unmap 后即被 Argus 池回收，必须拷走）
- 不再有 `read_frame().copy()` 的第二次拷贝
- 帧通过 `sink` 回调交给消费者（通常是 FrameQueue2），不再内部缓存 `self._frame`

借用视图契约:
    `FrameView.data` 是 `np.frombuffer` 的借用视图，**仅在 sink 调用期间有效**。
    sink 必须在返回前把自己需要的数据拷走。
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional, Tuple, List, Callable, NamedTuple
import logging

logger = logging.getLogger(__name__)

# PixelFormat 枚举值（与 cuda/include/style_cuda/frame.h 保持一致）
FMT_BGR8 = 0
FMT_RGB8 = 1
FMT_NV12 = 2
FMT_BGRX8 = 3
FMT_YUYV = 4


class FrameView(NamedTuple):
    """一帧的借用视图。data 在 sink 返回后失效。"""
    data: np.ndarray   # 借用视图，shape 取决于 fmt
    w: int             # 有效像素宽
    h: int             # 有效像素高
    pitch: int         # 每行字节数（可能 > w * bpp，含对齐填充）
    fmt: int           # FMT_*
    seq: int           # 帧序号
    ts: float          # monotonic 时间戳


class CSICamera:
    """CSI 摄像头类 - 用于 Jetson Orin Nano（IMX219 等）

    使用 GStreamer Python 绑定直接抓帧，不依赖 OpenCV 的 GStreamer 支持。

    用法（sink 模式，推荐 — 采集线程直接投递）:
        cam = CSICamera(camera_id=0, width=1280, height=720)
        cam.pipeline_size = (480, 360)
        cam.sink = my_sink_fn      # (FrameView) -> bool
        cam.open()

    用法（轮询模式，兼容旧接口）:
        ret, frame = cam.read_frame()   # 返回 BGR，内部只有一份缓存
    """

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ):
        """
        初始化摄像头

        Args:
            camera_id: 摄像头 ID（通常为 0 或 1）
            width: 传感器输出宽度
            height: 传感器输出高度
            fps: 帧率
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.is_opened = False

        # 硬件缩放目标（None = 不缩放，输出传感器原始尺寸）
        self.pipeline_size: Optional[Tuple[int, int]] = None

        # 帧接收器（sink 模式）。返回 False 表示丢弃该帧。
        self.sink: Optional[Callable[[FrameView], bool]] = None

        # 实际协商到的输出格式
        self.out_fmt: int = FMT_NV12
        self.out_w: int = 0
        self.out_h: int = 0

        # 轮询模式的单帧缓存（sink 未设置时才用）
        self._frame: Optional[np.ndarray] = None
        self._frame_bgr: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()

        # GStreamer
        self._pipeline = None
        self._appsink = None
        self._seq = 0
        self._frames_received = 0

    def _build_pipeline_str(self, out_format: str) -> str:
        """构造 GStreamer 管道字符串。out_format: 'NV12' 或 'BGRx'"""
        if self.pipeline_size:
            pw, ph = self.pipeline_size
            size_caps = f"width=(int){pw}, height=(int){ph}, "
        else:
            size_caps = ""

        # 关键：没有 videoconvert。nvvidconv 硬件缩放 + 格式转换，直接出 system 内存。
        return (
            f"nvarguscamerasrc sensor-id={self.camera_id} ! "
            f"video/x-raw(memory:NVMM), "
            f"width=(int){self.width}, height=(int){self.height}, "
            f"format=(string)NV12, framerate=(fraction){self.fps}/1 ! "
            f"nvvidconv ! "
            f"video/x-raw, {size_caps}format=(string){out_format} ! "
            f"appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
        )

    def open(self) -> bool:
        """
        打开 CSI 摄像头

        优先尝试 system 内存 NV12（1.5 B/px，融合预处理核原生支持）；
        caps 协商失败时回退 BGRx（4 B/px，同样走融合核）。

        Returns:
            是否成功打开
        """
        try:
            import gi
            gi.require_version('Gst', '1.0')
            from gi.repository import Gst, GLib

            Gst.init(None)
        except ImportError as e:
            logger.error(f"缺少 GStreamer Python 绑定: {e}")
            logger.error("请安装: sudo apt install python3-gi python3-gi-cairo gir1.2-gstreamer-1.0")
            return False

        # 依次尝试 NV12 / BGRx
        for fmt_name, fmt_enum in (("NV12", FMT_NV12), ("BGRx", FMT_BGRX8)):
            if self._try_open(Gst, fmt_name, fmt_enum):
                return True
            logger.warning(f"format={fmt_name} 管道启动失败，尝试下一个")

        logger.error("所有像素格式均协商失败，检查 nvarguscamerasrc 是否被占用")
        self.release()
        return False

    def _try_open(self, Gst, fmt_name: str, fmt_enum: int) -> bool:
        """尝试用指定像素格式打开管道。失败返回 False 并清理现场。"""
        from gi.repository import Gst as _Gst

        pipeline_str = self._build_pipeline_str(fmt_name)
        logger.info(f"创建 GStreamer 管道: {pipeline_str}")

        try:
            self._pipeline = Gst.parse_launch(pipeline_str)
        except Exception as e:
            logger.warning(f"管道解析失败 ({fmt_name}): {e}")
            self._pipeline = None
            return False

        if self._pipeline is None:
            return False

        self._appsink = self._pipeline.get_by_name("sink")
        if self._appsink is None:
            logger.error("无法获取 appsink")
            self._teardown()
            return False

        self._appsink.set_property("emit-signals", True)
        self._appsink.set_property("max-buffers", 1)
        self._appsink.set_property("drop", True)
        self._appsink.set_property("sync", False)

        self._new_frame_event.clear()
        self._appsink.connect("new-sample", self._on_new_sample)

        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            logger.warning(f"管道启动失败 ({fmt_name})")
            self._teardown()
            return False

        # 等第一帧（NV12 caps 在 Argus 上偶发协商不到，用短超时探测）
        if not self._new_frame_event.wait(timeout=5.0):
            logger.warning(f"{fmt_name} 超时无帧")
            self._teardown()
            return False

        self.out_fmt = fmt_enum
        self.is_opened = True
        logger.info(
            f"CSI 摄像头已打开: {self.out_w}x{self.out_h} format={fmt_name} "
            f"@ {self.fps}fps (硬件缩放: {self.width}x{self.height} → "
            f"{self.out_w}x{self.out_h})"
        )
        return True

    def _on_new_sample(self, appsink):
        """GStreamer appsink 回调（运行在 GStreamer streaming 线程）。

        只做一件事：把缓冲区交给 sink。**不分配第二份像素、不缓存 self._frame**
        （sink 模式下）。Gst 缓冲在 unmap 后立刻归还 Argus 池——这已经是
        采集侧的提前归还。
        """
        try:
            from gi.repository import Gst

            sample = appsink.emit("pull-sample")
            if sample is None:
                return Gst.FlowReturn.OK

            buf = sample.get_buffer()
            if buf is None:
                return Gst.FlowReturn.OK

            caps = sample.get_caps()
            struct = caps.get_structure(0)
            width = int(struct.get_value("width"))
            height = int(struct.get_value("height"))
            fmt_name = struct.get_value("format")

            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.OK

            try:
                nbytes = map_info.size
                # 从缓冲大小反推 pitch（nvvidconv 可能做 256B 行对齐）
                if fmt_name == "NV12":
                    rows = height * 3 // 2
                    bpp_num, bpp_den = 1, 1  # 每像素 1 字节（Y）；UV 半高
                    fmt = FMT_NV12
                else:  # BGRx / BGR
                    rows = height
                    bpp_num, bpp_den = (4, 1) if fmt_name == "BGRx" else (3, 1)
                    fmt = FMT_BGRX8 if fmt_name == "BGRx" else FMT_BGR8

                pitch = nbytes // rows if rows > 0 else 0
                if pitch <= 0:
                    return Gst.FlowReturn.OK

                # 借用视图 — sink 返回后失效
                raw = np.frombuffer(map_info.data, dtype=np.uint8, count=nbytes)
                view = raw.reshape(rows, pitch)

                self._seq += 1
                fv = FrameView(
                    data=view, w=width, h=height, pitch=pitch,
                    fmt=fmt, seq=self._seq, ts=time.monotonic(),
                )

                if self.sink is not None:
                    # sink 负责拷走需要的数据
                    self.sink(fv)
                else:
                    # 轮询模式：缓存一份（这是唯一一次像素拷贝）
                    self._cache_frame(fv)

                self._frames_received += 1
                self._new_frame_event.set()

            finally:
                buf.unmap(map_info)

            return Gst.FlowReturn.OK

        except Exception as e:
            logger.warning(f"处理帧时出错: {e}")
            try:
                from gi.repository import Gst
                return Gst.FlowReturn.ERROR
            except Exception:
                return False

    def _cache_frame(self, fv: FrameView):
        """轮询模式缓存一帧（sink 未设置时）。"""
        owned = fv.data.copy()
        if fv.fmt == FMT_NV12:
            # 提取紧凑 NV12 再转 BGR
            yuv = np.ascontiguousarray(fv.data[: fv.h * 3 // 2, : fv.w])
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        elif fv.fmt == FMT_BGRX8:
            bgrx = fv.data[: fv.h, : fv.w * 4].reshape(fv.h, fv.w, 4)
            bgr = cv2.cvtColor(bgrx, cv2.COLOR_BGRA2BGR)
        else:
            bgr = np.ascontiguousarray(fv.data[: fv.h, : fv.w * 3].reshape(fv.h, fv.w, 3))

        with self._lock:
            self._frame = owned
            self._frame_bgr = bgr

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        读取一帧（轮询模式，兼容旧接口）。

        Returns:
            (success, frame_bgr) — BGR 格式。返回的是内部缓存的**浅引用**，
            调用方若需长期持有请自行 `.copy()`。
        """
        if not self.is_opened:
            return False, None

        with self._lock:
            if self._frame_bgr is not None:
                return True, self._frame_bgr
        return False, None

    def read_frame_view(self) -> Tuple[bool, Optional[dict]]:
        """
        读取一帧的元信息（不含像素，供推理线程配合 copy_source 使用）。

        Returns:
            (success, meta) — meta 含 w/h/pitch/fmt/seq/ts 以及 'raw' 原始像素。
        """
        if not self.is_opened:
            return False, None

        with self._lock:
            if self._frame is not None:
                return True, {
                    "raw": self._frame,
                    "w": self.out_w, "h": self.out_h,
                    "pitch": self._frame.shape[1] if self._frame.ndim == 2 else 0,
                    "fmt": self.out_fmt,
                }
        return False, None

    def get_info(self) -> dict:
        """获取摄像头信息"""
        return {
            "camera_id": self.camera_id,
            "width": self.out_w if self.is_opened else 0,
            "height": self.out_h if self.is_opened else 0,
            "fps": self.fps,
            "is_opened": self.is_opened,
            "format": self.out_fmt,
            "frames_received": self._frames_received,
        }

    def _teardown(self):
        """内部：停管道并清空句柄（不改 is_opened）"""
        if self._pipeline is not None:
            try:
                from gi.repository import Gst
                self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        self._pipeline = None
        self._appsink = None

    def release(self):
        """释放摄像头资源"""
        self._teardown()
        self.is_opened = False
        logger.info(f"CSI 摄像头 {self.camera_id} 已释放")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


def list_cameras(max_cameras: int = 4) -> List[dict]:
    """
    列出可用的摄像头

    Args:
        max_cameras: 最大扫描的摄像头数量

    Returns:
        可用摄像头列表
    """
    cameras = []

    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst

        Gst.init(None)

        for i in range(max_cameras):
            try:
                # 探测管道也不带 videoconvert
                pipeline_str = (
                    f"nvarguscamerasrc sensor-id={i} ! "
                    f"video/x-raw(memory:NVMM), width=640, height=480, "
                    f"format=NV12, framerate=1/1 ! "
                    f"nvvidconv ! video/x-raw, format=NV12 ! fakesink"
                )
                pipeline = Gst.parse_launch(pipeline_str)
                ret = pipeline.set_state(Gst.State.PLAYING)

                if ret != Gst.StateChangeReturn.FAILURE:
                    cameras.append({
                        "id": i,
                        "name": f"CSI Camera {i}",
                        "type": "csi"
                    })

                pipeline.set_state(Gst.State.NULL)

            except Exception as e:
                logger.debug(f"扫描摄像头 {i} 失败: {e}")

    except ImportError:
        logger.warning("GStreamer Python 绑定不可用，无法扫描摄像头")

    return cameras
