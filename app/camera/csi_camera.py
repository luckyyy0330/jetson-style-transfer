"""CSI 摄像头模块 - Jetson Orin Nano 专用

支持 NVIDIA Jetson 的 CSI 接口摄像头（如 IMX219）
使用 GStreamer Python 绑定进行视频采集（绕过 OpenCV 的 GStreamer 后端）
"""

import cv2
import numpy as np
import threading
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class CSICamera:
    """CSI 摄像头类 - 用于 Jetson Orin Nano（IMX219 等）

    使用 GStreamer Python 绑定直接抓帧，不依赖 OpenCV 的 GStreamer 支持
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
            width: 分辨率宽度
            height: 分辨率高度
            fps: 帧率
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.is_opened = False

        # GStreamer 管道相关
        self._pipeline = None
        self._appsink = None
        self._bus = None
        self._loop = None
        self._frame = None
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()

    def open(self) -> bool:
        """
        打开 CSI 摄像头

        Returns:
            是否成功打开
        """
        try:
            import gi
            gi.require_version('Gst', '1.0')
            from gi.repository import Gst, GLib

            # 初始化 GStreamer
            Gst.init(None)

            # 构建管道字符串
            pipeline_str = (
                f"nvarguscamerasrc sensor-id={self.camera_id} ! "
                f"video/x-raw(memory:NVMM), "
                f"width=(int){self.width}, height=(int){self.height}, "
                f"format=(string)NV12, framerate=(fraction){self.fps}/1 ! "
                f"nvvidconv ! "
                f"video/x-raw, format=(string)BGRx ! "
                f"videoconvert ! "
                f"video/x-raw, format=(string)BGR ! "
                f"appsink name=sink emit-signals=true max-buffers=1 drop=true"
            )

            logger.info(f"创建 GStreamer 管道: {pipeline_str}")
            self._pipeline = Gst.parse_launch(pipeline_str)

            if self._pipeline is None:
                logger.error("无法创建 GStreamer 管道")
                return False

            # 获取 appsink
            self._appsink = self._pipeline.get_by_name("sink")
            if self._appsink is None:
                logger.error("无法获取 appsink")
                return False

            # 设置 appsink 属性
            self._appsink.set_property("emit-signals", True)
            self._appsink.set_property("max-buffers", 1)
            self._appsink.set_property("drop", True)

            # 连接回调
            self._appsink.connect("new-sample", self._on_new_sample)

            # 启动管道
            ret = self._pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.error("GStreamer 管道启动失败")
                self.release()
                return False

            # 等待第一帧（最多 5 秒）
            logger.info("等待摄像头第一帧...")
            if not self._new_frame_event.wait(timeout=5.0):
                logger.error("摄像头超时无帧，检查 nvarguscamerasrc 是否被占用")
                self.release()
                return False

            self.is_opened = True
            logger.info(f"CSI 摄像头已打开: {self.width}x{self.height} @ {self.fps}fps")
            return True

        except ImportError as e:
            logger.error(f"缺少 GStreamer Python 绑定: {e}")
            logger.error("请安装: sudo apt install python3-gi python3-gi-cairo gir1.2-gstreamer-1.0")
            return False
        except Exception as e:
            logger.error(f"打开 CSI 摄像头异常: {e}")
            self.release()
            return False

    def _on_new_sample(self, appsink):
        """GStreamer appsink 回调 - 接收新帧"""
        try:
            from gi.repository import Gst

            sample = appsink.emit("pull-sample")
            if sample is None:
                return False

            buf = sample.get_buffer()
            if buf is None:
                return False

            # 映射缓冲区数据
            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                return False

            try:
                # 获取视频信息
                caps = sample.get_caps()
                struct = caps.get_structure(0)
                width = struct.get_value("width")
                height = struct.get_value("height")

                # 将 BGR 数据转为 numpy 数组
                data = map_info.data
                frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))

                with self._lock:
                    self._frame = frame.copy()
                self._new_frame_event.set()

            finally:
                buf.unmap(map_info)

            return False

        except Exception as e:
            logger.warning(f"处理帧时出错: {e}")
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        读取一帧

        Returns:
            (success, frame) - frame 为 BGR 格式的 numpy 数组
        """
        if not self.is_opened:
            return False, None

        with self._lock:
            if self._frame is not None:
                return True, self._frame.copy()
        return False, None

    def get_info(self) -> dict:
        """获取摄像头信息"""
        if not self.is_opened:
            return {
                "camera_id": self.camera_id,
                "width": 0,
                "height": 0,
                "fps": 0,
                "is_opened": False
            }

        return {
            "camera_id": self.camera_id,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "is_opened": True
        }

    def release(self):
        """释放摄像头资源"""
        if self._pipeline is not None:
            try:
                from gi.repository import Gst
                self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            self._pipeline = None
            self._appsink = None
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
                pipeline_str = (
                    f"nvarguscamerasrc sensor-id={i} ! "
                    f"video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=1/1 ! "
                    f"nvvidconv ! videoconvert ! video/x-raw, format=BGR ! appsink"
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
