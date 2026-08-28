"""Jetson 风格转换设备 - 主程序入口

功能：
- 初始化 CSI 摄像头
- 加载 GAN 风格迁移模型（ONNX / TensorRT）
- 实时风格迁移（30-100+ FPS）
- OpenCV 窗口显示原始 + 风格化画面
- 每个风格对应独立的模型，切换风格即切换模型

启动：
  python -m app.main
  python -m app.main --camera 1
  python -m app.main --image test.jpg
"""

import os
import sys
import time
import signal
import logging
import argparse
from typing import Optional

import cv2
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.camera.csi_camera import CSICamera, list_cameras
from app.style.style_manager import StyleManager, Style
from app.style.engine import MultiBackendEngine, BackendType
from app.export.storage import StorageManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StyleTransferApp:
    """风格转换应用主类"""

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 1280,
        height: int = 720,
        models_dir: str = "models",
        image_path: str = None,
        preferred_backend: str = "auto",
        style_id: Optional[int] = None
    ):
        """
        初始化应用

        Args:
            camera_id: 摄像头 ID
            width: 摄像头宽度
            height: 摄像头高度
            models_dir: 模型目录
            image_path: 使用本地图片测试
            preferred_backend: 优先使用的推理后端
            style_id: 初始风格 ID，None 则使用默认
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.models_dir = models_dir
        self.image_path = image_path
        self.preferred_backend = preferred_backend
        self.initial_style_id = style_id

        # 组件
        self.camera: Optional[CSICamera] = None
        self.style_manager: Optional[StyleManager] = None
        self.engine: Optional[MultiBackendEngine] = None
        self.storage: Optional[StorageManager] = None

        # 状态
        self.is_running = False
        self.current_style: Optional[Style] = None
        self.last_frame: Optional[np.ndarray] = None
        self.last_styled_frame: Optional[np.ndarray] = None
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = 0

        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """信号处理（优雅退出）"""
        logger.info("收到退出信号，正在关闭...")
        self.stop()

    def initialize(self) -> bool:
        """
        初始化所有组件

        Returns:
            是否成功
        """
        logger.info("初始化应用...")

        # 1. 初始化存储管理器
        logger.info("初始化存储管理器...")
        self.storage = StorageManager()

        # 2. 加载风格配置
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "styles.json"
        )
        self.style_manager = StyleManager(
            config_path=config_path,
            models_dir=self.models_dir
        )

        # 设置初始风格
        if self.initial_style_id:
            self.style_manager.set_current_style(self.initial_style_id)

        current_style = self.style_manager.get_current_style()
        if current_style is None:
            logger.error("没有可用的风格配置")
            return False

        self.current_style = current_style

        # 解析模型路径
        resolved_model_path = self.style_manager.resolve_model_path(current_style)
        current_style.base_model = resolved_model_path
        logger.info(f"当前风格: {current_style.name} (模型: {resolved_model_path})")

        # 3. 加载推理引擎
        logger.info("加载推理引擎...")

        # 确定优先后端
        prefer_backend = None
        if self.preferred_backend == "tensorrt":
            prefer_backend = BackendType.TENSORRT
        elif self.preferred_backend == "onnx":
            prefer_backend = BackendType.ONNX_RUNTIME

        self.engine = MultiBackendEngine(
            model_dir=self.models_dir,
            input_size=current_style.input_size,
            prefer_backend=prefer_backend,
        )

        if not self.engine.load_model(style=current_style):
            logger.error("推理引擎加载失败")
            logger.error("请确保已下载模型: python scripts/download_models.py --all")
            return False

        logger.info(f"推理后端: {self.engine.current_backend.value}")

        # 4. 打开摄像头（图片模式跳过）
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
            if not self.camera.open():
                logger.error("摄像头打开失败")
                return False
            logger.info("摄像头打开成功")

        logger.info("初始化完成!")
        return True

    def run(self):
        """运行应用"""
        if not self.initialize():
            logger.error("初始化失败，退出")
            return

        self.is_running = True
        self.start_time = time.time()

        logger.info("应用已启动，按 Ctrl+C 退出")

        # 创建窗口（仅在摄像头模式下）
        if not self.image_path:
            cv2.namedWindow("Style Transfer", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Style Transfer", 800, 600)

        try:
            while self.is_running:
                self._process_frame()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _process_frame(self):
        """处理一帧"""
        # 读取摄像头帧
        if self.image_path:
            frame = cv2.imread(self.image_path)
            if frame is None:
                time.sleep(0.1)
                return
        else:
            ret, frame = self.camera.read_frame()
            if not ret or frame is None:
                if not hasattr(self, '_frame_warned'):
                    logger.warning("摄像头读取帧失败，请检查摄像头连接")
                    self._frame_warned = True
                time.sleep(0.01)
                return

        self.last_frame = frame

        # 执行风格迁移
        try:
            styled_frame, inference_time = self.engine.transfer(
                frame, style=self.current_style
            )
            self.last_styled_frame = styled_frame

            # 计算 FPS
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed if elapsed > 0 else 0

        except Exception as e:
            logger.error(f"风格迁移失败: {e}")
            self.last_styled_frame = frame

        # 显示结果
        if self.image_path:
            # 图片模式：保存结果并退出
            output_dir = "assets/captures"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            styled_path = os.path.join(
                output_dir,
                f"styled_{self.current_style.id}_{timestamp}.jpg"
            )
            cv2.imwrite(styled_path, self.last_styled_frame)
            logger.info(f"风格迁移完成，已保存: {styled_path}")
            self.stop()
            return
        else:
            # 摄像头模式：显示 OpenCV 窗口
            display = np.hstack([self.last_frame, self.last_styled_frame])

            # 显示后端和风格信息
            backend_info = self.engine.current_backend.value
            style_name = self.current_style.name if self.current_style else "N/A"
            cv2.putText(
                display,
                f"FPS: {self.fps:.1f} | {backend_info} | {style_name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            cv2.imshow("Style Transfer", display)
            key = cv2.waitKey(1) & 0xFF

            # 键盘快捷键
            if key == ord('q'):
                self.stop()
            elif key == ord('1'):
                self._on_style_change(1)
            elif key == ord('2'):
                self._on_style_change(2)
            elif key == ord('3'):
                self._on_style_change(3)
            elif key == ord('4'):
                self._on_style_change(4)
            elif key == ord('5'):
                self._on_style_change(5)
            elif key == ord(' '):
                self._on_capture()

    def _on_style_change(self, style_id: int):
        """风格切换回调"""
        new_style = self.style_manager.get_style(style_id)
        if new_style is None:
            logger.warning(f"风格 {style_id} 不存在")
            return

        # 解析为绝对路径
        resolved_path = self.style_manager.resolve_model_path(new_style)
        new_style.base_model = resolved_path

        logger.info(f"切换风格: {new_style.name} (模型: {resolved_path})")

        self.engine.switch_style(new_style)
        self.current_style = new_style
        self.style_manager.set_current_style(style_id)

    def _on_capture(self):
        """拍照回调"""
        if self.last_frame is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            # 保存原图
            original_path = self.storage.save_image(
                self.last_frame,
                f"original_{timestamp}.jpg"
            )

            # 保存风格化图
            if self.last_styled_frame is not None:
                styled_path = self.storage.save_image(
                    self.last_styled_frame,
                    f"styled_{timestamp}.jpg"
                )

                # 保存并排对比图
                comparison = np.hstack([self.last_frame, self.last_styled_frame])
                comp_path = self.storage.save_image(
                    comparison,
                    f"comparison_{timestamp}.jpg"
                )

                logger.info(f"已保存: {original_path}, {styled_path}")

    def stop(self):
        """停止应用"""
        self.is_running = False

        if self.camera:
            self.camera.release()

        if self.engine:
            self.engine.unload()

        if not self.image_path:
            cv2.destroyAllWindows()

        logger.info("应用已停止")


def main():
    parser = argparse.ArgumentParser(description="Jetson 风格转换设备")
    parser.add_argument("--camera", type=int, default=0, help="摄像头 ID")
    parser.add_argument("--usb", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--image", type=str, help="使用本地图片测试（无需摄像头）")
    parser.add_argument("--width", type=int, default=1280, help="摄像头宽度")
    parser.add_argument("--height", type=int, default=720, help="摄像头高度")
    parser.add_argument("--models", default="models", help="模型目录")
    parser.add_argument("--backend", choices=["tensorrt", "onnx", "auto"], default="auto",
                       help="推理后端 (默认: auto，自动检测最优)")
    parser.add_argument("--style", type=int, default=None, help="初始风格 ID (默认: 配置文件中的 default_style_id)")
    parser.add_argument("--list-cameras", action="store_true", help="列出可用摄像头")

    args = parser.parse_args()

    # 列出摄像头
    if args.list_cameras:
        cameras = list_cameras()
        print("可用摄像头:")
        for cam in cameras:
            print(f"  {cam['id']}: {cam['name']} ({cam['type']})")
        return

    # 创建并运行应用
    app = StyleTransferApp(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        models_dir=args.models,
        image_path=args.image,
        preferred_backend=args.backend,
        style_id=args.style
    )

    app.run()


if __name__ == "__main__":
    main()
