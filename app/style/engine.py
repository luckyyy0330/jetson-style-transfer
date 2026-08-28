"""多后端推理引擎

支持 TensorRT 和 ONNX Runtime 后端
自动检测并使用最优后端
每个风格对应独立的 GAN ONNX 模型
"""

import os
import logging
from typing import Optional, Tuple, Dict, List
from enum import Enum

import numpy as np

from app.style.style_manager import Style

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """推理后端类型"""
    TENSORRT = "tensorrt"
    ONNX_RUNTIME = "onnxruntime"
    NONE = "none"


class MultiBackendEngine:
    """多后端推理引擎

    自动检测并使用最优后端：
    1. TensorRT (最快，需要 .engine 文件)
    2. ONNX Runtime (通用，需要 .onnx 文件)

    每个风格对应独立的 GAN 模型
    """

    def __init__(
        self,
        model_dir: str = "models",
        input_size: int = 512,
        prefer_backend: Optional[BackendType] = None
    ):
        """
        初始化多后端引擎

        Args:
            model_dir: 模型目录
            input_size: 输入尺寸
            prefer_backend: 优先使用的后端
        """
        self.model_dir = model_dir
        self.input_size = input_size
        self.prefer_backend = prefer_backend

        # 当前后端和引擎实例
        self.current_backend: BackendType = BackendType.NONE
        self.backend_instance = None

        # 当前风格
        self._current_style: Optional[Style] = None

        # 可用后端
        self.available_backends: List[BackendType] = []

        # 检测可用后端
        self._detect_backends()

    def _detect_backends(self):
        """检测可用的推理后端"""
        if self._check_tensorrt():
            self.available_backends.append(BackendType.TENSORRT)
            logger.info("✓ TensorRT 可用")

        if self._check_onnxruntime():
            self.available_backends.append(BackendType.ONNX_RUNTIME)
            logger.info("✓ ONNX Runtime 可用")

        if not self.available_backends:
            logger.warning("没有可用的推理后端！请安装 onnxruntime-gpu")

        logger.info(f"可用后端: {[b.value for b in self.available_backends]}")

    def _check_tensorrt(self) -> bool:
        try:
            import tensorrt
            return True
        except ImportError:
            return False

    def _check_onnxruntime(self) -> bool:
        try:
            import onnxruntime
            return True
        except ImportError:
            return False

    def select_optimal_backend(self) -> BackendType:
        """选择最优后端"""
        if self.prefer_backend and self.prefer_backend in self.available_backends:
            logger.info(f"使用优先后端: {self.prefer_backend.value}")
            return self.prefer_backend

        # 优先级: TensorRT > ONNX Runtime
        priority = [
            BackendType.TENSORRT,
            BackendType.ONNX_RUNTIME,
        ]

        for backend in priority:
            if backend in self.available_backends:
                logger.info(f"自动选择后端: {backend.value}")
                return backend

        logger.warning("没有可用的推理后端")
        return BackendType.NONE

    def load_model(self, style: Style) -> bool:
        """
        根据风格加载模型

        Args:
            style: 风格配置

        Returns:
            是否成功
        """
        backend = self.select_optimal_backend()
        if backend == BackendType.NONE:
            logger.error("没有可用的推理后端")
            return False

        self.current_backend = backend
        self._current_style = style

        if backend == BackendType.TENSORRT:
            return self._load_tensorrt(style)
        elif backend == BackendType.ONNX_RUNTIME:
            return self._load_onnx(style)

        return False

    def _find_model_file(self, style: Style, extension: str) -> Optional[str]:
        """
        查找模型文件

        搜索顺序：
        1. style.base_model 指定的文件名
        2. style_{style_id}.{ext}
        3. styles 子目录下按名称查找

        Args:
            style: 风格配置
            extension: 文件扩展名（如 "onnx" 或 "engine"）

        Returns:
            模型文件路径，未找到返回 None
        """
        # 1. 直接使用 base_model
        model_path = os.path.join(self.model_dir, style.base_model)
        if os.path.exists(model_path):
            return model_path

        # 2. style_{id}.{ext}
        alt_path = os.path.join(
            self.model_dir, "styles", f"style_{style.id}.{extension}"
        )
        if os.path.exists(alt_path):
            return alt_path

        # 3. 如果 base_model 没有扩展名，尝试加上
        if not style.base_model.endswith(f".{extension}"):
            model_path = os.path.join(
                self.model_dir, f"{style.base_model}.{extension}"
            )
            if os.path.exists(model_path):
                return model_path

        return None

    def _load_tensorrt(self, style: Style) -> bool:
        """加载 TensorRT 引擎"""
        try:
            engine_path = self._find_model_file(style, "engine")
            if engine_path is None:
                logger.warning(
                    f"TensorRT 引擎不存在: {style.name} "
                    f"(请先运行 scripts/convert_tensorrt.py)"
                )
                return False

            from app.style.gan_engine import GANEngine

            self.backend_instance = GANEngine(
                model_path=engine_path,
                input_size=style.input_size,
                backend="tensorrt",
                normalize=style.normalize,
            )

            if not self.backend_instance.load():
                return False

            logger.info(f"TensorRT 引擎加载成功: {engine_path}")
            return True

        except Exception as e:
            logger.error(f"加载 TensorRT 引擎失败: {e}")
            return False

    def _load_onnx(self, style: Style) -> bool:
        """加载 ONNX 模型"""
        try:
            onnx_path = self._find_model_file(style, "onnx")
            if onnx_path is None:
                logger.warning(
                    f"ONNX 模型不存在: {style.name} "
                    f"(请先运行 scripts/download_models.py)"
                )
                return False

            from app.style.gan_engine import GANEngine

            self.backend_instance = GANEngine(
                model_path=onnx_path,
                input_size=style.input_size,
                backend="onnx",
                normalize=style.normalize,
            )

            if not self.backend_instance.load():
                return False

            logger.info(f"ONNX 模型加载成功: {onnx_path}")
            return True

        except Exception as e:
            logger.error(f"加载 ONNX 模型失败: {e}")
            return False

    def switch_style(self, new_style: Style) -> bool:
        """
        切换风格（可能需要重新加载模型）

        Args:
            new_style: 新风格配置

        Returns:
            是否成功
        """
        # 如果模型文件相同，只需更新 style 引用
        if (self._current_style and
            self._current_style.base_model == new_style.base_model and
            self.backend_instance is not None):
            logger.info(f"模型相同，更新风格: {new_style.name}")
            self._current_style = new_style
            return True

        # 模型不同，重新加载
        logger.info(f"切换风格: {new_style.name}")
        self.unload()
        return self.load_model(new_style)

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        执行推理

        Args:
            input_data: 输入数据 (NCHW 格式)

        Returns:
            输出数据
        """
        if self.backend_instance is None:
            raise RuntimeError("模型未加载")

        return self.backend_instance.session.run(
            [self.backend_instance.output_name],
            {self.backend_instance.input_name: input_data},
        )[0]

    def transfer(
        self,
        content_image: np.ndarray,
        style: Optional[Style] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        执行风格迁移（包含预处理和后处理）

        Args:
            content_image: 输入图像 (BGR 格式)
            style: 风格配置，None 则使用当前风格

        Returns:
            (styled_image, inference_time)
        """
        if self.backend_instance is None:
            raise RuntimeError("模型未加载")

        if style is not None and style.input_size != self.input_size:
            self.input_size = style.input_size

        return self.backend_instance.transfer(content_image)

    def unload(self):
        """卸载模型，释放资源"""
        if self.backend_instance is not None:
            self.backend_instance.unload()
            self.backend_instance = None

        self.current_backend = BackendType.NONE
        self._current_style = None
        logger.info("模型已卸载")

    def get_backend_info(self) -> Dict:
        """获取后端信息"""
        return {
            "current_backend": self.current_backend.value,
            "available_backends": [b.value for b in self.available_backends],
            "input_size": self.input_size,
            "current_style": self._current_style.name if self._current_style else None,
        }
