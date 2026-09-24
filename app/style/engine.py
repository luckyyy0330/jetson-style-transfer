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
        input_size: tuple = (512, 512),
        prefer_backend: Optional[BackendType] = None,
        strength: float = 1.0,
    ):
        """
        初始化多后端引擎

        Args:
            model_dir: 模型目录
            input_size: 输入尺寸 (width, height) 元组
            prefer_backend: 优先使用的后端
            strength: 风格强度
        """
        self.model_dir = model_dir
        self.input_size = input_size
        self.prefer_backend = prefer_backend
        self.strength = strength

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

        self._current_style = style

        # 按优先级尝试；TensorRT 引擎缺失/损坏时自动回退 ONNX Runtime
        candidates = []
        if backend == BackendType.TENSORRT:
            candidates = [BackendType.TENSORRT, BackendType.ONNX_RUNTIME]
        elif backend == BackendType.ONNX_RUNTIME:
            candidates = [BackendType.ONNX_RUNTIME]
        else:
            candidates = [backend]

        for cand in candidates:
            if cand not in self.available_backends:
                continue
            loader = self._load_tensorrt if cand == BackendType.TENSORRT else self._load_onnx
            if loader(style):
                self.current_backend = cand
                return True
            logger.warning(f"{cand.value} 加载失败，尝试下一个后端")

        logger.error("所有后端加载失败")
        self.current_backend = BackendType.NONE
        return False

    def _find_model_file(self, style: Style, extension: str) -> Optional[str]:
        """
        查找模型文件

        搜索顺序：
        1. trt/style_{id}_{w}x{h}_{prec}.engine（带分辨率/精度后缀的 TensorRT 引擎）
        2. trt/style_{id}.engine（旧命名兼容）
        3. base_model 指定的文件名

        Args:
            style: 风格配置
            extension: 文件扩展名（如 "onnx" 或 "engine"）

        Returns:
            模型文件路径，未找到返回 None
        """
        if extension == "engine":
            w, h = style.input_size
            # 新命名：style_{id}_{w}x{h}_{prec}.engine
            trt_dir = os.path.join(self.model_dir, "trt")
            if os.path.isdir(trt_dir):
                # 优先精确匹配 fp16
                for suffix in ("fp16", "fp32", "fp16_iofp16", "fp32_iofp16"):
                    p = os.path.join(trt_dir, f"style_{style.id}_{w}x{h}_{suffix}.engine")
                    if os.path.exists(p):
                        return p
                # 旧命名兼容
                legacy = os.path.join(trt_dir, f"style_{style.id}.engine")
                if os.path.exists(legacy):
                    return legacy
                # 任意精度后缀兜底
                import glob
                matches = sorted(glob.glob(os.path.join(trt_dir, f"style_{style.id}_*.engine")))
                if matches:
                    return matches[0]

        # base_model 直接使用（仅当扩展名匹配时）
        if style.base_model.endswith(f".{extension}"):
            model_path = os.path.join(self.model_dir, style.base_model)
            if os.path.exists(model_path):
                return model_path

        # 如果 base_model 没有扩展名，尝试加上
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
                sharpen=style.sharpen,
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
                sharpen=style.sharpen,
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

        **必须在推理线程内调用**，且调用前先 `quiesce()`。
        UI 线程请通过 mailbox 投递切换请求。

        Args:
            new_style: 新风格配置

        Returns:
            是否成功
        """
        # 先静默（等 pending + 丢 graph），再决定是否需要重载
        self.quiesce()

        # 如果模型文件相同，只需更新 style 引用
        if (self._current_style and
            self._current_style.base_model == new_style.base_model and
            self.backend_instance is not None):
            logger.info(f"模型相同，更新风格: {new_style.name}")
            self._current_style = new_style
            self.input_size = new_style.input_size
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

        # TensorRT 使用原生 API 推理
        if self.current_backend == BackendType.TENSORRT:
            return self.backend_instance._infer_tensorrt(input_data)

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
            # 同步到后端实例，避免换风格后预处理尺寸停留在旧值
            if self.backend_instance is not None:
                self.backend_instance.input_size = style.input_size
                self.backend_instance.input_width, self.backend_instance.input_height = style.input_size

        return self.backend_instance.transfer(
            content_image, strength=self.strength
        )

    # ---- CUDA 快路径三段式 API（Submit / ReleaseInput / Wait）----

    @property
    def use_gpu_path(self) -> bool:
        """是否走 CUDA Graph + 融合前后处理快路径"""
        return bool(
            self.backend_instance is not None
            and getattr(self.backend_instance, "use_gpu_path", False)
        )

    @property
    def graph_active(self) -> bool:
        """CUDA Graph 是否已实例化"""
        return bool(
            self.backend_instance is not None
            and getattr(self.backend_instance, "graph_active", False)
        )

    def copy_source(self, frame: np.ndarray) -> bool:
        """拷贝源帧到 mapped staging。见 GANEngine.copy_source。"""
        if self.backend_instance is None:
            return False
        fn = getattr(self.backend_instance, "copy_source", None)
        if fn is None:
            return False
        return fn(frame)

    def submit(self, src_w: int, src_h: int, src_pitch: int,
               fmt: int = 0, matrix: int = 0, full_range: bool = False) -> float:
        """Submit：set_source + run。staging 中必须已有该帧数据。"""
        if self.backend_instance is None:
            raise RuntimeError("模型未加载")
        return self.backend_instance.submit(src_w, src_h, src_pitch, fmt, matrix, full_range)

    def release_input(self) -> float:
        """提前归还输入缓冲（TRT 还在跑时）"""
        if self.backend_instance is None:
            return 0.0
        fn = getattr(self.backend_instance, "release_input", None)
        return fn() if fn else 0.0

    def wait(self) -> Tuple[np.ndarray, float]:
        """
        等待推理完成。

        Returns:
            (styled_bgr, wait_time_ms) — styled_bgr 是**已持有**的数组，
            可安全投递到显示队列 / 用于 strength 第二前向。
        """
        if self.backend_instance is None:
            raise RuntimeError("模型未加载")
        return self.backend_instance.wait()

    def quiesce(self):
        """
        静默引擎：等 pending 请求结束、丢弃 CUDA Graph。

        换风格/换分辨率前**必须**在推理线程内调用，之后再 `switch_style`。
        UI 线程绝不要直接碰 CUDA。
        """
        if self.backend_instance is not None:
            fn = getattr(self.backend_instance, "quiesce", None)
            if fn is not None:
                fn()

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
