"""GAN 风格迁移推理引擎

基于 ONNX Runtime / TensorRT 的轻量级 GAN 推理引擎。
支持 AnimeGANv2、CartoonGAN 等单次前向推理模型。

模型规格：
- 参数量: ~8M（远小于 SD 的 ~860M）
- 显存: ~200-500MB
- 推理: 单次前向，无需迭代去噪
"""

import os
import time
import logging
from typing import Optional, Tuple, Dict

import numpy as np
import cv2

logger = logging.getLogger(__name__)


class GANEngine:
    """GAN 风格迁移引擎

    支持 ONNX Runtime 和 TensorRT 后端
    每个风格对应独立的 ONNX 模型文件
    """

    def __init__(
        self,
        model_path: str,
        input_size: int = 512,
        backend: str = "onnx",
        dtype: str = "float16",
        normalize: str = "minus_one_to_one",
    ):
        """
        初始化 GAN 引擎

        Args:
            model_path: ONNX 模型文件路径
            input_size: 输入图像尺寸（正方形）
            backend: 推理后端 ("onnx" 或 "tensorrt")
            dtype: 数据类型 (float16/float32)
            normalize: 归一化方式 ("minus_one_to_one" 或 "zero_to_one")
        """
        self.model_path = model_path
        self.input_size = input_size
        self.backend = backend
        self.dtype = dtype
        self.normalize = normalize

        # 推理会话
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None
        self.is_loaded = False

        # 性能统计
        self.frame_count = 0
        self.total_inference_time = 0.0

    def load(self) -> bool:
        """
        加载模型

        Returns:
            是否成功
        """
        if not os.path.exists(self.model_path):
            logger.error(f"模型文件不存在: {self.model_path}")
            return False

        try:
            import onnxruntime as ort

            # 配置执行提供程序
            providers = []
            provider_options = []

            if self.backend == "tensorrt":
                # TensorRT 作为主要后端
                providers.append("TensorrtExecutionProvider")
                provider_options.append({
                    "trt_fp16_enable": self.dtype == "float16",
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": os.path.join(
                        os.path.dirname(self.model_path), "trt_cache"
                    ),
                    "trt_max_workspace_size": 1 << 30,  # 1GB
                })
                providers.append("CUDAExecutionProvider")
                provider_options.append({
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB
                })
            elif self.backend == "onnx":
                providers.append("CUDAExecutionProvider")
                provider_options.append({
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                })

            providers.append("CPUExecutionProvider")
            provider_options.append({})

            # 过滤掉不可用的提供程序
            available = ort.get_available_providers()
            final_providers = []
            final_options = []
            for p, opts in zip(providers, provider_options):
                if p in available:
                    final_providers.append(p)
                    final_options.append(opts)
                else:
                    logger.debug(f"提供程序不可用，跳过: {p}")

            # 配置会话选项
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            sess_options.intra_op_num_threads = 2
            sess_options.inter_op_num_threads = 1

            # 创建推理会话
            logger.info(f"加载 ONNX 模型: {self.model_path}")
            logger.info(f"执行提供程序: {final_providers}")

            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=final_providers,
            )

            # 获取输入输出信息
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_shape = self.session.get_outputs()[0].shape

            # 检查是否使用了期望的后端
            actual_providers = self.session.get_providers()
            if self.backend == "tensorrt" and "TensorrtExecutionProvider" in actual_providers:
                logger.info("✓ 使用 TensorRT 加速")
            elif "CUDAExecutionProvider" in actual_providers:
                logger.info("✓ 使用 CUDA 加速")
            else:
                logger.warning("⚠ 使用 CPU 推理（较慢）")

            self.is_loaded = True
            logger.info(
                f"模型加载完成: 输入={self.input_shape}, 输出={self.output_shape}"
            )
            return True

        except ImportError as e:
            logger.error(f"缺少依赖: {e}")
            logger.error("请安装: pip install onnxruntime-gpu")
            return False
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像

        Args:
            image: BGR 格式的 numpy 数组

        Returns:
            预处理后的 NCHW float32 数组
        """
        # BGR → RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # resize 到模型输入尺寸
        resized = cv2.resize(
            rgb,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )

        # 转 float32
        data = resized.astype(np.float32)

        # 归一化
        if self.normalize == "minus_one_to_one":
            data = data / 127.5 - 1.0  # [0,255] → [-1, 1]
        elif self.normalize == "zero_to_one":
            data = data / 255.0  # [0,255] → [0, 1]

        # HWC → NCHW
        data = np.transpose(data, (2, 0, 1))
        data = np.expand_dims(data, axis=0)

        return data

    def _postprocess(self, output: np.ndarray) -> np.ndarray:
        """
        后处理推理结果

        Args:
            output: 模型输出 (NCHW 或 CHW)

        Returns:
            BGR 格式的 uint8 图像
        """
        # 去掉 batch 维度
        result = np.squeeze(output)

        # CHW → HWC
        result = np.transpose(result, (1, 2, 0))

        # 反归一化
        if self.normalize == "minus_one_to_one":
            result = (result + 1.0) * 127.5  # [-1,1] → [0,255]
        elif self.normalize == "zero_to_one":
            result = result * 255.0  # [0,1] → [0,255]

        # 裁剪到有效范围并转换类型
        result = np.clip(result, 0, 255).astype(np.uint8)

        # RGB → BGR
        bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        return bgr

    def transfer(
        self,
        content_image: np.ndarray,
        input_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        执行风格迁移

        Args:
            content_image: 输入图像 (BGR 格式)
            input_size: 覆盖默认输入尺寸

        Returns:
            (styled_image, inference_time)
        """
        if not self.is_loaded or self.session is None:
            raise RuntimeError("模型未加载")

        start_time = time.perf_counter()

        try:
            # 保存原始尺寸
            orig_h, orig_w = content_image.shape[:2]

            # 预处理
            input_data = self._preprocess(content_image)

            # 推理
            output = self.session.run(
                [self.output_name],
                {self.input_name: input_data},
            )[0]

            # 后处理
            styled = self._postprocess(output)

            # resize 回原始尺寸
            styled = cv2.resize(
                styled,
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR,
            )

            inference_time = time.perf_counter() - start_time

            # 统计
            self.frame_count += 1
            self.total_inference_time += inference_time

            return styled, inference_time

        except Exception as e:
            logger.error(f"风格迁移失败: {e}")
            inference_time = time.perf_counter() - start_time
            return content_image, inference_time

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "model_path": self.model_path,
            "input_size": self.input_size,
            "backend": self.backend,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "normalize": self.normalize,
        }

    def get_fps(self) -> float:
        """获取平均 FPS"""
        if self.total_inference_time > 0:
            return self.frame_count / self.total_inference_time
        return 0.0

    def unload(self):
        """卸载模型，释放资源"""
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None
        self.is_loaded = False
        logger.info("GAN 模型已卸载")
