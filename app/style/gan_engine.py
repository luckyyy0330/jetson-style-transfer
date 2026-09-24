"""GAN 风格迁移推理引擎

基于 ONNX Runtime / TensorRT 的轻量级 GAN 推理引擎。
支持 AnimeGANv2、CartoonGAN 等单次前向推理模型。

快路径 (style_cuda + TensorRT):
    submit() → release_input() → wait()   — CUDA Graph + 融合前后处理
    采集/推理/显示可流水线重叠

慢路径 (ONNX Runtime CPU):
    transfer()  — 串行 CPU 前后处理 + session.run

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

# 尝试加载 CUDA 扩展
_STYLE_CUDA = None
try:
    import sys
    _ext_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cuda_ext")
    if _ext_dir not in sys.path:
        sys.path.insert(0, _ext_dir)
    import style_cuda as _STYLE_CUDA
    logger.info("✓ style_cuda CUDA 扩展已加载")
except ImportError:
    logger.info("style_cuda 不可用，使用 CPU 前后处理")


class GANEngine:
    """GAN 风格迁移引擎

    支持 ONNX Runtime 和 TensorRT 后端
    每个风格对应独立的 ONNX 模型文件
    """

    def __init__(
        self,
        model_path: str,
        input_size: tuple = (512, 512),
        backend: str = "onnx",
        dtype: str = "float16",
        normalize: str = "minus_one_to_one",
        sharpen: float = 0.0,
    ):
        """
        初始化 GAN 引擎

        Args:
            model_path: ONNX 模型文件路径
            input_size: 输入图像尺寸 (width, height) 元组
            backend: 推理后端 ("onnx" 或 "tensorrt")
            dtype: 数据类型 (float16/float32)
            normalize: 归一化方式 ("minus_one_to_one" 或 "zero_to_one")
        """
        self.model_path = model_path
        self.input_width, self.input_height = input_size
        self.input_size = input_size  # 保留元组形式
        self.backend = backend
        self.dtype = dtype
        self.normalize = normalize
        self.sharpen = sharpen

        # 推理会话
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None
        self.is_loaded = False

        # CUDA 快路径
        self._gpu_pipe = None
        self._use_gpu_path = False
        self._t_release_acc = 0.0

        # 性能统计
        self.frame_count = 0
        self.total_inference_time = 0.0
        self._t_h2d_acc = 0.0
        self._t_exec_acc = 0.0
        self._t_d2h_acc = 0.0

    def load(self) -> bool:
        """
        加载模型

        Returns:
            是否成功
        """
        if not os.path.exists(self.model_path):
            logger.error(f"模型文件不存在: {self.model_path}")
            return False

        # TensorRT 引擎：使用原生 TensorRT Python API
        if self.backend == "tensorrt":
            return self._load_tensorrt_engine()

        # ONNX 模型：使用 ONNX Runtime
        return self._load_onnx_session()

    def _load_tensorrt_engine(self) -> bool:
        """使用原生 TensorRT API 加载 .engine 文件"""
        try:
            import tensorrt as trt
            import torch

            logger.info(f"加载 TensorRT 引擎: {self.model_path}")

            trt_logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(trt_logger)

            with open(self.model_path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())

            if engine is None:
                logger.error("反序列化 TensorRT 引擎失败")
                return False

            context = engine.create_execution_context()

            # 获取输入输出形状
            self.input_shape = tuple(engine.get_tensor_shape(engine.get_tensor_name(0)))
            self.output_shape = tuple(engine.get_tensor_shape(engine.get_tensor_name(1)))
            self.input_name = engine.get_tensor_name(0)
            self.output_name = engine.get_tensor_name(1)

            # 释放旧的 GPU 缓冲区（如果存在）
            if hasattr(self, "_device_input") and self._device_input is not None:
                del self._device_input
            if hasattr(self, "_device_output") and self._device_output is not None:
                del self._device_output

            # 用 PyTorch 分配 GPU 内存（替代 pycuda）
            self._trt_engine = engine
            self._trt_context = context
            self._torch_stream = torch.cuda.Stream()

            import numpy as np
            input_size = int(np.prod(self.input_shape))
            output_size = int(np.prod(self.output_shape))

            # 尝试 CUDA 快路径（style_cuda GpuPipeline）
            self._try_init_gpu_path(engine, context)

            if not self._use_gpu_path:
                # 传统 PyTorch GPU 缓冲区（回退路径）
                self._device_input = torch.empty(
                    input_size, dtype=torch.float32, device="cuda"
                )
                self._device_output = torch.empty(
                    output_size, dtype=torch.float32, device="cuda"
                )
                # tensor 地址在 load 时设一次（地址此后不变，不必逐帧设置）
                context.set_tensor_address(
                    self.input_name, self._device_input.data_ptr()
                )
                context.set_tensor_address(
                    self.output_name, self._device_output.data_ptr()
                )
                logger.info(f"  GPU 缓冲区: 输入={input_size}, 输出={output_size}")
            else:
                self._device_input = None
                self._device_output = None

            self.is_loaded = True
            logger.info(f"✓ TensorRT 引擎加载成功")
            logger.info(f"  输入: {self.input_shape}, 输出: {self.output_shape}")
            logger.info(f"  快路径: {'CUDA Graph + 融合前后处理' if self._use_gpu_path else 'PyTorch 传统路径'}")

            return True

        except ImportError as e:
            logger.error(f"缺少 TensorRT 依赖: {e}")
            logger.error("请确保已安装: sudo apt install python3-libnvinfer")
            return False
        except Exception as e:
            logger.error(f"加载 TensorRT 引擎失败: {e}")
            return False

    def _try_init_gpu_path(self, engine, context):
        """尝试初始化 CUDA 快路径（style_cuda GpuPipeline）"""
        if _STYLE_CUDA is None:
            return

        try:
            # 检查形状是否静态
            shape = self.input_shape
            if len(shape) != 4 or any(d <= 0 for d in shape):
                logger.info("  输入形状非静态，跳过 CUDA 快路径")
                return

            # NCHW: (1, 3, H, W)
            h, w = int(shape[2]), int(shape[3])
            norm_mode = 0 if self.normalize == "minus_one_to_one" else 1

            # I/O dtype 必须与引擎构建时的 --inputIOFormats/--outputIOFormats 一致，
            # 否则前后处理核写的 float/half 会被引擎按另一种解释 → **静默输出垃圾**。
            io_dtype = self._probe_io_dtype(engine)
            logger.info(f"  引擎 I/O dtype → GpuPipeline io_dtype={io_dtype}")

            self._gpu_pipe = _STYLE_CUDA.GpuPipeline(
                width=w, height=h,
                norm=norm_mode,
                io_dtype=io_dtype,
            )

            # 设置 TRT tensor 地址到 GpuPipeline 的 device 缓冲（图地址稳定，只此一次）
            self._trt_context.set_tensor_address(
                self.input_name, self._gpu_pipe.input_ptr()
            )
            self._trt_context.set_tensor_address(
                self.output_name, self._gpu_pipe.output_ptr()
            )

            self._use_gpu_path = True
            logger.info(f"  CUDA 快路径初始化成功: {w}x{h}")

        except Exception as e:
            logger.warning(f"  CUDA 快路径初始化失败: {e}，回退传统路径")
            self._gpu_pipe = None
            self._use_gpu_path = False

    @staticmethod
    def _probe_io_dtype(engine) -> str:
        """探测 TensorRT 引擎的 I/O 张量 dtype → 'fp32' | 'fp16'

        与 `convert_tensorrt.py --io-dtype` 的构建选项对应。
        探测失败时保守返回 'fp32'（默认构建路径）。
        """
        try:
            # 只看第一个 input 张量的 dtype
            if hasattr(engine, "num_io_tensors") and engine.num_io_tensors > 0:
                for i in range(engine.num_io_tensors):
                    n = engine.get_tensor_name(i)
                    try:
                        mode = engine.get_tensor_mode(n)
                    except Exception:
                        mode = None
                    # 只看 input
                    if mode is not None and "INPUT" not in str(mode).upper():
                        continue
                    dt = engine.get_tensor_dtype(n)
                    dt_str = str(dt).upper()
                    if "HALF" in dt_str or "FP16" in dt_str or "FLOAT16" in dt_str:
                        return "fp16"
                    return "fp32"
            return "fp32"
        except Exception as e:
            logger.info(f"  I/O dtype 探测失败 ({e})，默认 fp32")
            return "fp32"

    def _load_onnx_session(self) -> bool:
        """使用 ONNX Runtime 加载 .onnx 文件"""
        try:
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]

            # 配置会话选项
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            sess_options.intra_op_num_threads = 2
            sess_options.inter_op_num_threads = 1

            logger.info(f"加载 ONNX 模型: {self.model_path}")

            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers,
            )

            # 获取输入输出信息
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_shape = self.session.get_outputs()[0].shape

            logger.info("✓ 使用 CPU 推理")

            self.is_loaded = True
            logger.info(
                f"模型加载完成: 输入={self.input_shape}, 输出={self.output_shape}"
            )
            return True

        except ImportError as e:
            logger.error(f"缺少依赖: {e}")
            logger.error("请安装: pip install onnxruntime")
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
            (self.input_width, self.input_height),
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

    def copy_source(self, frame: np.ndarray) -> bool:
        """
        拷贝源帧到 mapped staging（GPU 快路径）。

        可在采集线程或推理线程调用。staging 是单缓冲，预处理核读它期间不可写；
        `release_input()` 是唯一的复用许可。

        Args:
            frame: 像素缓冲（BGR / NV12 / BGRx），会被整体 memcpy 到 staging

        Returns:
            是否成功（False = 快路径不可用）
        """
        if not (self._use_gpu_path and self._gpu_pipe is not None):
            return False
        self._gpu_pipe.copy_source(frame)
        return True

    def submit(
        self,
        src_w: int,
        src_h: int,
        src_pitch: int,
        fmt: int = 0,
        matrix: int = 0,
        full_range: bool = False,
    ) -> float:
        """
        Submit：set_source + run（融合预处理 → TRT → 融合后处理，整条可被 Graph 捕获）。

        **staging 中必须已有该帧数据**（先调 `copy_source`）。
        `set_source` 只能在 idle 时调用（上一帧 `wait()` 完成之后）——
        `CudaRequest::idle()` 会在 pending 时抛异常。

        Args:
            src_w/src_h: 源帧有效像素尺寸
            src_pitch: 源帧每行字节数（可能含对齐填充）
            fmt: PixelFormat (0=BGR8, 1=RGB8, 2=NV12, 3=BGRX8, 4=YUYV)
            matrix: 色彩矩阵 (0=BT601, 1=BT709)
            full_range: 是否 full-range YUV

        Returns:
            提交耗时 (ms)
        """
        if not (self._use_gpu_path and self._gpu_pipe is not None):
            raise RuntimeError("快路径不可用，请使用 transfer()")

        t0 = time.perf_counter()
        self._gpu_pipe.set_source(src_w, src_h, src_pitch, fmt, matrix, full_range)

        def trt_enqueue():
            self._trt_context.execute_async_v3(
                stream_handle=self._gpu_pipe.stream_handle()
            )

        self._gpu_pipe.run(trt_enqueue)
        t1 = time.perf_counter()
        return (t1 - t0) * 1000

    def release_input(self) -> float:
        """
        提前归还输入缓冲（TRT 还在跑时）。

        只 `cudaEventSynchronize(inputRead)`——等融合预处理核读完 staging 即返回，
        之后采集线程即可写入下一帧。
        """
        if self._use_gpu_path and self._gpu_pipe is not None:
            t0 = time.perf_counter()
            self._gpu_pipe.release_input()
            t1 = time.perf_counter()
            ms = (t1 - t0) * 1000
            self._t_release_acc += ms
            return ms
        return 0.0

    def wait(self) -> Tuple[np.ndarray, float]:
        """
        等待推理完成，返回风格化图像。

        Returns:
            (styled_bgr, wait_time_ms)

        `styled_bgr` 是**已持有**的数组（C++ 侧已 memcpy 脱离 mapped 内存），
        可安全投递到显示队列 / 用于 strength 第二前向。
        """
        if self._use_gpu_path and self._gpu_pipe is not None:
            t0 = time.perf_counter()
            styled = self._gpu_pipe.wait()  # owned BGR uint8
            t1 = time.perf_counter()
            return styled, (t1 - t0) * 1000
        else:
            raise RuntimeError("快路径不可用，请使用 transfer()")

    def quiesce(self):
        """
        静默引擎：确保无 pending 请求并丢弃 CUDA Graph。

        换风格 / 换分辨率前**必须**调用（在推理线程内），否则旧 graph
        引用的 kernel 地址和 trt_enqueue 闭包会失效。
        """
        if self._gpu_pipe is not None:
            try:
                # 若有 pending 请求，等它结束
                self._gpu_pipe.wait()
            except Exception:
                pass
            self._gpu_pipe.set_graph(False)  # idle() + resetGraph()
        self._t_h2d_acc = 0.0
        self._t_exec_acc = 0.0
        self._t_d2h_acc = 0.0

    @property
    def use_gpu_path(self) -> bool:
        """是否走 CUDA Graph + 融合前后处理快路径"""
        return self._use_gpu_path and self._gpu_pipe is not None

    @property
    def graph_active(self) -> bool:
        """CUDA Graph 是否已实例化（False = stream 回退或未捕获）"""
        return self._use_gpu_path and self._gpu_pipe is not None and self._gpu_pipe.graph_active()

    def transfer(
        self,
        content_image: np.ndarray,
        input_size: Optional[int] = None,
        strength: float = 1.0,
    ) -> Tuple[np.ndarray, float]:
        """
        执行风格迁移（串行模式，快路径和慢路径通用）

        Args:
            content_image: 输入图像 (BGR 格式)
            input_size: 覆盖默认输入尺寸
            strength: 风格强度 (1.0=标准，>1 加强，<1 减弱，支持小数如1.5)

        Returns:
            (styled_image, inference_time) — styled_image 是**已持有**的数组
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载")

        # 快路径：copy_source + submit + release_input + wait
        if self._use_gpu_path and self._gpu_pipe is not None:
            return self._transfer_gpu(content_image, strength)

        # 慢路径：CPU 前后处理 + 传统推理
        return self._transfer_cpu(content_image, strength)

    def _transfer_gpu(self, content_image: np.ndarray, strength: float) -> Tuple[np.ndarray, float]:
        """GPU 快路径：CUDA Graph + 融合前后处理"""
        t0 = time.perf_counter()
        try:
            orig_h, orig_w = content_image.shape[:2]
            strength = max(0.1, strength)

            # 源帧元信息（BGR HWC；stride 可能含 padding）
            src_h, src_w = content_image.shape[:2]
            src_pitch = content_image.strides[0] if content_image.strides else src_w * 3

            # Submit + release_input + wait
            self.copy_source(content_image)
            self.submit(src_w, src_h, src_pitch, 0, 0, False)  # BGR8, BT601, limited
            self.release_input()
            styled, wait_ms = self.wait()  # owned 数组，已脱离 mapped 内存

            # strength 混合
            if strength > 1.0:
                self.copy_source(styled)
                self.submit(src_w, src_h, src_pitch, 0, 0, False)
                self.release_input()
                styled2, _ = self.wait()
                alpha = strength - 1.0
                styled = cv2.addWeighted(styled, 1.0 - alpha, styled2, alpha, 0)
            elif strength < 1.0:
                styled = cv2.addWeighted(content_image, 1.0 - strength, styled, strength, 0)

            # resize 回原始尺寸（同尺寸跳过）
            if styled.shape[1] != orig_w or styled.shape[0] != orig_h:
                styled = cv2.resize(styled, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            if self.sharpen > 0:
                blurred = cv2.GaussianBlur(styled, (0, 0), 2)
                styled = cv2.addWeighted(styled, 1.0 + self.sharpen, blurred, -self.sharpen, 0)

            t1 = time.perf_counter()
            inference_time = t1 - t0

            self.frame_count += 1
            self.total_inference_time += inference_time
            if self.frame_count % 60 == 0:
                total_ms = inference_time * 1000
                fps = 1000 / total_ms if total_ms > 0 else 0
                graph = 1 if self.graph_active else 0
                release_ms = self._t_release_acc / 60
                self._t_release_acc = 0.0
                logger.info(
                    f"[性能] graph:{graph} | release_input:{release_ms:.2f}ms | "
                    f"总耗时:{total_ms:.1f}ms | FPS:{fps:.1f}"
                )

            return styled, inference_time

        except Exception as e:
            logger.error(f"GPU 快路径失败: {e}")
            inference_time = time.perf_counter() - t0
            return content_image, inference_time

    def _transfer_cpu(self, content_image: np.ndarray, strength: float) -> Tuple[np.ndarray, float]:
        """CPU 慢路径：numpy/cv2 前后处理 + 传统推理"""
        t0 = time.perf_counter()

        try:
            # 保存原始尺寸
            orig_h, orig_w = content_image.shape[:2]

            strength = max(0.1, strength)

            # 第一轮：完整推理
            input_data = self._preprocess(content_image)
            t1 = time.perf_counter()

            if self.backend == "tensorrt":
                output = self._infer_tensorrt(input_data)
            else:
                output = self.session.run(
                    [self.output_name],
                    {self.input_name: input_data},
                )[0]
            t2 = time.perf_counter()

            styled = self._postprocess(output)
            t3 = time.perf_counter()

            if strength > 1.0:
                input_data2 = self._preprocess(styled)
                if self.backend == "tensorrt":
                    output2 = self._infer_tensorrt(input_data2)
                else:
                    output2 = self.session.run(
                        [self.output_name],
                        {self.input_name: input_data2},
                    )[0]
                styled2 = self._postprocess(output2)
                alpha = strength - 1.0
                styled = cv2.addWeighted(styled, 1.0 - alpha, styled2, alpha, 0)

            elif strength < 1.0:
                styled = cv2.addWeighted(content_image, 1.0 - strength, styled, strength, 0)

            # resize 回原始尺寸（同尺寸时跳过）
            if styled.shape[1] != orig_w or styled.shape[0] != orig_h:
                styled = cv2.resize(styled, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            if self.sharpen > 0:
                blurred = cv2.GaussianBlur(styled, (0, 0), 2)
                styled = cv2.addWeighted(styled, 1.0 + self.sharpen, blurred, -self.sharpen, 0)

            t4 = time.perf_counter()
            inference_time = t4 - t0

            # 详细计时日志（每60帧打印一次）
            self.frame_count += 1
            self.total_inference_time += inference_time
            if self.frame_count % 60 == 0:
                preprocess_ms = (t1 - t0) * 1000
                infer_ms = (t2 - t1) * 1000
                postprocess_ms = (t3 - t2) * 1000
                resize_ms = (t4 - t3) * 1000
                total_ms = inference_time * 1000
                fps = 1000 / total_ms if total_ms > 0 else 0

                n = 60
                h2d_ms = self._t_h2d_acc / n
                exec_ms = self._t_exec_acc / n
                d2h_ms = self._t_d2h_acc / n
                self._t_h2d_acc = 0.0
                self._t_exec_acc = 0.0
                self._t_d2h_acc = 0.0

                logger.info(
                    f"[性能] H2D:{h2d_ms:.1f}ms | GPU计算:{exec_ms:.1f}ms | D2H:{d2h_ms:.1f}ms | "
                    f"预处理:{preprocess_ms:.1f}ms | 推理:{infer_ms:.1f}ms | "
                    f"后处理:{postprocess_ms:.1f}ms | 缩放:{resize_ms:.1f}ms | "
                    f"总耗时:{total_ms:.1f}ms | FPS:{fps:.1f}"
                )

            return styled, inference_time

        except Exception as e:
            logger.error(f"风格迁移失败: {e}")
            inference_time = time.perf_counter() - t0
            return content_image, inference_time

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "model_path": self.model_path,
            "input_size": f"{self.input_width}x{self.input_height}",
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

    def _infer_tensorrt(self, input_data: np.ndarray) -> np.ndarray:
        """使用 TensorRT 原生 API 执行推理（**慢路径回退**，style_cuda 不可用时）

        快路径见 `copy_source/submit/release_input/wait`（CUDA Graph + 融合前后处理）。
        计时拆分：t_h2d / t_exec / t_d2h，便于识别传输税 vs 纯 GPU 计算。
        """
        import torch

        # 保存本地引用，防止 unload 并发置 None
        device_in = self._device_input
        device_out = self._device_output
        if device_in is None or device_out is None:
            raise RuntimeError("模型已卸载")

        # 安全检查：确保输入尺寸与 GPU 缓冲区匹配
        if int(input_data.size) != device_in.numel():
            raise RuntimeError(
                f"输入数据大小 ({input_data.size}) 与 GPU 缓冲区大小 "
                f"({device_in.numel()}) 不匹配。"
                f"引擎期望输入 {self.input_shape}，请检查 config/styles.json 中的 "
                f"input_size 是否与引擎一致。"
            )

        # --- H2D ---
        # 一次到位：host numpy → device 缓冲。
        # 旧实现先 .to("cuda") 再 device_in.copy_，多做一次 GPU 内 D2D；
        # 且 from_numpy 的内存未 pinned，non_blocking=True 实际是同步的。
        t_h2d0 = time.perf_counter()
        device_in.copy_(torch.from_numpy(input_data).reshape(-1))
        t_h2d1 = time.perf_counter()

        # tensor 地址已在 load 时设置（地址稳定，不必逐帧设置）

        # --- 执行推理 ---
        t_exec0 = time.perf_counter()
        self._trt_context.execute_async_v3(
            stream_handle=self._torch_stream.cuda_stream
        )
        self._torch_stream.synchronize()
        t_exec1 = time.perf_counter()

        # --- D2H ---
        t_d2h0 = time.perf_counter()
        output = device_out.cpu().numpy().reshape(self.output_shape)
        t_d2h1 = time.perf_counter()

        # 累积细粒度计时（每 60 帧汇总一次）
        self._t_h2d_acc += (t_h2d1 - t_h2d0) * 1000
        self._t_exec_acc += (t_exec1 - t_exec0) * 1000
        self._t_d2h_acc += (t_d2h1 - t_d2h0) * 1000

        return output

    def unload(self):
        """卸载模型，释放资源"""
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None
        self.is_loaded = False

        # 释放 CUDA 快路径
        self._gpu_pipe = None
        self._use_gpu_path = False

        # 释放 TensorRT 资源
        self._trt_engine = None
        self._trt_context = None
        self._torch_stream = None
        self._device_input = None
        self._device_output = None

        logger.info("GAN 模型已卸载")
