"""TensorRT 推理工具

支持 TensorRT 引擎的加载和推理
用于 Jetson 上的模型加速
"""

import os
import logging
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class TensorRTInference:
    """TensorRT 推理引擎"""

    def __init__(
        self,
        engine_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 512, 512),
        dtype: str = "float16"
    ):
        """
        初始化 TensorRT 推理引擎

        Args:
            engine_path: .engine 文件路径
            input_shape: 输入形状 (batch, channels, height, width)
            dtype: 数据类型 (float16/float32)
        """
        self.engine_path = engine_path
        self.input_shape = input_shape
        self.dtype = dtype

        self.engine = None
        self.context = None
        self.stream = None
        self.bindings = []
        self.inputs = []
        self.outputs = []

        self._loaded = False

    def load(self) -> bool:
        """
        加载 TensorRT 引擎

        Returns:
            是否成功加载
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit

            logger.info(f"加载 TensorRT 引擎: {self.engine_path}")

            # 创建 logger 和 runtime
            trt_logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(trt_logger)

            # 读取引擎文件
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()
            self.engine = runtime.deserialize_cuda_engine(engine_data)

            if self.engine is None:
                logger.error("TensorRT 引擎加载失败")
                return False

            # 创建执行上下文
            self.context = self.engine.create_execution_context()
            if self.context is None:
                logger.error("TensorRT 上下文创建失败")
                return False

            # 分配 GPU 内存
            self.stream = cuda.Stream()
            self._allocate_buffers()

            self._loaded = True
            logger.info("TensorRT 引擎加载成功")
            return True

        except ImportError as e:
            logger.error(f"缺少 TensorRT 依赖: {e}")
            logger.error("请安装: sudo apt install tensorrt python3-libnvinfer")
            return False
        except Exception as e:
            logger.error(f"加载 TensorRT 引擎失败: {e}")
            return False

    def _allocate_buffers(self):
        """分配 GPU 缓冲区"""
        import pycuda.driver as cuda

        self.bindings = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)

            # 计算缓冲区大小
            size = int(np.prod(shape))
            dtype_size = np.dtype(dtype).itemsize
            buffer_size = size * dtype_size

            # 分配主机和设备内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(buffer_size)

            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    'name': name,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape,
                    'dtype': dtype
                })
            else:
                self.outputs.append({
                    'name': name,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape,
                    'dtype': dtype
                })

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        执行推理

        Args:
            input_data: 输入数据 (NCHW 格式)

        Returns:
            输出数据
        """
        if not self._loaded:
            raise RuntimeError("TensorRT 引擎未加载")

        import pycuda.driver as cuda

        # 预处理输入
        input_data = input_data.astype(self.inputs[0]['dtype'])
        np.copyto(self.inputs[0]['host'], input_data.ravel())

        # 传输到 GPU
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)

        # 执行推理
        for i, inp in enumerate(self.inputs):
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for i, out in enumerate(self.outputs):
            self.context.set_tensor_address(out['name'], int(out['device']))

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # 传输结果回主机
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        # 同步
        self.stream.synchronize()

        # 返回输出
        return self.outputs[0]['host'].reshape(self.outputs[0]['shape'])

    def unload(self):
        """卸载引擎，释放资源"""
        self.engine = None
        self.context = None
        self.stream = None
        self.bindings = []
        self.inputs = []
        self.outputs = []
        self._loaded = False

        logger.info("TensorRT 引擎已卸载")

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def check_tensorrt_available() -> bool:
    """检查 TensorRT 是否可用"""
    try:
        import tensorrt
        logger.info(f"TensorRT 版本: {tensorrt.__version__}")
        return True
    except ImportError:
        return False


def get_tensorrt_version() -> Optional[str]:
    """获取 TensorRT 版本"""
    try:
        import tensorrt
        return tensorrt.__version__
    except ImportError:
        return None
