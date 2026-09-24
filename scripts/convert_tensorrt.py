"""转换 TensorRT 引擎

将 GAN ONNX 模型转换为 TensorRT 引擎，大幅提升推理速度。
TensorRT FP16 在 Jetson 上通常比 ONNX Runtime 快 2-3 倍。

用法:
  # 转换所有风格的 TensorRT 引擎
  python scripts/convert_tensorrt.py --all

  # 只转换指定风格
  python scripts/convert_tensorrt.py --style 1

  # 转换指定模型文件
  python scripts/convert_tensorrt.py --model models/animegan_v2.onnx

  # 使用 FP32 精度
  python scripts/convert_tensorrt.py --all --fp32
"""

import os
import sys
import json
import argparse
import logging
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def convert_with_trtexec(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    io_dtype: str = "fp32",
    opt_level: int = 3,
    force: bool = False,
) -> bool:
    """
    使用 trtexec 命令行工具转换 ONNX 到 TensorRT

    Args:
        onnx_path: ONNX 模型路径
        engine_path: 输出 engine 路径
        fp16: 是否使用 FP16 精度（内部计算精度）
        io_dtype: 外部 I/O 数据类型 (fp32/fp16/uint8)
        opt_level: builderOptimizationLevel（默认 3，参考工程配方）
        force: 强制覆盖已有 engine

    Returns:
        是否成功
    """
    if os.path.exists(engine_path) and not force:
        logger.info(f"Engine 已存在，跳过: {engine_path}（用 --force 强制重建）")
        return True

    # 检查 trtexec 是否可用
    import shutil
    trtexec_path = shutil.which("trtexec")

    # Jetson 上常见路径
    if trtexec_path is None:
        jetson_paths = [
            "/usr/src/tensorrt/bin/trtexec",
            "/usr/local/tensorrt/bin/trtexec",
        ]
        for p in jetson_paths:
            if os.path.isfile(p) and os.access(p, os.X_OK):
                trtexec_path = p
                break

    if trtexec_path is None:
        logger.error("trtexec 未找到，请确保 TensorRT 已安装")
        logger.error("Jetson 上通常位于 /usr/src/tensorrt/bin/trtexec")
        return False

    logger.info(f"找到 trtexec: {trtexec_path}")

    # timing cache：同硬件重复构建时大幅缩短 builder 时间
    cache_path = os.path.splitext(engine_path)[0] + ".cache"

    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--memPoolSize=workspace:4096MiB",
        f"--builderOptimizationLevel={opt_level}",
        "--skipInference",
        f"--timingCacheFile={cache_path}",
    ]

    if fp16:
        cmd.append("--fp16")
        logger.info("使用 FP16 内部精度")
    else:
        logger.info("使用 FP32 内部精度")

    # 外部 I/O 格式：fp16/uint8 可大幅减少传输量
    if io_dtype != "fp32":
        fmt = f"{io_dtype}:chw"
        cmd.append(f"--inputIOFormats={fmt}")
        cmd.append(f"--outputIOFormats={fmt}")
        logger.info(f"外部 I/O 精度: {io_dtype} (chw)")

    logger.info(f"执行转换: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        # 打印 GPU Compute Time（两种计时口径之一，不可与主机侧 steady_infer_ms 混用）
        for line in (result.stdout or "").splitlines():
            if "GPU Compute Time" in line or "mean =" in line and "ms" in line:
                logger.info(f"  trtexec: {line.strip()}")

        if result.returncode != 0:
            logger.error(f"转换失败:\n{result.stderr}")
            return False

        if not os.path.exists(engine_path):
            logger.error("转换后的 engine 文件不存在")
            return False

        engine_size = os.path.getsize(engine_path)
        logger.info(f"转换成功: {engine_path} ({engine_size // (1024*1024)}MB)")
        return True

    except subprocess.TimeoutExpired:
        logger.error("转换超时（10分钟）")
        return False
    except Exception as e:
        logger.error(f"转换失败: {e}")
        return False


def convert_with_python(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
) -> bool:
    """
    使用 TensorRT Python API 转换

    Args:
        onnx_path: ONNX 模型路径
        engine_path: 输出 engine 路径
        fp16: 是否使用 FP16 精度

    Returns:
        是否成功
    """
    try:
        import sys
        import glob as g
        system_site = g.glob("/usr/lib/python3*/dist-packages")
        if system_site:
            sys.path.insert(0, system_site[0])
        import tensorrt as trt

        logger.info(f"TensorRT 版本: {trt.__version__}")

        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, trt_logger)

        logger.info(f"解析 ONNX 模型: {onnx_path}")
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(f"ONNX 解析错误: {parser.get_error(error)}")
                return False

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("启用 FP16 精度")

        try:
            config.set_tactic_sources(
                1 << int(trt.TacticSource.CUDNN)
                | 1 << int(trt.TacticSource.CUBLAS)
            )
            logger.info("启用 cuDNN + cuBLAS 战术源")
        except Exception as e:
            logger.warning(f"设置战术源失败: {e}")

        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            input_shape = list(input_tensor.shape)
            for j, dim in enumerate(input_shape):
                if dim is None or dim < 0:
                    input_shape[j] = 1
            input_shape = tuple(input_shape)
            logger.info(f"  输入 {i}: {input_tensor.name}, shape={input_shape}")
            profile.set_shape(
                input_tensor.name,
                min=input_shape,
                opt=input_shape,
                max=input_shape,
            )
        config.add_optimization_profile(profile)

        logger.info("开始构建 TensorRT engine（可能需要几分钟）...")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            logger.error("构建 TensorRT engine 失败")
            return False

        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)

        engine_size = os.path.getsize(engine_path)
        logger.info(f"转换成功: {engine_path} ({engine_size // (1024*1024)}MB)")
        return True

    except ImportError:
        logger.error("未安装 TensorRT Python 绑定")
        logger.error("请安装: pip install tensorrt 或使用 trtexec")
        return False
    except Exception as e:
        logger.error(f"Python API 转换失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_onnx_to_trt(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    io_dtype: str = "fp32",
    opt_level: int = 3,
    force: bool = False,
) -> bool:
    """
    将 ONNX 模型转换为 TensorRT engine

    优先使用 trtexec，回退到 Python API

    Args:
        onnx_path: ONNX 模型路径
        engine_path: 输出 engine 路径
        fp16: 是否使用 FP16 精度
        io_dtype: 外部 I/O 数据类型
        opt_level: builder 优化等级
        force: 强制覆盖已有 engine

    Returns:
        是否成功
    """
    if convert_with_trtexec(onnx_path, engine_path, fp16, io_dtype, opt_level, force):
        return True

    logger.info("trtexec 不可用，尝试使用 Python API...")
    return convert_with_python(onnx_path, engine_path, fp16)


def main():
    parser = argparse.ArgumentParser(description="转换 GAN 模型为 TensorRT 引擎")
    parser.add_argument("--all", action="store_true", help="转换所有风格的模型")
    parser.add_argument("--style", type=int, help="转换指定风格 ID 的模型")
    parser.add_argument("--model", type=str, help="转换指定的 ONNX 模型文件")
    parser.add_argument("--models", default="models", help="模型目录")
    parser.add_argument("--config", default="config/styles.json", help="风格配置文件")
    parser.add_argument("--output", default=None, help="输出目录（默认: models/trt/）")
    parser.add_argument("--fp32", action="store_true", help="使用 FP32 内部精度（默认: FP16）")
    parser.add_argument("--io-dtype", choices=["fp32", "fp16", "uint8"], default="fp32",
                       help="外部 I/O 数据类型（默认: fp32；fp16 可减半传输量）")
    parser.add_argument("--opt-level", type=int, default=3,
                       help="builderOptimizationLevel（默认: 3）")
    parser.add_argument("--force", action="store_true", help="强制覆盖已有 engine")

    args = parser.parse_args()

    fp16 = not args.fp32
    io_dtype = args.io_dtype
    output_dir = args.output or os.path.join(args.models, "trt")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # 引擎命名带分辨率/精度后缀，避免改 input_size 后静默复用旧引擎
    prec_tag = "fp16" if fp16 else "fp32"
    if io_dtype != "fp32":
        prec_tag += f"_io{io_dtype}"

    def _engine_name(style_id: int, w: int, h: int) -> str:
        return f"style_{style_id}_{w}x{h}_{prec_tag}.engine"

    if args.model:
        onnx_path = args.model
        if not os.path.isabs(onnx_path):
            onnx_path = os.path.join(project_dir, onnx_path)

        if not os.path.exists(onnx_path):
            logger.error(f"ONNX 模型不存在: {onnx_path}")
            return

        model_name = os.path.splitext(os.path.basename(onnx_path))[0]
        engine_path = os.path.join(output_dir, f"{model_name}_{prec_tag}.engine")

        logger.info(f"转换模型: {onnx_path}")
        success = convert_onnx_to_trt(
            onnx_path, engine_path, fp16, io_dtype, args.opt_level, args.force
        )

        if success:
            logger.info("转换完成!")
        else:
            logger.error("转换失败")
        return

    config_path = os.path.join(project_dir, args.config)
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    styles = config.get('styles', [])
    if not styles:
        logger.error("没有找到风格配置")
        return

    if args.style:
        styles = [s for s in styles if s['id'] == args.style]
        if not styles:
            logger.error(f"风格 {args.style} 不存在")
            return

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for style in styles:
        style_id = style['id']
        style_name = style['name']
        model_name = style['base_model']

        onnx_path = os.path.join(args.models, model_name)

        # 解析 input_size（支持 [W,H] 或标量）
        raw_size = style.get('input_size', 512)
        if isinstance(raw_size, list):
            w, h = raw_size[0], raw_size[1]
        else:
            w = h = raw_size

        engine_path = os.path.join(output_dir, _engine_name(style_id, w, h))

        logger.info(f"\n=== 风格 {style_id}: {style_name} ({w}x{h}) ===")
        logger.info(f"ONNX: {onnx_path}")
        logger.info(f"Engine: {engine_path}")

        if not os.path.exists(onnx_path):
            logger.warning(f"ONNX 模型不存在，跳过: {onnx_path}")
            results[style_id] = False
            continue

        success = convert_onnx_to_trt(
            onnx_path, engine_path, fp16, io_dtype, args.opt_level, args.force
        )
        results[style_id] = success

    print("\n" + "=" * 50)
    print("转换结果:")
    print("=" * 50)
    for style_id, success in results.items():
        style_name = next(s['name'] for s in styles if s['id'] == style_id)
        status = "成功" if success else "失败"
        print(f"  风格 {style_id} ({style_name}): {status}")

    all_success = all(results.values())
    if all_success:
        print("\n全部转换完成!")
    else:
        print("\n部分转换失败，请检查日志")


if __name__ == "__main__":
    main()
