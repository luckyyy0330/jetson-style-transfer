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
import glob
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
) -> bool:
    """
    使用 trtexec 命令行工具转换 ONNX 到 TensorRT

    Args:
        onnx_path: ONNX 模型路径
        engine_path: 输出 engine 路径
        fp16: 是否使用 FP16 精度

    Returns:
        是否成功
    """
    # 检查 trtexec 是否可用
    try:
        result = subprocess.run(
            ["which", "trtexec"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            # Windows 上尝试 where
            result = subprocess.run(
                ["where", "trtexec"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                logger.error("trtexec 未找到，请确保 TensorRT 已安装")
                logger.error("Jetson 上通常位于 /usr/src/tensorrt/bin/trtexec")
                return False
    except FileNotFoundError:
        logger.error("trtexec 未找到，请确保 TensorRT 已安装")
        return False

    # 构建命令
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--workspace=2048",  # 2GB workspace
    ]

    if fp16:
        cmd.append("--fp16")
        logger.info("使用 FP16 精度")
    else:
        logger.info("使用 FP32 精度")

    logger.info(f"执行转换: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 分钟超时
        )

        if result.returncode != 0:
            logger.error(f"转换失败:\n{result.stderr}")
            return False

        # 验证输出文件
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
        import tensorrt as trt

        logger.info(f"TensorRT 版本: {trt.__version__}")

        # 创建 builder
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, trt_logger)

        # 解析 ONNX 模型
        logger.info(f"解析 ONNX 模型: {onnx_path}")
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(f"ONNX 解析错误: {parser.get_error(error)}")
                return False

        # 配置 builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("启用 FP16 精度")

        # 设置动态输入尺寸
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        input_shape = input_tensor.shape

        # 支持动态和静态形状
        if all(d > 0 for d in input_shape):
            # 静态形状
            profile.set_shape(
                input_tensor.name,
                min=input_shape,
                opt=input_shape,
                max=input_shape,
            )
        else:
            # 动态形状 - 使用 512x512 作为默认
            min_shape = (1, 3, 256, 256)
            opt_shape = (1, 3, 512, 512)
            max_shape = (1, 3, 1024, 1024)
            profile.set_shape(
                input_tensor.name,
                min=min_shape,
                opt=opt_shape,
                max=max_shape,
            )

        config.add_optimization_profile(profile)

        # 构建 engine
        logger.info("开始构建 TensorRT engine（可能需要几分钟）...")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            logger.error("构建 TensorRT engine 失败")
            return False

        # 保存 engine
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
        logger.error(f"转换失败: {e}")
        return False


def convert_onnx_to_trt(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
) -> bool:
    """
    将 ONNX 模型转换为 TensorRT engine

    优先使用 trtexec，如果不可用则使用 Python API

    Args:
        onnx_path: ONNX 模型路径
        engine_path: 输出 engine 路径
        fp16: 是否使用 FP16 精度

    Returns:
        是否成功
    """
    # 优先使用 trtexec
    if convert_with_trtexec(onnx_path, engine_path, fp16):
        return True

    # 回退到 Python API
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
    parser.add_argument("--fp32", action="store_true", help="使用 FP32 精度（默认: FP16）")

    args = parser.parse_args()

    fp16 = not args.fp32
    output_dir = args.output or os.path.join(args.models, "trt")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    if args.model:
        # 转换指定模型
        onnx_path = args.model
        if not os.path.isabs(onnx_path):
            onnx_path = os.path.join(project_dir, onnx_path)

        if not os.path.exists(onnx_path):
            logger.error(f"ONNX 模型不存在: {onnx_path}")
            return

        model_name = os.path.splitext(os.path.basename(onnx_path))[0]
        engine_path = os.path.join(output_dir, f"{model_name}.engine")

        logger.info(f"转换模型: {onnx_path}")
        success = convert_onnx_to_trt(onnx_path, engine_path, fp16)

        if success:
            logger.info("转换完成!")
        else:
            logger.error("转换失败")
        return

    # 从配置文件加载风格
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

    # 筛选要转换的风格
    if args.style:
        styles = [s for s in styles if s['id'] == args.style]
        if not styles:
            logger.error(f"风格 {args.style} 不存在")
            return

    # 转换每个风格
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for style in styles:
        style_id = style['id']
        style_name = style['name']
        model_name = style['base_model']

        onnx_path = os.path.join(args.models, model_name)
        engine_path = os.path.join(output_dir, f"style_{style_id}.engine")

        logger.info(f"\n=== 风格 {style_id}: {style_name} ===")
        logger.info(f"ONNX: {onnx_path}")

        if not os.path.exists(onnx_path):
            logger.warning(f"ONNX 模型不存在，跳过: {onnx_path}")
            results[style_id] = False
            continue

        # 检查 engine 是否已存在
        if os.path.exists(engine_path):
            logger.info(f"Engine 已存在，跳过: {engine_path}")
            results[style_id] = True
            continue

        success = convert_onnx_to_trt(onnx_path, engine_path, fp16)
        results[style_id] = success

    # 输出结果汇总
    print("\n" + "=" * 50)
    print("转换结果:")
    print("=" * 50)
    for style_id, success in results.items():
        style_name = next(s['name'] for s in styles if s['id'] == style_id)
        status = "✓ 成功" if success else "✗ 失败"
        print(f"  风格 {style_id} ({style_name}): {status}")

    all_success = all(results.values())
    if all_success:
        print("\n全部转换完成!")
    else:
        print("\n部分转换失败，请检查日志")


if __name__ == "__main__":
    main()
