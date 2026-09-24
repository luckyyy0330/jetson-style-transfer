"""审计 ONNX 模型和 TensorRT 引擎的 I/O 形状

验证 config/styles.json 的 input_size 与模型实际静态形状是否一致。
Phase 0 形状审计步骤。

用法:
    python3 scripts/audit_shapes.py
    python3 scripts/audit_shapes.py --model models/animegan_v2.onnx
"""

import os
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def audit_onnx(path: str):
    """打印 ONNX 模型的 I/O 形状"""
    try:
        import onnx
        m = onnx.load(path)
        print(f"\n=== ONNX: {path} ===")
        print(f"  opset: {m.opset_import[0].version if m.opset_import else '?'}")
        for inp in m.graph.input:
            dims = [d.dim_value if d.dim_value else d.dim_param or '?' for d in inp.type.tensor_type.shape.dim]
            print(f"  输入 {inp.name}: {dims}")
        for out in m.graph.output:
            dims = [d.dim_value if d.dim_value else d.dim_param or '?' for d in out.type.tensor_type.shape.dim]
            print(f"  输出 {out.name}: {dims}")
        return True
    except ImportError:
        logger.warning("onnx 未安装，跳过 ONNX 审计")
        return False
    except Exception as e:
        logger.error(f"ONNX 审计失败 {path}: {e}")
        return False


def audit_engine(path: str):
    """打印 TensorRT 引擎的 I/O 形状"""
    try:
        import tensorrt as trt
        logger.info(f"TensorRT 版本: {trt.__version__}")
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            logger.error(f"反序列化失败: {path}")
            return False

        print(f"\n=== TRT Engine: {path} ===")
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = tuple(engine.get_tensor_shape(name))
            dtype = engine.get_tensor_dtype(name)
            mode = engine.get_tensor_mode(name)
            print(f"  {'输入' if mode == trt.TensorIOMode.INPUT else '输出'} {name}: {shape} {dtype}")
        return True
    except ImportError:
        logger.warning("tensorrt 未安装，跳过 engine 审计")
        return False
    except Exception as e:
        logger.error(f"Engine 审计失败 {path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="审计模型 I/O 形状")
    parser.add_argument("--model", help="指定单个模型文件")
    parser.add_argument("--models", default="models", help="模型目录")
    parser.add_argument("--config", default="config/styles.json", help="风格配置")

    args = parser.parse_args()

    if args.model:
        if args.model.endswith(".onnx"):
            audit_onnx(args.model)
        elif args.model.endswith(".engine"):
            audit_engine(args.model)
        return

    # 读取配置，对照检查
    import json
    config_path = args.config
    styles = []
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            styles = json.load(f).get("styles", [])

    print("=" * 60)
    print("Phase 0 形状审计")
    print("=" * 60)

    for style in styles:
        sid = style["id"]
        name = style["name"]
        raw = style.get("input_size", 512)
        if isinstance(raw, list):
            w, h = raw[0], raw[1]
        else:
            w = h = raw
        print(f"\n风格 {sid} ({name}): config input_size = [{w}, {h}] (W,H)")
        print(f"  期望 ONNX shape = (1, 3, {h}, {w})  [N,C,H,W]")

        onnx_path = os.path.join(args.models, style["base_model"])
        if os.path.exists(onnx_path):
            audit_onnx(onnx_path)
        else:
            logger.warning(f"  ONNX 不存在: {onnx_path}")

        # 搜索对应 engine
        trt_dir = os.path.join(args.models, "trt")
        if os.path.isdir(trt_dir):
            import glob
            engines = glob.glob(os.path.join(trt_dir, f"style_{sid}*.engine"))
            for ep in engines:
                audit_engine(ep)

    print("\n" + "=" * 60)
    print("审计完成。检查 ONNX shape 是否为 (1, 3, H, W) 且 H/W 与 config 一致。")
    print("注意：ONNX 是 NCHW，config input_size 是 [W, H]。")


if __name__ == "__main__":
    main()
