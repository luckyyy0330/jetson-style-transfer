"""将 AnimeGAN2 PyTorch .pt 权重转换为 ONNX 格式

用法:
  python scripts/convert_to_onnx.py
  python scripts/convert_to_onnx.py --input models/celeba_distill.pt --output models/celeba_distill.onnx
"""

import os
import sys
import argparse
import torch
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AnimeGenerator(torch.nn.Module):
    """AnimeGAN2 生成器（简化版，用于推理）"""

    def __init__(self):
        super().__init__()
        # AnimeGAN2 使用 U-Net 风格的生成器
        # 具体结构在 torch.hub 加载时自动构建

    def forward(self, x):
        return x


def load_animegan2_model(pt_path: str):
    """加载 AnimeGAN2 PyTorch 模型"""
    # 尝试 torch.hub 加载
    model_name = os.path.basename(pt_path).replace(".pt", "")

    style_map = {
        "celeba_distill": "celeba_distill",
        "face_paint_512_v2": "face_paint_512_v2",
        "paprika": "paprika",
    }

    if model_name in style_map:
        print(f"通过 torch.hub 加载模型: {model_name}")
        model = torch.hub.load(
            "bryandlee/animegan2-pytorch:main",
            "generator",
            pretrained=model_name,
        )
        return model

    # 直接加载 .pt 文件
    print(f"直接加载 .pt 文件: {pt_path}")
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)

    # 检查 checkpoint 格式
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    return state_dict


def convert_to_onnx(model, output_path: str, input_size: int = 512):
    """将 PyTorch 模型转换为 ONNX"""
    model.eval()

    # 创建 dummy input (B, C, H, W)
    dummy_input = torch.randn(1, 3, input_size, input_size)

    print(f"转换为 ONNX (输入尺寸: {input_size}x{input_size})...")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    file_size = os.path.getsize(output_path) // 1024
    print(f"转换完成: {output_path} ({file_size}KB)")


def convert_state_dict_to_onnx(state_dict, output_path: str, input_size: int = 512):
    """如果只是 state_dict，提示用户需要完整模型"""
    print("错误: 只找到 state_dict（权重字典），不是完整的模型。")
    print("请使用 torch.hub 下载完整模型，或提供包含完整模型的 .pt 文件。")
    return False


def main():
    parser = argparse.ArgumentParser(description="将 AnimeGAN2 .pt 转为 ONNX")
    parser.add_argument(
        "--input", "-i", help="输入 .pt 文件路径（不指定则转换所有）"
    )
    parser.add_argument("--output", "-o", help="输出 .onnx 文件路径")
    parser.add_argument(
        "--models", default="models", help="模型目录（默认 models）"
    )
    parser.add_argument(
        "--input-size", type=int, default=384, help="ONNX 输入尺寸 (默认: 384)"
    )
    args = parser.parse_args()

    models_dir = args.models

    if args.input:
        # 转换指定文件
        output = args.output or args.input.replace(".pt", ".onnx")
        model = load_animegan2_model(args.input)
        if isinstance(model, dict):
            convert_state_dict_to_onnx(model, output)
        else:
            convert_to_onnx(model, output, input_size=args.input_size)
    else:
        # 转换目录下所有 .pt 文件
        pt_files = [f for f in os.listdir(models_dir) if f.endswith(".pt")]
        if not pt_files:
            print(f"在 {models_dir}/ 下没有找到 .pt 文件")
            return

        print(f"找到 {len(pt_files)} 个 .pt 文件:")
        for f in pt_files:
            print(f"  - {f}")
        print()

        for pt_file in pt_files:
            pt_path = os.path.join(models_dir, pt_file)
            onnx_path = os.path.join(models_dir, pt_file.replace(".pt", ".onnx"))

            if os.path.exists(onnx_path):
                print(f"跳过（已存在）: {onnx_path}")
                continue

            print(f"\n转换: {pt_file} -> {os.path.basename(onnx_path)}")
            try:
                model = load_animegan2_model(pt_path)
                if isinstance(model, dict):
                    convert_state_dict_to_onnx(model, onnx_path)
                else:
                    convert_to_onnx(model, onnx_path, input_size=args.input_size)
            except Exception as e:
                print(f"转换失败: {e}")

    print("\n下一步:")
    print("  python -m app.main --backend onnx")


if __name__ == "__main__":
    main()
