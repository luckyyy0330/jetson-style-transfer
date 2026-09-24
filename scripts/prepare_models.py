"""一键准备模型：下载 PyTorch 权重并转换为 ONNX

在 Windows 上运行：
  pip install torch torchvision
  python scripts/prepare_models.py

会自动下载模型并转换为 ONNX 格式，放到 models/ 目录下。
"""

import os
import sys
import torch

# 模型配置：名称 -> (torch.hub 预训练名, 输出文件名)
MODELS = {
    "animegan_v2": "celeba_distill",
    "animegan_v2_paprika": "paprika",
    "animegan_v2_face_paint": "face_paint_512_v2",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")


def download_and_convert(output_name: str, pretrained_name: str, input_size: tuple = (384, 384)) -> bool:
    """下载模型并转换为 ONNX

    Args:
        output_name: 输出文件名（不含扩展名）
        pretrained_name: torch.hub 预训练模型名
        input_size: 输入尺寸 (width, height) 元组
    """
    onnx_path = os.path.join(MODELS_DIR, f"{output_name}.onnx")

    # 始终重新导出，确保内嵌权重（不生成 .data 文件）
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
    # 清理可能存在的外部数据文件
    data_file = onnx_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)

    print(f"\n{'='*50}")
    print(f"下载模型: {pretrained_name}")
    print(f"输出文件: {output_name}.onnx")
    print(f"{'='*50}")

    try:
        # 通过 torch.hub 下载模型
        print("  从 GitHub 下载权重...")
        model = torch.hub.load(
            "bryandlee/animegan2-pytorch:main",
            "generator",
            pretrained=pretrained_name,
        )
        model.eval()
        print("  [OK] 权重下载完成")

        # 转换为 ONNX（使用 legacy exporter，确保权重内嵌）
        input_width, input_height = input_size
        print(f"  转换为 ONNX 格式 (输入尺寸: {input_width}x{input_height})...")
        dummy_input = torch.randn(1, 3, input_height, input_width)

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamo=False,  # 使用 legacy exporter，避免生成 .data 文件
        )

        file_size = os.path.getsize(onnx_path) // 1024
        print(f"  [OK] 转换完成: {onnx_path} ({file_size}KB)")

        # 验证没有生成外部数据文件
        if os.path.exists(data_file):
            print(f"  [WARN] 生成了外部数据文件 {data_file}，TensorRT 可能无法处理")
        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="下载 AnimeGAN2 模型并转换为 ONNX")
    parser.add_argument("--input-size", type=int, default=None, help="ONNX 输入尺寸（正方形简写，默认: 384）")
    parser.add_argument("--input-width", type=int, default=None, help="ONNX 输入宽度")
    parser.add_argument("--input-height", type=int, default=None, help="ONNX 输入高度")
    args = parser.parse_args()

    # 解析输入尺寸：优先使用 --input-width/--input-height，否则用 --input-size
    if args.input_width is not None and args.input_height is not None:
        input_size = (args.input_width, args.input_height)
    elif args.input_size is not None:
        input_size = (args.input_size, args.input_size)
    else:
        input_size = (384, 384)

    print("AnimeGAN2 模型准备工具")
    print(f"模型目录: {MODELS_DIR}")
    print(f"输入尺寸: {input_size[0]}x{input_size[1]}\n")

    os.makedirs(MODELS_DIR, exist_ok=True)

    success_count = 0
    for output_name, pretrained_name in MODELS.items():
        if download_and_convert(output_name, pretrained_name, input_size=input_size):
            success_count += 1

    print(f"\n{'='*50}")
    print(f"完成: {success_count}/{len(MODELS)} 个模型")
    print(f"{'='*50}")

    if success_count > 0:
        print("\n下一步:")
        print("  1. 将 models/ 目录传到 Jetson:")
        print("     scp -r models/* nvidia@<IP>:~/jetson-style-transfer/models/")
        print("  2. 在 Jetson 上转换 TensorRT engine:")
        print("     python3 scripts/convert_tensorrt.py --all")
        print("  3. 运行应用:")
        print("     python3 -m app.main")


if __name__ == "__main__":
    main()
