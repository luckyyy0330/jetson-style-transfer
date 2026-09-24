"""批量导出不同分辨率的 ONNX 模型，用于基准测试

用法:
  python scripts/benchmark_prepare.py --sizes 128 192 256 320 384 512

会对每个分辨率导出 ONNX 模型到 models/onnx_<SIZE>/ 目录。
"""

import os
import sys
import argparse

# 添加项目根目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from scripts.prepare_models import MODELS, download_and_convert


def main():
    parser = argparse.ArgumentParser(description="批量导出不同分辨率的 ONNX 模型")
    parser.add_argument(
        "--sizes", type=int, nargs="+",
        default=[128, 192, 256, 320, 384, 512],
        help="要测试的分辨率列表 (默认: 128 192 256 320 384 512)"
    )
    args = parser.parse_args()

    sizes = args.sizes
    models_dir = os.path.join(PROJECT_DIR, "models")

    print(f"将导出以下分辨率的 ONNX 模型: {sizes}")
    print(f"模型目录: {models_dir}\n")

    # 对每个分辨率导出 ONNX
    for size in sizes:
        onnx_dir = os.path.join(models_dir, f"onnx_{size}")
        os.makedirs(onnx_dir, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"导出分辨率: {size}x{size}")
        print(f"输出目录: {onnx_dir}")
        print(f"{'='*50}")

        # 临时修改 prepare_models 的输出目录
        import scripts.prepare_models as pm
        original_models_dir = pm.MODELS_DIR
        pm.MODELS_DIR = onnx_dir

        success_count = 0
        for output_name, pretrained_name in MODELS.items():
            if download_and_convert(output_name, pretrained_name, input_size=(size, size)):
                success_count += 1

        # 恢复原始目录
        pm.MODELS_DIR = original_models_dir

        print(f"\n{size}x{size}: {success_count}/{len(MODELS)} 个模型导出成功")

    # 打印 scp 传输命令
    print(f"\n{'='*50}")
    print("导出完成！下一步：")
    print(f"{'='*50}")
    print("\n1. 传输到 Jetson:")
    print(f"   scp -r models/onnx_* seeed@192.168.108.18:~/jetson-style-transfer/models/")
    print(f"   scp scripts/benchmark_run.sh seeed@192.168.108.18:~/jetson-style-transfer/scripts/")
    print("\n2. 在 Jetson 上运行基准测试:")
    print(f"   ssh seeed@192.168.108.18 \"cd ~/jetson-style-transfer && bash scripts/benchmark_run.sh\"")
    print("\n3. 查看结果:")
    print(f"   ssh seeed@192.168.108.18 \"cat ~/jetson-style-transfer/benchmark_results.csv\"")


if __name__ == "__main__":
    main()
