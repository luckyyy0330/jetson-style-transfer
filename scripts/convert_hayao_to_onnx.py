#!/usr/bin/env python3
"""
下载 TachibanaYoshino AnimeGANv2 Hayao (宫崎骏风格) checkpoint 并转换为 ONNX。

用法:
    python convert_hayao_to_onnx.py

依赖 (在 Jetson 上安装):
    pip install tensorflow tf2onnx
    # Jetson aarch64:
    pip install tensorflow-aarch64 tf2onnx

下载源:
    https://github.com/TachibanaYoshino/AnimeGANv2/tree/master/checkpoint/generator_Hayao_weight
"""

import os
import sys
import subprocess
import urllib.request

CKPT_DIR = os.path.join("models", "tf_checkpoints")
ONNX_OUTPUT = os.path.join("models", "hayao_ghibli.onnx")

HAYAO_BASE_URL = (
    "https://github.com/TachibanaYoshino/AnimeGANv2/raw/master/"
    "checkpoint/generator_Hayao_weight"
)

HAYAO_FILES = [
    "Hayao-99.ckpt.data-00000-of-00001",
    "Hayao-99.ckpt.index",
    "Hayao-99.ckpt.meta",
]


def download_file(url: str, dest: str) -> bool:
    """下载文件"""
    print(f"  下载: {os.path.basename(dest)} ...", end="", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
            with open(dest, "wb") as f:
                f.write(data)
            print(f" OK ({len(data) / 1024 / 1024:.1f} MB)")
            return True
    except Exception as e:
        print(f" 失败: {e}")
        return False


def download_checkpoint() -> bool:
    """下载 Hayao-99 checkpoint 文件"""
    os.makedirs(CKPT_DIR, exist_ok=True)

    all_exist = True
    for fname in HAYAO_FILES:
        path = os.path.join(CKPT_DIR, fname)
        if not os.path.exists(path):
            all_exist = False
            break

    if all_exist:
        print("Checkpoint 已存在，跳过下载。")
        return True

    print(f"正在下载 Hayao-99 checkpoint 到 {CKPT_DIR}/")
    for fname in HAYAO_FILES:
        dest = os.path.join(CKPT_DIR, fname)
        if os.path.exists(dest):
            print(f"  已存在: {fname}")
            continue
        url = f"{HAYAO_BASE_URL}/{fname}"
        if not download_file(url, dest):
            return False

    return True


def convert_via_meta_graph() -> bool:
    """方法 1: 使用 meta graph + tf2onnx 转换

    .meta 文件包含完整的 TF 图定义，可以直接转 ONNX。
    """
    ckpt_prefix = os.path.join(CKPT_DIR, "Hayao-99.ckpt")
    meta_file = ckpt_prefix + ".meta"

    if not os.path.exists(meta_file):
        print(f"Meta 文件不存在: {meta_file}")
        return False

    print("\n方法 1: 使用 meta graph → ONNX")

    # tf2onnx 命令行: --checkpoint 需要 .meta 文件
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--checkpoint", meta_file,
        "--output", ONNX_OUTPUT,
        "--opset", "13",
        "--inputs", "generator_input:0",
        "--outputs", "generator/Tanh:0",
    ]

    print(f"运行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode == 0:
        print("转换成功!")
        return True

    print(f"转换失败 (exit {result.returncode}):")
    print(result.stderr[:1000])

    # 尝试自动检测 input/output tensor 名称
    print("\n尝试自动检测 tensor 名称...")
    return convert_auto_detect_tensors()


def convert_auto_detect_tensors() -> bool:
    """使用 tf2onnx 的 --saved-model 方式，自动检测输入输出"""
    ckpt_prefix = os.path.join(CKPT_DIR, "Hayao-99.ckpt")
    meta_file = ckpt_prefix + ".meta"

    # 用 TF 加载 meta graph 并找到 input/output tensor
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow 未安装，无法自动检测。")
        return False

    tf.compat.v1.reset_default_graph()
    saver = tf.compat.v1.train.import_meta_graph(meta_file)
    graph = tf.compat.v1.get_default_graph()

    # 列出所有 placeholder (输入)
    placeholders = [
        op.name + ":0"
        for op in graph.get_operations()
        if op.type == "Placeholder"
    ]
    print(f"检测到的 Placeholder: {placeholders}")

    # 列出可能的输出 (最后一层的输出)
    all_ops = list(graph.get_operations())
    conv_ops = [
        op for op in all_ops
        if "Conv2D" in op.type or "out" in op.name.lower()
    ]
    print(f"Conv2D 操作数: {len(conv_ops)}")

    # 找 generator/G_MODEL/out 或类似名称
    candidate_outputs = []
    for op in all_ops:
        name = op.name.lower()
        if "out" in name or "generator" in name:
            if op.type in ("Conv2D", "Tanh", "Sigmoid", "BiasAdd"):
                candidate_outputs.append(f"{op.name}:0")

    print(f"候选输出 tensors: {candidate_outputs[:10]}")

    # 尝试各种组合
    input_tensor = placeholders[0] if placeholders else None
    if not input_tensor:
        print("未找到输入 tensor")
        return False

    # 按优先级尝试输出
    output_candidates = candidate_outputs or [
        "generator/Tanh:0",
        "generator/G_MODEL/out:0",
        "generator/out:0",
        "generator/decoder/conv_last/BiasAdd:0",
    ]

    for output_tensor in output_candidates:
        print(f"\n尝试: input={input_tensor}, output={output_tensor}")
        cmd = [
            sys.executable, "-m", "tf2onnx.convert",
            "--checkpoint", meta_file,
            "--output", ONNX_OUTPUT,
            "--opset", "13",
            "--inputs", input_tensor,
            "--outputs", output_tensor,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"成功! 输入: {input_tensor}, 输出: {output_tensor}")
            return True
        else:
            # 打印简短错误
            err_line = result.stderr.strip().split("\n")[-1] if result.stderr else ""
            print(f"  失败: {err_line[:200]}")

    return False


def convert_via_frozen_graph() -> bool:
    """方法 2: 先冻结图 (pb)，再用 tf2onnx 转换"""
    ckpt_prefix = os.path.join(CKPT_DIR, "Hayao-99.ckpt")
    meta_file = ckpt_prefix + ".meta"
    pb_file = os.path.join(CKPT_DIR, "hayao_frozen.pb")

    print("\n方法 2: 冻冻图 → ONNX")

    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow 未安装。")
        return False

    # 加载 meta graph 并冻结
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.import_meta_graph(meta_file)
        saver.restore(sess, ckpt_prefix)

        graph = tf.compat.v1.get_default_graph()

        # 找到所有可能的输出
        all_ops = graph.get_operations()
        output_names = []
        for op in all_ops:
            name_lower = op.name.lower()
            if ("out" in name_lower and "generator" in name_lower):
                output_names.append(op.name)
            elif op.type == "Tanh" and "generator" in name_lower:
                output_names.append(op.name)

        if not output_names:
            # 回退: 找最后一个 BiasAdd 或 Conv2D
            for op in reversed(all_ops):
                if op.type in ("BiasAdd", "Conv2D") and "generator" in op.name.lower():
                    output_names.append(op.name)
                    break

        print(f"候选输出节点: {output_names}")

        # 冻结图
        output_node_names = [name.split(":")[0] for name in output_names[:3]]
        if not output_node_names:
            print("未找到输出节点")
            return False

        print(f"冻结图，输出节点: {output_node_names}")
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), output_node_names
        )

        # 保存 frozen pb
        with open(pb_file, "wb") as f:
            f.write(frozen_graph.SerializeToString())
        print(f"冻结图已保存: {pb_file}")

    # 用 tf2onnx 转换 frozen pb
    print("转换 frozen pb → ONNX ...")
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--graphdef", pb_file,
        "--output", ONNX_OUTPUT,
        "--opset", "13",
    ]
    # 添加 inputs/outputs
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        tf.import_graph_def(frozen_graph, name="")
    placeholders = [
        op.name + ":0"
        for op in graph.get_operations()
        if op.type == "Placeholder"
    ]
    if placeholders:
        cmd.extend(["--inputs", placeholders[0]])
    if output_names:
        cmd.extend(["--outputs", ",".join(output_names[:3])])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode == 0:
        print("转换成功!")
        return True
    else:
        print(f"转换失败: {result.stderr[:500]}")
        return False


def verify_onnx():
    """验证 ONNX 模型"""
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(ONNX_OUTPUT)
        inp = sess.get_inputs()[0]
        out = sess.get_outputs()[0]
        print(f"\nONNX 模型验证:")
        print(f"  输入: {inp.name}, shape={inp.shape}")
        print(f"  输出: {out.name}, shape={out.shape}")

        # 测试推理
        dummy = np.random.randn(*inp.shape).astype(np.float32)
        result = sess.run(None, {inp.name: dummy})
        print(f"  推理测试: 输出 shape={result[0].shape}, OK!")

        # 计算参数量
        import onnx
        model = onnx.load(ONNX_OUTPUT)
        total_params = sum(np.prod(init.dims) for init in model.graph.initializer)
        print(f"  参数量: {total_params / 1e6:.1f}M")
        return True
    except Exception as e:
        print(f"  验证失败: {e}")
        return False


def main():
    print("=" * 60)
    print("  TachibanaYoshino AnimeGANv2 Hayao → ONNX 转换工具")
    print("  宫崎骏风格优化版 (基于《千与千寻》《龙猫》等)")
    print("=" * 60)

    # 检查依赖
    try:
        import tensorflow as tf
        print(f"TensorFlow 版本: {tf.__version__}")
    except ImportError:
        print("错误: 需要安装 TensorFlow")
        print("  pip install tensorflow       # x86_64")
        print("  pip install tensorflow-aarch64  # Jetson aarch64")
        return 1

    try:
        import onnxruntime
        print(f"ONNX Runtime 版本: {onnxruntime.__version__}")
    except ImportError:
        print("警告: onnxruntime 未安装，将跳过验证")

    # 步骤 1: 下载 checkpoint
    print("\n" + "=" * 40)
    print("步骤 1/3: 下载 Hayao-99 checkpoint")
    print("=" * 40)
    if not download_checkpoint():
        print("下载失败!")
        return 1

    # 步骤 2: 转换 ONNX
    print("\n" + "=" * 40)
    print("步骤 2/3: 转换为 ONNX")
    print("=" * 40)

    success = False
    # 尝试方法 1: meta graph 直接转
    if not success:
        success = convert_via_meta_graph()

    # 尝试方法 2: 冻结图再转
    if not success:
        success = convert_via_frozen_graph()

    if not success:
        print("\n所有转换方法都失败了。")
        print("可能的原因:")
        print("  1. TF/tf2onnx 版本不兼容")
        print("  2. 输入/输出 tensor 名称不匹配")
        print("  3. 模型使用了 tf2onnx 不支持的 op")
        return 1

    # 步骤 3: 验证
    print("\n" + "=" * 40)
    print("步骤 3/3: 验证 ONNX 模型")
    print("=" * 40)
    verify_onnx()

    print("\n" + "=" * 60)
    print("转换完成!")
    print(f"  ONNX 模型: {ONNX_OUTPUT}")
    print()
    print("下一步:")
    print("  1. 更新 config/styles.json:")
    print(f'     "base_model": "hayao_ghibli.onnx"')
    print("  2. 删除旧 TensorRT engine:")
    print("     rm models/trt/style_1.engine")
    print("  3. 重新编译 TensorRT engine:")
    print("     python3 scripts/convert_tensorrt.py --all")
    print("  4. 运行测试:")
    print("     python3 -m app.main")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
