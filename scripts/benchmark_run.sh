#!/bin/bash
# Jetson 端自动基准测试脚本
# 测试不同分辨率对帧率的影响
#
# 用法:
#   bash scripts/benchmark_run.sh
#
# 前置条件:
#   - models/onnx_<SIZE>/ 目录已存在（由 benchmark_prepare.py 生成）
#   - 已安装 TensorRT、摄像头已连接

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# 支持正方形 "WxW" 或非正方形 "WxH"；生产路径是 480x360
RESOLUTIONS=("480x360" "128x128" "192x192" "256x256" "320x320" "384x384" "512x512")
TEST_DURATION=30
RESULTS_FILE="benchmark_results.csv"

echo "分辨率基准测试"
echo "============="
echo "测试分辨率: ${RESOLUTIONS[*]}"
echo "每档测试时长: ${TEST_DURATION}秒"
echo "结果文件: ${RESULTS_FILE}"
echo ""

# 初始化 CSV
echo "resolution,infer_ms,total_ms,fps" > "$RESULTS_FILE"

# 备份原始配置
cp config/styles.json config/styles.json.bak

for RES in "${RESOLUTIONS[@]}"; do
    W="${RES%x*}"
    H="${RES#*x}"
    ONNX_DIR="models/onnx_${W}x${H}"
    echo ""
    echo "=========================================="
    echo "测试分辨率: ${W}x${H}"
    echo "=========================================="

    # 1. 更新 config/styles.json —— input_size 必须是 (W,H) 元组
    python3 -c "
import json
with open('config/styles.json', 'r') as f:
    config = json.load(f)
for style in config['styles']:
    style['input_size'] = [${W}, ${H}]
with open('config/styles.json', 'w') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print(f'  配置已更新: input_size = [${W}, ${H}]')
"

    # 2. 复制对应分辨率的 ONNX 模型
    if [ -d "${ONNX_DIR}" ]; then
        cp ${ONNX_DIR}/*.onnx models/
        echo "  ONNX 模型已复制: ${ONNX_DIR}/ → models/"
    else
        echo "  [错误] ONNX 模型目录不存在: ${ONNX_DIR}/"
        echo "  请先在 Windows 端运行: python scripts/benchmark_prepare.py"
        continue
    fi

    # 3. 删除旧引擎
    rm -f models/trt/style_*.engine
    echo "  旧 TensorRT 引擎已删除"

    # 4. 重建 TensorRT 引擎
    echo "  正在重建 TensorRT 引擎（可能需要几分钟）..."
    python3 scripts/convert_tensorrt.py --all 2>&1 | tail -5

    # 5. 运行测试
    echo "  正在运行测试 (${TEST_DURATION}秒)..."
    timeout ${TEST_DURATION} python3 app/main.py --camera 0 --backend tensorrt 2>&1 \
        | grep '\[性能\]' > /tmp/bench_logs_${RES}.txt || true

    # 6. 解析结果（取最后3行的平均值）
    if [ -s /tmp/bench_logs_${RES}.txt ]; then
        RESULT=$(tail -3 /tmp/bench_logs_${RES}.txt | python3 -c "
import sys, re
lines = sys.stdin.readlines()
infer_list, total_list, fps_list = [], [], []
for line in lines:
    m = re.search(r'推理:(\d+\.?\d*)ms', line)
    if m: infer_list.append(float(m.group(1)))
    m = re.search(r'总耗时:(\d+\.?\d*)ms', line)
    if m: total_list.append(float(m.group(1)))
    m = re.search(r'FPS:(\d+\.?\d*)', line)
    if m: fps_list.append(float(m.group(1)))
if infer_list:
    avg_infer = sum(infer_list) / len(infer_list)
    avg_total = sum(total_list) / len(total_list)
    avg_fps = sum(fps_list) / len(fps_list)
    print(f'{avg_infer:.1f},{avg_total:.1f},{avg_fps:.1f}')
else:
    print('N/A,N/A,N/A')
")
        echo "${W}x${H},${RESULT}" >> "$RESULTS_FILE"
        INFER=$(echo "$RESULT" | cut -d, -f1)
        TOTAL=$(echo "$RESULT" | cut -d, -f2)
        FPS=$(echo "$RESULT" | cut -d, -f3)
        echo "  结果: ${W}x${H} → 推理=${INFER}ms, 总耗时=${TOTAL}ms, FPS=${FPS}"
    else
        echo "  [警告] 未捕获到性能日志"
        echo "${W}x${H},N/A,N/A,N/A" >> "$RESULTS_FILE"
    fi
done

# 恢复原始配置
cp config/styles.json.bak config/styles.json
rm config/styles.json.bak
echo ""
echo "配置已恢复为原始值"

# 打印汇总表格
echo ""
echo "=========================================="
echo "测试结果汇总"
echo "=========================================="
printf "%-12s %-12s %-12s %-10s\n" "分辨率" "推理时间" "总帧时间" "FPS"
printf "%-12s %-12s %-12s %-10s\n" "--------" "--------" "--------" "---"
tail -n +2 "$RESULTS_FILE" | while IFS=, read -r res infer total fps; do
    printf "%-12s %-12s %-12s %-10s\n" "${res}" "${infer}ms" "${total}ms" "$fps"
done

echo ""
echo "详细结果已保存到: ${RESULTS_FILE}"
