#!/bin/bash
# 最终修复脚本：加载缺失的摄像头模块
# 适用于 JetPack 7.2.1 + Jetson Orin Nano Super

set -e

RED='\e[0;31m'
GREEN='\e[0;32m'
YELLOW='\e[1;33m'
NC='\e[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "=========================================="
echo "  修复缺失的摄像头模块"
echo "=========================================="
echo ""

# 1. 停止相关服务
log_info "1. 停止 nvargus-daemon..."
sudo systemctl stop nvargus-daemon 2>/dev/null || true

# 2. 加载 VI5 模块
log_info "2. 加载 vi5 模块..."
sudo modprobe nvhost_vi5
sleep 1

# 3. 加载 NVCSI 模块
log_info "3. 加载 nvcsi 模块..."
sudo modprobe nvhost_nvcsi
sleep 1

# 4. 加载 ISP 模块
log_info "4. 加载 isp 模块..."
sudo modprobe nvhost_isp5 2>/dev/null || true
sudo modprobe tegra_capture_isp
sleep 1

# 5. 加载 tegra_camera 模块
log_info "5. 加载 tegra_camera 模块..."
sudo modprobe tegra_camera
sleep 1

# 6. 加载 IMX219 模块
log_info "6. 加载 imx219 模块..."
sudo modprobe nv_imx219
sleep 2

# 7. 验证模块加载
log_info "7. 验证模块加载状态..."
echo ""
echo "IMX219: $(lsmod | grep nv_imx219 | awk '{print $1}')"
echo "VI5: $(lsmod | grep nvhost_vi5 | awk '{print $1}')"
echo "NVCSI: $(lsmod | grep nvhost_nvcsi | awk '{print $1}')"
echo "TEGRA_CAMERA: $(lsmod | grep tegra_camera | awk '{print $1}')"
echo ""

# 8. 检查设备节点
log_info "8. 检查设备节点..."
if ls /dev/video* 2>/dev/null; then
    log_info "摄像头设备节点已恢复！"
else
    log_warn "设备节点仍未出现"
fi

# 9. 重启 nvargus-daemon
log_info "9. 重启 nvargus-daemon..."
sudo systemctl start nvargus-daemon
sleep 2

# 10. 测试摄像头
log_info "10. 测试摄像头..."
if nvgstcapture-1.0 -d /dev/video0 --file=/tmp/test.jpg -m auto 2>/dev/null; then
    log_info "摄像头测试成功！"
    rm -f /tmp/test.jpg
else
    log_warn "nvargus 测试失败"
    echo ""
    echo "尝试 v4l2-ctl:"
    v4l2-ctl --list-devices 2>/dev/null || echo "v4l2-ctl 不可用"
fi

echo ""
echo "=========================================="
echo "  修复完成"
echo "=========================================="
