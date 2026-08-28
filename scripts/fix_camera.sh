#!/bin/bash
# 修复 CSI 摄像头无法识别的问题
# 适用于 Jetson Orin Nano Super + IMX219

set -e

RED='\e[0;31m'
GREEN='\e[0;32m'
YELLOW='\e[1;33m'
NC='\e[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "=========================================="
echo "  Jetson CSI 摄像头修复工具"
echo "=========================================="
echo ""

# 检查是否在 Jetson 上
if ! grep -qi "nvidia" /etc/nv_tegra_release 2>/dev/null; then
    log_error "未检测到 Jetson 设备"
    exit 1
fi

# 1. 停止 nvargus-daemon
log_info "1. 停止 nvargus-daemon..."
sudo systemctl stop nvargus-daemon 2>/dev/null || true

# 2. 卸载摄像头模块
log_info "2. 卸载摄像头模块..."
sudo rmmod nv_imx219 2>/dev/null || true
sudo rmmod tegra_camera 2>/dev/null || true
sudo rmmod tegra_capture_isp 2>/dev/null || true
sudo rmmod nvhost_nvcsi 2>/dev/null || true
sudo rmmod nvhost_vi5 2>/dev/null || true

sleep 1

# 3. 重新加载模块（按依赖顺序）
log_info "3. 重新加载摄像头模块..."
sudo modprobe host1x
sudo modprobe nvhost_vi5
sudo modprobe nvhost_nvcsi
sudo modprobe tegra_capture_isp
sudo modprobe tegra_camera
sudo modprobe nv_imx219

sleep 2

# 4. 检查设备节点
log_info "4. 检查设备节点..."
if ls /dev/video* 2>/dev/null; then
    log_info "摄像头设备节点已恢复！"
else
    log_warn "设备节点仍未出现，尝试其他方法..."
fi

# 5. 尝试通过 nvargus 测试
log_info "5. 测试摄像头..."
if nvgstcapture-1.0 -d /dev/video0 --file=test.jpg -m auto 2>/dev/null; then
    log_info "摄像头测试成功！"
    rm -f test.jpg
else
    log_warn "nvargus 测试失败，尝试 v4l2-ctl..."
    v4l2-ctl --list-devices 2>/dev/null || true
fi

# 6. 重启 nvargus-daemon
log_info "6. 重启 nvargus-daemon..."
sudo systemctl start nvargus-daemon 2>/dev/null || true

echo ""
echo "=========================================="
echo "  修复完成"
echo "=========================================="
echo ""
echo "如果摄像头仍无法识别，请执行以下命令并把输出发给我："
echo "  sudo dmesg | tail -50"
echo "  sudo cat /sys/kernel/debug/clk/clk_summary | grep -i vi"
echo ""
echo "或者直接重启系统："
echo "  sudo reboot"
