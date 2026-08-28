#!/bin/bash
# 修复 DCE (Display and Camera Engine) RPC 失败导致摄像头无法识别
# 适用于 Jetson Orin Nano Super

set -e

RED='\e[0;31m'
GREEN='\e[0;32m'
YELLOW='\e[1;33m'
NC='\e[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "=========================================="
echo "  Jetson DCE + Camera 修复工具"
echo "=========================================="
echo ""

# 检查是否在 Jetson 上
if ! grep -qi "nvidia" /etc/nv_tegra_release 2>/dev/null; then
    log_error "未检测到 Jetson 设备"
    exit 1
fi

# 1. 检查当前 nvpmodel 模式
log_info "1. 检查当前电源模式..."
CURRENT_PM=$(sudo nvpmodel -p 2>/dev/null | grep -oP 'Power Mode.*?:\s*\K.*' || echo "unknown")
log_info "当前电源模式: $CURRENT_PM"

# 2. 设置正确的电源模式（MAXN_SUPER = ID=2）
log_info "2. 设置 MAXN_SUPER 电源模式..."
sudo nvpmodel -m 2
sleep 2

# 3. 恢复时钟（撤销 jetson_clocks）
log_info "3. 恢复时钟设置..."
sudo jetson_clocks --restore 2>/dev/null || log_warn "无法恢复时钟（可能未保存过）"

# 4. 重启 DCE 相关服务
log_info "4. 重启 DCE 相关服务..."
sudo systemctl stop nvargus-daemon 2>/dev/null || true
sudo systemctl stop gdm 2>/dev/null || true  # 如果有桌面环境

sleep 2

# 5. 重新加载摄像头模块
log_info "5. 重新加载摄像头模块..."
sudo rmmod nv_imx219 2>/dev/null || true
sudo rmmod tegra_camera 2>/dev/null || true
sudo rmmod tegra_capture_isp 2>/dev/null || true
sudo rmmod nvhost_nvcsi 2>/dev/null || true
sudo rmmod nvhost_vi5 2>/dev/null || true

sleep 1

sudo modprobe host1x
sudo modprobe nvhost_vi5
sudo modprobe nvhost_nvcsi
sudo modprobe tegra_capture_isp
sudo modprobe tegra_camera
sudo modprobe nv_imx219

sleep 2

# 6. 重启 nvargus-daemon
log_info "6. 重启 nvargus-daemon..."
sudo systemctl start nvargus-daemon

sleep 2

# 7. 检查设备节点
log_info "7. 检查设备节点..."
if ls /dev/video* 2>/dev/null; then
    log_info "摄像头设备节点已恢复！"
else
    log_warn "设备节点仍未出现"
fi

# 8. 检查 DCE 错误是否消失
log_info "8. 检查 DCE 状态..."
DCE_ERRORS=$(sudo dmesg | grep -c "NVRM_RPC_DCE.*Failed" 2>/dev/null || echo "0")
log_info "DCE 错误数量: $DCE_ERRORS"

if [ "$DCE_ERRORS" -gt 0 ]; then
    log_warn "仍存在 DCE 错误，可能需要重启系统"
fi

# 9. 测试摄像头
log_info "9. 测试摄像头..."
if nvgstcapture-1.0 -d /dev/video0 --file=/tmp/test_camera.jpg -m auto 2>/dev/null; then
    log_info "摄像头测试成功！"
    rm -f /tmp/test_camera.jpg
else
    log_warn "nvargus 测试失败"
    v4l2-ctl --list-devices 2>/dev/null || true
fi

echo ""
echo "=========================================="
echo "  修复完成"
echo "=========================================="
echo ""
echo "如果问题仍然存在，请执行："
echo "  sudo reboot"
echo ""
echo "重启后运行："
echo "  sudo dmesg | grep -iE 'dce|nvcsi|vi5|imx219' | tail -30"
echo ""
echo "然后把输出发给我。"
