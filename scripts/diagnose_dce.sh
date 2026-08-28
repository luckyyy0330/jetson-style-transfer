#!/bin/bash
# 诊断 DCE 固件和摄像头问题
# 请在 Jetson 上执行并把输出发给我

echo "=========================================="
echo "  DCE + Camera 完整诊断"
echo "=========================================="
echo ""

echo "=== 1. 系统信息 ==="
cat /etc/nv_tegra_release
uname -a
echo ""

echo "=== 2. JetPack 版本 ==="
apt-cache show nvidia-jetpack 2>/dev/null | grep -E "^(Package|Version):" | head -6
echo ""

echo "=== 3. DCE 固件检查 ==="
echo "固件目录内容:"
ls -la /lib/firmware/nvidia/tegra234/ 2>/dev/null || echo "目录不存在"
echo ""
echo "DCE 相关文件:"
find /lib/firmware -name "*dce*" -o -name "*DCE*" 2>/dev/null || echo "无"
echo ""
echo "DCE 设备节点:"
ls -la /dev/nvidia-dce* 2>/dev/null || echo "无 DCE 设备节点"
echo ""
echo "DCE 内核模块:"
lsmod | grep -i dce
echo ""

echo "=== 4. Camera 驱动状态 ==="
echo "IMX219 模块:"
lsmod | grep imx
echo ""
echo "VI/NVCSI 模块:"
lsmod | grep -E "vi5|nvcsi"
echo ""
echo "Camera I2C 驱动:"
ls /sys/bus/i2c/drivers/ | grep -i imx
echo ""

echo "=== 5. 设备节点 ==="
echo "/dev/video*:"
ls -la /dev/video* 2>/dev/null || echo "无"
echo ""
echo "V4L2 设备:"
v4l2-ctl --list-devices 2>/dev/null || echo "v4l2-ctl 不可用"
echo ""

echo "=== 6. nvpmodel 状态 ==="
cat /etc/nvpmodel.conf | grep -A3 "POWER_MODEL ID=" | head -20
echo ""
sudo nvpmodel -p 2>/dev/null
echo ""

echo "=== 7. 最近的 DCE 错误 ==="
sudo dmesg | grep -i "dce" | tail -20
echo ""

echo "=== 8. 最近的 camera 错误 ==="
sudo dmesg | grep -iE "imx219|csi|nvcsi|vi5|camera" | tail -20
echo ""

echo "=========================================="
echo "  诊断完成，请把以上输出发给我"
echo "=========================================="
