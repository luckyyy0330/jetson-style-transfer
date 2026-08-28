# Jetson 风格转换设备 — 完整搭建指南

> 适用于 Jetson Orin Nano 8GB Developer Kit

---

## 一、硬件准备

### 1.1 所需硬件

| 硬件 | 规格 | 用途 |
|------|------|------|
| Jetson Orin Nano 8GB | Developer Kit | 主控板 |
| CSI 摄像头 | IMX219 模块 | 拍照/录像 |
| HDMI 显示器 | 任意尺寸 | 显示 OpenCV 窗口预览 |
| NVMe SSD | M.2 2230 规格 | 存储模型和照片 |
| microSD 卡 | 32GB+ | 系统启动盘 |
| 电源适配器 | 12V/3A DC | 供电 |
| 网线 | RJ45 | SSH 连接 |

### 1.2 接线顺序

```
1. 网线 → 以太网口（必须，用于 SSH 连接）
2. CSI 摄像头排线 → CSI 口（蓝色面朝下，卡扣扣紧）
3. HDMI 线 → HDMI 口（可选，有 SSH 就不需要）
4. USB 键盘鼠标 → USB 口（可选）
5. 12V 电源线 → USB-C 口（最后插，插上即开机）
```

---

## 二、烧录系统

### 2.1 下载 JetPack 镜像

1. 打开 NVIDIA 官网：https://developer.nvidia.com/embedded/jetson-linux
2. 下载 **JetPack 6.x SD Card Image**（约 10GB）
3. 解压得到 `.img` 文件

### 2.2 烧录到 SD 卡

**在 Windows 上操作：**

1. 下载 Balena Etcher：https://www.balena.io/etcher/
2. 插入 microSD 卡（32GB+）
3. 打开 Balena Etcher：
   - **Flash from file** → 选择解压后的 `.img` 文件
   - **Select target** → 选择你的 SD 卡
   - **Flash!** → 开始烧录（约 5-10 分钟）

### 2.3 首次启动

1. 将烧录好的 SD 卡插入 Jetson 的 SD 卡槽
2. 插上 12V 电源线（Jetson 自动开机）
3. 等待系统启动（首次约 1-2 分钟）
4. 连接显示器，进入 Ubuntu 初始设置向导：
   - 选择语言、键盘
   - 创建用户名密码（默认 `nvidia` / `nvidia`）
   - 完成后进入桌面

---

## 三、网络配置

### 3.1 获取 Jetson IP 地址

在 Jetson 终端中执行：

```bash
hostname -I
# 输出类似：192.168.108.46
```

### 3.2 从电脑 SSH 连接

**在 Windows PowerShell 中：**

```powershell
ssh nvidia@<Jetson的IP>
# 密码：nvidia
```

> 第一次连接会问 `Are you sure you want to continue connecting?`，输入 `yes`。

---

## 四、Jetson 环境配置

### 4.1 设置性能模式

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

### 4.2 安装基础依赖

```bash
sudo apt install -y git python3-pip python3-dev
```

### 4.3 安装 Python 依赖

```bash
pip3 install --break-system-packages numpy opencv-python-headless Pillow onnxruntime
```

> **⚠️ 重要：不要执行 `sudo apt upgrade`！**
>
> `apt upgrade` 会更新内核和设备树文件，导致摄像头配置丢失（`/dev/video*` 消失）。
> Jetson JetPack 已预装稳定版本的驱动和内核，无需额外升级。
```

---

## 五、部署代码

### 5.1 从电脑传代码到 Jetson

**在 Windows PowerShell 中：**

​```powershell
# 先在 Jetson 上创建目录
ssh nvidia@192.168.108.71 "mkdir -p ~/jetson-style-transfer"

# 复制整个项目
scp -r D:\desktop\jetson-style-transfer\* nvidia@192.168.108.71:~/jetson-style-transfer/
```

### 5.2 下载并转换模型

**方法一：在 Jetson 上下载模型（较慢）**

```bash
cd ~/jetson-style-transfer
python3 scripts/download_models.py --output models
```

**方法二：在 Windows 上下载模型再传过去（推荐）**

在 Windows PowerShell 中：

```powershell
# 1. 安装 PyTorch（如果没有）
pip install torch torchvision

# 2. 下载模型
cd D:\desktop\jetson-style-transfer
python scripts\download_models.py --output models

# 3. 传到 Jetson
ssh nvidia@<IP> "mkdir -p ~/jetson-style-transfer/models/sd-v1-5"
scp -r D:\desktop\jetson-style-transfer\models\sd-v1-5\* nvidia@<IP>:~/jetson-style-transfer/models/sd-v1-5/
```

### 5.3 转换 TensorRT 引擎（推荐，大幅提升性能）

在 Jetson 上执行：

```bash
cd ~/jetson-style-transfer

# 安装 TensorRT（如果没装）
sudo apt install -y tensorrt python3-libnvinfer

# 安装 ONNX 依赖
pip3 install onnx onnxscript

# 转换所有模型
python3 scripts/convert_tensorrt.py --all --models models
```

转换完成后检查：

```bash
ls models/trt/
# 应该看到 .engine 文件（每个约 3.8MB）
```

### 5.4 验证安装

```bash
# 检查 TensorRT 是否可用
python3 -c "import tensorrt; print(tensorrt.__version__)"

# 检查模型文件
ls -la models/trt/

# 测试启动（Ctrl+C 退出）
python3 -m app.main --list-cameras
```

---

## 六、连接摄像头

### 6.1 检查摄像头是否识别

```bash
ls /dev/video*
# 应该看到 /dev/video0
```

### 6.2 测试 CSI 摄像头

```bash
nvgstcapture-1.0 -m wb:sunny
# 应该能看到实时画面
```

### 6.3 排查摄像头问题

如果 `/dev/video*` 不存在：

1. 检查 CSI 排线是否插紧（蓝色面朝下）
2. 重启 Jetson：`sudo reboot`
3. 检查摄像头模块是否支持：`ls /dev/nvhost-*`

---

## 七、启动应用

### 7.1 启动应用

```bash
cd ~/jetson-style-transfer

# 自动检测最优后端（推荐）
python3 -m app.main

# 指定使用 TensorRT 后端（如果已转换）
python3 -m app.main --backend tensorrt

# 指定使用 ONNX 后端
python3 -m app.main --backend onnx

# 指定使用 PyTorch 后端
python3 -m app.main --backend pytorch
```

### 7.2 键盘操作

| 按键 | 功能 |
|------|------|
| `1` | 切换到动漫风格 |
| `空格` | 拍照 |
| `Q` | 退出 |

---

## 八、开机自启动（可选）

让应用开机自动运行：

```bash
# 创建启动脚本
cat > ~/start_style_transfer.sh << 'EOF'
#!/bin/bash
cd /home/nvidia/jetson-style-transfer
python3 -m app.main
EOF

chmod +x ~/start_style_transfer.sh

# 添加到自启动
echo "@/home/nvidia/start_style_transfer.sh" >> ~/.config/autostart.desktop
```

---

## 九、常见问题

### Q: nvidia-smi 报错 "Driver/library version mismatch"

```bash
# 重启试试
sudo reboot

# 如果还不行，检查是否有冲突的驱动包
dpkg -l | grep nvidia | grep -v "$(cat /etc/nv_tegra_release | head -1)"
```

### Q: TensorRT 转换失败

```bash
# 检查 TensorRT 版本
python3 -c "import tensorrt; print(tensorrt.__version__)"

# 检查 ONNX 文件
ls -la models/onnx/
# 确认有 .onnx 和 .onnx.data 文件
```

### Q: 应用启动报错 "No module named 'torch'"

这是正常的——推理不需要 PyTorch。如果报其他 import 错误，检查依赖：

```bash
pip3 install numpy opencv-python-headless Pillow onnxruntime
```

### Q: 摄像头画面卡顿

```bash
# 检查 GPU 使用情况
tegrastats

# 设置风扇全速
echo 255 | sudo tee /sys/devices/pwm-fan/target_duty_cycle
```

### Q: 处理速度慢

1. **转换 TensorRT 引擎**（推荐，提升 3-5 倍）：
   ```bash
   python3 scripts/convert_tensorrt.py --all --models models
   ```

2. **指定使用 TensorRT 后端**：
   ```bash
   python3 -m app.main --backend tensorrt
   ```

3. **设置性能模式**：
   ```bash
   sudo nvpmodel -m 0 && sudo jetson_clocks
   ```

4. **检查散热**：
   ```bash
   cat /sys/class/thermal/thermal_zone*/temp
   ```

### Q: TensorRT 引擎不存在

```bash
# 查看可用的 TensorRT 引擎
ls -la models/trt/

# 如果没有，需要先转换
python3 scripts/convert_tensorrt.py --all --models models

# 或者使用 ONNX 后端
python3 -m app.main --backend onnx
```

---

## 十、项目文件说明

```
jetson-style-transfer/
├── app/                         # 核心应用代码
│   ├── main.py                  # 程序入口
│   ├── camera/
│   │   └── csi_camera.py        # CSI/USB 摄像头采集
│   ├── style/
│   │   ├── engine.py            # 多后端推理引擎（TensorRT/ONNX/PyTorch）
│   │   ├── tensorrt_utils.py    # TensorRT 推理工具
│   │   ├── streamdiffusion_engine.py # StreamDiffusion 后端
│   │   └── lora_manager.py      # LoRA 风格管理（可选）
│   └── export/
│       └── storage.py           # 文件保存/导出
│
├── models/                      # 模型文件
│   ├── sd-v1-5/                 # PyTorch 模型（首次运行自动下载）
│   ├── onnx/                    # ONNX 模型（转换生成）
│   └── trt/                     # TensorRT 引擎（转换生成）
│
├── scripts/
│   ├── download_models.py       # 下载模型
│   ├── convert_tensorrt.py      # 转换 TensorRT 引擎
│   └── setup_device.py          # 设备初始化
│
├── config/
│   ├── styles.json              # 风格配置
│   └── device.json              # 设备参数
│
└── assets/
    └── captures/                # 拍照/录像保存目录
```

---

## 十一、性能参考

| 指标 | 数值 |
|------|------|
| 单张图片推理时间 | ~20-50ms（TensorRT FP16） |
| FPS | 20-50 |
| 模型大小 | ~3.8MB × 5 |
| GPU 温度（满载） | 60-75°C |
| 功耗 | 7-15W |

---

*最后更新：2026-08-26*
