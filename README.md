# Jetson 风格转换设备

基于 NVIDIA Jetson Orin Nano 的实时艺术风格转换硬件产品。使用 GAN 模型（AnimeGANv2）对摄像头画面进行风格迁移，支持拍照保存和 SD 卡/USB 导出。

## 功能特性

- **实时风格转换** - GAN 单次前向推理，TensorRT 加速下 80-150 FPS
- **预设风格切换** - 3 种预设风格（动漫、宫崎骏、人脸动漫），按键 1/2/3 即可切换
- **摄像头预览** - OpenCV 窗口显示原始 + 风格化画面并排对比
- **一键拍照** - 保存风格化结果
- **多种输入源** - CSI 摄像头（IMX219）、USB 摄像头、本地图片
- **TensorRT GPU 加速** - 原生 TensorRT Python API，直接在 GPU 上推理

## 技术方案

基于 **GAN（生成对抗网络）** 的轻量级风格迁移，每个风格使用独立的 ONNX 模型：

| 特性 | 说明 |
|------|------|
| 模型架构 | AnimeGANv2 |
| 推理后端 | TensorRT（推荐，GPU 加速）> ONNX Runtime（CPU） |
| 输入 | 512x512 图像 |
| 模型大小 | ~8.5MB（ONNX），~5MB（TensorRT Engine） |
| 显存占用 | ~200-500MB |
| 速度 | TensorRT FP16: 80-150 FPS |

## 性能

| 后端 | 单帧推理 | FPS (512x512) | 说明 |
|------|----------|---------------|------|
| TensorRT FP16 | ~8ms | 80-150 | GPU 加速，推荐 |
| ONNX Runtime (CPU) | ~15ms | 40-70 | 无需 GPU，兼容性好 |

## 硬件要求

| 硬件 | 规格 | 用途 |
|------|------|------|
| Jetson Orin Nano 8GB | Developer Kit | 主控板 |
| CSI 摄像头 | IMX219 模块 | 拍照/录像 |
| HDMI 显示器 | 任意尺寸 | 显示 OpenCV 窗口预览 |
| NVMe SSD | M.2 2230 规格 | 存储模型和照片 |
| microSD 卡 | 32GB+ | 系统启动盘 |
| 电源适配器 | 12V/3A DC | 供电 |
| 网线 | RJ45 | SSH 连接 |

## 项目结构

```
jetson-style-transfer/
├── app/                             # 核心应用代码
│   ├── main.py                      # 程序入口，主循环
│   ├── camera/
│   │   └── csi_camera.py            # CSI/USB 摄像头采集（GStreamer）
│   ├── style/
│   │   ├── style_manager.py         # 风格预设管理器
│   │   ├── engine.py                # 多后端推理引擎（TensorRT/ONNX）
│   │   └── gan_engine.py            # GAN 推理引擎（原生 TensorRT + ONNX Runtime）
│   └── export/
│       └── storage.py               # 文件保存、SD 卡/USB 同步
│
├── models/                          # 模型文件
│   ├── animegan_v2.onnx             # 动漫风格模型
│   ├── animegan_v2_paprika.onnx     # 宫崎骏风格模型
│   ├── animegan_v2_face_paint.onnx  # 人脸动漫风格模型
│   └── trt/                         # TensorRT 引擎（转换生成）
│
├── scripts/                         # 工具脚本
│   ├── prepare_models.py            # 下载模型并转换为 ONNX（推荐）
│   ├── download_models.py           # 按风格下载 ONNX 模型
│   ├── convert_tensorrt.py          # 转换 TensorRT 引擎
│   └── convert_to_onnx.py           # .pt 转 .onnx 工具
│
├── config/
│   └── styles.json                  # 风格配置（模型路径和参数）
│
├── assets/
│   └── captures/                    # 拍照/录像保存目录
│
├── requirements.txt                 # Python 依赖
└── README.md                        # 本文件
```

---

## 搭建指南

### 1. 硬件接线

```
1. 网线 → 以太网口（必须，用于 SSH 连接）
2. CSI 摄像头排线 → CSI 口（蓝色面朝下，卡扣扣紧）
3. HDMI 线 → HDMI 口（可选，有 SSH 就不需要）
4. USB 键盘鼠标 → USB 口（可选）
5. 12V 电源线 → USB-C 口（最后插，插上即开机）
```

### 2. 烧录系统

**下载 JetPack 镜像：**

1. 打开 NVIDIA 官网：https://developer.nvidia.com/embedded/jetson-linux
2. 下载 **JetPack 7.x SD Card Image**（约 10GB）
3. 解压得到 `.img` 文件

**烧录到 SD 卡（Windows）：**

1. 下载 Balena Etcher：https://www.balena.io/etcher/
2. 插入 microSD 卡（32GB+）
3. 打开 Balena Etcher：
   - **Flash from file** → 选择解压后的 `.img` 文件
   - **Select target** → 选择你的 SD 卡
   - **Flash!** → 开始烧录（约 5-10 分钟）

**首次启动：**

1. 将烧录好的 SD 卡插入 Jetson 的 SD 卡槽
2. 插上 12V 电源线（Jetson 自动开机）
3. 等待系统启动（首次约 1-2 分钟）
4. 连接显示器，进入 Ubuntu 初始设置向导：
   - 选择语言、键盘
   - 创建用户名密码（默认 `nvidia` / `nvidia`）
   - 完成后进入桌面

### 3. 网络配置

**获取 Jetson IP 地址：**

在 Jetson 终端中执行：

```bash
hostname -I
# 输出类似：192.168.108.46
```

**从电脑 SSH 连接（Windows PowerShell）：**

```powershell
ssh nvidia@<Jetson的IP>
# 密码：nvidia
```

> 第一次连接会问 `Are you sure you want to continue connecting?`，输入 `yes`。
>
> 如果报 "REMOTE HOST IDENTIFICATION HAS CHANGED"，执行：
> ```powershell
> ssh-keygen -R <Jetson的IP>
> ```

### 4. Jetson 环境配置

**设置性能模式：**

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

**安装基础依赖：**

```bash
sudo apt install -y git python3-pip python3-dev python3-libnvinfer
```

**创建虚拟环境：**

> **重要：必须使用 `--system-site-packages`，否则 TensorRT 和 PyTorch 无法使用**

```bash
cd ~/jetson-style-transfer
python3 -m venv --system-site-packages venv
source venv/bin/activate
```

**安装 Python 依赖：**

```bash
pip install -r requirements.txt
```

如果 pip 安装超时，使用国内镜像源：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

> **⚠️ 重要：不要执行 `sudo apt upgrade`！**
>
> `apt upgrade` 会更新内核和设备树文件，导致摄像头配置丢失（`/dev/video*` 消失）。
> Jetson JetPack 已预装稳定版本的驱动和内核，无需额外升级。

### 5. 部署代码

**从电脑传代码到 Jetson（Windows PowerShell）：**

```powershell
# 先在 Jetson 上创建目录
ssh seeed@192.168.108.23> "mkdir -p ~/jetson-style-transfer"

# 复制整个项目
scp -r D:\desktop\jetson-style-transfer\* seeed@192.168.108.18:~/jetson-style-transfer/


```

**准备模型（推荐在 Windows 上操作）：**

```powershell
# 1. 安装 PyTorch（如果没有）
pip install torch torchvision

# 2. 一键下载模型并转换为 ONNX
cd D:\desktop\jetson-style-transfer
python scripts\prepare_models.py

# 3. 传到 Jetson
scp models\*.onnx seeed@192.168.108.23:~/jetson-style-transfer/models/
```

**或者在 Jetson 上下载：**

```bash
cd ~/jetson-style-transfer
source venv/bin/activate
python3 scripts/download_models.py --all
```

**转换 TensorRT 引擎（推荐，大幅提升性能）：**

在 Jetson 上执行：

```bash
cd ~/jetson-style-transfer
source venv/bin/activate

# 转换所有模型（使用 Python API，约 2-3 分钟/模型）
python3 scripts/convert_tensorrt.py --all
```

转换完成后检查：

```bash
ls -la models/trt/
# 应该看到 .engine 文件（每个约 5MB）
```

**验证安装：**

```bash
cd ~/jetson-style-transfer
source venv/bin/activate

# 检查 TensorRT 是否可用
python3 -c "import tensorrt; print(tensorrt.__version__)"

# 检查模型文件
ls -la models/trt/

# 检查 CUDA 是否可用
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 6. 连接摄像头

**检查摄像头是否识别：**

```bash
ls /dev/video*
# 应该看到 /dev/video0
```

**测试 CSI 摄像头：**

```bash
nvgstcapture-1.0 -m wb:sunny
# 应该能看到实时画面
```

**排查摄像头问题：**

如果 `/dev/video*` 不存在：

1. 检查 CSI 排线是否插紧（蓝色面朝下）
2. 重启 Jetson：`sudo reboot`
3. 检查摄像头模块是否支持：`ls /dev/nvhost-*`

### 7. 启动应用

```bash
cd ~/jetson-style-transfer
source venv/bin/activate

# 使用 TensorRT 后端（推荐，GPU 加速）
python3 -m app.main --backend tensorrt

# 使用图片测试（无需摄像头）
python3 -m app.main --image test.jpg --backend tensorrt
```

---

## 操作方式

### 键盘快捷键

| 按键 | 功能 |
|------|------|
| `空格` | 拍照 |
| `r` | 开始/停止录像 |
| `Q` | 退出 |

### 启动参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--camera` | 0 | 摄像头 ID |
| `--image` | - | 使用本地图片测试 |
| `--backend` | auto | 推理后端 (auto/tensorrt/onnx) |
| `--style` | 配置文件默认值 | 初始风格 ID |
| `--models` | models | 模型目录 |

---

## 风格配置

编辑 `config/styles.json` 添加或修改风格：

```json
{
  "styles": [
    {
      "id": 1,
      "name": "动漫",
      "icon": "🎨",
      "base_model": "animegan_v2.onnx",
      "model_url": "",
      "input_size": 384,
      "normalize": "minus_one_to_one",
      "sharpen": 0.3
    }
  ],
  "settings": {
    "default_style_id": 1,
    "max_fps": 60
  }
}
```

**字段说明：**

| 字段 | 说明 |
|------|------|
| `base_model` | ONNX 模型文件名（相对于 `models/`） |
| `model_url` | 模型下载地址（可选） |
| `input_size` | 模型输入尺寸（默认 384） |
| `normalize` | 归一化方式（`minus_one_to_one` 或 `zero_to_one`） |
| `sharpen` | 锐化强度（0=关闭，0.3=轻度，0.5=中度），补偿 GAN 输出模糊 |

## 添加自定义风格

1. 准备一个 GAN 风格迁移的 ONNX 模型
2. 将模型文件放到 `models/` 目录
3. 在 `config/styles.json` 中添加风格配置
4. 运行 `python3 scripts/convert_tensorrt.py --style <ID>` 转换 TensorRT 引擎
5. 运行应用即可使用新风格

## 参数调优指南

### 可调参数

| 参数 | 文件位置 | 默认值 | 说明 |
|------|---------|--------|------|
| `strength` | `app/main.py` | 1.0 | 风格强度，>1会推理两次，变慢 |
| `skip_frames` | `app/main.py` | 0 | 跳帧数，视觉更流畅但不提高实际帧率 |
| `sharpen` | `config/styles.json` | 0.3 | 锐化强度（0=关闭，0.3=轻度，0.5=中度） |
| 输出分辨率 | `app/main.py` | 800x600 | 摄像头/显示/录制分辨率 |

### 帧率参考（AnimeGANv2，Jetson Orin Super Nano）

| input_size | 帧率 | 画质 |
|-----------|------|------|
| 384 | ~12-15 FPS | 标准 |

### 其他参数直接生效，无需重编

`strength`、`skip_frames`、`sharpen`、输出分辨率这些参数改完直接运行即可，不需要重编 engine。

---

## 系统优化（Jetson 上）

```bash
# 设置性能模式
sudo nvpmodel -m 0
sudo jetson_clocks

# 设置风扇速度（0-255）
echo 200 | sudo tee /sys/devices/pwm-fan/target_duty_cycle
```

---

## 开机自启动（可选）

让应用开机自动运行：

```bash
# 创建启动脚本
cat > ~/start_style_transfer.sh << 'EOF'
#!/bin/bash
cd /home/nvidia/jetson-style-transfer
source venv/bin/activate
python3 -m app.main --backend tensorrt
EOF

chmod +x ~/start_style_transfer.sh

# 添加到自启动
mkdir -p ~/.config/autostart
cat > ~/.config/autostart/style-transfer.desktop << 'EOF'
[Desktop Entry]
Type=Application
Name=Style Transfer
Exec=/home/nvidia/start_style_transfer.sh
X-GNOME-Autostart-enabled=true
EOF
```

---

## 常见问题

### nvidia-smi 报错 "Driver/library version mismatch"

```bash
sudo reboot
```

### TensorRT 转换失败 "Could not find any implementation"

ONNX 模型包含动态维度，TensorRT 无法处理。确保使用 `prepare_models.py` 生成的 ONNX 文件（不含 dynamic_axes），或重新在 Windows 上运行：

```powershell
del models\*.onnx
python scripts\prepare_models.py
scp models\*.onnx nvidia@<IP>:~/jetson-style-transfer/models/
```

### "torch.cuda.Stream requires CUDA support"

安装的是 CPU 版 PyTorch，需要 CUDA 版本：

```bash
pip uninstall torch torchvision -y
pip install torch torchvision --extra-index-url https://pypi.nvidia.com
```

### CUDA 推理报错 "no kernel image"?

ONNX Runtime 的 CUDA 提供程序在 Jetson 上可能不兼容。使用 TensorRT 后端：

```bash
python3 -m app.main --backend tensorrt
```

### 摄像头画面卡顿

```bash
# 检查 GPU 使用情况
tegrastats

# 设置风扇全速
echo 255 | sudo tee /sys/devices/pwm-fan/target_duty_cycle
```

### 处理速度慢

1. **确保使用 TensorRT 后端**：
   ```bash
   python3 -m app.main --backend tensorrt
   ```

2. **设置性能模式**：
   ```bash
   sudo nvpmodel -m 0 && sudo jetson_clocks
   ```

3. **检查散热**：
   ```bash
   cat /sys/class/thermal/thermal_zone*/temp
   ```

### TensorRT 引擎不存在

```bash
# 查看可用的 TensorRT 引擎
ls -la models/trt/

# 如果没有，需要先转换
python3 scripts/convert_tensorrt.py --all
```

### 模型下载失败？

GitHub 在国内可能不稳定。使用 `prepare_models.py` 在 Windows 上下载后传过去，或使用浏览器手动下载后放到 `models/` 目录。

---

## 调试技巧

```bash
# 查看 GPU 使用情况
tegrastats

# 查看温度
cat /sys/class/thermal/thermal_zone*/temp

# 查看摄像头设备
ls /dev/video*

# 测试 CSI 摄像头
nvgstcapture-1.0 -m wb:sunny

# 检查 TensorRT 版本
python3 -c "import tensorrt; print(tensorrt.__version__)"

# 检查 CUDA 是否可用
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## 性能参考

| 指标 | 数值 |
|------|------|
| 单张图片推理时间 | ~8-15ms（TensorRT FP16） |
| FPS | 80-150 |
| 模型大小 | ~8.5MB x 3 |
| TensorRT Engine | ~5MB x 3 |
| GPU 温度（满载） | 60-75°C |
| 功耗 | 7-15W |

---

## License

MIT License
