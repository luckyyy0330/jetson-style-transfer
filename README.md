# Jetson 风格转换设备

基于 NVIDIA Jetson Orin Nano 的实时艺术风格转换硬件产品。使用 GAN 模型（AnimeGANv2 等）对摄像头画面进行风格迁移，支持拍照保存和 SD 卡/USB 导出。

## 功能特性

- **实时风格转换** - GAN 单次前向推理，512×512 下 80-150 FPS
- **预设风格切换** - 每个风格对应独立的 ONNX 模型，按键 1/5 即可切换
- **摄像头预览** - OpenCV 窗口显示原始 + 风格化画面并排对比
- **一键拍照** - 保存原图、风格化图、并排对比图
- **多种输入源** - CSI 摄像头（IMX219）、USB 摄像头、本地图片
- **SD 卡/USB 导出** - 拍摄内容自动同步到外部存储
- **TensorRT 加速** - 自动检测并使用最优后端，帧率再提升 2-3 倍

## 技术方案

基于 **GAN（生成对抗网络）** 的轻量级风格迁移，每个风格使用独立的 ONNX 模型：

| 特性 | 说明 |
|------|------|
| 模型架构 | GAN（AnimeGANv2 / CartoonGAN 等） |
| 推理 | 多后端支持：TensorRT（推荐）> ONNX Runtime |
| 输入 | 512×512 图像（可配置） |
| 模型大小 | ~8.5MB（远小于 SD 的 ~4GB） |
| 显存占用 | ~200-500MB |
| 速度 | ONNX: 40-70 FPS，TensorRT: 80-150 FPS |
| 风格 | 每个风格配置独立的 ONNX 模型文件 |

## 性能

| 后端 | 单帧推理 | FPS (512×512) | 显存占用 |
|------|----------|---------------|----------|
| ONNX Runtime (CUDA) | ~15ms | 40-70 | ~300MB |
| TensorRT FP16 | ~8ms | 80-150 | ~200MB |
| TensorRT FP32 | ~12ms | 60-100 | ~400MB |

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 输入尺寸 | 512 | 模型输入分辨率（正方形） |
| 归一化 | minus_one_to_one | [-1, 1] 范围（大多数 GAN 模型） |

## 硬件要求

| 硬件 | 规格 |
|------|------|
| 主控板 | Jetson Orin Nano 8GB Developer Kit |
| 摄像头 | CSI IMX219 摄像头模块（或 USB 摄像头） |
| 显示器 | HDMI 显示器（用于 OpenCV 窗口预览） |
| 存储 | 64GB NVMe SSD + 32GB microSD |
| 电源 | 12V/3A DC 适配器 |

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
│   │   ├── gan_engine.py            # GAN 推理引擎（ONNX Runtime）
│   │   └── tensorrt_utils.py        # TensorRT 推理工具
│   └── export/
│       └── storage.py               # 文件保存、SD 卡/USB 同步
│
├── models/                          # 模型文件（通过 download_models.py 下载）
│   ├── animegan_v2.onnx             # 动漫风格模型
│   ├── animegan_v2_paprika.onnx     # 宫崎骏风格模型
│   ├── trt/                         # TensorRT 引擎（转换生成）
│   └── trt_cache/                   # TensorRT 缓存
│
├── scripts/                         # 工具脚本
│   ├── download_models.py           # 按风格下载 ONNX 模型
│   ├── convert_tensorrt.py          # 转换 TensorRT 引擎
│   └── setup_device.py              # 设备初始化
│
├── config/
│   ├── styles.json                  # 风格配置（模型路径和参数）
│   └── device.json                  # 设备参数
│
├── docs/
│   ├── architecture.md              # 架构文档
│   └── setup_guide.md               # 完整搭建指南
│
├── assets/
│   └── captures/                    # 拍照/录像保存目录
│
├── requirements.txt                 # Python 依赖
└── README.md                        # 本文件
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 如果 ONNX Runtime GPU 安装有问题
pip install onnxruntime-gpu
```

### 2. 下载模型

```bash
# 下载所有风格所需的模型
python scripts/download_models.py --all

# 只下载指定风格的模型
python scripts/download_models.py --style 1    # 动漫
python scripts/download_models.py --style 2    # 写实
```

### 3. 转换 TensorRT 引擎（推荐，大幅提升性能）

```bash
# 转换所有风格的 TensorRT 引擎
python scripts/convert_tensorrt.py --all

# 只转换指定风格
python scripts/convert_tensorrt.py --style 1
```

### 4. 运行应用

```bash
# 启动应用（自动检测最优后端）
python -m app.main

# 指定初始风格
python -m app.main --style 1

# 指定使用 TensorRT 后端
python -m app.main --backend tensorrt

# 使用本地图片测试
python -m app.main --image test.jpg
```

## 操作方式

### 键盘快捷键

| 按键 | 功能 |
|------|------|
| `1` | 切换到动漫风格 |
| `2` | 切换到写实风格 |
| `3` | 切换到素描风格 |
| `4` | 切换到宫崎骏风格 |
| `5` | 切换到人脸动漫风格 |
| `空格` | 拍照 |
| `Q` | 退出 |

### 启动参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--camera` | 0 | 摄像头 ID |
| `--usb` | false | 使用 USB 摄像头 |
| `--image` | - | 使用本地图片测试 |
| `--backend` | auto | 推理后端 (auto/tensorrt/onnx) |
| `--style` | 配置文件默认值 | 初始风格 ID |
| `--models` | models | 模型目录 |

## 风格配置

编辑 `config/styles.json` 添加或修改风格。每个风格指定独立的 ONNX 模型：

```json
{
  "styles": [
    {
      "id": 1,
      "name": "动漫",
      "icon": "🎨",
      "base_model": "animegan_v2.onnx",
      "model_url": "https://github.com/.../animegan_v2.onnx",
      "input_size": 512,
      "normalize": "minus_one_to_one"
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
| `input_size` | 模型输入尺寸（默认 512） |
| `normalize` | 归一化方式（`minus_one_to_one` 或 `zero_to_one`） |

## 添加自定义风格

1. 准备一个 GAN 风格迁移的 ONNX 模型
2. 将模型文件放到 `models/` 目录
3. 在 `config/styles.json` 中添加风格配置
4. 运行应用即可使用新风格

支持的模型格式：
- ONNX（推荐，直接使用）
- TensorRT engine（通过 `convert_tensorrt.py` 转换）

## 系统优化（Jetson 上）

```bash
# 设置性能模式
sudo nvpmodel -m 0
sudo jetson_clocks

# 设置风扇速度（0-255）
echo 200 | sudo tee /sys/devices/pwm-fan/target_duty_cycle
```

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

# 列出可用摄像头
python -m app.main --list-cameras

# 查看 ONNX 模型信息
python -c "import onnxruntime as ort; print(ort.InferenceSession('models/animegan_v2.onnx').get_inputs())"
```

## 常见问题

### ONNX Runtime 安装失败？

```bash
# Jetson 上安装 GPU 版本
pip install onnxruntime-gpu

# 验证安装
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

### 摄像头无法打开？

```bash
# 检查设备节点
ls /dev/video*

# CSI 摄像头：检查排线是否插紧（蓝色面朝下），重启 Jetson
# USB 摄像头：加 --usb 参数
```

### 模型下载失败？

```bash
# 检查网络连接
ping github.com

# 手动下载模型后放到 models/ 目录
# 查看 config/styles.json 中的 model_url 获取下载地址
```

### 处理速度太慢？

1. **转换 TensorRT 引擎**（推荐，提升 2-3 倍）：
   ```bash
   python scripts/convert_tensorrt.py --all
   ```
2. 降低输入分辨率（修改 `config/styles.json` 中的 `input_size`）
3. 设置性能模式：`sudo nvpmodel -m 0 && sudo jetson_clocks`
4. 检查散热：温度过高会降频

### 显存不足？

- GAN 模型显存占用很低（~300MB），一般不会不足
- 如果不够，尝试关闭其他 GPU 应用

## 搭建指南

详细的硬件准备、系统烧录、环境配置、部署流程请参考 [docs/setup_guide.md](docs/setup_guide.md)。

## License

MIT License
