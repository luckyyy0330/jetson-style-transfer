# 风格迁移程序性能分析

本文档分析程序每一步的耗时和算力消耗，基于 Jetson Orin Nano + TensorRT 后端。

---

## 每帧处理流程

```
摄像头采集 → 跳帧判断 → 预处理 → GAN推理 → 后处理 → 缩放回原尺寸 → 显示渲染 → 录像写入
```

---

## 各步骤详解

### 1. 摄像头采集

| 项目 | 详情 |
|------|------|
| 耗时 | ~3-8ms |
| CPU 占用 | <5%（仅 DMA 搬运） |
| GPU 占用 | ISP 硬件处理，NVMM 显存传递 |
| 代码位置 | `app/camera/csi_camera.py` GStreamer 管道 |

**处理链路：**

```
CSI 传感器 → Bayer 原始数据 → ISP 转 NV12 → nvvidconv 硬件转 BGRx → videoconvert 去 alpha → BGR 帧
```

- `nvarguscamerasrc`：NVIDIA CSI 摄像头驱动，读取传感器原始数据
- `nvvidconv`：硬件色彩空间转换，走 NVMM 显存，不占 CPU
- `videoconvert`：纯 CPU 格式微调（BGRx → BGR），耗时 <1ms

---

### 2. 跳帧判断

| 项目 | 详情 |
|------|------|
| 耗时 | <0.1ms |
| CPU 占用 | 可忽略 |
| GPU 占用 | 无 |
| 代码位置 | `app/main.py` `_process_frame()` |

- 一次取模运算，跳帧时直接复用上一帧结果，跳过后续所有处理

---

### 3. 预处理

| 项目 | 详情 |
|------|------|
| 耗时 | ~2-5ms |
| CPU 占用 | numpy 运算，单核 |
| GPU 占用 | 无 |
| 代码位置 | `app/style/gan_engine.py` `_preprocess()` |

**处理步骤：**

| 步骤 | 操作 | 耗时 |
|------|------|------|
| 1 | BGR → RGB（`cv2.cvtColor`） | ~1ms |
| 2 | Resize 到模型输入尺寸（384×384） | ~1ms |
| 3 | 归一化到 [-1, 1]（`/127.5 - 1.0`） | <1ms |
| 4 | HWC → NCHW（`np.transpose`） | <1ms |
| 5 | 增加 batch 维度（`np.expand_dims`） | <1ms |

---

### 4. GAN 推理（主要瓶颈）

| 项目 | 详情 |
|------|------|
| 耗时 | ~30-80ms/次 |
| CPU 占用 | <10% |
| GPU 占用 | **90%+**，显存 200-500MB |
| 代码位置 | `app/style/gan_engine.py` `transfer()` → `_infer_tensorrt()` |

**推理流程：**

```
预处理数据(NCHW) → 复制到 GPU(PyTorch CUDA) → TensorRT execute_async_v3 → 同步等待 → 读取结果回 CPU
```

**strength 对推理的影响：**

| strength | 推理次数 | 推理总耗时 | 说明 |
|----------|---------|-----------|------|
| 1.0 | 1 次 | ~30-80ms | 标准强度 |
| 1.5 | 2 次 | ~60-160ms | 第二次推理结果与第一次混合 50% |
| 2.0 | 2 次 | ~60-160ms | 两次完整推理结果直接叠加 |

**模型信息：**

| 项目 | 详情 |
|------|------|
| 模型 | AnimeGANv2 生成器 |
| 参数量 | ~8M |
| 输入 | 1×3×384×384 float32 |
| 输出 | 1×3×384×384 float32 |
| 推理后端 | TensorRT FP16（Jetson）/ ONNX Runtime CPU |

---

### 5. 后处理

| 项目 | 详情 |
|------|------|
| 耗时 | ~2-5ms |
| CPU 占用 | numpy 运算 |
| GPU 占用 | 无 |
| 代码位置 | `app/style/gan_engine.py` `_postprocess()` |

**处理步骤：**

| 步骤 | 操作 | 耗时 |
|------|------|------|
| 1 | 去 batch 维度（`np.squeeze`） | <1ms |
| 2 | NCHW → HWC（`np.transpose`） | <1ms |
| 3 | 反归一化 [-1,1] → [0,255]（`(result + 1.0) × 127.5`） | <1ms |
| 4 | Clip 到 [0, 255] 并转 uint8 | <1ms |
| 5 | RGB → BGR（`cv2.cvtColor`） | ~1ms |

---

### 6. 缩放回原尺寸

| 项目 | 详情 |
|------|------|
| 耗时 | ~1-3ms |
| CPU 占用 | OpenCV `resize` |
| GPU 占用 | 无 |
| 代码位置 | `app/style/gan_engine.py` `transfer()` 末尾 |

- 从模型输入尺寸（384×384）缩放回摄像头分辨率（800×600）
- 使用 `cv2.INTER_LINEAR` 双线性插值

---

### 7. 显示渲染

| 项目 | 详情 |
|------|------|
| 耗时 | ~3-8ms |
| CPU 占用 | numpy 拼接 + OpenCV 绘制 |
| GPU 占用 | OpenGL 窗口渲染 |
| 代码位置 | `app/main.py` `_process_frame()` |

**处理步骤：**

| 步骤 | 操作 | 耗时 |
|------|------|------|
| 1 | `np.hstack` 原图 + 风格化图左右拼接 | ~1-2ms |
| 2 | `cv2.putText` 叠加 FPS/后端/风格信息 | <1ms |
| 3 | `cv2.imshow` 渲染到窗口（800×600） | ~2-5ms |
| 4 | `cv2.waitKey` 处理键盘事件 | <1ms |

---

### 8. 录像写入（仅录制时）

| 项目 | 详情 |
|------|------|
| 耗时 | ~3-8ms |
| CPU 占用 | 视频编码 |
| GPU 占用 | 无 |
| 代码位置 | `app/main.py` `_process_frame()` 录像段 |

- 同时写入两路视频：原始帧 + 风格化帧
- 编码器：mp4v（优先）或 XVID（备选）
- 帧率：使用实际处理帧率，避免视频加速

---

## 汇总

### 实测数据（Jetson Orin Super Nano 8G，TensorRT FP16，strength=1.0）

#### transfer() 内部各步骤（实测，去掉首次预热）

| 步骤 | 平均耗时 | 占比 |
|------|---------|------|
| 预处理 | ~2.2ms | 3.3% |
| **GAN 推理** | **~61.5ms** | **91.9%** |
| 后处理 | ~2.5ms | 3.7% |
| 混合 | ~0.0ms | 0% |
| 缩放回原尺寸 | ~1.3ms | 1.9% |
| **transfer() 合计** | **~67.5ms** | 100% |

#### transfer() 外部

| 步骤 | 平均耗时 | 占比 |
|------|---------|------|
| 摄像头采集 | ~5ms | 7% |
| 显示渲染 | ~5ms | 7% |
| 其他 | ~0.5ms | 1% |

#### 整帧汇总

| 指标 | 数值 |
|------|------|
| **transfer() 内部** | **~67.5ms** |
| **整帧总计（含采集+显示）** | **~78ms** |
| **实际帧率** | **~13 FPS** |

#### 推理周期性波动

推理耗时存在 ~4ms 的周期性波动（60ms ↔ 64ms），每5帧出现一次峰值，可能与 GPU 调度或显存对齐有关。
| strength=1.5，录像中 | ~85-200ms | 5-11 FPS |
| strength=2.0，录像中 | ~85-200ms | 5-11 FPS |

### 资源占用

| 资源 | 用途 | 占用量 |
|------|------|--------|
| GPU | GAN 推理（TensorRT/CUDA） | 推理期间 90%+，显存 200-500MB |
| CPU | 预处理/后处理/显示/录像编码 | ~1-2 核 |
| 显存 | 模型权重 + 输入输出缓冲区 | ~200-500MB |
| 内存 | 帧缓冲（原始帧 + 风格化帧 + 拼接帧） | ~15-30MB |
| ISP | 摄像头采集 | 硬件独立，不占 CPU/GPU |

### 瓶颈排序

```
GAN 推理 (61.5ms, 79%) >>>>>> 摄像头 (5ms, 6%) ≈ 显示 (5ms, 6%) > 后处理 (2.5ms) ≈ 预处理 (2.2ms) > 缩放 (1.3ms)
```

**实测 GAN 推理占整帧时间的 79%**，是唯一的性能瓶颈。要提升帧率，最有效的手段是减少推理次数（降低 strength）或降低推理分辨率。

---
---

# Phase 0 / Phase 1 — 工程优化与上机验收

> 上面的 78ms/13fps 是 **Phase 0 之前的基线**。本节记录 Phase 0 修掉的污染源、
> Phase 1 的工程改造，以及在 Jetson 上的验收步骤。

## 关于 "67 TOPS 只用上 1 TOPS"

**这个说法本身不该作为优化目标。** 67 TOPS 是 INT8-sparse 的营销峰值：
- FP32/FP16 模型根本用不到 INT8 峰值，两者不能直接比较
- AnimeGANv2 生成器 ~8M 参数 @480×360 约 80–150 GFLOPs，本来只需要 ~1 TOPS 级算力
- 视频分析盒子一路 YOLO 也就 ~1 TOPS，靠**并发 N 路**用满芯片，而不是把单模型顶到峰值

**目标不是提高 TOPS 利用率，而是把 78ms/帧压到 ≤33ms/帧（30fps）。**
61.5ms 的"推理"里有一半是工程税（H2D/D2D/D2H/同步/launch），不是模型算不动。

## Phase 0 — 修掉污染基线的 bug

| Bug | 影响 | 修复 |
|---|---|---|
| `style_manager._load_config` 漏传 `sharpen` | `Style.sharpen` 永远是 dataclass 默认 `0.3`，`config/styles.json` 的 `0.0` 无效 → **每帧白跑一次 `GaussianBlur`+`addWeighted`**；同时也是画质 bug（强制锐化），会污染后续所有画质闸门 | `sharpen=float(style_data.get('sharpen', 0.0))`，dataclass 默认改为 `0.0` |
| `pipeline_size == input_size` 时 `cv2.resize` 回原尺寸 | 同尺寸空转 | 加短路条件 |
| TRT engine 缺失时**不回退 ONNX** | 启动直接失败 | `load_model` 按 `TENSORRT → ONNX_RUNTIME` 顺序尝试 |
| 引擎名 `style_{id}.engine` 不含分辨率/精度 | 改 `input_size` 会**静默复用旧引擎** | 新命名 `style_{id}_{w}x{h}_{prec}.engine` + `--force` |
| `benchmark_run.sh` 写 `input_size` 为标量 | `csi_camera.py:70` / `main.py` 期望 `(W,H)` 元组 | 改为 `[W, H]`，分辨率表加 `480x360` |
| `_infer_tensorrt` 计时窗口包了 H2D+多余 D2D+`synchronize()`+整图 D2H | "推理 61.5ms" 不是 GPU 计算 | 拆成 `t_h2d` / `t_exec` / `t_d2h` 三段 |

### 计时口径（两种不可混用）

| 口径 | 来源 | 含义 |
|---|---|---|
| **GPU Compute Time** | `trtexec` stdout | 纯 GPU 计算下限，不含主机侧 launch/同步 |
| **`steady_infer_ms`** | 主机侧 `[性能]` 日志 | 含 H2D/D2H/launch/sync 的墙钟时间 |

首帧（TRT 懒初始化 + Graph 捕获 + 首次 DMA 注册）**排除**在稳态统计外。

## Phase 1 — 工程改造（已完成的代码改动）

### 1.1 新增 pybind11 CUDA 扩展 `style_cuda`（`cuda/`）

照抄参考工程 `jetson_yolo` 的设计，**不自创抽象**：

| 文件 | 移植自 | 内容 |
|---|---|---|
| `cuda/include/style_cuda/cuda_request.h` | `jetson_media/cuda_request.h` | `CudaRequest` 语义照搬：warmup `enqueue()`+`cudaStreamSynchronize` 冲 TRT 懒初始化 → `cudaStreamBeginCapture(ThreadLocal)` → `enqueue()` → `EndCapture` → `Instantiate`；捕获中 `inputRead` 用 **`cudaEventRecordExternal`**（普通事件无法通知主机）；失败 `failed_=true` **永久**回退 stream 模式 |
| `cuda/include/style_cuda/cuda_descriptor.h` | `jetson_media/cuda_descriptor.h` | `CudaDescriptor<T>`（integrated GPU 上 mapped host 参数块）⇒ 改分辨率/换风格**不触发 Graph 重捕获** |
| `cuda/include/style_cuda/pixel.cuh` | `jetson_media/pixel.cuh` | `rgb(DeviceImage)`：BT.601/709 × limited/full 的 NV12 + BGR8/RGB8/BGRX8 |
| `cuda/src/prepost.cu` | `letterbox.cu` 的 `letterbox_media_tile`（去 letterbox pad） | **融合预处理核（5-in-1）**：源格式→RGB + 双线性 half-pixel resize + 归一化 + HWC→NCHW + 类型转换，一次读一次写，2×2 tile。**融合后处理核（4-in-1）**：NCHW fp16/fp32 → 反归一化 + clip + round-uint8 + NCHW→HWC + RGB→BGR，直写 `cudaHostAllocMapped` 输出（免 D2H） |
| `cuda/src/bindings.cpp` | 新写 | `GpuPipeline`：一个 `CudaRequest` + 描述符 + mapped 缓冲的薄属主 |

**Python API（三段式）：**
```python
pipe = style_cuda.GpuPipeline(width=480, height=360, norm=0, io_dtype="fp32")
pipe.copy_source(frame)                      # memcpy 到 mapped staging
pipe.set_source(w, h, pitch, fmt, ...)       # 仅 idle 时可调（CudaRequest::idle 会校验）
pipe.run(trt_enqueue)                        # Submit：warmup+capture 或 cudaGraphLaunch 或 stream 回退
pipe.release_input()                         # 只 sync(inputRead) → staging 可复用，TRT 还在跑
out_bgr = pipe.wait()                        # sync(completed) → **owned** BGR uint8 数组
out_view = pipe.wait_view()                  # 零拷贝 view（alias mapped，下帧覆写，自担风险）
pipe.graph_active()                          # → graph=1|0 写入 [性能] 日志
```

**关键设计约束（踩过的坑）：**
1. `set_tensor_address` **在 load 时做一次**，之后不再改 —— 这是 Graph 地址稳定的前提
2. `set_source` 只能在 idle 时调（`CudaRequest::idle()` 在 pending 时抛异常）
3. `wait()` **返回持有副本**，不是 mapped view。原因：`np.ascontiguousarray` 对**已经连续**的 view 不拷贝，别名根本断不开 —— 下一帧后处理核会静默覆写"已保存"的图。要零拷贝用 `wait_view()` 并自己保证在下次 `wait()` 前拷走。
4. `cudaHostAllocMapped` staging 在预处理核读它期间 CPU 不可写，`release_input()` 是唯一的复用许可
5. 绝不在队列锁内 `cudaFree` / `cudaFreeHost` / Gst teardown

### 1.2 `GANEngine` 三段式拆分（`app/style/gan_engine.py`）

- 快路径（`backend=="tensorrt"` 且 `style_cuda` 可导入且 shape 静态）：持有 `GpuPipeline`
- **删掉** `torch.from_numpy(...).to("cuda", non_blocking=True)`（未 pinned ⇒ `non_blocking` 是假的）
- **删掉** 多余的 `device_in.copy_` 和两个 `torch.empty(..., float32)`、`self._torch_stream`
- `transfer()` 拆成 **`copy_source` / `submit` / `release_input` / `wait`** 四段
- 新增 `quiesce()`：等 pending 请求结束 + `resetGraph()`。换风格前必须在推理线程内调用
- **ONNX-CPU 回退原样保留**（`_preprocess` / `_postprocess` / `session.run`）

### 1.3 采集去税（`app/camera/csi_camera.py`）

| 之前 | 现在 |
|---|---|
| `nvvidconv ! BGRx ! **videoconvert** ! BGR ! appsink` | `nvvidconv ! **NV12** ! appsink`（回退 BGRx） |
| CPU `videoconvert` 强制 NVMM→system 下载 + CPU BGRx→BGR | **无 CPU 色彩转换**，融合预处理核直接吃 NV12（1.5 B/px） |
| 回调里 `np.frombuffer().reshape()` + `frame.copy()` | 回调里 **一次** `fv.data.copy()`（Gst 缓冲 unmap 后即被 Argus 池回收，必须拷走） |
| `read_frame()` 再 `self._frame.copy()` | **删除**。改 `sink` 回调直投 `FrameQueue2` |

**借用视图契约**：`FrameView.data` 是 `np.frombuffer` 的借用视图，**仅在 sink 调用期间有效**。sink 必须在返回前拷走需要的数据。

### 1.4 三线程双槽流水（`app/main.py` + `app/pipeline/frame_queue.py`）

```
GStreamer streaming 线程 ──push──> FrameQueue2(Latest)   [采集 ~5ms 移出关键路径]
                                        │ acquire
                                        ▼
                                推理线程: copy_source → submit → release_input → wait
                                        │                 ↑ staging 提前归还
                                        │ push (orig, styled)
                                        ▼
显示（主线程）  <──acquire──  FrameQueue2(Latest)        [显示 ~5ms 移出关键路径]
        │
        └──mailbox──> 推理线程（切风格/改强度）
```

- `FrameQueue2` 照搬 `jetson_media/src/pipeline.cpp`：恰好 2 槽、`Empty|Ready|Processing`、`Latest` 只淘汰最旧的 **Ready**（**绝不碰 Processing**）、释放资源在**锁外**
- UI 线程**绝不**直接调用 `switch_style` —— 通过 mailbox 投递，推理线程内 `quiesce()` + `switch_style`
- HUD 显示 `graph:0|1` 与 `steady_infer_ms`，让 CUDA Graph 静默回退可见
- 录像改 `nvv4l2h264enc` 硬编码（回退 `cv2.VideoWriter`）

## 上机验收步骤

### Step 0 — 形状审计（必须先做）

`config/styles.json` 是 `[480,360]`，但 `models/onnx_*` 全是**正方形**导出。必须确认 `models/animegan_v2.onnx` 真实输入是 `(1,3,360,480)`：

```bash
python3 scripts/audit_shapes.py
```

不对就按 `input_size=(480,360)` 重导 ONNX（ONNX shape `(1,3,360,480)`，**注意 W,H 顺序**）。

### Step 1 — 构建 CUDA 扩展

```bash
bash scripts/build_cuda_ext.sh
python3 -c "import sys; sys.path.insert(0,'app/cuda_ext'); import style_cuda; print('ok', style_cuda.version())"
```

### Step 2 — 重建 TensorRT 引擎

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
python3 scripts/convert_tensorrt.py --all --force --io-dtype fp16 --opt-level 3
```

引擎命名 `style_{id}_{w}x{h}_{prec}.engine`。`--io-dtype fp16` 让 I/O 节点免 fp32 往返；
`--opt-level 3` 是参考工程的配方（5 是构建期成本且收益不稳）。

### Step 3 — trtexec GPU 侧对照（模型下限）

```bash
/usr/src/tensorrt/bin/trtexec --loadEngine=models/trt/style_1_480x360_fp16.engine \
    --warmUp=1000 --iterations=300 --dumpProfile
/usr/src/tensorrt/bin/trtexec --loadEngine=models/trt/style_1_480x360_fp16.engine \
    --warmUp=1000 --iterations=300 --useCudaGraph
```

**记 GPU Compute Time，不是 GPU Host Time。** stream 模式与 `--useCudaGraph` 各记一个。
`--useCudaGraph` 必须先跑通，否则运行时不该指望 `graph=1`。

### Step 4 — 运行并抓 [性能] 日志

```bash
python3 app/main.py --backend tensorrt --style 1
```

**新日志格式**（向后兼容旧字段，`benchmark_run.sh` 的正则仍匹配 `推理:` / `总耗时:` / `FPS:`）：

```
[性能] graph:1 | release_input:0.25ms | 总耗时:37.2ms | FPS:26.8
```

慢路径（无 style_cuda）仍是：
```
[性能] H2D:3.1ms | GPU计算:52.0ms | D2H:4.2ms | 预处理:2.2ms | 推理:61.5ms | 后处理:2.5ms | 缩放:0.0ms | 总耗时:67.5ms | FPS:14.8
```

### Step 5 — 帧率闸门

| 闸门 | 判据 |
|---|---|
| **Phase 1 成功** | ≥22fps 连续 60s（以 `[性能] FPS` 为准，**不用 HUD 平均**，HUD 含预热） |
| **Phase 1 优秀** | ≥25fps |
| 画质 | `test.jpg` 人眼并排无可见差异；换风格按键 2/3 允许一帧卡顿后 `graph=1` 恢复 |
| 回退回归 | `python3 app/main.py --backend onnx --image test.jpg` 仍能出图 |
| strength=1.5 | 能跑，fps ≈ strength=1.0 的一半（**已文档化，30fps 闸门是 strength=1.0**） |

### Step 6 — 若 Phase 1 < 30fps，进 Phase 2

| 手段 | 收益 | 风险 |
|---|---|---|
| **2a** 把 denorm/uint8/layout 折进 ONNX 尾部（graphsurgeon） | −1.5~−3ms | **无**（数学等价，PSNR≥45dB 闸门） |
| **2b** INT8 PTQ + 画质闸门 | 计算 ×0.6–0.7 | **GAN 上偏高**（色偏/棋盘格）。硬闸门：vs FP16 的 PSNR≥25dB **且** SSIM≥0.90，**必须人眼 A/B**。闸门不过就不把 INT8 设默认 |
| **2c** 通道剪枝 + 蒸馏 | 1.5–2.5× | **最高**（风格身份漂移）。1–2 周训练项目，**2a+2b 不够再启动** |

用 INT8 只为 2× 计算，**不为"TOPS 利用率"**。

## Phase 1 后预期账（480×360, strength=1.0, FP16）

| 阶段 | Phase 0 前 | Phase 1 后 | 跑在哪 |
|---|---|---|---|
| 采集 | 5ms | 2–3ms（流水线隐藏） | GStreamer 线程 |
| CPU 预处理 | 2.2 | **0**（融合核 0.1–0.3） | GPU，在图内 |
| H2D+D2H | 3–5（含在 61.5） | **~0**（mapped staging + mapped out） | — |
| TRT 计算 | （含在 61.5） | 35–55ms（权重不变；fp16 I/O + timing cache 省 2–6） | GPU，在图内 |
| CPU 后处理+resize | 3.8 | **0**（融合核；同尺寸跳过 resize） | GPU，在图内 |
| 显示 | 5ms | 4–6ms（流水线隐藏） | 主线程 |
| host/launch/sync | ~10 | **~0.5**（`cudaGraphLaunch` + 2 事件同步） | 推理线程 |
| **wall/frame** | **78ms → 13fps** | **~37–57ms → 18–27fps** | |

**诚实上限**：若 Step 3 测出 GPU Compute Time 是 55ms，Phase 1 顶到 18–20fps，
Phase 2b（INT8）就是 30fps 的必经之路 —— 这不是工程没做到位，是模型算不动。

