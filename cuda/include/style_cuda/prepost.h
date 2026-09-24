// Fused preprocess / postprocess kernel declarations
// 5-in-1 preprocess: color + resize + normalize + layout + type, one read one write
// 4-in-1 postprocess: denorm + clip + layout + RGB→BGR, writes to mapped host buffer
//
// 锁纪律 / 生命周期（参考工程 C10 内存台账）:
//   - staging (cudaHostAllocMapped) 在预处理核读它期间 CPU 不可写
//   - release_input() = cudaEventSynchronize(inputRead) 是唯一的复用许可
//   - wait() 返回的 mapped 输出被下一帧后处理核覆写 → 2 槽纪律保护正在显示的帧
//   - 绝不在队列锁内 cudaFree / cudaFreeHost
#pragma once
#include <cuda_runtime.h>
#include "frame.h"

namespace style_cuda {

// 归一化模式
enum class NormMode : int {
    MINUS_ONE_TO_ONE = 0,  // x / 127.5 - 1
    ZERO_TO_ONE = 1,       // x / 255
};

// 预处理描述符（稳定地址，Graph 不重捕获）
struct PreDesc {
    DeviceImage image;
    int4 meta;          // x=dst_w, y=dst_h, z=0, w=0 (纯 stretch，无 letterbox pad)
    float norm_a, norm_b; // out = pixel * norm_a + norm_b
    int out_dtype;      // 0=float32, 1=float16
};

// 后处理描述符
struct PostDesc {
    int4 meta;          // x=w, y=h, z=0, w=0
    float denorm_a, denorm_b; // pixel = out * denorm_a + denorm_b
    int in_dtype;       // 0=float32, 1=float16
};

// 融合预处理：源 → NCHW 归一化 float32/16
cudaError_t cuda_preprocess(const PreDesc* desc, void* dst, cudaStream_t stream);

// 融合后处理：NCHW float32/16 → BGR uint8（写 mapped host buffer）
cudaError_t cuda_postprocess(const PostDesc* desc, const void* src,
                             unsigned char* dst_bgr, cudaStream_t stream);

} // namespace style_cuda
