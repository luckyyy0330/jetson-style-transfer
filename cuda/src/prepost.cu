// Fused preprocess / postprocess CUDA kernels
// 移植自 jetson_yolodetect/src/letterbox.cu（去掉 letterbox pad，改为纯 stretch resize）
// 5-in-1 preprocess: color + resize + normalize + HWC→NCHW + type
// 4-in-1 postprocess: denorm + clip + NCHW→HWC + RGB→BGR
#include "style_cuda/prepost.h"
#include "style_cuda/pixel.cuh"

#include <cuda_fp16.h>

namespace style_cuda {
namespace {

// 融合预处理核：每线程 2×2 tile，一次读一次写
__device__ __forceinline__ void preprocess_tile(const DeviceImage& image,
                                                 void* __restrict__ dst,
                                                 int dst_width, int dst_height,
                                                 float norm_a, float norm_b,
                                                 int out_dtype) {
    const int src_width = image.width, src_height = image.height;
    const int base_x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int base_y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    const float scale_x = src_width * __frcp_rn(dst_width);
    const float scale_y = src_height * __frcp_rn(dst_height);
    const size_t plane = static_cast<size_t>(dst_width) * dst_height;

#pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        const int py = base_y + dy;
        if (py >= dst_height) continue;
#pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            const int px = base_x + dx;
            if (px >= dst_width) continue;

            // half-pixel 坐标，clamp 到边界（border replicate）
            const float sx = fminf(fmaxf(__fmaf_rn(px + 0.5f, scale_x, -0.5f), 0.0f), src_width - 1.0f);
            const float sy = fminf(fmaxf(__fmaf_rn(py + 0.5f, scale_y, -0.5f), 0.0f), src_height - 1.0f);
            const int x0 = __float2int_rd(sx);
            const int y0 = __float2int_rd(sy);
            const int x1 = min(x0 + 1, src_width - 1);
            const int y1 = min(y0 + 1, src_height - 1);
            const float a = sx - x0, b = sy - y0;
            const float w00 = (1.0f - a) * (1.0f - b);
            const float w10 = a * (1.0f - b);
            const float w01 = (1.0f - a) * b;
            const float w11 = a * b;

            const uchar3 p00 = rgb(image, x0, y0);
            const uchar3 p10 = rgb(image, x1, y0);
            const uchar3 p01 = rgb(image, x0, y1);
            const uchar3 p11 = rgb(image, x1, y1);

            float r = fmaf(w00, p00.x, fmaf(w10, p10.x, fmaf(w01, p01.x, w11 * p11.x)));
            float g = fmaf(w00, p00.y, fmaf(w10, p10.y, fmaf(w01, p01.y, w11 * p11.y)));
            float b = fmaf(w00, p00.z, fmaf(w10, p10.z, fmaf(w01, p01.z, w11 * p11.z)));

            // 归一化
            r = fmaf(r, norm_a, norm_b);
            g = fmaf(g, norm_a, norm_b);
            b = fmaf(b, norm_a, norm_b);

            // HWC→NCHW，R 在 plane 0，G 在 plane 1，B 在 plane 2
            if (out_dtype == 0) {
                auto* out = static_cast<float*>(dst) + static_cast<size_t>(py) * dst_width + px;
                out[0] = r;
                out[plane] = g;
                out[2 * plane] = b;
            } else {
                auto* out = static_cast<__half*>(dst) + static_cast<size_t>(py) * dst_width + px;
                out[0] = __float2half(r);
                out[plane] = __float2half(g);
                out[2 * plane] = __float2half(b);
            }
        }
    }
}

__global__ void preprocess_kernel(const PreDesc* descs, void* dst) {
    const auto& desc = descs[blockIdx.z];
    void* output = static_cast<char*>(dst) +
        static_cast<size_t>(blockIdx.z) * 3 * desc.meta.x * desc.meta.y *
        (desc.out_dtype == 0 ? sizeof(float) : sizeof(__half));
    preprocess_tile(desc.image, output, desc.meta.x, desc.meta.y,
                    desc.norm_a, desc.norm_b, desc.out_dtype);
}

// 融合后处理核：NCHW float32/16 → BGR uint8 HWC
__device__ __forceinline__ void postprocess_tile(const void* __restrict__ src,
                                                  unsigned char* __restrict__ dst,
                                                  int width, int height,
                                                  float denorm_a, float denorm_b,
                                                  int in_dtype) {
    const int base_x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int base_y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    const size_t plane = static_cast<size_t>(width) * height;
    const size_t dst_pitch = static_cast<size_t>(width) * 3;

#pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        const int py = base_y + dy;
        if (py >= height) continue;
#pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            const int px = base_x + dx;
            if (px >= width) continue;

            const size_t idx = static_cast<size_t>(py) * width + px;
            float r, g, b;
            if (in_dtype == 0) {
                const auto* in = static_cast<const float*>(src);
                r = in[idx];
                g = in[idx + plane];
                b = in[idx + 2 * plane];
            } else {
                const auto* in = static_cast<const __half*>(src);
                r = __half2float(in[idx]);
                g = __half2float(in[idx + plane]);
                b = __half2float(in[idx + 2 * plane]);
            }

            // 反归一化 + clip
            r = fmaf(r, denorm_a, denorm_b);
            g = fmaf(g, denorm_a, denorm_b);
            b = fmaf(b, denorm_a, denorm_b);

            // RGB → BGR，写 HWC uint8
            unsigned char* out = dst + py * dst_pitch + px * 3;
            out[0] = byte(b);
            out[1] = byte(g);
            out[2] = byte(r);
        }
    }
}

__global__ void postprocess_kernel(const PostDesc* descs, const void* src,
                                    unsigned char* dst_bgr) {
    const auto& desc = descs[0]; // 单帧
    postprocess_tile(src, dst_bgr, desc.meta.x, desc.meta.y,
                     desc.denorm_a, desc.denorm_b, desc.in_dtype);
}

} // namespace

cudaError_t cuda_preprocess(const PreDesc* desc, void* dst, cudaStream_t stream) {
    if (!desc || !dst) return cudaErrorInvalidValue;
    const dim3 block(16, 16);
    const dim3 grid((desc->meta.x + 31) / 32, (desc->meta.y + 31) / 32, 1);
    preprocess_kernel<<<grid, block, 0, stream>>>(desc, dst);
    return cudaGetLastError();
}

cudaError_t cuda_postprocess(const PostDesc* desc, const void* src,
                              unsigned char* dst_bgr, cudaStream_t stream) {
    if (!desc || !src || !dst_bgr) return cudaErrorInvalidValue;
    const dim3 block(16, 16);
    const dim3 grid((desc->meta.x + 31) / 32, (desc->meta.y + 31) / 32, 1);
    postprocess_kernel<<<grid, block, 0, stream>>>(desc, src, dst_bgr);
    return cudaGetLastError();
}

} // namespace style_cuda
