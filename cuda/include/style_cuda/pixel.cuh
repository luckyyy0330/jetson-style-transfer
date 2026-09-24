// Device-side YUV→RGB conversion (BT.601/709 × limited/full)
// 移植自 jetson_media/include/jetson_media/pixel.cuh
#pragma once
#include "frame.h"

namespace style_cuda {

__device__ inline unsigned char byte(float value) {
    return static_cast<unsigned char>(fminf(255.f, fmaxf(0.f, floorf(value + .5f))));
}

__device__ inline uchar3 rgb(const DeviceImage& image, int x, int y) {
    if (image.format == PixelFormat::BGR8 || image.format == PixelFormat::RGB8) {
        const auto* p = image.planes[0] + static_cast<size_t>(y) * image.pitch[0] + 3 * x;
        return image.format == PixelFormat::BGR8 ? make_uchar3(p[2], p[1], p[0]) : make_uchar3(p[0], p[1], p[2]);
    }
    if (image.format == PixelFormat::BGRX8) {
        const auto* p = image.planes[0] + static_cast<size_t>(y) * image.pitch[0] + 4 * x;
        return make_uchar3(p[2], p[1], p[0]);
    }
    float luma, u, v;
    if (image.format == PixelFormat::YUYV) {
        const auto* p = image.planes[0] + static_cast<size_t>(y) * image.pitch[0] + (x / 2) * 4;
        luma = p[(x % 2) * 2]; u = p[1] - 128.f; v = p[3] - 128.f;
    } else { // NV12
        luma = image.planes[0][static_cast<size_t>(y) * image.pitch[0] + x];
        const auto* uv = image.planes[1] + static_cast<size_t>(y / 2) * image.pitch[1] + (x / 2) * 2;
        u = uv[0] - 128.f; v = uv[1] - 128.f;
    }
    const bool hd = image.matrix == Matrix::BT709;
    if (image.fullRange) {
        return make_uchar3(byte(luma + (hd ? 1.5748f : 1.402f) * v),
                           byte(luma - (hd ? .187324f : .344136f) * u - (hd ? .468124f : .714136f) * v),
                           byte(luma + (hd ? 1.8556f : 1.772f) * u));
    }
    luma = (luma - 16.f) * (255.f / 219.f);
    return make_uchar3(byte(luma + (hd ? 1.792741f : 1.596027f) * v),
                       byte(luma - (hd ? .213249f : .391762f) * u - (hd ? .532909f : .812968f) * v),
                       byte(luma + (hd ? 2.112402f : 2.017232f) * u));
}

} // namespace style_cuda
