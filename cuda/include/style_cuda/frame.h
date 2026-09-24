// DeviceImage — borrowed CUDA view of a frame (no ownership)
// 移植自 jetson_media/include/jetson_media/frame.h
#pragma once
#include <cuda_runtime.h>

namespace style_cuda {

enum class PixelFormat : int {
    BGR8 = 0,
    RGB8 = 1,
    NV12 = 2,
    BGRX8 = 3,
    YUYV = 4,
};

enum class Matrix : int {
    BT601 = 0,
    BT709 = 1,
};

struct DeviceImage {
    const unsigned char* planes[2] = {nullptr, nullptr};
    size_t pitch[2] = {0, 0};
    cudaTextureObject_t textures[2] = {0, 0};
    int width = 0, height = 0;
    PixelFormat format = PixelFormat::BGR8;
    Matrix matrix = Matrix::BT601;
    bool fullRange = false;
};

} // namespace style_cuda
