// Stable kernel argument storage — CUDA Graph never recaptures on geometry change
// 移植自 jetson_media/include/jetson_media/cuda_descriptor.h
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <type_traits>

namespace style_cuda {

template<class T> class CudaDescriptor {
    T* host_ = nullptr;
    T* device_ = nullptr;
    bool mapped_ = false;
    static void check(cudaError_t e) {
        if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    }
public:
    static_assert(std::is_trivially_copyable<T>::value, "Descriptor must be trivially copyable");
    CudaDescriptor() {
        int id; cudaDeviceProp prop{};
        check(cudaGetDevice(&id)); check(cudaGetDeviceProperties(&prop,id));
        mapped_ = prop.integrated && prop.canMapHostMemory;
        check(cudaHostAlloc(reinterpret_cast<void**>(&host_),sizeof(T),
              mapped_ ? cudaHostAllocMapped : cudaHostAllocDefault));
        try {
            if (mapped_) check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_),host_,0));
            else check(cudaMalloc(reinterpret_cast<void**>(&device_),sizeof(T)));
        } catch (...) { cudaFreeHost(host_); throw; }
        *host_ = T{};
    }
    CudaDescriptor(const CudaDescriptor&) = delete;
    CudaDescriptor& operator=(const CudaDescriptor&) = delete;
    ~CudaDescriptor() { if (!mapped_) cudaFree(device_); cudaFreeHost(host_); }
    void update(const T& value, cudaStream_t stream) {
        *host_ = value;
        if (!mapped_) check(cudaMemcpyAsync(device_,host_,sizeof(T),cudaMemcpyHostToDevice,stream));
    }
    const T* device() const { return device_; }
    T* host() { return host_; }
};

} // namespace style_cuda
