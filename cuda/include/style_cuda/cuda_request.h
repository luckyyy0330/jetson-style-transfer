// CUDA Graph request state machine — one in-flight request per instance
// 移植自 jetson_media/include/jetson_media/cuda_request.h
// 关键设计:
//   - warmup enqueue() + cudaStreamSynchronize 冲掉 TRT 懒初始化
//   - cudaStreamBeginCapture(ThreadLocal) → enqueue() → EndCapture → Instantiate
//   - 捕获中 inputRead 用 cudaEventRecordExternal（普通事件无法通知主机）
//   - 失败置 failed_=true 永久回退 stream 模式
//   - releaseInput() 只 sync(inputRead_)，缓冲可提前归还，TRT 还在跑
#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace style_cuda {

class CudaRequest {
    cudaEvent_t inputRead_ = nullptr, completed_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
    bool enabled_ = true, failed_ = false, pending_ = false;
    static void check(cudaError_t e) {
        if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    }
public:
    CudaRequest() {
        check(cudaEventCreateWithFlags(&inputRead_, cudaEventDisableTiming));
        try { check(cudaEventCreateWithFlags(&completed_, cudaEventDisableTiming)); }
        catch (...) { cudaEventDestroy(inputRead_); throw; }
    }
    CudaRequest(const CudaRequest&) = delete;
    CudaRequest& operator=(const CudaRequest&) = delete;
    ~CudaRequest() {
        resetGraph(); cudaEventDestroy(inputRead_); cudaEventDestroy(completed_);
    }
    void idle() const {
        if (pending_) throw std::logic_error("Wait must finish before another Submit or configuration change");
    }
    void begin() { idle(); pending_ = true; }

    void inputRead(cudaStream_t stream) {
        cudaStreamCaptureStatus capture;
        check(cudaStreamIsCapturing(stream, &capture));
        check(cudaEventRecordWithFlags(inputRead_, stream,
              capture == cudaStreamCaptureStatusActive ? cudaEventRecordExternal : cudaEventRecordDefault));
    }
    void resetGraph() noexcept {
        if (executable_) cudaGraphExecDestroy(executable_);
        if (graph_) cudaGraphDestroy(graph_);
        executable_ = nullptr; graph_ = nullptr; failed_ = false;
    }
    void setGraph(bool enabled) { idle(); resetGraph(); enabled_ = enabled; }
    bool graphEnabled() const { return enabled_; }
    bool graphActive() const { return executable_ != nullptr; }
    bool graphFailed() const { return failed_; }

    template<class Enqueue> void compute(cudaStream_t stream, Enqueue enqueue) {
        if (enabled_ && !failed_ && !executable_) {
            enqueue(); // Flush TensorRT lazy initialization outside capture.
            check(cudaStreamSynchronize(stream));
            cudaError_t status = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
            if (status == cudaSuccess) {
                bool ok = true;
                try { enqueue(); } catch (...) { ok = false; }
                status = cudaStreamEndCapture(stream, &graph_);
                if (ok && status == cudaSuccess)
                    status = cudaGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0);
                else if (status == cudaSuccess) status = cudaErrorStreamCaptureInvalidated;
            }
            if (status != cudaSuccess) {
                resetGraph(); failed_ = true; cudaGetLastError();
                std::cerr << "CUDA Graph unavailable; using stream pipeline: "
                          << cudaGetErrorString(status) << '\n';
            }
        }
        if (executable_) check(cudaGraphLaunch(executable_, stream));
        else enqueue();
    }
    void submitted(cudaStream_t stream) { check(cudaEventRecord(completed_, stream)); }
    void releaseInput() {
        if (!pending_) throw std::logic_error("No pending inference request");
        check(cudaEventSynchronize(inputRead_));
    }
    void wait() {
        if (!pending_) throw std::logic_error("No pending inference request");
        check(cudaEventSynchronize(completed_)); pending_ = false;
    }
    void abort(cudaStream_t stream) noexcept {
        cudaStreamSynchronize(stream); pending_ = false;
    }
};

} // namespace style_cuda
