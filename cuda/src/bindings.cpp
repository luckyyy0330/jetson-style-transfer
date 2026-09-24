// pybind11 bindings for style_cuda — GpuPipeline
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "style_cuda/cuda_request.h"
#include "style_cuda/cuda_descriptor.h"
#include "style_cuda/prepost.h"
#include "style_cuda/frame.h"

#include <cuda_runtime.h>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace py = pybind11;

namespace style_cuda {

static void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

// GpuPipeline: 一个 CudaRequest + 稳定描述符 + pinned/mapped 缓冲
class GpuPipeline {
public:
    GpuPipeline(int width, int height, int norm_mode, const std::string& io_dtype)
        : width_(width), height_(height) {
        if (width <= 0 || height <= 0)
            throw std::invalid_argument("width/height must be positive");

        out_dtype_ = (io_dtype == "fp16") ? 1 : 0;
        in_dtype_ = out_dtype_;

        // 归一化参数
        if (norm_mode == 0) { // minus_one_to_one
            norm_a_ = 1.0f / 127.5f; norm_b_ = -1.0f;
            denorm_a_ = 127.5f; denorm_b_ = 127.5f;
        } else { // zero_to_one
            norm_a_ = 1.0f / 255.0f; norm_b_ = 0.0f;
            denorm_a_ = 255.0f; denorm_b_ = 0.0f;
        }

        // 创建 CUDA stream
        check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

        // 分配 device 输入缓冲 (NCHW)
        size_t in_bytes = 3ULL * width * height * (out_dtype_ == 0 ? 4 : 2);
        check(cudaMalloc(&device_input_, in_bytes));

        // 分配 device 输出缓冲 (NCHW, TRT 输出)
        check(cudaMalloc(&device_output_, in_bytes));

        // 分配 mapped host 输出缓冲 (BGR uint8, 免 D2H)
        size_t out_bytes = 3ULL * width * height;
        check(cudaHostAlloc(reinterpret_cast<void**>(&host_output_), out_bytes,
                            cudaHostAllocMapped));
        check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&host_output_dev_),
                                       host_output_, 0));

        // 分配 mapped host staging (输入帧，采集线程 memcpy 到这里)
        // 预留最大 1920x1080 的 BGRx (4 B/px) 或 NV12 (1.5 B/px)
        staging_capacity_ = 1920ULL * 1080 * 4;
        check(cudaHostAlloc(reinterpret_cast<void**>(&staging_), staging_capacity_,
                            cudaHostAllocMapped));
        check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&staging_dev_),
                                       staging_, 0));
    }

    ~GpuPipeline() {
        req_.abort(stream_);
        if (device_input_) cudaFree(device_input_);
        if (device_output_) cudaFree(device_output_);
        if (host_output_) cudaFreeHost(host_output_);
        if (staging_) cudaFreeHost(staging_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    GpuPipeline(const GpuPipeline&) = delete;
    GpuPipeline& operator=(const GpuPipeline&) = delete;

    // 设置源帧参数（仅 idle 时调用）
    void set_source(int src_w, int src_h, size_t src_pitch, int fmt, int matrix, bool full_range) {
        req_.idle();
        src_w_ = src_w; src_h_ = src_h; src_pitch_ = src_pitch;
        fmt_ = static_cast<PixelFormat>(fmt);
        matrix_ = static_cast<Matrix>(matrix);
        full_range_ = full_range;
    }

    // 拷贝源帧到 staging（采集线程调用）
    void copy_source(py::buffer src) {
        auto info = src.request();
        size_t bytes = static_cast<size_t>(info.size) * info.itemsize;
        if (bytes > staging_capacity_)
            throw std::runtime_error("source frame too large for staging");
        std::memcpy(staging_, info.ptr, bytes);
    }

    // Submit: 预处理 + TRT 推理 + 后处理（整条 GPU 路径可被 Graph 捕获）
    // trt_enqueue: Python callable，内部调用 context.execute_async_v3
    void run(py::function trt_enqueue) {
        req_.begin();

        // 更新预处理描述符（稳定地址，Graph 不重捕获）
        PreDesc pre{};
        pre.image.planes[0] = static_cast<const unsigned char*>(staging_dev_);
        pre.image.pitch[0] = src_pitch_;
        pre.image.width = src_w_;
        pre.image.height = src_h_;
        pre.image.format = fmt_;
        pre.image.matrix = matrix_;
        pre.image.fullRange = full_range_;
        if (fmt_ == PixelFormat::NV12) {
            pre.image.planes[1] = pre.image.planes[0] + src_pitch_ * src_h_;
            pre.image.pitch[1] = src_pitch_;
        }
        pre.meta = make_int4(width_, height_, 0, 0);
        pre.norm_a = norm_a_; pre.norm_b = norm_b_;
        pre.out_dtype = out_dtype_;
        pre_desc_.update(pre, stream_);

        PostDesc post{};
        post.meta = make_int4(width_, height_, 0, 0);
        post.denorm_a = denorm_a_; post.denorm_b = denorm_b_;
        post.in_dtype = in_dtype_;
        post_desc_.update(post, stream_);

        // Graph 捕获整条 GPU 路径
        req_.compute(stream_, [&]() {
            // 预处理
            check(cuda_preprocess(pre_desc_.device(), device_input_, stream_));
            // 记录 inputRead 事件（捕获中用 RecordExternal）
            req_.inputRead(stream_);
            // TRT 推理（Python callable）
            trt_enqueue();
            // 后处理
            check(cuda_postprocess(post_desc_.device(), device_output_,
                                   static_cast<unsigned char*>(host_output_dev_), stream_));
        });

        req_.submitted(stream_);
    }

    // 提前归还输入缓冲（TRT 还在跑时）
    void release_input() { req_.releaseInput(); }

    // 等待完成，返回**已持有**的 BGR uint8 数组（一次 memcpy，脱离 mapped 内存）。
    // 绝不能返回 alias host_output_ 的 view：下一帧后处理核会覆写它，
    // 且 np.ascontiguousarray 对已连续 view 不拷贝 —— 别名不会断开。
    py::array_t<uint8_t> wait() {
        req_.wait();
        py::array_t<uint8_t> out({height_, width_, 3});
        std::memcpy(out.mutable_data(), host_output_,
                    static_cast<size_t>(height_) * width_ * 3);
        return out;
    }

    // 零拷贝 view（alias mapped host 输出）。调用方必须在下一次 wait() 前拷走。
    py::array_t<uint8_t> wait_view() {
        req_.wait();
        return py::array_t<uint8_t>(
            {height_, width_, 3},
            {width_ * 3, 3, 1},
            host_output_
        );
    }

    bool graph_active() const { return req_.graphActive(); }
    bool graph_failed() const { return req_.graphFailed(); }
    int width() const { return width_; }
    int height() const { return height_; }
    int64_t stream_handle() const { return reinterpret_cast<int64_t>(stream_); }
    int64_t input_ptr() const { return reinterpret_cast<int64_t>(device_input_); }
    int64_t output_ptr() const { return reinterpret_cast<int64_t>(device_output_); }

    void set_graph(bool enabled) { req_.setGraph(enabled); }

private:
    int width_, height_;
    int out_dtype_ = 0, in_dtype_ = 0;
    float norm_a_ = 1/127.5f, norm_b_ = -1.0f;
    float denorm_a_ = 127.5f, denorm_b_ = 127.5f;

    int src_w_ = 0, src_h_ = 0;
    size_t src_pitch_ = 0;
    PixelFormat fmt_ = PixelFormat::BGR8;
    Matrix matrix_ = Matrix::BT601;
    bool full_range_ = false;

    cudaStream_t stream_ = nullptr;
    void* device_input_ = nullptr;
    void* device_output_ = nullptr;
    void* host_output_ = nullptr;
    void* host_output_dev_ = nullptr;
    void* staging_ = nullptr;
    void* staging_dev_ = nullptr;
    size_t staging_capacity_ = 0;

    CudaRequest req_;
    CudaDescriptor<PreDesc> pre_desc_;
    CudaDescriptor<PostDesc> post_desc_;
};

} // namespace style_cuda

PYBIND11_MODULE(style_cuda, m) {
    m.doc() = "CUDA fused preprocess/postprocess + CUDA Graph pipeline for style transfer";

    m.def("version", []() { return "0.1.0"; });

    py::class_<style_cuda::GpuPipeline>(m, "GpuPipeline")
        .def(py::init<int, int, int, const std::string&>(),
             py::arg("width"), py::arg("height"),
             py::arg("norm") = 0, py::arg("io_dtype") = "fp32")
        .def("set_source", &style_cuda::GpuPipeline::set_source,
             py::arg("src_w"), py::arg("src_h"), py::arg("src_pitch"),
             py::arg("fmt"), py::arg("matrix") = 0, py::arg("full_range") = false)
        .def("copy_source", &style_cuda::GpuPipeline::copy_source)
        .def("run", &style_cuda::GpuPipeline::run)
        .def("release_input", &style_cuda::GpuPipeline::release_input)
        .def("wait", &style_cuda::GpuPipeline::wait)
        .def("wait_view", &style_cuda::GpuPipeline::wait_view)
        .def("graph_active", &style_cuda::GpuPipeline::graph_active)
        .def("graph_failed", &style_cuda::GpuPipeline::graph_failed)
        .def("width", &style_cuda::GpuPipeline::width)
        .def("height", &style_cuda::GpuPipeline::height)
        .def("stream_handle", &style_cuda::GpuPipeline::stream_handle)
        .def("input_ptr", &style_cuda::GpuPipeline::input_ptr)
        .def("output_ptr", &style_cuda::GpuPipeline::output_ptr)
        .def("set_graph", &style_cuda::GpuPipeline::set_graph);

    // PixelFormat 枚举
    py::enum_<style_cuda::PixelFormat>(m, "PixelFormat")
        .value("BGR8", style_cuda::PixelFormat::BGR8)
        .value("RGB8", style_cuda::PixelFormat::RGB8)
        .value("NV12", style_cuda::PixelFormat::NV12)
        .value("BGRX8", style_cuda::PixelFormat::BGRX8)
        .value("YUYV", style_cuda::PixelFormat::YUYV);
}
