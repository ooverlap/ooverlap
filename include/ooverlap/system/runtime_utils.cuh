#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace ooverlap {
namespace system {
namespace runtime {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << what << " failed: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

inline void set_device(int dev) {
    check_cuda(cudaSetDevice(dev), "cudaSetDevice");
}

inline void ensure_context_on_device(int dev) {
    set_device(dev);
    check_cuda(cudaFree(nullptr), "cudaFree(nullptr)");
}

inline cudaStream_t create_stream_on_device(
    int dev,
    unsigned int flags = cudaStreamNonBlocking) {
    set_device(dev);
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, flags),
               "cudaStreamCreateWithFlags");
    return stream;
}

inline void destroy_stream_on_device(int dev, cudaStream_t& stream) {
    if (stream != nullptr) {
        set_device(dev);
        check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
        stream = nullptr;
    }
}

inline void sync_stream_on_device(int dev, cudaStream_t stream, const char* what) {
    set_device(dev);
    check_cuda(cudaStreamSynchronize(stream), what);
}

} // namespace runtime
} // namespace system
} // namespace ooverlap
