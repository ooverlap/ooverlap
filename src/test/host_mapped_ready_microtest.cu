#include "test/host_mapped_ready_microtest.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ooverlap {
namespace {

void check_cuda(
    cudaError_t err,
    const char* what) {
    if (err == cudaSuccess) {
        return;
    }

    std::ostringstream oss;
    oss << what << " failed: " << cudaGetErrorString(err);
    throw std::runtime_error(oss.str());
}

__global__ void host_mapped_ready_publish_kernel(
    int* ready,
    int value) {
    if (threadIdx.x != 0) {
        return;
    }

    volatile int* p =
        reinterpret_cast<volatile int*>(ready);

    p[0] = value;

#if defined(__CUDA_ARCH__)
    __threadfence_system();
#endif
}

__global__ void host_mapped_ready_wait_kernel(
    const int* ready,
    int value,
    int* result,
    unsigned long long max_iters) {
    if (threadIdx.x != 0) {
        return;
    }

    const volatile int* p =
        reinterpret_cast<const volatile int*>(ready);

    for (unsigned long long i = 0; i < max_iters; ++i) {
        const int observed = p[0];

        if (observed >= value) {
            result[0] = observed;
            return;
        }

#if defined(__CUDA_ARCH__)
        if ((i & 255ull) == 0ull) {
            __nanosleep(256);
        }
#endif
    }

    result[0] = -1;
}

} // namespace

std::map<std::string, long long> host_mapped_ready_signal_roundtrip(
    int dev_publish,
    int dev_wait,
    int value,
    unsigned long long max_iters) {
    int device_count = 0;
    check_cuda(
        cudaGetDeviceCount(&device_count),
        "cudaGetDeviceCount");

    if (dev_publish < 0 || dev_publish >= device_count ||
        dev_wait < 0 || dev_wait >= device_count ||
        value <= 0 ||
        max_iters == 0) {
        throw std::invalid_argument(
            "host_mapped_ready_signal_roundtrip: invalid arguments");
    }

    void* host_ready_void = nullptr;

    check_cuda(
        cudaHostAlloc(
            &host_ready_void,
            sizeof(int),
            cudaHostAllocMapped | cudaHostAllocPortable),
        "cudaHostAlloc(mapped ready)");

    int* host_ready =
        reinterpret_cast<int*>(host_ready_void);

    host_ready[0] = 0;

    int* publish_ready = nullptr;
    int* wait_ready = nullptr;
    int* wait_result = nullptr;

    cudaStream_t publish_stream = nullptr;
    cudaStream_t wait_stream = nullptr;

    int observed = -999;
    bool ok = false;

    try {
        check_cuda(
            cudaSetDevice(dev_publish),
            "cudaSetDevice(dev_publish)");

        void* publish_ready_void = nullptr;

        check_cuda(
            cudaHostGetDevicePointer(
                &publish_ready_void,
                host_ready,
                0),
            "cudaHostGetDevicePointer(publish)");

        publish_ready =
            reinterpret_cast<int*>(publish_ready_void);

        check_cuda(
            cudaStreamCreateWithFlags(
                &publish_stream,
                cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags(publish)");

        check_cuda(
            cudaSetDevice(dev_wait),
            "cudaSetDevice(dev_wait)");

        void* wait_ready_void = nullptr;

        check_cuda(
            cudaHostGetDevicePointer(
                &wait_ready_void,
                host_ready,
                0),
            "cudaHostGetDevicePointer(wait)");

        wait_ready =
            reinterpret_cast<int*>(wait_ready_void);

        check_cuda(
            cudaMalloc(
                &wait_result,
                sizeof(int)),
            "cudaMalloc(wait_result)");

        check_cuda(
            cudaMemset(
                wait_result,
                0,
                sizeof(int)),
            "cudaMemset(wait_result)");

        check_cuda(
            cudaStreamCreateWithFlags(
                &wait_stream,
                cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags(wait)");

        host_mapped_ready_wait_kernel<<<1, 32, 0, wait_stream>>>(
            wait_ready,
            value,
            wait_result,
            max_iters);

        check_cuda(
            cudaGetLastError(),
            "host_mapped_ready_wait_kernel launch");

        check_cuda(
            cudaSetDevice(dev_publish),
            "cudaSetDevice(dev_publish before publish launch)");

        host_mapped_ready_publish_kernel<<<1, 32, 0, publish_stream>>>(
            publish_ready,
            value);

        check_cuda(
            cudaGetLastError(),
            "host_mapped_ready_publish_kernel launch");

        check_cuda(
            cudaStreamSynchronize(publish_stream),
            "cudaStreamSynchronize(publish)");

        check_cuda(
            cudaSetDevice(dev_wait),
            "cudaSetDevice(dev_wait before wait sync)");

        check_cuda(
            cudaStreamSynchronize(wait_stream),
            "cudaStreamSynchronize(wait)");

        check_cuda(
            cudaMemcpy(
                &observed,
                wait_result,
                sizeof(int),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(wait_result -> host)");

        ok = observed >= value;
    } catch (...) {
        if (publish_stream != nullptr) {
            cudaSetDevice(dev_publish);
            cudaStreamDestroy(publish_stream);
        }

        if (wait_stream != nullptr) {
            cudaSetDevice(dev_wait);
            cudaStreamDestroy(wait_stream);
        }

        if (wait_result != nullptr) {
            cudaSetDevice(dev_wait);
            cudaFree(wait_result);
        }

        cudaFreeHost(host_ready);
        throw;
    }

    if (publish_stream != nullptr) {
        cudaSetDevice(dev_publish);
        cudaStreamDestroy(publish_stream);
    }

    if (wait_stream != nullptr) {
        cudaSetDevice(dev_wait);
        cudaStreamDestroy(wait_stream);
    }

    if (wait_result != nullptr) {
        cudaSetDevice(dev_wait);
        cudaFree(wait_result);
    }

    const int host_value_after =
        host_ready[0];

    const std::uintptr_t publish_ptr_value =
        reinterpret_cast<std::uintptr_t>(publish_ready);
    const std::uintptr_t wait_ptr_value =
        reinterpret_cast<std::uintptr_t>(wait_ready);

    cudaFreeHost(host_ready);

    std::map<std::string, long long> result;
    result["ok"] = ok ? 1 : 0;
    result["observed"] = static_cast<long long>(observed);
    result["host_value_after"] = static_cast<long long>(host_value_after);
    result["dev_publish"] = static_cast<long long>(dev_publish);
    result["dev_wait"] = static_cast<long long>(dev_wait);
    result["publish_device_ptr"] =
        static_cast<long long>(publish_ptr_value);
    result["wait_device_ptr"] =
        static_cast<long long>(wait_ptr_value);
    result["same_device_ptr_value"] =
        publish_ptr_value == wait_ptr_value ? 1 : 0;
    result["max_iters"] =
        static_cast<long long>(max_iters);

    return result;
}

} // namespace ooverlap
