#pragma once

#include "ooverlap/testing/checks.cuh"
#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_runtime.h>

#include <functional>

namespace ooverlap {
namespace testing {

inline double elapsed_one_rank_ms(
    int device,
    cudaStream_t stream,
    const std::function<void()>& launch_once) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    system::runtime::set_device(device);

    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");

    launch_once();

    check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float ms = 0.0f;

    check_cuda(
        cudaEventElapsedTime(&ms, start, stop),
        "cudaEventElapsedTime");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return static_cast<double>(ms);
}

template <typename Fn>
inline double elapsed_one_rank_ms(
    int device,
    cudaStream_t stream,
    Fn&& launch_once) {
    return elapsed_one_rank_ms(
        device,
        stream,
        std::function<void()>(std::forward<Fn>(launch_once)));
}

} // namespace testing
} // namespace ooverlap
