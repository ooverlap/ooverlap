#pragma once

#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/testing/checks.cuh"
#include "ooverlap/testing/collective_test_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>

namespace ooverlap {
namespace testing {
namespace detail {

template <typename LaunchOnce>
inline double elapsed_ms_two_stream_max_impl(
    int dev0,
    cudaStream_t stream0,
    int dev1,
    cudaStream_t stream1,
    int iters,
    LaunchOnce&& launch_once) {
    cudaEvent_t start0 = nullptr;
    cudaEvent_t stop0 = nullptr;
    cudaEvent_t start1 = nullptr;
    cudaEvent_t stop1 = nullptr;

    system::runtime::set_device(dev0);
    check_cuda(cudaEventCreate(&start0), "cudaEventCreate(start0)");
    check_cuda(cudaEventCreate(&stop0), "cudaEventCreate(stop0)");
    check_cuda(cudaEventRecord(start0, stream0), "cudaEventRecord(start0)");

    system::runtime::set_device(dev1);
    check_cuda(cudaEventCreate(&start1), "cudaEventCreate(start1)");
    check_cuda(cudaEventCreate(&stop1), "cudaEventCreate(stop1)");
    check_cuda(cudaEventRecord(start1, stream1), "cudaEventRecord(start1)");

    for (int i = 0; i < iters; ++i) {
        launch_once(i);
    }

    system::runtime::set_device(dev0);
    check_cuda(cudaEventRecord(stop0, stream0), "cudaEventRecord(stop0)");

    system::runtime::set_device(dev1);
    check_cuda(cudaEventRecord(stop1, stream1), "cudaEventRecord(stop1)");

    system::runtime::set_device(dev0);
    check_cuda(cudaEventSynchronize(stop0), "cudaEventSynchronize(stop0)");

    system::runtime::set_device(dev1);
    check_cuda(cudaEventSynchronize(stop1), "cudaEventSynchronize(stop1)");

    float ms0 = 0.0f;
    float ms1 = 0.0f;

    system::runtime::set_device(dev0);
    check_cuda(
        cudaEventElapsedTime(&ms0, start0, stop0),
        "cudaEventElapsedTime(ms0)");

    system::runtime::set_device(dev1);
    check_cuda(
        cudaEventElapsedTime(&ms1, start1, stop1),
        "cudaEventElapsedTime(ms1)");

    system::runtime::set_device(dev0);
    cudaEventDestroy(start0);
    cudaEventDestroy(stop0);

    system::runtime::set_device(dev1);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);

    return static_cast<double>(std::max(ms0, ms1));
}

} // namespace detail

inline void sync_two_streams(
    int dev0,
    cudaStream_t stream0,
    int dev1,
    cudaStream_t stream1,
    const char* what) {
    system::runtime::sync_stream_on_device(dev0, stream0, what);
    system::runtime::sync_stream_on_device(dev1, stream1, what);
}

inline void fill_two_rank_sources_fp16(
    half* rank0,
    half* rank1,
    int64_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1) {
    fill_rank_source_fp16(rank0, numel, 0, dev0, stream0);
    fill_rank_source_fp16(rank1, numel, 1, dev1, stream1);
}

inline void copy_two_buffers_async(
    const half* rank0_src,
    const half* rank1_src,
    half* rank0_dst,
    half* rank1_dst,
    size_t bytes,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1) {
    system::runtime::set_device(dev0);
    check_cuda(
        cudaMemcpyAsync(
            rank0_dst,
            rank0_src,
            bytes,
            cudaMemcpyDeviceToDevice,
            stream0),
        "cudaMemcpyAsync(rank0_src -> rank0_dst)");

    system::runtime::set_device(dev1);
    check_cuda(
        cudaMemcpyAsync(
            rank1_dst,
            rank1_src,
            bytes,
            cudaMemcpyDeviceToDevice,
            stream1),
        "cudaMemcpyAsync(rank1_src -> rank1_dst)");
}

inline void prepare_two_work_buffers(
    const half* rank0_src,
    const half* rank1_src,
    half* rank0_work,
    half* rank1_work,
    size_t bytes,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1) {
    copy_two_buffers_async(
        rank0_src,
        rank1_src,
        rank0_work,
        rank1_work,
        bytes,
        dev0,
        dev1,
        stream0,
        stream1);

    sync_two_streams(
        dev0,
        stream0,
        dev1,
        stream1,
        "sync prepare_two_work_buffers");
}

inline double elapsed_ms_two_stream_max(
    int dev0,
    cudaStream_t stream0,
    int dev1,
    cudaStream_t stream1,
    int iters,
    const std::function<void(int)>& launch_once) {
    return detail::elapsed_ms_two_stream_max_impl(
        dev0,
        stream0,
        dev1,
        stream1,
        iters,
        launch_once);
}

template <typename Fn>
inline double elapsed_ms_two_stream_max(
    int dev0,
    cudaStream_t stream0,
    int dev1,
    cudaStream_t stream1,
    int iters,
    Fn&& launch_once) {
    return detail::elapsed_ms_two_stream_max_impl(
        dev0,
        stream0,
        dev1,
        stream1,
        iters,
        std::forward<Fn>(launch_once));
}

inline void cuda_malloc_half_on_device(
    int device,
    half** out,
    size_t bytes,
    const char* what) {
    if (out == nullptr) {
        throw std::invalid_argument("cuda_malloc_half_on_device: out is null");
    }

    *out = nullptr;

    system::runtime::set_device(device);
    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(out), bytes),
        what);
}

inline void cuda_free_on_device(
    int device,
    half*& ptr) {
    if (ptr == nullptr) {
        return;
    }

    system::runtime::set_device(device);
    cudaFree(ptr);
    ptr = nullptr;
}

inline void destroy_stream_on_device(
    int device,
    cudaStream_t& stream) {
    if (stream == nullptr) {
        return;
    }

    system::runtime::destroy_stream_on_device(device, stream);
    stream = nullptr;
}

inline void destroy_nccl_comms(
    ncclComm_t* comms,
    int count) {
    if (comms == nullptr) {
        return;
    }

    for (int i = 0; i < count; ++i) {
        if (comms[i] != nullptr) {
            ncclCommDestroy(comms[i]);
            comms[i] = nullptr;
        }
    }
}

} // namespace testing
} // namespace ooverlap
