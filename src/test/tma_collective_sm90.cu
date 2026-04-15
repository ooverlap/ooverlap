#include "test/tma_collective_sm90.h"
#include "overlap/bulk_tma_copy_sm90.cuh"

#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/system/peer_buffer.cuh"
#include "ooverlap/testing/test_utils.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <vector>

namespace ooverlap {
namespace {

__global__ void add_inplace_kernel(
    half* dst,
    const half* src,
    int64_t n) {
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < n;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        float a = __half2float(dst[idx]);
        float b = __half2float(src[idx]);
        dst[idx] = __float2half_rn(a + b);
    }
}

} // namespace

cudaError_t enqueue_fp16_add_inplace_sm90(
    half* dst,
    const half* src,
    size_t numel,
    cudaStream_t stream) {
    if (dst == nullptr || src == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }
    if (numel == 0) {
        return cudaSuccess;
    }

    constexpr int kThreads = 256;
    int blocks = static_cast<int>((numel + kThreads - 1) / kThreads);
    add_inplace_kernel<<<blocks, kThreads, 0, stream>>>(
        dst, src, static_cast<int64_t>(numel));
    return cudaGetLastError();
}

cudaError_t enqueue_two_gpu_all_reduce_tma_sm90(
    half* rank0_buf,
    half* rank1_buf,
    half* rank0_inbox,
    half* rank1_inbox,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1) {

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(
        enqueue_bulk_tma_copy_sm90(rank0_buf, rank1_inbox, numel, stream0),
        "enqueue_bulk_tma_copy_sm90 rank0->rank1_inbox");

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(
        enqueue_bulk_tma_copy_sm90(rank1_buf, rank0_inbox, numel, stream1),
        "enqueue_bulk_tma_copy_sm90 rank1->rank0_inbox");

    system::runtime::sync_stream_on_device(dev0, stream0, "cudaStreamSynchronize(stream0 copy)");
    system::runtime::sync_stream_on_device(dev1, stream1, "cudaStreamSynchronize(stream1 copy)");

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(
        enqueue_fp16_add_inplace_sm90(rank0_buf, rank0_inbox, numel, stream0),
        "enqueue_fp16_add_inplace_sm90 rank0");

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(
        enqueue_fp16_add_inplace_sm90(rank1_buf, rank1_inbox, numel, stream1),
        "enqueue_fp16_add_inplace_sm90 rank1");

    system::runtime::sync_stream_on_device(dev0, stream0, "cudaStreamSynchronize(stream0 add)");
    system::runtime::sync_stream_on_device(dev1, stream1, "cudaStreamSynchronize(stream1 add)");

    return cudaSuccess;
}

cudaError_t enqueue_two_gpu_all_gather_tma_sm90(
    const half* rank0_shard,
    const half* rank1_shard,
    half* rank0_full_out,
    half* rank1_full_out,
    size_t shard_numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1) {

    half* rank0_peer_slot = rank0_full_out + shard_numel;
    half* rank1_peer_slot = rank1_full_out + 0;

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(
        cudaMemcpyAsync(rank0_full_out, rank0_shard, shard_numel * sizeof(half),
                        cudaMemcpyDeviceToDevice, stream0),
        "cudaMemcpyAsync rank0 local shard");

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(
        cudaMemcpyAsync(rank1_full_out + shard_numel, rank1_shard, shard_numel * sizeof(half),
                        cudaMemcpyDeviceToDevice, stream1),
        "cudaMemcpyAsync rank1 local shard");

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(
        enqueue_bulk_tma_copy_sm90(const_cast<half*>(rank0_shard), rank1_peer_slot, shard_numel, stream0),
        "enqueue_bulk_tma_copy_sm90 rank0 shard -> rank1 output");

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(
        enqueue_bulk_tma_copy_sm90(const_cast<half*>(rank1_shard), rank0_peer_slot, shard_numel, stream1),
        "enqueue_bulk_tma_copy_sm90 rank1 shard -> rank0 output");

    system::runtime::sync_stream_on_device(dev0, stream0, "cudaStreamSynchronize(stream0 allgather)");
    system::runtime::sync_stream_on_device(dev1, stream1, "cudaStreamSynchronize(stream1 allgather)");

    return cudaSuccess;
}

bool tma_two_gpu_all_reduce_smoke_test(
    int64_t numel,
    int dev0,
    int dev1) {

    if (numel <= 0) {
        throw std::invalid_argument("tma_two_gpu_all_reduce_smoke_test: numel must be > 0");
    }

    int ndev = 0;
    system::runtime::check_cuda(cudaGetDeviceCount(&ndev), "cudaGetDeviceCount");
    if (ndev < 2) {
        throw std::runtime_error("tma_two_gpu_all_reduce_smoke_test: need at least 2 GPUs");
    }
    if (dev0 == dev1) {
        throw std::invalid_argument("tma_two_gpu_all_reduce_smoke_test: dev0 and dev1 must differ");
    }

    system::runtime::ensure_context_on_device(dev0);
    system::runtime::ensure_context_on_device(dev1);

    cudaStream_t stream0 = system::runtime::create_stream_on_device(dev0);
    cudaStream_t stream1 = system::runtime::create_stream_on_device(dev1);

    const size_t bytes = static_cast<size_t>(numel) * sizeof(half);

    half* rank0 = nullptr;
    half* rank1 = nullptr;

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(cudaMalloc(&rank0, bytes), "cudaMalloc(rank0)");

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(cudaMalloc(&rank1, bytes), "cudaMalloc(rank1)");

    std::vector<int> access_devices = {dev0, dev1};
    system::mapped_peer_buffer inbox0 =
        system::alloc_peer_visible_buffer(bytes, dev0, access_devices);
    system::mapped_peer_buffer inbox1 =
        system::alloc_peer_visible_buffer(bytes, dev1, access_devices);

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(cudaMemset(inbox0.ptr, 0, inbox0.mapped_size), "cudaMemset(inbox0)");

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(cudaMemset(inbox1.ptr, 0, inbox1.mapped_size), "cudaMemset(inbox1)");

    system::runtime::set_device(dev0);
    testing::fill_pattern(rank0, numel, 0.25f, 1.0f, stream0);

    system::runtime::set_device(dev1);
    testing::fill_pattern(rank1, numel, 0.50f, 2.0f, stream1);

    system::runtime::sync_stream_on_device(dev0, stream0, "cudaStreamSynchronize(fill rank0)");
    system::runtime::sync_stream_on_device(dev1, stream1, "cudaStreamSynchronize(fill rank1)");

    system::runtime::check_cuda(
        enqueue_two_gpu_all_reduce_tma_sm90(
            rank0,
            rank1,
            reinterpret_cast<half*>(inbox0.ptr),
            reinterpret_cast<half*>(inbox1.ptr),
            static_cast<size_t>(numel),
            dev0,
            dev1,
            stream0,
            stream1),
        "enqueue_two_gpu_all_reduce_tma_sm90");

    auto got0 = testing::copy_half_device_to_host_float(rank0, numel, dev0);
    auto got1 = testing::copy_half_device_to_host_float(rank1, numel, dev1);

    auto ref0 = testing::host_reference_pattern_fp16(numel, 0.25f, 1.0f);
    auto ref1 = testing::host_reference_pattern_fp16(numel, 0.50f, 2.0f);
    std::vector<float> ref(static_cast<size_t>(numel));
    for (int64_t i = 0; i < numel; ++i) {
        float acc = ref0[static_cast<size_t>(i)];
        acc = testing::round_to_half(acc + ref1[static_cast<size_t>(i)]);
        ref[static_cast<size_t>(i)] = acc;
    }

    testing::expect_allclose(got0, ref, "all_reduce rank0");
    testing::expect_allclose(got1, ref, "all_reduce rank1");

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(cudaFree(rank0), "cudaFree(rank0)");
    system::runtime::destroy_stream_on_device(dev0, stream0);

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(cudaFree(rank1), "cudaFree(rank1)");
    system::runtime::destroy_stream_on_device(dev1, stream1);

    system::free_peer_visible_buffer(inbox0);
    system::free_peer_visible_buffer(inbox1);

    return true;
}

bool tma_two_gpu_all_gather_smoke_test(
    int64_t shard_numel,
    int dev0,
    int dev1) {

    if (shard_numel <= 0) {
        throw std::invalid_argument("tma_two_gpu_all_gather_smoke_test: shard_numel must be > 0");
    }

    int ndev = 0;
    system::runtime::check_cuda(cudaGetDeviceCount(&ndev), "cudaGetDeviceCount");
    if (ndev < 2) {
        throw std::runtime_error("tma_two_gpu_all_gather_smoke_test: need at least 2 GPUs");
    }
    if (dev0 == dev1) {
        throw std::invalid_argument("tma_two_gpu_all_gather_smoke_test: dev0 and dev1 must differ");
    }

    system::runtime::ensure_context_on_device(dev0);
    system::runtime::ensure_context_on_device(dev1);

    cudaStream_t stream0 = system::runtime::create_stream_on_device(dev0);
    cudaStream_t stream1 = system::runtime::create_stream_on_device(dev1);

    const size_t shard_bytes = static_cast<size_t>(shard_numel) * sizeof(half);
    const size_t full_bytes = 2 * shard_bytes;

    half* rank0_shard = nullptr;
    half* rank1_shard = nullptr;

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(cudaMalloc(&rank0_shard, shard_bytes), "cudaMalloc(rank0_shard)");

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(cudaMalloc(&rank1_shard, shard_bytes), "cudaMalloc(rank1_shard)");

    std::vector<int> access_devices = {dev0, dev1};
    system::mapped_peer_buffer out0 =
        system::alloc_peer_visible_buffer(full_bytes, dev0, access_devices);
    system::mapped_peer_buffer out1 =
        system::alloc_peer_visible_buffer(full_bytes, dev1, access_devices);

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(cudaMemset(out0.ptr, 0, out0.mapped_size), "cudaMemset(out0)");

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(cudaMemset(out1.ptr, 0, out1.mapped_size), "cudaMemset(out1)");

    system::runtime::set_device(dev0);
    testing::fill_pattern(rank0_shard, shard_numel, 1.0f, 10.0f, stream0);

    system::runtime::set_device(dev1);
    testing::fill_pattern(rank1_shard, shard_numel, 1.0f, 100.0f, stream1);

    system::runtime::sync_stream_on_device(dev0, stream0, "cudaStreamSynchronize(fill rank0 shard)");
    system::runtime::sync_stream_on_device(dev1, stream1, "cudaStreamSynchronize(fill rank1 shard)");

    system::runtime::check_cuda(
        enqueue_two_gpu_all_gather_tma_sm90(
            rank0_shard,
            rank1_shard,
            reinterpret_cast<half*>(out0.ptr),
            reinterpret_cast<half*>(out1.ptr),
            static_cast<size_t>(shard_numel),
            dev0,
            dev1,
            stream0,
            stream1),
        "enqueue_two_gpu_all_gather_tma_sm90");

    auto got0 = testing::copy_half_device_to_host_float(
        reinterpret_cast<half*>(out0.ptr), 2 * shard_numel, dev0);
    auto got1 = testing::copy_half_device_to_host_float(
        reinterpret_cast<half*>(out1.ptr), 2 * shard_numel, dev1);

    auto ref_shard0 = testing::host_reference_pattern_fp16(shard_numel, 1.0f, 10.0f);
    auto ref_shard1 = testing::host_reference_pattern_fp16(shard_numel, 1.0f, 100.0f);
    std::vector<float> ref_full(static_cast<size_t>(2 * shard_numel));
    for (int64_t i = 0; i < shard_numel; ++i) {
        ref_full[static_cast<size_t>(i)] = ref_shard0[static_cast<size_t>(i)];
        ref_full[static_cast<size_t>(i + shard_numel)] = ref_shard1[static_cast<size_t>(i)];
    }

    testing::expect_allclose(got0, ref_full, "all_gather rank0");
    testing::expect_allclose(got1, ref_full, "all_gather rank1");

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(cudaFree(rank0_shard), "cudaFree(rank0_shard)");
    system::runtime::destroy_stream_on_device(dev0, stream0);

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(cudaFree(rank1_shard), "cudaFree(rank1_shard)");
    system::runtime::destroy_stream_on_device(dev1, stream1);

    system::free_peer_visible_buffer(out0);
    system::free_peer_visible_buffer(out1);

    return true;
}

} // namespace ooverlap
