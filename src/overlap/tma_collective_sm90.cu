#include "overlap/tma_collective_sm90.h"
#include "overlap/bulk_tma_copy_sm90.cuh"
#include "ooverlap/system/vmm.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace ooverlap {
namespace {

struct mapped_peer_buffer {
    void* ptr = nullptr;
    size_t mapped_size = 0;
};

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << what << " failed: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

inline void ensure_context_on_device(int dev) {
    check_cuda(cudaSetDevice(dev), "cudaSetDevice");
    check_cuda(cudaFree(nullptr), "cudaFree(nullptr)");
}

inline cudaStream_t create_stream_on_device(int dev) {
    check_cuda(cudaSetDevice(dev), "cudaSetDevice");
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
    return stream;
}

inline mapped_peer_buffer alloc_peer_visible_buffer(
    size_t bytes,
    int owner_device,
    const std::vector<int>& access_devices) {
    mapped_peer_buffer out{};
    ooverlap::system::vmm::vm_alloc_map_set_access(
        &out.ptr,
        &out.mapped_size,
        bytes,
        owner_device,
        access_devices);
    return out;
}

inline void free_peer_visible_buffer(mapped_peer_buffer& buf) {
    if (buf.ptr != nullptr && buf.mapped_size != 0) {
        ooverlap::system::vmm::vm_unmap(buf.ptr, buf.mapped_size);
        buf.ptr = nullptr;
        buf.mapped_size = 0;
    }
}

__global__ void fill_pattern_kernel(
    half* ptr,
    int64_t n,
    float scale,
    float bias) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = bias + scale * static_cast<float>(idx % 1024);
    ptr[idx] = __float2half_rn(x);
}

__global__ void add_inplace_kernel(
    half* dst,
    const half* src,
    int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float a = __half2float(dst[idx]);
    float b = __half2float(src[idx]);
    dst[idx] = __float2half_rn(a + b);
}

inline void fill_pattern(
    half* ptr,
    int64_t n,
    float scale,
    float bias,
    cudaStream_t stream) {
    constexpr int kThreads = 256;
    int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    fill_pattern_kernel<<<blocks, kThreads, 0, stream>>>(ptr, n, scale, bias);
    check_cuda(cudaGetLastError(), "fill_pattern_kernel launch");
}

inline std::vector<float> host_reference_pattern(
    int64_t n,
    float scale,
    float bias) {
    std::vector<float> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        out[static_cast<size_t>(i)] = bias + scale * static_cast<float>(i % 1024);
    }
    return out;
}

inline std::vector<float> copy_half_device_to_host_float(
    const half* ptr,
    int64_t n,
    int dev) {
    std::vector<half> tmp(static_cast<size_t>(n));
    check_cuda(cudaSetDevice(dev), "cudaSetDevice");
    check_cuda(cudaMemcpy(tmp.data(), ptr, static_cast<size_t>(n) * sizeof(half),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(device->host)");

    std::vector<float> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        out[static_cast<size_t>(i)] = __half2float(tmp[static_cast<size_t>(i)]);
    }
    return out;
}

inline void expect_allclose(
    const std::vector<float>& got,
    const std::vector<float>& ref,
    const char* what,
    float atol = 1e-2f) {
    if (got.size() != ref.size()) {
        std::ostringstream oss;
        oss << what << " size mismatch: got=" << got.size() << " ref=" << ref.size();
        throw std::runtime_error(oss.str());
    }

    for (size_t i = 0; i < got.size(); ++i) {
        float diff = std::fabs(got[i] - ref[i]);
        if (diff > atol) {
            std::ostringstream oss;
            oss << what << " mismatch at i=" << i
                << " got=" << got[i]
                << " ref=" << ref[i]
                << " diff=" << diff;
            throw std::runtime_error(oss.str());
        }
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

    check_cuda(cudaSetDevice(dev0), "cudaSetDevice(dev0)");
    check_cuda(enqueue_bulk_tma_copy_sm90(rank0_buf, rank1_inbox, numel, stream0),
               "enqueue_bulk_tma_copy_sm90 rank0->rank1_inbox");

    check_cuda(cudaSetDevice(dev1), "cudaSetDevice(dev1)");
    check_cuda(enqueue_bulk_tma_copy_sm90(rank1_buf, rank0_inbox, numel, stream1),
               "enqueue_bulk_tma_copy_sm90 rank1->rank0_inbox");

    check_cuda(cudaSetDevice(dev0), "cudaSetDevice(dev0)");
    check_cuda(cudaStreamSynchronize(stream0), "cudaStreamSynchronize(stream0 copy)");

    check_cuda(cudaSetDevice(dev1), "cudaSetDevice(dev1)");
    check_cuda(cudaStreamSynchronize(stream1), "cudaStreamSynchronize(stream1 copy)");

    check_cuda(cudaSetDevice(dev0), "cudaSetDevice(dev0)");
    check_cuda(enqueue_fp16_add_inplace_sm90(rank0_buf, rank0_inbox, numel, stream0),
               "enqueue_fp16_add_inplace_sm90 rank0");

    check_cuda(cudaSetDevice(dev1), "cudaSetDevice(dev1)");
    check_cuda(enqueue_fp16_add_inplace_sm90(rank1_buf, rank1_inbox, numel, stream1),
               "enqueue_fp16_add_inplace_sm90 rank1");

    check_cuda(cudaSetDevice(dev0), "cudaSetDevice(dev0)");
    check_cuda(cudaStreamSynchronize(stream0), "cudaStreamSynchronize(stream0 add)");

    check_cuda(cudaSetDevice(dev1), "cudaSetDevice(dev1)");
    check_cuda(cudaStreamSynchronize(stream1), "cudaStreamSynchronize(stream1 add)");

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

    // Local placement
    check_cuda(cudaSetDevice(dev0), "cudaSetDevice(dev0)");
    check_cuda(cudaMemcpyAsync(rank0_full_out, rank0_shard, shard_numel * sizeof(half),
                               cudaMemcpyDeviceToDevice, stream0),
               "cudaMemcpyAsync rank0 local shard");

    check_cuda(cudaSetDevice(dev1), "cudaSetDevice(dev1)");
    check_cuda(cudaMemcpyAsync(rank1_full_out + shard_numel, rank1_shard, shard_numel * sizeof(half),
                               cudaMemcpyDeviceToDevice, stream1),
               "cudaMemcpyAsync rank1 local shard");

    // Peer sends into the remote output slot
    check_cuda(cudaSetDevice(dev0), "cudaSetDevice(dev0)");
    check_cuda(enqueue_bulk_tma_copy_sm90(rank0_shard, rank1_peer_slot, shard_numel, stream0),
               "enqueue_bulk_tma_copy_sm90 rank0 shard -> rank1 output");

    check_cuda(cudaSetDevice(dev1), "cudaSetDevice(dev1)");
    check_cuda(enqueue_bulk_tma_copy_sm90(rank1_shard, rank0_peer_slot, shard_numel, stream1),
               "enqueue_bulk_tma_copy_sm90 rank1 shard -> rank0 output");

    check_cuda(cudaSetDevice(dev0), "cudaSetDevice(dev0)");
    check_cuda(cudaStreamSynchronize(stream0), "cudaStreamSynchronize(stream0 allgather)");

    check_cuda(cudaSetDevice(dev1), "cudaSetDevice(dev1)");
    check_cuda(cudaStreamSynchronize(stream1), "cudaStreamSynchronize(stream1 allgather)");

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
    check_cuda(cudaGetDeviceCount(&ndev), "cudaGetDeviceCount");
    if (ndev < 2) {
        throw std::runtime_error("tma_two_gpu_all_reduce_smoke_test: need at least 2 GPUs");
    }
    if (dev0 == dev1) {
        throw std::invalid_argument("tma_two_gpu_all_reduce_smoke_test: dev0 and dev1 must differ");
    }

    ensure_context_on_device(dev0);
    ensure_context_on_device(dev1);

    cudaStream_t stream0 = create_stream_on_device(dev0);
    cudaStream_t stream1 = create_stream_on_device(dev1);

    const size_t bytes = static_cast<size_t>(numel) * sizeof(half);

    half* rank0 = nullptr;
    half* rank1 = nullptr;

    check_cuda(cudaSetDevice(dev0), "cudaSetDevice(dev0)");
    check_cuda(cudaMalloc(&rank0, bytes), "cudaMalloc(rank0)");

    check_cuda(cudaSetDevice(dev1), "cudaSetDevice(dev1)");
    check_cuda(cudaMalloc(&rank1, bytes), "cudaMalloc(rank1)");

    std::vector<int> access_devices = {dev0, dev1};
    mapped_peer_buffer inbox0 = alloc_peer_visible_buffer(bytes, dev0, access_devices);
    mapped_peer_buffer inbox1 = alloc_peer_visible_buffer(bytes, dev1, access_devices);

    check_cuda(cudaSetDevice(dev0), "cudaSetDevice(dev0)");
    check_cuda(cudaMemset(inbox0.ptr, 0, inbox0.mapped_size), "cudaMemset(inbox0)");

    check_cuda(cudaSetDevice(dev1), "cudaSetDevice(dev1)");
    check_cuda(cudaMemset(inbox1.ptr, 0, inbox1.mapped_size), "cudaMemset(inbox1)");

    // Fill rank-local inputs
    check_cuda(cudaSetDevice(dev0), "cudaSetDevice(dev0)");
    fill_pattern(rank0, numel, 0.25f, 1.0f, stream0);

    check_cuda(cudaSetDevice(dev1), "cudaSetDevice(dev1)");
    fill_pattern(rank1, numel, 0.50f, 2.0f, stream1);

    check_cuda(cudaSetDevice(dev0), "cudaSetDevice(dev0)");
    check_cuda(cudaStreamSynchronize(stream0), "cudaStreamSynchronize(fill rank0)");

    check_cuda(cudaSetDevice(dev1), "cudaSetDevice(dev1)");
    check_cuda(cudaStreamSynchronize(stream1), "cudaStreamSynchronize(fill rank1)");

    check_cuda(
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

    auto got0 = copy_half_device_to_host_float(rank0, numel, dev0);
    auto got1 = copy_half_device_to_host_float(rank1, numel, dev1);

    auto ref0 = host_reference_pattern(numel, 0.25f, 1.0f);
    auto ref1 = host_reference_pattern(numel, 0.50f, 2.0f);
    std::vector<float> ref(static_cast<size_t>(numel));
    for (int64_t i = 0; i < numel; ++i) {
        ref[static_cast<size_t>(i)] =
            ref0[static_cast<size_t>(i)] + ref1[static_cast<size_t>(i)];
    }

    expect_allclose(got0, ref, "all_reduce rank0");
    expect_allclose(got1, ref, "all_reduce rank1");

    check_cuda(cudaSetDevice(dev0), "cudaSetDevice(dev0)");
    cudaStreamDestroy(stream0);
    cudaFree(rank0);

    check_cuda(cudaSetDevice(dev1), "cudaSetDevice(dev1)");
    cudaStreamDestroy(stream1);
    cudaFree(rank1);

    free_peer_visible_buffer(inbox0);
    free_peer_visible_buffer(inbox1);

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
    check_cuda(cudaGetDeviceCount(&ndev), "cudaGetDeviceCount");
    if (ndev < 2) {
        throw std::runtime_error("tma_two_gpu_all_gather_smoke_test: need at least 2 GPUs");
    }
    if (dev0 == dev1) {
        throw std::invalid_argument("tma_two_gpu_all_gather_smoke_test: dev0 and dev1 must differ");
    }

    ensure_context_on_device(dev0);
    ensure_context_on_device(dev1);

    cudaStream_t stream0 = create_stream_on_device(dev0);
    cudaStream_t stream1 = create_stream_on_device(dev1);

    const size_t shard_bytes = static_cast<size_t>(shard_numel) * sizeof(half);
    const size_t full_bytes = 2 * shard_bytes;

    half* rank0_shard = nullptr;
    half* rank1_shard = nullptr;

    check_cuda(cudaSetDevice(dev0), "cudaSetDevice(dev0)");
    check_cuda(cudaMalloc(&rank0_shard, shard_bytes), "cudaMalloc(rank0_shard)");

    check_cuda(cudaSetDevice(dev1), "cudaSetDevice(dev1)");
    check_cuda(cudaMalloc(&rank1_shard, shard_bytes), "cudaMalloc(rank1_shard)");

    std::vector<int> access_devices = {dev0, dev1};
    mapped_peer_buffer out0 = alloc_peer_visible_buffer(full_bytes, dev0, access_devices);
    mapped_peer_buffer out1 = alloc_peer_visible_buffer(full_bytes, dev1, access_devices);

    check_cuda(cudaSetDevice(dev0), "cudaSetDevice(dev0)");
    check_cuda(cudaMemset(out0.ptr, 0, out0.mapped_size), "cudaMemset(out0)");

    check_cuda(cudaSetDevice(dev1), "cudaSetDevice(dev1)");
    check_cuda(cudaMemset(out1.ptr, 0, out1.mapped_size), "cudaMemset(out1)");

    check_cuda(cudaSetDevice(dev0), "cudaSetDevice(dev0)");
    fill_pattern(rank0_shard, shard_numel, 1.0f, 10.0f, stream0);

    check_cuda(cudaSetDevice(dev1), "cudaSetDevice(dev1)");
    fill_pattern(rank1_shard, shard_numel, 1.0f, 100.0f, stream1);

    check_cuda(cudaSetDevice(dev0), "cudaSetDevice(dev0)");
    check_cuda(cudaStreamSynchronize(stream0), "cudaStreamSynchronize(fill rank0 shard)");

    check_cuda(cudaSetDevice(dev1), "cudaSetDevice(dev1)");
    check_cuda(cudaStreamSynchronize(stream1), "cudaStreamSynchronize(fill rank1 shard)");

    check_cuda(
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

    auto got0 = copy_half_device_to_host_float(reinterpret_cast<half*>(out0.ptr), 2 * shard_numel, dev0);
    auto got1 = copy_half_device_to_host_float(reinterpret_cast<half*>(out1.ptr), 2 * shard_numel, dev1);

    auto ref_shard0 = host_reference_pattern(shard_numel, 1.0f, 10.0f);
    auto ref_shard1 = host_reference_pattern(shard_numel, 1.0f, 100.0f);
    std::vector<float> ref_full(static_cast<size_t>(2 * shard_numel));
    for (int64_t i = 0; i < shard_numel; ++i) {
        ref_full[static_cast<size_t>(i)] = ref_shard0[static_cast<size_t>(i)];
        ref_full[static_cast<size_t>(i + shard_numel)] = ref_shard1[static_cast<size_t>(i)];
    }

    expect_allclose(got0, ref_full, "all_gather rank0");
    expect_allclose(got1, ref_full, "all_gather rank1");

    check_cuda(cudaSetDevice(dev0), "cudaSetDevice(dev0)");
    cudaStreamDestroy(stream0);
    cudaFree(rank0_shard);

    check_cuda(cudaSetDevice(dev1), "cudaSetDevice(dev1)");
    cudaStreamDestroy(stream1);
    cudaFree(rank1_shard);

    free_peer_visible_buffer(out0);
    free_peer_visible_buffer(out1);

    return true;
}

} // namespace ooverlap
