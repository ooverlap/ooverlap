#include "comm/tma_two_gpu_peer_allreduce_fast_gmem_sm90.h"

#include "ooverlap/system/runtime_utils.cuh"

#include "comm/launch_config.h"
#include "comm/params.h"
#include "comm/pipeline_tma_reduce.h"
#include "comm/utils.h"
#include "comm/window_pipeline_sm90.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace ooverlap {
namespace {

struct SignalCacheEntry {
    int* ptr = nullptr;
    size_t capacity = 0;
};

__host__ __device__ __forceinline__ size_t dtype_size_bytes(
    oo_dtype_t dtype) {
    switch (dtype) {
        case OO_DTYPE_FLOAT16:
            return sizeof(half);
        case OO_DTYPE_BFLOAT16:
            return sizeof(__nv_bfloat16);
        case OO_DTYPE_FLOAT32:
            return sizeof(float);
        default:
            return 0;
    }
}

__host__ __device__ __forceinline__ int overlap_pair_count_for_windows(
    int owned_windows,
    int max_ctas) {
    if (owned_windows <= 0 || max_ctas < TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA) {
        return 0;
    }

    return comm::utils::min_int(
        owned_windows,
        max_ctas / TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA);
}

template <typename ReduceApply, int ElemBytes>
__global__ void tma_then_fastcopy_rank_kernel_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    int rank,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    int ctas_per_rank,
    int window_chunks) {
    comm::window_pipeline::wait_for_collective_ready(
        local_ready_signal,
        peer_ready_signal,
        collective_epoch);

    const int cta_idx = static_cast<int>(blockIdx.x);

    if (ctas_per_rank <= 0 || cta_idx >= ctas_per_rank) {
        return;
    }

    const size_t total_bytes = count * static_cast<size_t>(ElemBytes);

    const int num_chunks =
        comm::utils::ceil_div_int64_to_int(
            total_bytes,
            TMA_TWO_GPU_PEER_CHUNK_BYTES);

    const int num_windows =
        comm::utils::window_count_for_chunks(
            num_chunks,
            window_chunks);

    const comm::utils::WindowRange rank_range =
        comm::utils::rank_window_range(num_windows, rank);

    const comm::utils::WindowRange cta_range =
        comm::utils::cta_window_range(
            cta_idx,
            ctas_per_rank,
            rank_range);

    if (cta_range.begin >= cta_range.end) {
        return;
    }

    extern __shared__ uint4 shared_storage_u4[];

    unsigned char* shared_raw =
        reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore barriers[TMA_TWO_GPU_PEER_BARRIER_COUNT];

    comm::window_pipeline::run_window_range<
        TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH,
        TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP,
        TMA_TWO_GPU_PEER_CHUNK_BYTES,
        ReduceApply>(
            local_in,
            peer_buf,
            total_bytes,
            cta_range.begin,
            cta_range.end,
            window_chunks,
            shared_raw,
            barriers);

    comm::window_pipeline::copy_window_range_gmem<
        uint4,
        TMA_TWO_GPU_PEER_FAST_COPY_UNROLL,
        TMA_TWO_GPU_PEER_CHUNK_BYTES>(
            peer_buf,
            local_buf,
            total_bytes,
            cta_range.begin,
            cta_range.end,
            window_chunks);
}

template <typename ReduceApply, int ElemBytes>
__global__ void tma_overlap_fastcopy_rank_kernel_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    int rank,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    int* window_ready_flags,
    int pair_count,
    int window_chunks) {
    comm::window_pipeline::wait_for_collective_ready(
        local_ready_signal,
        peer_ready_signal,
        collective_epoch);

    const int pair_idx =
        static_cast<int>(blockIdx.x) /
        TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA;

    const int role =
        static_cast<int>(blockIdx.x) %
        TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA;

    if (pair_count <= 0 || pair_idx >= pair_count) {
        return;
    }

    const size_t total_bytes = count * static_cast<size_t>(ElemBytes);

    const int num_chunks =
        comm::utils::ceil_div_int64_to_int(
            total_bytes,
            TMA_TWO_GPU_PEER_CHUNK_BYTES);

    const int num_windows =
        comm::utils::window_count_for_chunks(
            num_chunks,
            window_chunks);

    const comm::utils::WindowRange rank_range =
        comm::utils::rank_window_range(num_windows, rank);

    const comm::utils::WindowRange pair_range =
        comm::utils::cta_window_range(
            pair_idx,
            pair_count,
            rank_range);

    if (pair_range.begin >= pair_range.end) {
        return;
    }

    if (role == 1) {
        comm::window_pipeline::copy_window_range_gmem_after_ready<
            uint4,
            TMA_TWO_GPU_PEER_FAST_COPY_UNROLL,
            TMA_TWO_GPU_PEER_CHUNK_BYTES>(
                peer_buf,
                local_buf,
                total_bytes,
                pair_range.begin,
                pair_range.end,
                window_chunks,
                window_ready_flags,
                rank_range.begin);
        return;
    }

    extern __shared__ uint4 shared_storage_u4[];

    unsigned char* shared_raw =
        reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore barriers[TMA_TWO_GPU_PEER_BARRIER_COUNT];

    comm::window_pipeline::run_window_range_signal<
        TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH,
        TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP,
        TMA_TWO_GPU_PEER_CHUNK_BYTES,
        ReduceApply>(
            local_in,
            peer_buf,
            total_bytes,
            pair_range.begin,
            pair_range.end,
            window_chunks,
            window_ready_flags,
            rank_range.begin,
            shared_raw,
            barriers);
}

SignalCacheEntry& signal_cache_for_device(int device) {
    static std::mutex mutex;
    static std::unordered_map<int, SignalCacheEntry> cache;

    std::lock_guard<std::mutex> lock(mutex);
    return cache[device];
}

int* ensure_signal_capacity(int device, size_t required_count) {
    if (required_count == 0) {
        return nullptr;
    }

    SignalCacheEntry& entry = signal_cache_for_device(device);

    if (entry.ptr != nullptr && entry.capacity >= required_count) {
        return entry.ptr;
    }

    system::runtime::set_device(device);

    if (entry.ptr != nullptr) {
        cudaFree(entry.ptr);
        entry.ptr = nullptr;
        entry.capacity = 0;
    }

    system::runtime::check_cuda(
        cudaMalloc(&entry.ptr, required_count * sizeof(int)),
        "cudaMalloc(overlap window ready flags)");

    entry.capacity = required_count;
    return entry.ptr;
}

template <typename ReduceApply, int ElemBytes, bool Overlap>
void configure_kernel_once_for(int device) {
    struct CacheEntry {
        bool configured = false;
        size_t dynamic_smem_bytes = 0;
    };

    static std::mutex mutex;
    static std::unordered_map<int, CacheEntry> cache;

    const size_t dynamic_smem_bytes =
        TMA_TWO_GPU_PEER_DYNAMIC_SHARED_BYTES;

    const size_t total_smem_bytes =
        dynamic_smem_bytes + TMA_TWO_GPU_PEER_STATIC_SHARED_BYTES;

    std::lock_guard<std::mutex> lock(mutex);

    auto it = cache.find(device);

    if (it != cache.end() &&
        it->second.configured &&
        it->second.dynamic_smem_bytes == dynamic_smem_bytes) {
        return;
    }

    system::runtime::set_device(device);

    cudaDeviceProp prop{};

    system::runtime::check_cuda(
        cudaGetDeviceProperties(&prop, device),
        "cudaGetDeviceProperties");

    if (total_smem_bytes >
        static_cast<size_t>(prop.sharedMemPerBlockOptin)) {
        throw std::runtime_error(
            "tma_fastcopy_allreduce_configure_kernel_once: requested shared memory exceeds opt-in limit");
    }

    if constexpr (Overlap) {
        if (dynamic_smem_bytes >
            static_cast<size_t>(prop.sharedMemPerBlock)) {
            system::runtime::check_cuda(
                cudaFuncSetAttribute(
                    tma_overlap_fastcopy_rank_kernel_sm90<
                        ReduceApply,
                        ElemBytes>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(dynamic_smem_bytes)),
                "cudaFuncSetAttribute(MaxDynamicSharedMemorySize overlap)");
        }

        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                tma_overlap_fastcopy_rank_kernel_sm90<
                    ReduceApply,
                    ElemBytes>,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                100),
            "cudaFuncSetAttribute(PreferredSharedMemoryCarveout overlap)");
    } else {
        if (dynamic_smem_bytes >
            static_cast<size_t>(prop.sharedMemPerBlock)) {
            system::runtime::check_cuda(
                cudaFuncSetAttribute(
                    tma_then_fastcopy_rank_kernel_sm90<
                        ReduceApply,
                        ElemBytes>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(dynamic_smem_bytes)),
                "cudaFuncSetAttribute(MaxDynamicSharedMemorySize seq)");
        }

        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                tma_then_fastcopy_rank_kernel_sm90<
                    ReduceApply,
                    ElemBytes>,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                100),
            "cudaFuncSetAttribute(PreferredSharedMemoryCarveout seq)");
    }

    cache[device] = {true, dynamic_smem_bytes};
}

template <typename ReduceApply, int ElemBytes, bool Overlap>
cudaError_t launch_rank_kernel_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    int rank,
    int dev0,
    int dev1,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    comm::LaunchConfig launch_config) {
    if constexpr (Overlap) {
        if (!comm::launch_config_valid_for_overlap(launch_config)) {
            return cudaErrorInvalidValue;
        }
    } else {
        if (!comm::launch_config_valid(launch_config)) {
            return cudaErrorInvalidValue;
        }
    }

    const int device = (rank == 0) ? dev0 : dev1;

    const size_t total_bytes = count * static_cast<size_t>(ElemBytes);

    const int num_chunks =
        comm::utils::ceil_div_int64_to_int(
            total_bytes,
            TMA_TWO_GPU_PEER_CHUNK_BYTES);

    const int num_windows =
        comm::utils::window_count_for_chunks(
            num_chunks,
            launch_config.window_chunks);

    const int owned_windows =
        comm::utils::rank_window_count(num_windows, rank);

    const bool needs_rendezvous =
        local_ready_signal != nullptr &&
        peer_ready_signal != nullptr &&
        collective_epoch > 0;

    configure_kernel_once_for<ReduceApply, ElemBytes, Overlap>(device);

    system::runtime::set_device(device);

    if constexpr (Overlap) {
        const int pair_count =
            overlap_pair_count_for_windows(
                owned_windows,
                launch_config.max_ctas);

        int num_blocks =
            pair_count * TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA;

        if (needs_rendezvous) {
            num_blocks = std::max(1, num_blocks);
        }

        if (num_blocks <= 0) {
            return cudaSuccess;
        }

        int* ready_flags = nullptr;

        if (owned_windows > 0) {
            ready_flags =
                ensure_signal_capacity(
                    device,
                    static_cast<size_t>(owned_windows));

            system::runtime::check_cuda(
                cudaMemsetAsync(
                    ready_flags,
                    0,
                    static_cast<size_t>(owned_windows) * sizeof(int),
                    stream),
                "cudaMemsetAsync(overlap window ready flags)");
        }

        tma_overlap_fastcopy_rank_kernel_sm90<
            ReduceApply,
            ElemBytes><<<
                num_blocks,
                launch_config.threads,
                TMA_TWO_GPU_PEER_DYNAMIC_SHARED_BYTES,
                stream>>>(
                    local_in,
                    local_buf,
                    peer_buf,
                    count,
                    rank,
                    local_ready_signal,
                    peer_ready_signal,
                    collective_epoch,
                    ready_flags,
                    pair_count,
                    launch_config.window_chunks);
    } else {
        const int ctas_per_rank =
            comm::utils::cta_count_for_windows(
                owned_windows,
                launch_config.max_ctas);

        int num_blocks =
            needs_rendezvous ? std::max(1, ctas_per_rank) : ctas_per_rank;

        if (num_blocks <= 0) {
            return cudaSuccess;
        }

        tma_then_fastcopy_rank_kernel_sm90<
            ReduceApply,
            ElemBytes><<<
                num_blocks,
                launch_config.threads,
                TMA_TWO_GPU_PEER_DYNAMIC_SHARED_BYTES,
                stream>>>(
                    local_in,
                    local_buf,
                    peer_buf,
                    count,
                    rank,
                    local_ready_signal,
                    peer_ready_signal,
                    collective_epoch,
                    ctas_per_rank,
                    launch_config.window_chunks);
    }

    return cudaGetLastError();
}

template <typename ReduceOp, int ElemBytes, bool Overlap>
cudaError_t launch_reduce_op_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    int rank,
    int dev0,
    int dev1,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    comm::LaunchConfig launch_config) {
    using ReduceApply = comm::PipelineTMAReduce<
        TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH,
        TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP,
        ReduceOp>;

    return launch_rank_kernel_sm90<ReduceApply, ElemBytes, Overlap>(
        local_in,
        local_buf,
        peer_buf,
        count,
        rank,
        dev0,
        dev1,
        stream,
        local_ready_signal,
        peer_ready_signal,
        collective_epoch,
        launch_config);
}

template <bool Overlap>
cudaError_t dispatch_rank_kernel_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int rank,
    int dev0,
    int dev1,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    comm::LaunchConfig launch_config) {
    if (dtype == OO_DTYPE_FLOAT16) {
        if (op == OO_REDUCE_ADD) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceAddNoFtzF16,
                static_cast<int>(sizeof(half)),
                Overlap>(
                    local_in, local_buf, peer_buf, count, rank, dev0, dev1,
                    stream, local_ready_signal, peer_ready_signal,
                    collective_epoch, launch_config);
        }

        if (op == OO_REDUCE_MIN) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceMinF16,
                static_cast<int>(sizeof(half)),
                Overlap>(
                    local_in, local_buf, peer_buf, count, rank, dev0, dev1,
                    stream, local_ready_signal, peer_ready_signal,
                    collective_epoch, launch_config);
        }

        if (op == OO_REDUCE_MAX) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceMaxF16,
                static_cast<int>(sizeof(half)),
                Overlap>(
                    local_in, local_buf, peer_buf, count, rank, dev0, dev1,
                    stream, local_ready_signal, peer_ready_signal,
                    collective_epoch, launch_config);
        }
    }

    if (dtype == OO_DTYPE_BFLOAT16) {
        if (op == OO_REDUCE_ADD) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceAddBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                Overlap>(
                    local_in, local_buf, peer_buf, count, rank, dev0, dev1,
                    stream, local_ready_signal, peer_ready_signal,
                    collective_epoch, launch_config);
        }

        if (op == OO_REDUCE_MIN) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceMinBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                Overlap>(
                    local_in, local_buf, peer_buf, count, rank, dev0, dev1,
                    stream, local_ready_signal, peer_ready_signal,
                    collective_epoch, launch_config);
        }

        if (op == OO_REDUCE_MAX) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceMaxBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                Overlap>(
                    local_in, local_buf, peer_buf, count, rank, dev0, dev1,
                    stream, local_ready_signal, peer_ready_signal,
                    collective_epoch, launch_config);
        }
    }

    if (dtype == OO_DTYPE_FLOAT32) {
        if (op == OO_REDUCE_ADD) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceAddF32,
                static_cast<int>(sizeof(float)),
                Overlap>(
                    local_in, local_buf, peer_buf, count, rank, dev0, dev1,
                    stream, local_ready_signal, peer_ready_signal,
                    collective_epoch, launch_config);
        }
    }

    return cudaErrorInvalidValue;
}

cudaError_t validate_args(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    oo_dtype_t dtype,
    int rank,
    int dev0,
    int dev1) {
    if (local_in == nullptr || local_buf == nullptr || peer_buf == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }

    if (count == 0) {
        return cudaErrorInvalidValue;
    }

    if (rank != 0 && rank != 1) {
        return cudaErrorInvalidValue;
    }

    if (dev0 == dev1) {
        return cudaErrorInvalidValue;
    }

    if (dtype_size_bytes(dtype) == 0) {
        return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}

template <bool Overlap>
cudaError_t enqueue_rank_kernel_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int rank,
    int dev0,
    int dev1,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    comm::LaunchConfig launch_config) {
    cudaError_t status = validate_args(
        local_in,
        local_buf,
        peer_buf,
        count,
        dtype,
        rank,
        dev0,
        dev1);

    if (status != cudaSuccess) {
        return status;
    }

    if constexpr (Overlap) {
        if (!comm::launch_config_valid_for_overlap(launch_config)) {
            return cudaErrorInvalidValue;
        }
    } else {
        if (!comm::launch_config_valid(launch_config)) {
            return cudaErrorInvalidValue;
        }
    }

    return dispatch_rank_kernel_sm90<Overlap>(
        local_in,
        local_buf,
        peer_buf,
        count,
        dtype,
        op,
        rank,
        dev0,
        dev1,
        stream,
        local_ready_signal,
        peer_ready_signal,
        collective_epoch,
        launch_config);
}

} // namespace

cudaError_t enqueue_tma_two_gpu_peer_allreduce_rank_seq_fastcopy_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int rank,
    int dev0,
    int dev1,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    comm::LaunchConfig launch_config) {
    return enqueue_rank_kernel_sm90<false>(
        local_in,
        local_buf,
        peer_buf,
        count,
        dtype,
        op,
        rank,
        dev0,
        dev1,
        stream,
        local_ready_signal,
        peer_ready_signal,
        collective_epoch,
        launch_config);
}

cudaError_t enqueue_tma_two_gpu_peer_allreduce_rank_overlap_fastcopy_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int rank,
    int dev0,
    int dev1,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    comm::LaunchConfig launch_config) {
    return enqueue_rank_kernel_sm90<true>(
        local_in,
        local_buf,
        peer_buf,
        count,
        dtype,
        op,
        rank,
        dev0,
        dev1,
        stream,
        local_ready_signal,
        peer_ready_signal,
        collective_epoch,
        launch_config);
}

} // namespace ooverlap
