#include "comm/tma_two_gpu_peer_allreduce_fast_gmem_sm90.h"

#include "ooverlap/system/runtime_utils.cuh"

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

__host__ __device__ __forceinline__ size_t window_begin_byte_sm90(
    comm::utils::Window window) {
    return static_cast<size_t>(window.start_chunk) *
           TMA_TWO_GPU_PEER_CHUNK_BYTES;
}

__host__ __device__ __forceinline__ size_t window_end_byte_sm90(
    comm::utils::Window window,
    size_t total_bytes) {
    const size_t end =
        static_cast<size_t>(window.start_chunk + window.chunk_count) *
        TMA_TWO_GPU_PEER_CHUNK_BYTES;

    return comm::utils::min_sz(end, total_bytes);
}

__host__ __device__ __forceinline__ size_t window_size_bytes_sm90(
    comm::utils::Window window,
    size_t total_bytes) {
    const size_t begin = window_begin_byte_sm90(window);
    const size_t end = window_end_byte_sm90(window, total_bytes);

    return (begin < end) ? (end - begin) : 0;
}

template <typename ReduceApply>
__device__ void reduce_window_tma_sm90(
    const unsigned char* local_in_bytes,
    unsigned char* peer_buf_bytes,
    comm::utils::Window window,
    size_t total_bytes,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    const size_t begin = window_begin_byte_sm90(window);
    const size_t bytes = window_size_bytes_sm90(window, total_bytes);

    comm::window_pipeline::run_window<
        TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH,
        TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP,
        TMA_TWO_GPU_PEER_CHUNK_BYTES,
        ReduceApply>(
            local_in_bytes + begin,
            peer_buf_bytes + begin,
            bytes,
            shared_raw,
            barriers);
}

template <typename ReduceApply>
__device__ void reduce_window_tma_and_signal_sm90(
    const unsigned char* local_in_bytes,
    unsigned char* peer_buf_bytes,
    comm::utils::Window window,
    size_t total_bytes,
    int* ready_count,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    const size_t begin = window_begin_byte_sm90(window);
    const size_t bytes = window_size_bytes_sm90(window, total_bytes);

    comm::window_pipeline::run_window_signal<
        TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH,
        TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP,
        TMA_TWO_GPU_PEER_CHUNK_BYTES,
        TMA_TWO_GPU_PEER_OVERLAP_SIGNAL_BATCH_CHUNKS,
        ReduceApply>(
            local_in_bytes + begin,
            peer_buf_bytes + begin,
            bytes,
            ready_count,
            shared_raw,
            barriers);
}

__device__ void fast_copy_window_cta_sm90(
    const unsigned char* __restrict__ src_bytes,
    unsigned char* __restrict__ dst_bytes,
    comm::utils::Window window,
    size_t total_bytes) {
    const size_t begin = window_begin_byte_sm90(window);
    const size_t bytes = window_size_bytes_sm90(window, total_bytes);

    comm::window_pipeline::copy_window_gmem<
        uint4,
        TMA_TWO_GPU_PEER_FAST_COPY_UNROLL>(
            src_bytes + begin,
            dst_bytes + begin,
            bytes);
}

__device__ void fast_copy_window_after_reduce_ready_sm90(
    const unsigned char* __restrict__ peer_buf_bytes,
    unsigned char* __restrict__ local_buf_bytes,
    comm::utils::Window window,
    size_t total_bytes,
    const int* ready_count) {
    constexpr int BatchChunks =
        TMA_TWO_GPU_PEER_OVERLAP_SIGNAL_BATCH_CHUNKS;

    static_assert(BatchChunks > 0, "BatchChunks must be > 0");

    if (window.chunk_count <= 0 || ready_count == nullptr) {
        return;
    }

    const size_t window_begin = window_begin_byte_sm90(window);
    const size_t window_bytes = window_size_bytes_sm90(window, total_bytes);

    const unsigned char* peer_window = peer_buf_bytes + window_begin;
    unsigned char* local_window = local_buf_bytes + window_begin;

    for (int begin = 0; begin < window.chunk_count; begin += BatchChunks) {
        int end = begin + BatchChunks;

        if (end > window.chunk_count) {
            end = window.chunk_count;
        }

        comm::window_pipeline::wait_reduce_ready_count(
            ready_count,
            end);

        const size_t begin_byte =
            comm::window_pipeline::chunk_range_begin_byte<
                TMA_TWO_GPU_PEER_CHUNK_BYTES>(begin);

        const size_t bytes =
            comm::window_pipeline::chunk_range_size_bytes<
                TMA_TWO_GPU_PEER_CHUNK_BYTES>(
                    begin,
                    end,
                    window_bytes);

        comm::window_pipeline::copy_window_gmem_range_no_fence<
            uint4,
            TMA_TWO_GPU_PEER_FAST_COPY_UNROLL>(
                peer_window,
                local_window,
                begin_byte,
                bytes);
    }

    comm::window_pipeline::finish_window_gmem_copy();
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
    int owned_windows) {
    comm::window_pipeline::wait_for_collective_ready(
        local_ready_signal,
        peer_ready_signal,
        collective_epoch);

    const int owned_window_idx = static_cast<int>(blockIdx.x);

    if (owned_window_idx >= owned_windows) {
        return;
    }

    const size_t total_bytes = count * static_cast<size_t>(ElemBytes);

    const int num_chunks =
        comm::utils::ceil_div_int64_to_int(
            total_bytes,
            TMA_TWO_GPU_PEER_CHUNK_BYTES);

    const int num_windows = comm::utils::window_num_chunks(num_chunks);
    const int window_idx = 2 * owned_window_idx + rank;

    if (window_idx >= num_windows) {
        return;
    }

    const comm::utils::Window window =
        comm::utils::make_window(window_idx, num_chunks, num_windows);

    if (window.chunk_count <= 0) {
        return;
    }

    extern __shared__ uint4 shared_storage_u4[];

    unsigned char* shared_raw =
        reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore barriers[TMA_TWO_GPU_PEER_BARRIER_COUNT];

    const unsigned char* local_in_bytes =
        reinterpret_cast<const unsigned char*>(local_in);

    unsigned char* local_buf_bytes =
        reinterpret_cast<unsigned char*>(local_buf);

    unsigned char* peer_buf_bytes =
        reinterpret_cast<unsigned char*>(peer_buf);

    reduce_window_tma_sm90<ReduceApply>(
        local_in_bytes,
        peer_buf_bytes,
        window,
        total_bytes,
        shared_raw,
        barriers);

    fast_copy_window_cta_sm90(
        peer_buf_bytes,
        local_buf_bytes,
        window,
        total_bytes);
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
    int* window_ready_counts,
    int owned_windows) {
    comm::window_pipeline::wait_for_collective_ready(
        local_ready_signal,
        peer_ready_signal,
        collective_epoch);

    const int owned_window_idx =
        static_cast<int>(blockIdx.x) /
        TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_WINDOW;

    const int role =
        static_cast<int>(blockIdx.x) %
        TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_WINDOW;

    if (owned_window_idx >= owned_windows) {
        return;
    }

    const size_t total_bytes = count * static_cast<size_t>(ElemBytes);

    const int num_chunks =
        comm::utils::ceil_div_int64_to_int(
            total_bytes,
            TMA_TWO_GPU_PEER_CHUNK_BYTES);

    const int num_windows = comm::utils::window_num_chunks(num_chunks);
    const int window_idx = 2 * owned_window_idx + rank;

    if (window_idx >= num_windows) {
        return;
    }

    const comm::utils::Window window =
        comm::utils::make_window(window_idx, num_chunks, num_windows);

    if (window.chunk_count <= 0) {
        return;
    }

    const unsigned char* local_in_bytes =
        reinterpret_cast<const unsigned char*>(local_in);

    unsigned char* local_buf_bytes =
        reinterpret_cast<unsigned char*>(local_buf);

    unsigned char* peer_buf_bytes =
        reinterpret_cast<unsigned char*>(peer_buf);

    int* ready_count =
        (window_ready_counts != nullptr)
            ? window_ready_counts + owned_window_idx
            : nullptr;

    if (role == 1) {
        fast_copy_window_after_reduce_ready_sm90(
            peer_buf_bytes,
            local_buf_bytes,
            window,
            total_bytes,
            ready_count);
        return;
    }

    extern __shared__ uint4 shared_storage_u4[];

    unsigned char* shared_raw =
        reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore barriers[TMA_TWO_GPU_PEER_BARRIER_COUNT];

    reduce_window_tma_and_signal_sm90<ReduceApply>(
        local_in_bytes,
        peer_buf_bytes,
        window,
        total_bytes,
        ready_count,
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
        "cudaMalloc(overlap ready counts)");

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
    int collective_epoch) {
    const int device = (rank == 0) ? dev0 : dev1;

    const size_t total_bytes = count * static_cast<size_t>(ElemBytes);

    const int num_chunks =
        comm::utils::ceil_div_int64_to_int(
            total_bytes,
            TMA_TWO_GPU_PEER_CHUNK_BYTES);

    const int num_windows = comm::utils::window_num_chunks(num_chunks);

    const int owned_windows =
        (rank == 0) ? ((num_windows + 1) / 2) : (num_windows / 2);

    const bool needs_rendezvous =
        local_ready_signal != nullptr &&
        peer_ready_signal != nullptr &&
        collective_epoch > 0;

    int num_blocks = 0;

    if constexpr (Overlap) {
        num_blocks =
            owned_windows * TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_WINDOW;
    } else {
        num_blocks = owned_windows;
    }

    if (needs_rendezvous) {
        num_blocks = std::max(1, num_blocks);
    }

    if (num_blocks <= 0) {
        return cudaSuccess;
    }

    configure_kernel_once_for<ReduceApply, ElemBytes, Overlap>(device);

    system::runtime::set_device(device);

    if constexpr (Overlap) {
        int* ready_counts = nullptr;

        if (owned_windows > 0) {
            ready_counts =
                ensure_signal_capacity(
                    device,
                    static_cast<size_t>(owned_windows));

            system::runtime::check_cuda(
                cudaMemsetAsync(
                    ready_counts,
                    0,
                    static_cast<size_t>(owned_windows) * sizeof(int),
                    stream),
                "cudaMemsetAsync(overlap ready counts)");
        }

        tma_overlap_fastcopy_rank_kernel_sm90<
            ReduceApply,
            ElemBytes><<<
                num_blocks,
                TMA_TWO_GPU_PEER_THREADS,
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
                    ready_counts,
                    owned_windows);
    } else {
        tma_then_fastcopy_rank_kernel_sm90<
            ReduceApply,
            ElemBytes><<<
                num_blocks,
                TMA_TWO_GPU_PEER_THREADS,
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
                    owned_windows);
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
    int collective_epoch) {
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
        collective_epoch);
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
    int collective_epoch) {
    if (dtype == OO_DTYPE_FLOAT16) {
        if (op == OO_REDUCE_ADD) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceAddNoFtzF16,
                static_cast<int>(sizeof(half)),
                Overlap>(
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
                    collective_epoch);
        }

        if (op == OO_REDUCE_MIN) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceMinF16,
                static_cast<int>(sizeof(half)),
                Overlap>(
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
                    collective_epoch);
        }

        if (op == OO_REDUCE_MAX) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceMaxF16,
                static_cast<int>(sizeof(half)),
                Overlap>(
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
                    collective_epoch);
        }
    }

    if (dtype == OO_DTYPE_BFLOAT16) {
        if (op == OO_REDUCE_ADD) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceAddBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                Overlap>(
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
                    collective_epoch);
        }

        if (op == OO_REDUCE_MIN) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceMinBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                Overlap>(
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
                    collective_epoch);
        }

        if (op == OO_REDUCE_MAX) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceMaxBF16,
                static_cast<int>(sizeof(__nv_bfloat16)),
                Overlap>(
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
                    collective_epoch);
        }
    }

    if (dtype == OO_DTYPE_FLOAT32) {
        if (op == OO_REDUCE_ADD) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceAddF32,
                static_cast<int>(sizeof(float)),
                Overlap>(
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
                    collective_epoch);
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
    int collective_epoch) {
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
        collective_epoch);
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
    int collective_epoch) {
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
        collective_epoch);
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
    int collective_epoch) {
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
        collective_epoch);
}

} // namespace ooverlap
