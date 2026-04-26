#include "comm/tma_two_gpu_peer_allreduce_pivot_sm90.h"

#include "ooverlap/system/runtime_utils.cuh"

#include "comm/pipeline_stage.h"
#include "comm/pipeline_tma_load.h"
#include "comm/pipeline_tma_copy.h"
#include "comm/pipeline_tma_reduce.h"
#include "comm/fast_gmem_copy.cuh"

#include "comm/params.h"
#include "comm/utils.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#ifndef OOVERLAP_PIVOT_FAST_COPY_BATCH_CHUNKS
#define OOVERLAP_PIVOT_FAST_COPY_BATCH_CHUNKS 16
#endif

namespace ooverlap {
namespace {

struct PivotSignalCacheEntry {
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

__device__ __forceinline__ unsigned char* stage_ptr(
    unsigned char* shared_raw,
    int stage) {
    return shared_raw +
           static_cast<size_t>(stage) * TMA_TWO_GPU_PEER_CHUNK_BYTES;
}

__device__ __forceinline__ void wait_for_collective_ready_sm90(
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch) {
    if (local_ready_signal == nullptr ||
        peer_ready_signal == nullptr ||
        collective_epoch <= 0) {
        return;
    }

    if (threadIdx.x == 0) {
        atomicMax(local_ready_signal, collective_epoch);
        __threadfence_system();

        const volatile int* peer_ready =
            reinterpret_cast<const volatile int*>(peer_ready_signal);

        while (peer_ready[0] < collective_epoch) {
#if defined(__CUDA_ARCH__)
            __nanosleep(64);
#endif
        }
    }

    __syncthreads();
}

__device__ __forceinline__ int compute_pivot_chunk_count(
    int window_chunk_count,
    int pivot_numerator,
    int pivot_denominator) {
    if (window_chunk_count <= 0 ||
        pivot_numerator <= 0 ||
        pivot_denominator <= 0) {
        return 0;
    }

    int pivot =
        static_cast<int>(
            (static_cast<long long>(window_chunk_count) *
             static_cast<long long>(pivot_numerator)) /
            static_cast<long long>(pivot_denominator));

    if (pivot < 0) {
        pivot = 0;
    }

    if (pivot > window_chunk_count) {
        pivot = window_chunk_count;
    }

    return pivot;
}

__device__ __forceinline__ comm::utils::Window make_sub_window_sm90(
    comm::utils::Window window,
    int local_start,
    int local_count) {
    comm::utils::Window out{};
    out.index = window.index;
    out.start_chunk = window.start_chunk + local_start;
    out.chunk_count = local_count;
    out.owner_rank = window.owner_rank;
    return out;
}

__device__ __forceinline__ void publish_pivot_ready_count_sm90(
    int* ready_count,
    int count) {
    if (ready_count == nullptr || count <= 0) {
        return;
    }

    __threadfence();
    atomicMax(ready_count, count);
}

__device__ __forceinline__ void fast_copy_chunk_u128_no_allocate_sm90(
    const unsigned char* __restrict__ src_bytes,
    unsigned char* __restrict__ dst_bytes,
    size_t bytes) {
    comm::fast_copy::copy_byte_range<uint4, 8>(
        src_bytes,
        dst_bytes,
        0,
        bytes,
        static_cast<size_t>(threadIdx.x),
        static_cast<size_t>(blockDim.x));
}

/*
 * Original full-CTA TMA pipeline, kept in the same shape as the normal kernel.
 */
template <int StageDepth, int FillDepth, typename Apply>
__device__ void run_tma_window_pipeline_sm90(
    const unsigned char* src_bytes,
    unsigned char* dst_bytes,
    comm::utils::Window window,
    size_t total_bytes,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    comm::PipelineTMALoad load{};
    Apply apply{};

    for (int warm = 0; warm < FillDepth; ++warm) {
        if (warm >= window.chunk_count) {
            break;
        }

        const int chunk = window.start_chunk + warm;
        const int slot = warm;

        const size_t offset =
            static_cast<size_t>(chunk) * TMA_TWO_GPU_PEER_CHUNK_BYTES;

        const size_t bytes =
            comm::utils::min_sz(
                TMA_TWO_GPU_PEER_CHUNK_BYTES,
                total_bytes - offset);

        comm::PipelineStage stage = comm::make_pipeline_stage(
            comm::make_pipeline_chunk(
                src_bytes + offset,
                dst_bytes + offset,
                bytes),
            stage_ptr(shared_raw, slot),
            &barriers[slot]);

        if (threadIdx.x == 0) {
            load.issue(&stage);
        }

        __syncthreads();
    }

    for (int iter = 0; iter < window.chunk_count; ++iter) {
        const int chunk = window.start_chunk + iter;
        const int cur_slot = iter % StageDepth;

        const size_t offset =
            static_cast<size_t>(chunk) * TMA_TWO_GPU_PEER_CHUNK_BYTES;

        const size_t bytes =
            comm::utils::min_sz(
                TMA_TWO_GPU_PEER_CHUNK_BYTES,
                total_bytes - offset);

        comm::PipelineStage cur_stage = comm::make_pipeline_stage(
            comm::make_pipeline_chunk(
                src_bytes + offset,
                dst_bytes + offset,
                bytes),
            stage_ptr(shared_raw, cur_slot),
            &barriers[cur_slot]);

        if (threadIdx.x == 0) {
            load.wait_ready(&cur_stage);
        }

        __syncthreads();

        const int future_iter = iter + FillDepth;

        if (future_iter < window.chunk_count) {
            const int future_chunk = window.start_chunk + future_iter;
            const int future_slot = future_iter % StageDepth;

            const size_t future_offset =
                static_cast<size_t>(future_chunk) *
                TMA_TWO_GPU_PEER_CHUNK_BYTES;

            const size_t future_bytes =
                comm::utils::min_sz(
                    TMA_TWO_GPU_PEER_CHUNK_BYTES,
                    total_bytes - future_offset);

            comm::PipelineStage future_stage = comm::make_pipeline_stage(
                comm::make_pipeline_chunk(
                    src_bytes + future_offset,
                    dst_bytes + future_offset,
                    future_bytes),
                stage_ptr(shared_raw, future_slot),
                &barriers[future_slot]);

            if (threadIdx.x == 0) {
                if (iter >= FillDepth) {
                    apply.wait_before_stage_reuse();
                }

                load.issue(&future_stage);
            }
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            apply.issue_bulk(&cur_stage);
        }

        apply.finish_tail(&cur_stage);

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        apply.wait_complete();
        __threadfence_system();
    }

    __syncthreads();
}

/*
 * Full-CTA reduce pipeline plus per-window producer signal.
 *
 * Important:
 * For chunks copied by the fast-copy CTA, we need the reduced destination
 * chunk to be truly complete before publishing the counter. Therefore for
 * those pivot chunks we use reduce_async_wait<FillDepth - 1>(), not only
 * reduce_async_read_wait<FillDepth - 1>().
 */
template <typename ReduceApply>
__device__ void reduce_window_and_signal_pivot_cta_sm90(
    const unsigned char* src_bytes,
    unsigned char* dst_bytes,
    comm::utils::Window window,
    size_t total_bytes,
    int pivot_count,
    int* ready_count,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    constexpr int StageDepth = TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH;
    constexpr int FillDepth = TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP;

    static_assert(StageDepth > 0, "StageDepth must be > 0");
    static_assert(FillDepth > 0, "FillDepth must be > 0");
    static_assert(FillDepth <= StageDepth, "FillDepth must be <= StageDepth");

    if (window.chunk_count <= 0) {
        return;
    }

    comm::PipelineTMALoad load{};
    ReduceApply apply{};

   
    auto publish_completed = [&](int completed_count) {
        if (threadIdx.x != 0 ||
            ready_count == nullptr ||
            pivot_count <= 0 ||
            completed_count <= 0) {
            return;
        }
    
        int publish_count = completed_count;
    
        if (publish_count > pivot_count) {
            publish_count = pivot_count;
        }
    
        constexpr int kBatchChunks = OOVERLAP_PIVOT_FAST_COPY_BATCH_CHUNKS;
    
        /*
         * Publish only on batch boundaries, or for the final pivot chunk.
         * This avoids one global fence/atomic per chunk.
         */
        if (publish_count < pivot_count &&
            (publish_count % kBatchChunks) != 0) {
            return;
        }
    
        publish_pivot_ready_count_sm90(ready_count, publish_count);
    };

    for (int warm = 0; warm < FillDepth; ++warm) {
        if (warm >= window.chunk_count) {
            break;
        }

        const int chunk = window.start_chunk + warm;
        const int slot = warm;

        const size_t offset =
            static_cast<size_t>(chunk) * TMA_TWO_GPU_PEER_CHUNK_BYTES;

        const size_t bytes =
            comm::utils::min_sz(
                TMA_TWO_GPU_PEER_CHUNK_BYTES,
                total_bytes - offset);

        comm::PipelineStage stage = comm::make_pipeline_stage(
            comm::make_pipeline_chunk(
                src_bytes + offset,
                dst_bytes + offset,
                bytes),
            stage_ptr(shared_raw, slot),
            &barriers[slot]);

        if (threadIdx.x == 0) {
            load.issue(&stage);
        }

        __syncthreads();
    }

    for (int iter = 0; iter < window.chunk_count; ++iter) {
        const int chunk = window.start_chunk + iter;
        const int cur_slot = iter % StageDepth;

        const size_t offset =
            static_cast<size_t>(chunk) * TMA_TWO_GPU_PEER_CHUNK_BYTES;

        const size_t bytes =
            comm::utils::min_sz(
                TMA_TWO_GPU_PEER_CHUNK_BYTES,
                total_bytes - offset);

        comm::PipelineStage cur_stage = comm::make_pipeline_stage(
            comm::make_pipeline_chunk(
                src_bytes + offset,
                dst_bytes + offset,
                bytes),
            stage_ptr(shared_raw, cur_slot),
            &barriers[cur_slot]);

        if (threadIdx.x == 0) {
            load.wait_ready(&cur_stage);
        }

        __syncthreads();

        const int future_iter = iter + FillDepth;

        if (future_iter < window.chunk_count) {
            const int future_chunk = window.start_chunk + future_iter;
            const int future_slot = future_iter % StageDepth;

            const size_t future_offset =
                static_cast<size_t>(future_chunk) *
                TMA_TWO_GPU_PEER_CHUNK_BYTES;

            const size_t future_bytes =
                comm::utils::min_sz(
                    TMA_TWO_GPU_PEER_CHUNK_BYTES,
                    total_bytes - future_offset);

            comm::PipelineStage future_stage = comm::make_pipeline_stage(
                comm::make_pipeline_chunk(
                    src_bytes + future_offset,
                    dst_bytes + future_offset,
                    future_bytes),
                stage_ptr(shared_raw, future_slot),
                &barriers[future_slot]);

            if (threadIdx.x == 0) {
                if (iter >= FillDepth) {
                    const int completed_count = iter - FillDepth + 1;
                 
                    if (completed_count <= pivot_count) {
                        constexpr int kBatchChunks =
                            OOVERLAP_PIVOT_FAST_COPY_BATCH_CHUNKS;
                 
                        const bool publish_now =
                            completed_count == pivot_count ||
                            ((completed_count % kBatchChunks) == 0);
                 
                        if (publish_now) {
                            /*
                             * Consumer CTA may read these chunks now, so wait for
                             * the reduce result to be globally complete.
                             */
                            tma::reduce_async_read_wait<FillDepth - 1>();
                            publish_completed(completed_count);
                        } else {
                            /*
                             * No consumer reads this chunk yet. We only need the normal
                             * stage-reuse wait.
                             */
                            apply.wait_before_stage_reuse();
                        }
                    } else {
                        apply.wait_before_stage_reuse();
                    }
                } 

                load.issue(&future_stage);
            }
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            apply.issue_bulk(&cur_stage);
        }

        apply.finish_tail(&cur_stage);

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        apply.wait_complete();
        __threadfence();

        if (pivot_count > 0) {
            publish_pivot_ready_count_sm90(ready_count, pivot_count);
        }

        __threadfence_system();
    }

    __syncthreads();
}

template <typename ReduceApply>
__device__ void reduce_window_to_local_sm90(
    const unsigned char* peer_buf_bytes,
    unsigned char* local_buf_bytes,
    comm::utils::Window window,
    size_t total_bytes,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    run_tma_window_pipeline_sm90<
        TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH,
        TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP,
        ReduceApply>(
            peer_buf_bytes,
            local_buf_bytes,
            window,
            total_bytes,
            shared_raw,
            barriers);
}

__device__ void copy_window_tma_sm90(
    const unsigned char* src_bytes,
    unsigned char* dst_bytes,
    comm::utils::Window window,
    size_t total_bytes,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    using CopyApply = comm::PipelineTMACopy<
        TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH,
        TMA_TWO_GPU_PEER_COPY_STAGE_GAP>;

    run_tma_window_pipeline_sm90<
        TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH,
        TMA_TWO_GPU_PEER_COPY_STAGE_GAP,
        CopyApply>(
            src_bytes,
            dst_bytes,
            window,
            total_bytes,
            shared_raw,
            barriers);
}

__device__ void fast_copy_pivot_prefix_cta_sm90(
    const unsigned char* local_buf_bytes,
    unsigned char* peer_buf_bytes,
    comm::utils::Window window,
    size_t total_bytes,
    int pivot_count,
    const int* ready_count) {
    if (pivot_count <= 0 || ready_count == nullptr) {
        return;
    }

    constexpr int kBatchChunks = OOVERLAP_PIVOT_FAST_COPY_BATCH_CHUNKS;

    const volatile int* ready =
        reinterpret_cast<const volatile int*>(ready_count);

    for (int begin = 0; begin < pivot_count; begin += kBatchChunks) {
        int end = begin + kBatchChunks;

        if (end > pivot_count) {
            end = pivot_count;
        }

        /*
         * Wait until the producer has completed the whole batch.
         */
        while (ready[0] < end) {
#if defined(__CUDA_ARCH__)
            __nanosleep(64);
#endif
        }

        const int begin_chunk = window.start_chunk + begin;
        const int end_chunk = window.start_chunk + end;

        const size_t begin_offset =
            static_cast<size_t>(begin_chunk) *
            TMA_TWO_GPU_PEER_CHUNK_BYTES;

        size_t end_offset =
            static_cast<size_t>(end_chunk) *
            TMA_TWO_GPU_PEER_CHUNK_BYTES;

        if (end_offset > total_bytes) {
            end_offset = total_bytes;
        }

        if (begin_offset >= end_offset) {
            continue;
        }

        const size_t bytes = end_offset - begin_offset;

        /*
         * Copy the whole batch as one contiguous byte range.
         * This should reduce loop overhead and make the fast path closer to
         * the standalone fast copy benchmark.
         */
        fast_copy_chunk_u128_no_allocate_sm90(
            local_buf_bytes + begin_offset,
            peer_buf_bytes + begin_offset,
            bytes);
    }

    if (threadIdx.x == 0) {
        __threadfence_system();
    }
}
    
template <typename ReduceApply, int ElemBytes>
__global__ void tma_two_gpu_allreduce_rank_pivot_kernel_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    int rank,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    int pivot_numerator,
    int pivot_denominator,
    int* pivot_ready_counts,
    int owned_windows,
    int blocks_per_window) {
    wait_for_collective_ready_sm90(
        local_ready_signal,
        peer_ready_signal,
        collective_epoch);

    const bool cta_split = blocks_per_window == 2;

    const int owned_window_idx =
        cta_split
            ? static_cast<int>(blockIdx.x) / 2
            : static_cast<int>(blockIdx.x);

    const int role =
        cta_split
            ? static_cast<int>(blockIdx.x) & 1
            : 0;

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

    (void)local_in_bytes;

    const int pivot_count =
        cta_split
            ? compute_pivot_chunk_count(
                  window.chunk_count,
                  pivot_numerator,
                  pivot_denominator)
            : 0;

    int* window_ready_count =
        (pivot_ready_counts != nullptr)
            ? pivot_ready_counts + owned_window_idx
            : nullptr;

    /*
     * role 1: consumer CTA.
     *
     * It only copies the prefix [0, pivot_count). It has no shared-memory TMA
     * state, but the launch still reserves dynamic smem because this is one
     * combined kernel. That is fine for this test.
     */
    if (role == 1) {
        fast_copy_pivot_prefix_cta_sm90(
            local_buf_bytes,
            peer_buf_bytes,
            window,
            total_bytes,
            pivot_count,
            window_ready_count);
        return;
    }

    /*
     * role 0: producer CTA.
     *
     * This is intentionally full CTA, not one warp. That is the key difference
     * from the previous slow pivot implementation.
     */
    extern __shared__ uint4 shared_storage_u4[];

    unsigned char* shared_raw =
        reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore barriers[TMA_TWO_GPU_PEER_BARRIER_COUNT];

    if (pivot_count <= 0) {
        reduce_window_to_local_sm90<ReduceApply>(
            peer_buf_bytes,
            local_buf_bytes,
            window,
            total_bytes,
            shared_raw,
            barriers);

        copy_window_tma_sm90(
            local_buf_bytes,
            peer_buf_bytes,
            window,
            total_bytes,
            shared_raw,
            barriers);

        return;
    }

    reduce_window_and_signal_pivot_cta_sm90<ReduceApply>(
        peer_buf_bytes,
        local_buf_bytes,
        window,
        total_bytes,
        pivot_count,
        window_ready_count,
        shared_raw,
        barriers);

    const int rest_count = window.chunk_count - pivot_count;

    if (rest_count > 0) {
        const comm::utils::Window rest_window =
            make_sub_window_sm90(
                window,
                pivot_count,
                rest_count);

        copy_window_tma_sm90(
            local_buf_bytes,
            peer_buf_bytes,
            rest_window,
            total_bytes,
            shared_raw,
            barriers);
    }
}

PivotSignalCacheEntry& pivot_signal_cache_for_device(int device) {
    static std::mutex mutex;
    static std::unordered_map<int, PivotSignalCacheEntry> cache;

    std::lock_guard<std::mutex> lock(mutex);
    return cache[device];
}

int* ensure_pivot_signal_capacity(int device, size_t required_count) {
    if (required_count == 0) {
        return nullptr;
    }

    PivotSignalCacheEntry& entry = pivot_signal_cache_for_device(device);

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
        "cudaMalloc(pivot ready counts)");

    entry.capacity = required_count;
    return entry.ptr;
}

template <typename ReduceApply, int ElemBytes>
void configure_pivot_kernel_once_for(int device) {
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
            "tma_two_gpu_peer_allreduce_pivot_configure_kernel_once: requested shared memory exceeds opt-in limit");
    }

    if (total_smem_bytes >
        static_cast<size_t>(prop.sharedMemPerBlock)) {
        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                tma_two_gpu_allreduce_rank_pivot_kernel_sm90<
                    ReduceApply,
                    ElemBytes>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(dynamic_smem_bytes)),
            "cudaFuncSetAttribute(MaxDynamicSharedMemorySize pivot)");
    }

    system::runtime::check_cuda(
        cudaFuncSetAttribute(
            tma_two_gpu_allreduce_rank_pivot_kernel_sm90<
                ReduceApply,
                ElemBytes>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            100),
        "cudaFuncSetAttribute(PreferredSharedMemoryCarveout pivot)");

    cache[device] = {true, dynamic_smem_bytes};
}

template <typename ReduceApply, int ElemBytes>
cudaError_t launch_rank_pivot_kernel_sm90(
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
    int pivot_numerator,
    int pivot_denominator) {
    const int device = (rank == 0) ? dev0 : dev1;

    const size_t total_bytes = count * static_cast<size_t>(ElemBytes);

    const int num_chunks =
        comm::utils::ceil_div_int64_to_int(
            total_bytes,
            TMA_TWO_GPU_PEER_CHUNK_BYTES);

    const int num_windows = comm::utils::window_num_chunks(num_chunks);

    const int owned_windows =
        (rank == 0) ? ((num_windows + 1) / 2) : (num_windows / 2);

    const bool use_cta_split =
        pivot_numerator > 0 &&
        pivot_denominator > 0 &&
        owned_windows > 0;

    const int blocks_per_window = use_cta_split ? 2 : 1;

    const bool needs_rendezvous =
        local_ready_signal != nullptr &&
        peer_ready_signal != nullptr &&
        collective_epoch > 0;

    int num_blocks = owned_windows * blocks_per_window;

    if (needs_rendezvous) {
        num_blocks = std::max(1, num_blocks);
    }

    if (num_blocks <= 0) {
        return cudaSuccess;
    }

    configure_pivot_kernel_once_for<ReduceApply, ElemBytes>(device);

    system::runtime::set_device(device);

    int* pivot_ready_counts = nullptr;

    if (use_cta_split) {
        pivot_ready_counts =
            ensure_pivot_signal_capacity(
                device,
                static_cast<size_t>(owned_windows));

        system::runtime::check_cuda(
            cudaMemsetAsync(
                pivot_ready_counts,
                0,
                static_cast<size_t>(owned_windows) * sizeof(int),
                stream),
            "cudaMemsetAsync(pivot ready counts)");
    }

    tma_two_gpu_allreduce_rank_pivot_kernel_sm90<
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
                pivot_numerator,
                pivot_denominator,
                pivot_ready_counts,
                owned_windows,
                blocks_per_window);

    return cudaGetLastError();
}

template <typename ReduceOp, int ElemBytes>
cudaError_t launch_reduce_op_pivot_sm90(
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
    int pivot_numerator,
    int pivot_denominator) {
    using ReduceApply = comm::PipelineTMAReduce<
        TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH,
        TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP,
        ReduceOp>;

    return launch_rank_pivot_kernel_sm90<ReduceApply, ElemBytes>(
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
        pivot_numerator,
        pivot_denominator);
}

cudaError_t dispatch_rank_pivot_kernel_sm90(
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
    int pivot_numerator,
    int pivot_denominator) {
    if (dtype == OO_DTYPE_FLOAT16) {
        if (op == OO_REDUCE_ADD) {
            return launch_reduce_op_pivot_sm90<
                comm::PipelineReduceAddNoFtzF16,
                static_cast<int>(sizeof(half))>(
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
                    pivot_numerator,
                    pivot_denominator);
        }

        if (op == OO_REDUCE_MIN) {
            return launch_reduce_op_pivot_sm90<
                comm::PipelineReduceMinF16,
                static_cast<int>(sizeof(half))>(
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
                    pivot_numerator,
                    pivot_denominator);
        }

        if (op == OO_REDUCE_MAX) {
            return launch_reduce_op_pivot_sm90<
                comm::PipelineReduceMaxF16,
                static_cast<int>(sizeof(half))>(
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
                    pivot_numerator,
                    pivot_denominator);
        }
    }

    if (dtype == OO_DTYPE_BFLOAT16) {
        if (op == OO_REDUCE_ADD) {
            return launch_reduce_op_pivot_sm90<
                comm::PipelineReduceAddBF16,
                static_cast<int>(sizeof(__nv_bfloat16))>(
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
                    pivot_numerator,
                    pivot_denominator);
        }

        if (op == OO_REDUCE_MIN) {
            return launch_reduce_op_pivot_sm90<
                comm::PipelineReduceMinBF16,
                static_cast<int>(sizeof(__nv_bfloat16))>(
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
                    pivot_numerator,
                    pivot_denominator);
        }

        if (op == OO_REDUCE_MAX) {
            return launch_reduce_op_pivot_sm90<
                comm::PipelineReduceMaxBF16,
                static_cast<int>(sizeof(__nv_bfloat16))>(
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
                    pivot_numerator,
                    pivot_denominator);
        }
    }

    if (dtype == OO_DTYPE_FLOAT32) {
        if (op == OO_REDUCE_ADD) {
            return launch_reduce_op_pivot_sm90<
                comm::PipelineReduceAddF32,
                static_cast<int>(sizeof(float))>(
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
                    pivot_numerator,
                    pivot_denominator);
        }
    }

    return cudaErrorInvalidValue;
}

} // namespace

cudaError_t enqueue_tma_two_gpu_peer_allreduce_rank_pivot_sm90(
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
    int pivot_numerator,
    int pivot_denominator) {
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

    return dispatch_rank_pivot_kernel_sm90(
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
        pivot_numerator,
        pivot_denominator);
}

} // namespace ooverlap
