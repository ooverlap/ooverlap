#pragma once

#include "comm/fast_gmem_copy.cuh"
#include "comm/pipeline_stage.h"
#include "comm/pipeline_tma_copy.h"
#include "comm/pipeline_tma_load.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {
namespace window_pipeline {

__host__ __device__ __forceinline__ size_t min_size(
    size_t a,
    size_t b) {
    return (a < b) ? a : b;
}

template <size_t ChunkBytes>
__host__ __device__ __forceinline__ int chunk_count_for_bytes(
    size_t byte_count) {
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");

    if (byte_count == 0) {
        return 0;
    }

    return static_cast<int>(
        (byte_count + static_cast<size_t>(ChunkBytes) - 1) /
        static_cast<size_t>(ChunkBytes));
}

template <size_t ChunkBytes>
__host__ __device__ __forceinline__ size_t chunk_offset_bytes(
    int chunk_idx) {
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");

    return static_cast<size_t>(chunk_idx) *
           static_cast<size_t>(ChunkBytes);
}

template <size_t ChunkBytes>
__host__ __device__ __forceinline__ size_t chunk_size_bytes(
    int chunk_idx,
    size_t window_bytes) {
    const size_t offset = chunk_offset_bytes<ChunkBytes>(chunk_idx);

    if (offset >= window_bytes) {
        return 0;
    }

    return min_size(
        static_cast<size_t>(ChunkBytes),
        window_bytes - offset);
}

template <size_t ChunkBytes>
__device__ __forceinline__ unsigned char* stage_smem_ptr(
    unsigned char* shared_raw,
    int stage_idx) {
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");

    return shared_raw +
           static_cast<size_t>(stage_idx) *
               static_cast<size_t>(ChunkBytes);
}

template <size_t ChunkBytes>
__device__ __forceinline__ PipelineStage make_stage_for_chunk(
    const unsigned char* src_window,
    unsigned char* dst_window,
    size_t window_bytes,
    int chunk_idx,
    int slot,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    const size_t offset = chunk_offset_bytes<ChunkBytes>(chunk_idx);
    const size_t bytes = chunk_size_bytes<ChunkBytes>(
        chunk_idx,
        window_bytes);

    return make_pipeline_stage(
        make_pipeline_chunk(
            src_window + offset,
            dst_window + offset,
            bytes),
        stage_smem_ptr<ChunkBytes>(shared_raw, slot),
        &barriers[slot]);
}

__device__ __forceinline__ void wait_for_collective_ready(
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

__device__ __forceinline__ void publish_reduce_ready_count(
    int* reduce_ready_count,
    int ready_chunks) {
    if (reduce_ready_count == nullptr || ready_chunks <= 0) {
        return;
    }

    __threadfence_system();
    atomicMax(reduce_ready_count, ready_chunks);
}

__device__ __forceinline__ void wait_reduce_ready_count(
    const int* reduce_ready_count,
    int ready_chunks) {
    if (reduce_ready_count == nullptr || ready_chunks <= 0) {
        return;
    }

    const volatile int* ready =
        reinterpret_cast<const volatile int*>(reduce_ready_count);

    while (ready[0] < ready_chunks) {
#if defined(__CUDA_ARCH__)
        __nanosleep(64);
#endif
    }
}

template <int SignalBatchChunks>
__device__ __forceinline__ bool should_publish_reduce_ready_count(
    int completed_chunks,
    int total_chunks) {
    static_assert(
        SignalBatchChunks > 0,
        "SignalBatchChunks must be > 0");

    if (completed_chunks <= 0) {
        return false;
    }

    if (completed_chunks >= total_chunks) {
        return true;
    }

    return (completed_chunks % SignalBatchChunks) == 0;
}

/*
 * Generic TMA-load + apply pipeline over one byte window.
 *
 * The caller gives addresses already pointing at the beginning of the window.
 * window_bytes is the exact number of bytes in that window.
 *
 * Apply must provide:
 *
 *   void wait_before_stage_reuse() const;
 *   void issue_bulk(const PipelineStage*) const;
 *   void finish_tail(const PipelineStage*) const;
 *   void wait_complete() const;
 *
 * This matches PipelineTMAReduce and PipelineTMACopy.
 */
template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes,
    typename Apply>
__device__ void run_window(
    const void* src_window,
    void* dst_window,
    size_t window_bytes,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    static_assert(StageDepth > 0, "StageDepth must be > 0");
    static_assert(FillDepth > 0, "FillDepth must be > 0");
    static_assert(FillDepth <= StageDepth, "FillDepth must be <= StageDepth");
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");

    if (window_bytes == 0) {
        return;
    }

    const unsigned char* src_bytes =
        reinterpret_cast<const unsigned char*>(src_window);

    unsigned char* dst_bytes =
        reinterpret_cast<unsigned char*>(dst_window);

    const int chunk_count =
        chunk_count_for_bytes<ChunkBytes>(window_bytes);

    if (chunk_count <= 0) {
        return;
    }

    PipelineTMALoad load{};
    Apply apply{};

    for (int warm = 0; warm < FillDepth; ++warm) {
        if (warm >= chunk_count) {
            break;
        }

        PipelineStage stage = make_stage_for_chunk<ChunkBytes>(
            src_bytes,
            dst_bytes,
            window_bytes,
            warm,
            warm,
            shared_raw,
            barriers);

        if (threadIdx.x == 0) {
            load.issue(&stage);
        }

        __syncthreads();
    }

    for (int iter = 0; iter < chunk_count; ++iter) {
        const int cur_slot = iter % StageDepth;

        PipelineStage cur_stage = make_stage_for_chunk<ChunkBytes>(
            src_bytes,
            dst_bytes,
            window_bytes,
            iter,
            cur_slot,
            shared_raw,
            barriers);

        if (threadIdx.x == 0) {
            load.wait_ready(&cur_stage);
        }

        __syncthreads();

        const int future_iter = iter + FillDepth;

        if (future_iter < chunk_count) {
            const int future_slot = future_iter % StageDepth;

            PipelineStage future_stage = make_stage_for_chunk<ChunkBytes>(
                src_bytes,
                dst_bytes,
                window_bytes,
                future_iter,
                future_slot,
                shared_raw,
                barriers);

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
 * Reduce producer pipeline.
 *
 * This is the only helper here that publishes a per-window progress signal.
 * The signal means:
 *
 *   reduce_ready_count == N
 *
 * Chunks [0, N) in dst_window are globally visible to a consumer.
 *
 * This should be used for the TMA-reduce phase, not for copy.
 */
template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes,
    int SignalBatchChunks,
    typename ReduceApply>
__device__ void run_window_signal(
    const void* src_window,
    void* dst_window,
    size_t window_bytes,
    int* reduce_ready_count,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    static_assert(StageDepth > 0, "StageDepth must be > 0");
    static_assert(FillDepth > 0, "FillDepth must be > 0");
    static_assert(FillDepth <= StageDepth, "FillDepth must be <= StageDepth");
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");
    static_assert(
        SignalBatchChunks > 0,
        "SignalBatchChunks must be > 0");

    if (window_bytes == 0) {
        return;
    }

    const unsigned char* src_bytes =
        reinterpret_cast<const unsigned char*>(src_window);

    unsigned char* dst_bytes =
        reinterpret_cast<unsigned char*>(dst_window);

    const int chunk_count =
        chunk_count_for_bytes<ChunkBytes>(window_bytes);

    if (chunk_count <= 0) {
        return;
    }

    PipelineTMALoad load{};
    ReduceApply apply{};

    for (int warm = 0; warm < FillDepth; ++warm) {
        if (warm >= chunk_count) {
            break;
        }

        PipelineStage stage = make_stage_for_chunk<ChunkBytes>(
            src_bytes,
            dst_bytes,
            window_bytes,
            warm,
            warm,
            shared_raw,
            barriers);

        if (threadIdx.x == 0) {
            load.issue(&stage);
        }

        __syncthreads();
    }

    for (int iter = 0; iter < chunk_count; ++iter) {
        const int cur_slot = iter % StageDepth;

        PipelineStage cur_stage = make_stage_for_chunk<ChunkBytes>(
            src_bytes,
            dst_bytes,
            window_bytes,
            iter,
            cur_slot,
            shared_raw,
            barriers);

        if (threadIdx.x == 0) {
            load.wait_ready(&cur_stage);
        }

        __syncthreads();

        const int future_iter = iter + FillDepth;

        if (future_iter < chunk_count) {
            const int future_slot = future_iter % StageDepth;

            PipelineStage future_stage = make_stage_for_chunk<ChunkBytes>(
                src_bytes,
                dst_bytes,
                window_bytes,
                future_iter,
                future_slot,
                shared_raw,
                barriers);

            if (threadIdx.x == 0) {
                if (iter >= FillDepth) {
                    apply.wait_before_stage_reuse();

                    const int completed_chunks =
                        iter - FillDepth + 1;

                    if (should_publish_reduce_ready_count<
                            SignalBatchChunks>(
                            completed_chunks,
                            chunk_count)) {
                        publish_reduce_ready_count(
                            reduce_ready_count,
                            completed_chunks);
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

        publish_reduce_ready_count(
            reduce_ready_count,
            chunk_count);

        __threadfence_system();
    }

    __syncthreads();
}

/*
 * Explicit TMA copy helper.
 *
 * This is the TMA-copy phase from the non-fast path:
 *
 *   TMA load from src_window into shared memory
 *   TMA store from shared memory into dst_window
 *
 * No progress signal is involved.
 */
template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes>
__device__ void copy_window_tma(
    const void* src_window,
    void* dst_window,
    size_t window_bytes,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    using CopyApply = PipelineTMACopy<StageDepth, FillDepth>;

    run_window<
        StageDepth,
        FillDepth,
        ChunkBytes,
        CopyApply>(
            src_window,
            dst_window,
            window_bytes,
            shared_raw,
            barriers);
}

/*
 * Fast global-memory copy over a byte range inside one window.
 *
 * This intentionally does not fence or synchronize. It is useful when a caller
 * wants to copy several reduce-ready batches and fence once at the end.
 */
template <typename VecT, int Unroll>
__device__ __forceinline__ void copy_window_gmem_range_no_fence(
    const void* __restrict__ src_window,
    void* __restrict__ dst_window,
    size_t begin_byte,
    size_t byte_count) {
    static_assert(Unroll > 0, "Unroll must be > 0");

    if (byte_count == 0) {
        return;
    }

    comm::fast_copy::copy_byte_range<VecT, Unroll>(
        src_window,
        dst_window,
        begin_byte,
        byte_count,
        static_cast<size_t>(threadIdx.x),
        static_cast<size_t>(blockDim.x));
}

/*
 * Complete fast global-memory copy over one window.
 *
 * No progress signal is involved.
 */
template <typename VecT, int Unroll>
__device__ void copy_window_gmem(
    const void* __restrict__ src_window,
    void* __restrict__ dst_window,
    size_t window_bytes) {
    copy_window_gmem_range_no_fence<VecT, Unroll>(
        src_window,
        dst_window,
        0,
        window_bytes);

    __syncthreads();

    if (threadIdx.x == 0) {
        __threadfence_system();
    }

    __syncthreads();
}

/*
 * Finish a sequence of fast-copy ranges.
 *
 * This is separated from copy_window_gmem_range_no_fence so the overlap
 * consumer can wait on reduce progress, copy multiple ready ranges, and then
 * fence once.
 */
__device__ __forceinline__ void finish_window_gmem_copy() {
    __syncthreads();

    if (threadIdx.x == 0) {
        __threadfence_system();
    }

    __syncthreads();
}

template <size_t ChunkBytes>
__host__ __device__ __forceinline__ size_t chunk_range_begin_byte(
    int begin_chunk) {
    return static_cast<size_t>(begin_chunk) *
           static_cast<size_t>(ChunkBytes);
}

template <size_t ChunkBytes>
__host__ __device__ __forceinline__ size_t chunk_range_end_byte(
    int end_chunk,
    size_t window_bytes) {
    const size_t raw_end =
        static_cast<size_t>(end_chunk) *
        static_cast<size_t>(ChunkBytes);

    return min_size(raw_end, window_bytes);
}

template <size_t ChunkBytes>
__host__ __device__ __forceinline__ size_t chunk_range_size_bytes(
    int begin_chunk,
    int end_chunk,
    size_t window_bytes) {
    const size_t begin =
        chunk_range_begin_byte<ChunkBytes>(begin_chunk);

    const size_t end =
        chunk_range_end_byte<ChunkBytes>(end_chunk, window_bytes);

    if (begin >= end) {
        return 0;
    }

    return end - begin;
}

} // namespace window_pipeline
} // namespace comm
} // namespace ooverlap
