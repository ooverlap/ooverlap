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

struct ChunkRange {
    int begin = 0;
    int end = 0;
};

__host__ __device__ __forceinline__ size_t min_size(
    size_t a,
    size_t b) {
    return (a < b) ? a : b;
}

__host__ __device__ __forceinline__ int min_int(
    int a,
    int b) {
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
__host__ __device__ __forceinline__ size_t chunk_size_bytes_abs(
    int chunk_idx,
    size_t total_bytes) {
    const size_t offset = chunk_offset_bytes<ChunkBytes>(chunk_idx);

    if (offset >= total_bytes) {
        return 0;
    }

    return min_size(
        static_cast<size_t>(ChunkBytes),
        total_bytes - offset);
}

__host__ __device__ __forceinline__ int window_begin_chunk(
    int window_idx,
    int window_chunks) {
    if (window_idx < 0 || window_chunks <= 0) {
        return 0;
    }

    return window_idx * window_chunks;
}

__host__ __device__ __forceinline__ int window_end_chunk_raw(
    int window_idx,
    int window_chunks) {
    if (window_idx < 0 || window_chunks <= 0) {
        return 0;
    }

    return (window_idx + 1) * window_chunks;
}

__host__ __device__ __forceinline__ int window_end_chunk_clamped(
    int window_idx,
    int total_chunks,
    int window_chunks) {
    return min_int(
        window_end_chunk_raw(window_idx, window_chunks),
        total_chunks);
}

__host__ __device__ __forceinline__ ChunkRange chunk_range_for_window_range(
    int begin_window,
    int end_window,
    int total_chunks,
    int window_chunks) {
    ChunkRange range{};

    if (begin_window >= end_window ||
        total_chunks <= 0 ||
        window_chunks <= 0) {
        return range;
    }

    range.begin = min_int(
        window_begin_chunk(begin_window, window_chunks),
        total_chunks);

    range.end = min_int(
        window_begin_chunk(end_window, window_chunks),
        total_chunks);

    if (range.begin > range.end) {
        range.begin = range.end;
    }

    return range;
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
__device__ __forceinline__ PipelineStage make_stage_for_abs_chunk(
    const unsigned char* src_base,
    unsigned char* dst_base,
    size_t total_bytes,
    int abs_chunk_idx,
    int slot,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    const size_t offset =
        chunk_offset_bytes<ChunkBytes>(abs_chunk_idx);

    const size_t bytes =
        chunk_size_bytes_abs<ChunkBytes>(
            abs_chunk_idx,
            total_bytes);

    return make_pipeline_stage(
        make_pipeline_chunk(
            src_base + offset,
            dst_base + offset,
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

__device__ __forceinline__ void publish_window_ready(
    int* window_ready) {
    if (window_ready == nullptr) {
        return;
    }

    __threadfence_system();
    atomicMax(window_ready, 1);
}

__device__ __forceinline__ void wait_window_ready(
    const int* window_ready) {
    if (window_ready == nullptr) {
        return;
    }

    const volatile int* ready =
        reinterpret_cast<const volatile int*>(window_ready);

    while (ready[0] < 1) {
#if defined(__CUDA_ARCH__)
        __nanosleep(64);
#endif
    }
}

struct WindowSignalCursor {
    int begin_window = 0;
    int end_window = 0;
    int next_window = 0;
    int next_window_end_chunk = 0;
    int total_chunks = 0;
    int window_chunks = 0;
    int ready_window_base = 0;
    int* window_ready_flags = nullptr;
};

__device__ __forceinline__ WindowSignalCursor make_window_signal_cursor(
    int begin_window,
    int end_window,
    int total_chunks,
    int window_chunks,
    int* window_ready_flags,
    int ready_window_base) {
    WindowSignalCursor cursor{};
    cursor.begin_window = begin_window;
    cursor.end_window = end_window;
    cursor.next_window = begin_window;
    cursor.total_chunks = total_chunks;
    cursor.window_chunks = window_chunks;
    cursor.ready_window_base = ready_window_base;
    cursor.window_ready_flags = window_ready_flags;

    cursor.next_window_end_chunk =
        window_end_chunk_clamped(
            begin_window,
            total_chunks,
            window_chunks);

    return cursor;
}

__device__ __forceinline__ void advance_window_signal_cursor(
    WindowSignalCursor* cursor,
    int safe_completed_chunk_exclusive) {
    if (threadIdx.x != 0 || cursor == nullptr) {
        return;
    }

    if (cursor->window_ready_flags == nullptr) {
        return;
    }

    if (cursor->next_window >= cursor->end_window) {
        return;
    }

    if (safe_completed_chunk_exclusive < cursor->next_window_end_chunk) {
        return;
    }

    const int flag_idx =
        cursor->next_window - cursor->ready_window_base;

    if (flag_idx >= 0) {
        publish_window_ready(cursor->window_ready_flags + flag_idx);
    }

    ++cursor->next_window;

    if (cursor->next_window < cursor->end_window) {
        cursor->next_window_end_chunk =
            window_end_chunk_clamped(
                cursor->next_window,
                cursor->total_chunks,
                cursor->window_chunks);
    }
}

__device__ __forceinline__ void publish_remaining_windows(
    WindowSignalCursor* cursor) {
    if (threadIdx.x != 0 || cursor == nullptr) {
        return;
    }

    if (cursor->window_ready_flags == nullptr) {
        return;
    }

    for (; cursor->next_window < cursor->end_window;
         ++cursor->next_window) {
        const int flag_idx =
            cursor->next_window - cursor->ready_window_base;

        if (flag_idx >= 0) {
            publish_window_ready(cursor->window_ready_flags + flag_idx);
        }
    }
}

/*
 * Streaming TMA-load + apply pipeline over an absolute chunk range.
 */
template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes,
    typename Apply>
__device__ void run_chunk_range(
    const void* src_base,
    void* dst_base,
    size_t total_bytes,
    int begin_chunk,
    int end_chunk,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    static_assert(StageDepth > 0, "StageDepth must be > 0");
    static_assert(FillDepth > 0, "FillDepth must be > 0");
    static_assert(FillDepth <= StageDepth, "FillDepth must be <= StageDepth");
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");

    if (begin_chunk >= end_chunk || total_bytes == 0) {
        return;
    }

    const unsigned char* src_bytes =
        reinterpret_cast<const unsigned char*>(src_base);

    unsigned char* dst_bytes =
        reinterpret_cast<unsigned char*>(dst_base);

    PipelineTMALoad load{};
    Apply apply{};

    const int total_range_chunks = end_chunk - begin_chunk;

    for (int warm = 0; warm < FillDepth; ++warm) {
        if (warm >= total_range_chunks) {
            break;
        }

        const int abs_chunk = begin_chunk + warm;
        const int slot = warm;

        PipelineStage stage = make_stage_for_abs_chunk<ChunkBytes>(
            src_bytes,
            dst_bytes,
            total_bytes,
            abs_chunk,
            slot,
            shared_raw,
            barriers);

        if (threadIdx.x == 0) {
            load.issue(&stage);
        }

        __syncthreads();
    }

    for (int iter = 0; iter < total_range_chunks; ++iter) {
        const int abs_chunk = begin_chunk + iter;
        const int cur_slot = iter % StageDepth;

        PipelineStage cur_stage = make_stage_for_abs_chunk<ChunkBytes>(
            src_bytes,
            dst_bytes,
            total_bytes,
            abs_chunk,
            cur_slot,
            shared_raw,
            barriers);

        if (threadIdx.x == 0) {
            load.wait_ready(&cur_stage);
        }

        __syncthreads();

        const int future_iter = iter + FillDepth;

        if (future_iter < total_range_chunks) {
            const int future_abs_chunk = begin_chunk + future_iter;
            const int future_slot = future_iter % StageDepth;

            PipelineStage future_stage = make_stage_for_abs_chunk<ChunkBytes>(
                src_bytes,
                dst_bytes,
                total_bytes,
                future_abs_chunk,
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
 * Streaming TMA-load + apply pipeline over a runtime window range.
 */
template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes,
    typename Apply>
__device__ void run_window_range(
    const void* src_base,
    void* dst_base,
    size_t total_bytes,
    int begin_window,
    int end_window,
    int window_chunks,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    const int total_chunks =
        chunk_count_for_bytes<ChunkBytes>(total_bytes);

    const ChunkRange chunks =
        chunk_range_for_window_range(
            begin_window,
            end_window,
            total_chunks,
            window_chunks);

    run_chunk_range<
        StageDepth,
        FillDepth,
        ChunkBytes,
        Apply>(
            src_base,
            dst_base,
            total_bytes,
            chunks.begin,
            chunks.end,
            shared_raw,
            barriers);
}

/*
 * Streaming reduce producer over a runtime window range.
 */
template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes,
    typename ReduceApply>
__device__ void run_window_range_signal(
    const void* src_base,
    void* dst_base,
    size_t total_bytes,
    int begin_window,
    int end_window,
    int window_chunks,
    int* window_ready_flags,
    int ready_window_base,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    static_assert(StageDepth > 0, "StageDepth must be > 0");
    static_assert(FillDepth > 0, "FillDepth must be > 0");
    static_assert(FillDepth <= StageDepth, "FillDepth must be <= StageDepth");
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");

    if (begin_window >= end_window ||
        window_chunks <= 0 ||
        total_bytes == 0) {
        return;
    }

    const int total_chunks =
        chunk_count_for_bytes<ChunkBytes>(total_bytes);

    const ChunkRange chunks =
        chunk_range_for_window_range(
            begin_window,
            end_window,
            total_chunks,
            window_chunks);

    if (chunks.begin >= chunks.end) {
        return;
    }

    const unsigned char* src_bytes =
        reinterpret_cast<const unsigned char*>(src_base);

    unsigned char* dst_bytes =
        reinterpret_cast<unsigned char*>(dst_base);

    PipelineTMALoad load{};
    ReduceApply apply{};

    const int total_range_chunks = chunks.end - chunks.begin;

    WindowSignalCursor signal_cursor =
        make_window_signal_cursor(
            begin_window,
            end_window,
            total_chunks,
            window_chunks,
            window_ready_flags,
            ready_window_base);

    for (int warm = 0; warm < FillDepth; ++warm) {
        if (warm >= total_range_chunks) {
            break;
        }

        const int abs_chunk = chunks.begin + warm;
        const int slot = warm;

        PipelineStage stage = make_stage_for_abs_chunk<ChunkBytes>(
            src_bytes,
            dst_bytes,
            total_bytes,
            abs_chunk,
            slot,
            shared_raw,
            barriers);

        if (threadIdx.x == 0) {
            load.issue(&stage);
        }

        __syncthreads();
    }

    for (int iter = 0; iter < total_range_chunks; ++iter) {
        const int abs_chunk = chunks.begin + iter;
        const int cur_slot = iter % StageDepth;

        PipelineStage cur_stage = make_stage_for_abs_chunk<ChunkBytes>(
            src_bytes,
            dst_bytes,
            total_bytes,
            abs_chunk,
            cur_slot,
            shared_raw,
            barriers);

        if (threadIdx.x == 0) {
            load.wait_ready(&cur_stage);
        }

        __syncthreads();

        const int future_iter = iter + FillDepth;

        if (future_iter < total_range_chunks) {
            const int future_abs_chunk = chunks.begin + future_iter;
            const int future_slot = future_iter % StageDepth;

            PipelineStage future_stage = make_stage_for_abs_chunk<ChunkBytes>(
                src_bytes,
                dst_bytes,
                total_bytes,
                future_abs_chunk,
                future_slot,
                shared_raw,
                barriers);

            if (threadIdx.x == 0) {
                if (iter >= FillDepth) {
                    apply.wait_before_stage_reuse();

                    const int safe_completed_chunk_exclusive =
                        chunks.begin + (iter - FillDepth + 1);

                    advance_window_signal_cursor(
                        &signal_cursor,
                        safe_completed_chunk_exclusive);
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

        publish_remaining_windows(&signal_cursor);

        __threadfence_system();
    }

    __syncthreads();
}

__device__ __forceinline__ void wait_window_signal_for_window(
    const int* window_ready_flags,
    int window_idx,
    int ready_window_base) {
    if (window_ready_flags == nullptr) {
        return;
    }

    const int flag_idx = window_idx - ready_window_base;

    const int* window_ready =
        (flag_idx >= 0) ? window_ready_flags + flag_idx : nullptr;

    if (threadIdx.x == 0) {
        wait_window_ready(window_ready);
    }

    __syncthreads();
}

/*
 * Streaming TMA-load + apply pipeline over a runtime window range, but each
 * window is allowed to enter the issue stream only after its signal is ready.
 *
 * This is the consumer side for:
 *
 *   CopyTMASignal -> ReduceTMAAfterSignal
 *
 * The pipeline is still one flowing pipeline across the whole window range.
 * We only wait when the first chunk of a new window is about to be issued.
 */
template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes,
    typename Apply>
__device__ void run_window_range_after_ready(
    const void* src_base,
    void* dst_base,
    size_t total_bytes,
    int begin_window,
    int end_window,
    int window_chunks,
    const int* window_ready_flags,
    int ready_window_base,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    static_assert(StageDepth > 0, "StageDepth must be > 0");
    static_assert(FillDepth > 0, "FillDepth must be > 0");
    static_assert(FillDepth <= StageDepth, "FillDepth must be <= StageDepth");
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");

    if (begin_window >= end_window ||
        window_chunks <= 0 ||
        total_bytes == 0) {
        return;
    }

    const int total_chunks =
        chunk_count_for_bytes<ChunkBytes>(total_bytes);

    const ChunkRange chunks =
        chunk_range_for_window_range(
            begin_window,
            end_window,
            total_chunks,
            window_chunks);

    if (chunks.begin >= chunks.end) {
        return;
    }

    const unsigned char* src_bytes =
        reinterpret_cast<const unsigned char*>(src_base);

    unsigned char* dst_bytes =
        reinterpret_cast<unsigned char*>(dst_base);

    PipelineTMALoad load{};
    Apply apply{};

    const int total_range_chunks = chunks.end - chunks.begin;

    int last_waited_window = -1;

    for (int warm = 0; warm < FillDepth; ++warm) {
        if (warm >= total_range_chunks) {
            break;
        }

        const int abs_chunk = chunks.begin + warm;
        const int window_idx = abs_chunk / window_chunks;

        if (window_idx != last_waited_window) {
            wait_window_signal_for_window(
                window_ready_flags,
                window_idx,
                ready_window_base);

            last_waited_window = window_idx;
        }

        const int slot = warm;

        PipelineStage stage = make_stage_for_abs_chunk<ChunkBytes>(
            src_bytes,
            dst_bytes,
            total_bytes,
            abs_chunk,
            slot,
            shared_raw,
            barriers);

        if (threadIdx.x == 0) {
            load.issue(&stage);
        }

        __syncthreads();
    }

    for (int iter = 0; iter < total_range_chunks; ++iter) {
        const int abs_chunk = chunks.begin + iter;
        const int cur_slot = iter % StageDepth;

        PipelineStage cur_stage = make_stage_for_abs_chunk<ChunkBytes>(
            src_bytes,
            dst_bytes,
            total_bytes,
            abs_chunk,
            cur_slot,
            shared_raw,
            barriers);

        if (threadIdx.x == 0) {
            load.wait_ready(&cur_stage);
        }

        __syncthreads();

        const int future_iter = iter + FillDepth;

        if (future_iter < total_range_chunks) {
            const int future_abs_chunk = chunks.begin + future_iter;
            const int future_window_idx = future_abs_chunk / window_chunks;

            if (future_window_idx != last_waited_window) {
                wait_window_signal_for_window(
                    window_ready_flags,
                    future_window_idx,
                    ready_window_base);

                last_waited_window = future_window_idx;
            }

            const int future_slot = future_iter % StageDepth;

            PipelineStage future_stage = make_stage_for_abs_chunk<ChunkBytes>(
                src_bytes,
                dst_bytes,
                total_bytes,
                future_abs_chunk,
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

template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes>
__device__ void copy_window_range_tma_signal(
    const void* src_base,
    void* dst_base,
    size_t total_bytes,
    int begin_window,
    int end_window,
    int window_chunks,
    int* window_ready_flags,
    int ready_window_base,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    using CopyApply = PipelineTMACopy<StageDepth, FillDepth>;

    run_window_range_signal<
        StageDepth,
        FillDepth,
        ChunkBytes,
        CopyApply>(
            src_base,
            dst_base,
            total_bytes,
            begin_window,
            end_window,
            window_chunks,
            window_ready_flags,
            ready_window_base,
            shared_raw,
            barriers);
}

template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes,
    typename ReduceApply>
__device__ void reduce_window_range_tma_after_ready(
    const void* src_base,
    void* dst_base,
    size_t total_bytes,
    int begin_window,
    int end_window,
    int window_chunks,
    const int* window_ready_flags,
    int ready_window_base,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    run_window_range_after_ready<
        StageDepth,
        FillDepth,
        ChunkBytes,
        ReduceApply>(
            src_base,
            dst_base,
            total_bytes,
            begin_window,
            end_window,
            window_chunks,
            window_ready_flags,
            ready_window_base,
            shared_raw,
            barriers);
}

template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes>
__device__ void copy_window_range_tma(
    const void* src_base,
    void* dst_base,
    size_t total_bytes,
    int begin_window,
    int end_window,
    int window_chunks,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    using CopyApply = PipelineTMACopy<StageDepth, FillDepth>;

    run_window_range<
        StageDepth,
        FillDepth,
        ChunkBytes,
        CopyApply>(
            src_base,
            dst_base,
            total_bytes,
            begin_window,
            end_window,
            window_chunks,
            shared_raw,
            barriers);
}

template <typename VecT, int Unroll>
__device__ __forceinline__ void copy_gmem_range_no_fence(
    const void* __restrict__ src_base,
    void* __restrict__ dst_base,
    size_t begin_byte,
    size_t byte_count) {
    static_assert(Unroll > 0, "Unroll must be > 0");

    if (byte_count == 0) {
        return;
    }

    comm::fast_copy::copy_byte_range<VecT, Unroll>(
        src_base,
        dst_base,
        begin_byte,
        byte_count,
        static_cast<size_t>(threadIdx.x),
        static_cast<size_t>(blockDim.x));
}

template <size_t ChunkBytes>
__host__ __device__ __forceinline__ size_t window_range_begin_byte(
    int begin_window,
    size_t total_bytes,
    int window_chunks) {
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");

    if (begin_window <= 0 || window_chunks <= 0) {
        return 0;
    }

    const size_t begin =
        static_cast<size_t>(begin_window) *
        static_cast<size_t>(window_chunks) *
        static_cast<size_t>(ChunkBytes);

    return min_size(begin, total_bytes);
}

template <size_t ChunkBytes>
__host__ __device__ __forceinline__ size_t window_range_end_byte(
    int end_window,
    size_t total_bytes,
    int window_chunks) {
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");

    if (end_window <= 0 || window_chunks <= 0) {
        return 0;
    }

    const size_t end =
        static_cast<size_t>(end_window) *
        static_cast<size_t>(window_chunks) *
        static_cast<size_t>(ChunkBytes);

    return min_size(end, total_bytes);
}

template <size_t ChunkBytes>
__host__ __device__ __forceinline__ size_t window_range_size_bytes(
    int begin_window,
    int end_window,
    size_t total_bytes,
    int window_chunks) {
    const size_t begin =
        window_range_begin_byte<ChunkBytes>(
            begin_window,
            total_bytes,
            window_chunks);

    const size_t end =
        window_range_end_byte<ChunkBytes>(
            end_window,
            total_bytes,
            window_chunks);

    return (begin < end) ? (end - begin) : 0;
}

template <
    typename VecT,
    int Unroll,
    size_t ChunkBytes>
__device__ void copy_window_range_gmem(
    const void* __restrict__ src_base,
    void* __restrict__ dst_base,
    size_t total_bytes,
    int begin_window,
    int end_window,
    int window_chunks) {
    const size_t begin =
        window_range_begin_byte<ChunkBytes>(
            begin_window,
            total_bytes,
            window_chunks);

    const size_t bytes =
        window_range_size_bytes<ChunkBytes>(
            begin_window,
            end_window,
            total_bytes,
            window_chunks);

    copy_gmem_range_no_fence<VecT, Unroll>(
        src_base,
        dst_base,
        begin,
        bytes);

    __syncthreads();

    if (threadIdx.x == 0) {
        __threadfence_system();
    }

    __syncthreads();
}

template <
    typename VecT,
    int Unroll,
    size_t ChunkBytes>
__device__ void copy_window_range_gmem_after_ready(
    const void* __restrict__ src_base,
    void* __restrict__ dst_base,
    size_t total_bytes,
    int begin_window,
    int end_window,
    int window_chunks,
    const int* window_ready_flags,
    int ready_window_base) {
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");

    if (begin_window >= end_window || window_chunks <= 0) {
        return;
    }

    for (int window_idx = begin_window;
         window_idx < end_window;
         ++window_idx) {
        const int flag_idx = window_idx - ready_window_base;

        const int* window_ready =
            (window_ready_flags != nullptr && flag_idx >= 0)
                ? window_ready_flags + flag_idx
                : nullptr;

        wait_window_ready(window_ready);

        const size_t begin =
            window_range_begin_byte<ChunkBytes>(
                window_idx,
                total_bytes,
                window_chunks);

        const size_t bytes =
            window_range_size_bytes<ChunkBytes>(
                window_idx,
                window_idx + 1,
                total_bytes,
                window_chunks);

        copy_gmem_range_no_fence<VecT, Unroll>(
            src_base,
            dst_base,
            begin,
            bytes);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        __threadfence_system();
    }

    __syncthreads();
}

} // namespace window_pipeline
} // namespace comm
} // namespace ooverlap
