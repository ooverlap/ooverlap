#pragma once

#include "ooverlap/mem/fast_add.cuh"
#include "ooverlap/mem/fast_copy.cuh"
#include "comm/pipeline/pipeline_stage.h"
#include "comm/pipeline/pipeline_tma_copy.h"
#include "comm/pipeline/pipeline_tma_load.h"
#include "ooverlap/tma/tma.cuh"
#include "ooverlap/tma/tma_reduce.cuh"

#include "comm/utils/utils.h"
#include "comm/params.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {
namespace pipeline {

struct ChunkRange {
    int begin = 0;
    int end = 0;
};

struct ByteRange {
    size_t begin = 0;
    size_t bytes = 0;
};

template <size_t ChunkBytes>
__host__ __device__ __forceinline__ int chunk_count_for_bytes(
    size_t byte_count) {
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");

    return comm::utils::ceil_div_int64_to_int(
        byte_count,
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
    const size_t offset =
        chunk_offset_bytes<ChunkBytes>(chunk_idx);

    if (offset >= total_bytes) {
        return 0;
    }

    return comm::utils::min_sz(
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
    return comm::utils::min_int(
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

    range.begin =
        comm::utils::min_int(
            window_begin_chunk(begin_window, window_chunks),
            total_chunks);

    range.end =
        comm::utils::min_int(
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

__device__ __forceinline__ void publish_window_ready(
    int* window_ready) {
    if (window_ready == nullptr) {
        return;
    }

    __threadfence();
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

template <size_t ChunkBytes>
__host__ __device__ __forceinline__ bool chunk_range_is_16b_bulk_aligned(
    size_t total_bytes,
    int begin_chunk,
    int end_chunk) {
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");

    if ((ChunkBytes % 16) != 0) {
        return false;
    }

    if (begin_chunk >= end_chunk || total_bytes == 0) {
        return false;
    }

    const size_t begin_byte =
        chunk_offset_bytes<ChunkBytes>(begin_chunk);

    if (begin_byte >= total_bytes) {
        return false;
    }

    const size_t raw_end_byte =
        chunk_offset_bytes<ChunkBytes>(end_chunk);

    const size_t end_byte =
        comm::utils::min_sz(raw_end_byte, total_bytes);

    if (begin_byte >= end_byte) {
        return false;
    }

    return ((begin_byte & static_cast<size_t>(15)) == 0) &&
           ((end_byte & static_cast<size_t>(15)) == 0);
}

/*
 * FillDepth stays as the apply-side async depth for compatibility.
 * LoadFillDepth is the load-side warm-ahead depth.
 * Safe slot reuse requires LoadFillDepth + FillDepth <= StageDepth.
 */


template <size_t ChunkBytes>
__host__ __device__ __forceinline__ ByteRange byte_range_for_window_range(
    int begin_window,
    int end_window,
    size_t total_bytes,
    int window_chunks) {
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");

    ByteRange range{};

    if (begin_window >= end_window ||
        window_chunks <= 0 ||
        total_bytes == 0) {
        return range;
    }

    const size_t begin =
        static_cast<size_t>(begin_window) *
        static_cast<size_t>(window_chunks) *
        static_cast<size_t>(ChunkBytes);

    const size_t end =
        static_cast<size_t>(end_window) *
        static_cast<size_t>(window_chunks) *
        static_cast<size_t>(ChunkBytes);

    range.begin = comm::utils::min_sz(begin, total_bytes);

    const size_t clamped_end =
        comm::utils::min_sz(end, total_bytes);

    range.bytes =
        range.begin < clamped_end ? clamped_end - range.begin : 0;

    return range;
}

__device__ __forceinline__ PipelineStage make_stage_for_byte_range(
    const unsigned char* src_base,
    unsigned char* dst_base,
    size_t begin_byte,
    size_t byte_count,
    unsigned char* shared_raw,
    sync::semaphore* barrier) {
    return make_pipeline_stage(
        make_pipeline_chunk(
            src_base + begin_byte,
            dst_base + begin_byte,
            byte_count),
        shared_raw,
        barrier);
}


__host__ __device__ __forceinline__ bool byte_range_is_16b_bulk_aligned(
    size_t begin_byte,
    size_t byte_count) {
    if (byte_count == 0) {
        return false;
    }

    const size_t end_byte =
        begin_byte + byte_count;

    return ((begin_byte & static_cast<size_t>(15)) == 0) &&
           ((end_byte & static_cast<size_t>(15)) == 0);
}

template <typename Apply>
__device__ __forceinline__ void run_byte_range_single_tma_16b_aligned_thread0(
    const void* src_base,
    void* dst_base,
    size_t begin_byte,
    size_t byte_count,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {

    if (threadIdx.x != 0) {
        return;
    }

    if (byte_count == 0) {
        return;
    }

    const unsigned char* src_bytes =
        reinterpret_cast<const unsigned char*>(src_base);

    unsigned char* dst_bytes =
        reinterpret_cast<unsigned char*>(dst_base);

    PipelineTMALoad load{};
    Apply apply{};

    PipelineStage stage =
        make_stage_for_byte_range(
            src_bytes,
            dst_bytes,
            begin_byte,
            byte_count,
            shared_raw,
            &barriers[0]);

    load.issue(&stage);
    load.wait_ready(&stage);

    apply.issue_bulk(&stage);
    apply.wait_complete();
}

template <typename Apply>
__device__ void run_byte_range_single_tma(
    const void* src_base,
    void* dst_base,
    size_t begin_byte,
    size_t byte_count,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    if (byte_count == 0) {
        return;
    }

    if (byte_range_is_16b_bulk_aligned(begin_byte, byte_count)) {
        run_byte_range_single_tma_16b_aligned_thread0<Apply>(
            src_base,
            dst_base,
            begin_byte,
            byte_count,
            shared_raw,
            barriers);
        return;
    }

    const unsigned char* src_bytes =
        reinterpret_cast<const unsigned char*>(src_base);

    unsigned char* dst_bytes =
        reinterpret_cast<unsigned char*>(dst_base);

    PipelineTMALoad load{};
    Apply apply{};

    PipelineStage stage =
        make_stage_for_byte_range(
            src_bytes,
            dst_bytes,
            begin_byte,
            byte_count,
            shared_raw,
            &barriers[0]);

    if (threadIdx.x == 0) {
        load.issue(&stage);
        load.wait_ready(&stage);
        apply.issue_bulk(&stage);
    }

    apply.finish_tail(&stage);

    __syncthreads();

    if (threadIdx.x == 0) {
        apply.wait_complete();
    }

    __syncthreads();
}

__device__ __forceinline__ void publish_window_range_ready(
    int* window_ready_flags,
    int begin_window,
    int end_window,
    int ready_window_base) {
    if (threadIdx.x != 0 || window_ready_flags == nullptr) {
        return;
    }

    for (int window_idx = begin_window;
         window_idx < end_window;
         ++window_idx) {
        const int flag_idx =
            window_idx - ready_window_base;

        if (flag_idx >= 0) {
            publish_window_ready(window_ready_flags + flag_idx);
        }
    }
}

template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes,
    typename Apply,
    int LoadFillDepth = FillDepth>
__device__ __forceinline__ void run_chunk_range_16b_aligned_thread0(
    const void* src_base,
    void* dst_base,
    size_t total_bytes,
    int begin_chunk,
    int end_chunk,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    static_assert(StageDepth > 0, "StageDepth must be > 0");
    static_assert(FillDepth > 0, "FillDepth must be > 0");
    static_assert(LoadFillDepth > 0, "LoadFillDepth must be > 0");
    static_assert(LoadFillDepth + FillDepth <= StageDepth,
                  "LoadFillDepth + FillDepth must be <= StageDepth");
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");

    if (threadIdx.x != 0) {
        return;
    }

    if (begin_chunk >= end_chunk || total_bytes == 0) {
        return;
    }

    const unsigned char* src_bytes =
        reinterpret_cast<const unsigned char*>(src_base);

    unsigned char* dst_bytes =
        reinterpret_cast<unsigned char*>(dst_base);

    PipelineTMALoad load{};
    Apply apply{};

    const int total_range_chunks =
        end_chunk - begin_chunk;

    for (int warm = 0; warm < LoadFillDepth; ++warm) {
        if (warm >= total_range_chunks) {
            break;
        }

        const int abs_chunk =
            begin_chunk + warm;

        PipelineStage stage =
            make_stage_for_abs_chunk<ChunkBytes>(
                src_bytes,
                dst_bytes,
                total_bytes,
                abs_chunk,
                warm,
                shared_raw,
                barriers);

        load.issue(&stage);
    }

    for (int iter = 0; iter < total_range_chunks; ++iter) {
        const int abs_chunk =
            begin_chunk + iter;

        const int cur_slot =
            iter % StageDepth;

        PipelineStage cur_stage =
            make_stage_for_abs_chunk<ChunkBytes>(
                src_bytes,
                dst_bytes,
                total_bytes,
                abs_chunk,
                cur_slot,
                shared_raw,
                barriers);

        load.wait_ready(&cur_stage);

        const int future_iter =
            iter + LoadFillDepth;

        if (future_iter < total_range_chunks) {
            const int future_abs_chunk =
                begin_chunk + future_iter;

            const int future_slot =
                future_iter % StageDepth;

            PipelineStage future_stage =
                make_stage_for_abs_chunk<ChunkBytes>(
                    src_bytes,
                    dst_bytes,
                    total_bytes,
                    future_abs_chunk,
                    future_slot,
                    shared_raw,
                    barriers);

            if (future_iter >= StageDepth) {
                apply.wait_before_stage_reuse();
            }

            load.issue(&future_stage);
        }

        apply.issue_bulk(&cur_stage);
    }

    apply.wait_complete();
}

template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes,
    typename Apply,
    int LoadFillDepth = FillDepth>
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
    static_assert(LoadFillDepth > 0, "LoadFillDepth must be > 0");
    static_assert(LoadFillDepth + FillDepth <= StageDepth,
                  "LoadFillDepth + FillDepth must be <= StageDepth");
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");

    if (begin_chunk >= end_chunk || total_bytes == 0) {
        return;
    }

    if (chunk_range_is_16b_bulk_aligned<ChunkBytes>(
            total_bytes,
            begin_chunk,
            end_chunk)) {
        run_chunk_range_16b_aligned_thread0<
            StageDepth,
            FillDepth,
            ChunkBytes,
            Apply,
            LoadFillDepth>(
                src_base,
                dst_base,
                total_bytes,
                begin_chunk,
                end_chunk,
                shared_raw,
                barriers);
        return;
    }

    const unsigned char* src_bytes =
        reinterpret_cast<const unsigned char*>(src_base);

    unsigned char* dst_bytes =
        reinterpret_cast<unsigned char*>(dst_base);

    PipelineTMALoad load{};
    Apply apply{};

    const int total_range_chunks =
        end_chunk - begin_chunk;

    for (int warm = 0; warm < LoadFillDepth; ++warm) {
        if (warm >= total_range_chunks) {
            break;
        }

        const int abs_chunk =
            begin_chunk + warm;

        PipelineStage stage =
            make_stage_for_abs_chunk<ChunkBytes>(
                src_bytes,
                dst_bytes,
                total_bytes,
                abs_chunk,
                warm,
                shared_raw,
                barriers);

        if (threadIdx.x == 0) {
            load.issue(&stage);
        }

        __syncthreads();
    }

    for (int iter = 0; iter < total_range_chunks; ++iter) {
        const int abs_chunk =
            begin_chunk + iter;

        const int cur_slot =
            iter % StageDepth;

        PipelineStage cur_stage =
            make_stage_for_abs_chunk<ChunkBytes>(
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

        const int future_iter =
            iter + LoadFillDepth;

        if (future_iter < total_range_chunks) {
            const int future_abs_chunk =
                begin_chunk + future_iter;

            const int future_slot =
                future_iter % StageDepth;

            PipelineStage future_stage =
                make_stage_for_abs_chunk<ChunkBytes>(
                    src_bytes,
                    dst_bytes,
                    total_bytes,
                    future_abs_chunk,
                    future_slot,
                    shared_raw,
                    barriers);

            if (threadIdx.x == 0) {
                if (future_iter >= StageDepth) {
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
    }

    __syncthreads();
}


// -----------------------------------------------------------------------------
// TMA fanout window pipeline
// -----------------------------------------------------------------------------
//
// OOVERLAP_FANOUT_WINDOW_PIPELINE_PATCH:
//
// First implementation for WindowTaskOp::{CopyTMAFanout, ReduceTMAFanout}.
//
// Deliberately minimal:
//   - no signal variants
//   - no small-task special path
//   - only 16-byte aligned TMA bulk ranges
//   - no scalar tail path
//
// The source SMEM tile is produced by TMA load and immediately consumed by TMA
// store/reduce, so these helpers use *_op_nofence and one commit_group per
// chunk.  Do not use this path if threads modify the SMEM tile between load and
// outgoing fanout.
// -----------------------------------------------------------------------------

__device__ __forceinline__ tma::TmaReduceScope
fanout_reduce_scope_from_u8(uint8_t scope_value) {
    switch (static_cast<int>(scope_value)) {
        case 1:
            return tma::TmaReduceScope::Cta;
        case 2:
            return tma::TmaReduceScope::Cluster;
        case 3:
            return tma::TmaReduceScope::Gpu;
        case 4:
            return tma::TmaReduceScope::Sys;
        case 0:
        default:
            return tma::TmaReduceScope::Default;
    }
}

__device__ __forceinline__ void issue_reduce_add_noftz_f16_fanout_one_nofence(
    void* dst_gmem,
    void* src_smem,
    uint32_t size_bytes,
    tma::TmaReduceScope scope) {
    if (dst_gmem == nullptr || src_smem == nullptr || size_bytes == 0) {
        return;
    }

    switch (scope) {
        case tma::TmaReduceScope::Cta:
            if constexpr (OOVERLAP_TMA_REDUCE_HAS_PTX93_SCOPE) {
                tma::reduce_add_noftz_f16_async_op_nofence<
                    tma::TmaReduceScope::Cta>(
                        dst_gmem,
                        src_smem,
                        size_bytes);
            } else {
                tma::reduce_add_noftz_f16_async_op_nofence<
                    tma::TmaReduceScope::Default>(
                        dst_gmem,
                        src_smem,
                        size_bytes);
            }
            break;

        case tma::TmaReduceScope::Cluster:
            if constexpr (OOVERLAP_TMA_REDUCE_HAS_PTX93_SCOPE) {
                tma::reduce_add_noftz_f16_async_op_nofence<
                    tma::TmaReduceScope::Cluster>(
                        dst_gmem,
                        src_smem,
                        size_bytes);
            } else {
                tma::reduce_add_noftz_f16_async_op_nofence<
                    tma::TmaReduceScope::Default>(
                        dst_gmem,
                        src_smem,
                        size_bytes);
            }
            break;

        case tma::TmaReduceScope::Gpu:
            if constexpr (OOVERLAP_TMA_REDUCE_HAS_PTX93_SCOPE) {
                tma::reduce_add_noftz_f16_async_op_nofence<
                    tma::TmaReduceScope::Gpu>(
                        dst_gmem,
                        src_smem,
                        size_bytes);
            } else {
                tma::reduce_add_noftz_f16_async_op_nofence<
                    tma::TmaReduceScope::Default>(
                        dst_gmem,
                        src_smem,
                        size_bytes);
            }
            break;

        case tma::TmaReduceScope::Sys:
            if constexpr (OOVERLAP_TMA_REDUCE_HAS_PTX93_SCOPE) {
                tma::reduce_add_noftz_f16_async_op_nofence<
                    tma::TmaReduceScope::Sys>(
                        dst_gmem,
                        src_smem,
                        size_bytes);
            } else {
                tma::reduce_add_noftz_f16_async_op_nofence<
                    tma::TmaReduceScope::Default>(
                        dst_gmem,
                        src_smem,
                        size_bytes);
            }
            break;

        case tma::TmaReduceScope::Default:
        default:
            tma::reduce_add_noftz_f16_async_op_nofence<
                tma::TmaReduceScope::Default>(
                    dst_gmem,
                    src_smem,
                    size_bytes);
            break;
    }
}

__device__ __forceinline__ void issue_reduce_add_noftz_f16_fanout_nofence(
    void* src_smem,
    uint32_t size_bytes,
    void* const* dst_gmems,
    const uint8_t* reduce_scopes,
    int dst_count) {
    if (src_smem == nullptr ||
        size_bytes == 0 ||
        dst_gmems == nullptr ||
        dst_count <= 0) {
        return;
    }

    for (int i = 0; i < dst_count; ++i) {
        const tma::TmaReduceScope scope =
            reduce_scopes != nullptr
                ? fanout_reduce_scope_from_u8(reduce_scopes[i])
                : tma::TmaReduceScope::Default;

        issue_reduce_add_noftz_f16_fanout_one_nofence(
            dst_gmems[i],
            src_smem,
            size_bytes,
            scope);
    }
}

template <size_t ChunkBytes>
__device__ __forceinline__ PipelineStage make_fanout_stage_for_abs_chunk(
    const unsigned char* src_base,
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
            nullptr,
            bytes),
        stage_smem_ptr<ChunkBytes>(shared_raw, slot),
        &barriers[slot]);
}

template <int MaxFanoutDsts>
__device__ __forceinline__ int clamp_fanout_dst_count(int dst_count) {
    if (dst_count < 0) {
        return 0;
    }

    return dst_count > MaxFanoutDsts ? MaxFanoutDsts : dst_count;
}

template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes,
    int LoadFillDepth = FillDepth>
__device__ __forceinline__ void run_chunk_range_tma_copy_fanout_16b_aligned_thread0(
    const void* src_base,
    void* const* fanout_dsts,
    int fanout_dst_count,
    size_t total_bytes,
    int begin_chunk,
    int end_chunk,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    static_assert(StageDepth > 0, "StageDepth must be > 0");
    static_assert(FillDepth > 0, "FillDepth must be > 0");
    static_assert(LoadFillDepth > 0, "LoadFillDepth must be > 0");
    static_assert(LoadFillDepth + FillDepth <= StageDepth,
                  "LoadFillDepth + FillDepth must be <= StageDepth");
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");

    if (threadIdx.x != 0) {
        return;
    }

    const int dst_count =
        clamp_fanout_dst_count<TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS>(
            fanout_dst_count);

    if (begin_chunk >= end_chunk ||
        total_bytes == 0 ||
        src_base == nullptr ||
        fanout_dsts == nullptr ||
        dst_count <= 0) {
        return;
    }

    const unsigned char* src_bytes =
        reinterpret_cast<const unsigned char*>(src_base);

    PipelineTMALoad load{};

    const int total_range_chunks =
        end_chunk - begin_chunk;

    #pragma unroll 16
    for (int warm = 0; warm < LoadFillDepth; ++warm) {
        if (warm >= total_range_chunks) {
            break;
        }

        const int abs_chunk =
            begin_chunk + warm;

        PipelineStage stage =
            make_fanout_stage_for_abs_chunk<ChunkBytes>(
                src_bytes,
                total_bytes,
                abs_chunk,
                warm,
                shared_raw,
                barriers);

        load.issue(&stage);
    }

    #pragma unroll 8
    for (int iter = 0; iter < total_range_chunks; ++iter) {
        const int abs_chunk =
            begin_chunk + iter;

        const int cur_slot =
            iter % StageDepth;

        PipelineStage cur_stage =
            make_fanout_stage_for_abs_chunk<ChunkBytes>(
                src_bytes,
                total_bytes,
                abs_chunk,
                cur_slot,
                shared_raw,
                barriers);

        load.wait_ready(&cur_stage);

        const int future_iter =
            iter + LoadFillDepth;

        if (future_iter < total_range_chunks) {
            const int future_abs_chunk =
                begin_chunk + future_iter;

            const int future_slot =
                future_iter % StageDepth;

            PipelineStage future_stage =
                make_fanout_stage_for_abs_chunk<ChunkBytes>(
                    src_bytes,
                    total_bytes,
                    future_abs_chunk,
                    future_slot,
                    shared_raw,
                    barriers);

            if (future_iter >= StageDepth) {
                tma::store_async_read_wait<FillDepth - 1>();
            }

            load.issue(&future_stage);
        }

        const size_t bulk_bytes =
            pipeline_stage_bulk_bytes(&cur_stage);

        if (bulk_bytes != 0) {
            void* chunk_dsts[TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS] = {};
            const size_t offset =
                chunk_offset_bytes<ChunkBytes>(abs_chunk);


            #pragma unroll 4
            for (int dst_idx = 0; dst_idx < dst_count; ++dst_idx) {
                if (fanout_dsts[dst_idx] == nullptr) {
                    chunk_dsts[dst_idx] = nullptr;
                } else {
                    chunk_dsts[dst_idx] =
                        reinterpret_cast<unsigned char*>(
                            fanout_dsts[dst_idx]) + offset;
                }
            }

            tma::store_async_fanout_array_op_nofence(
                cur_stage.smem,
                static_cast<uint32_t>(bulk_bytes),
                chunk_dsts,
                dst_count);

            tma::store_commit_group();
        }
    }

    tma::store_async_wait<0>();
}

template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes,
    int LoadFillDepth = FillDepth>
__device__ __forceinline__ void run_chunk_range_tma_reduce_fanout_16b_aligned_thread0(
    const void* src_base,
    void* const* fanout_dsts,
    const uint8_t* fanout_reduce_scope,
    int fanout_dst_count,
    size_t total_bytes,
    int begin_chunk,
    int end_chunk,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    static_assert(StageDepth > 0, "StageDepth must be > 0");
    static_assert(FillDepth > 0, "FillDepth must be > 0");
    static_assert(LoadFillDepth > 0, "LoadFillDepth must be > 0");
    static_assert(LoadFillDepth + FillDepth <= StageDepth,
                  "LoadFillDepth + FillDepth must be <= StageDepth");
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");

    if (threadIdx.x != 0) {
        return;
    }

    const int dst_count =
        clamp_fanout_dst_count<TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS>(
            fanout_dst_count);

    if (begin_chunk >= end_chunk ||
        total_bytes == 0 ||
        src_base == nullptr ||
        fanout_dsts == nullptr ||
        dst_count <= 0) {
        return;
    }

    const unsigned char* src_bytes =
        reinterpret_cast<const unsigned char*>(src_base);

    PipelineTMALoad load{};

    const int total_range_chunks =
        end_chunk - begin_chunk;

    #pragma unroll 16
    for (int warm = 0; warm < LoadFillDepth; ++warm) {
        if (warm >= total_range_chunks) {
            break;
        }

        const int abs_chunk =
            begin_chunk + warm;

        PipelineStage stage =
            make_fanout_stage_for_abs_chunk<ChunkBytes>(
                src_bytes,
                total_bytes,
                abs_chunk,
                warm,
                shared_raw,
                barriers);

        load.issue(&stage);
    }

    #pragma unroll 8
    for (int iter = 0; iter < total_range_chunks; ++iter) {
        const int abs_chunk =
            begin_chunk + iter;

        const int cur_slot =
            iter % StageDepth;

        PipelineStage cur_stage =
            make_fanout_stage_for_abs_chunk<ChunkBytes>(
                src_bytes,
                total_bytes,
                abs_chunk,
                cur_slot,
                shared_raw,
                barriers);

        load.wait_ready(&cur_stage);

        const int future_iter =
            iter + LoadFillDepth;

        if (future_iter < total_range_chunks) {
            const int future_abs_chunk =
                begin_chunk + future_iter;

            const int future_slot =
                future_iter % StageDepth;

            PipelineStage future_stage =
                make_fanout_stage_for_abs_chunk<ChunkBytes>(
                    src_bytes,
                    total_bytes,
                    future_abs_chunk,
                    future_slot,
                    shared_raw,
                    barriers);

            if (future_iter >= StageDepth) {
                tma::reduce_async_read_wait<FillDepth - 1>();
            }

            load.issue(&future_stage);
        }

        const size_t bulk_bytes =
            pipeline_stage_bulk_bytes(&cur_stage);

        if (bulk_bytes != 0) {
            void* chunk_dsts[TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS] = {};
            const size_t offset =
                chunk_offset_bytes<ChunkBytes>(abs_chunk);

            #pragma unroll 4
            for (int dst_idx = 0; dst_idx < dst_count; ++dst_idx) {
                if (fanout_dsts[dst_idx] == nullptr) {
                    chunk_dsts[dst_idx] = nullptr;
                } else {
                    chunk_dsts[dst_idx] =
                        reinterpret_cast<unsigned char*>(
                            fanout_dsts[dst_idx]) + offset;
                }
            }

            issue_reduce_add_noftz_f16_fanout_nofence(
                cur_stage.smem,
                static_cast<uint32_t>(bulk_bytes),
                chunk_dsts,
                fanout_reduce_scope,
                dst_count);

            tma::reduce_commit_group();
        }
    }

    tma::reduce_async_wait<0>();
}

template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes,
    int LoadFillDepth = FillDepth>
__device__ void copy_window_range_tma_fanout(
    const void* src_base,
    void* const* fanout_dsts,
    int fanout_dst_count,
    size_t total_bytes,
    int begin_window,
    int end_window,
    int window_chunks,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    static_assert(StageDepth > 0, "StageDepth must be > 0");
    static_assert(FillDepth > 0, "FillDepth must be > 0");
    static_assert(LoadFillDepth > 0, "LoadFillDepth must be > 0");
    static_assert(LoadFillDepth + FillDepth <= StageDepth,
                  "LoadFillDepth + FillDepth must be <= StageDepth");
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");

    if (begin_window >= end_window ||
        window_chunks <= 0 ||
        total_bytes == 0 ||
        fanout_dst_count <= 0) {
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

    /*
     * First pass: only support pure bulk-aligned fanout.  Tail handling for
     * fanout copy/reduce can be added later if needed.
     */
    if (!chunk_range_is_16b_bulk_aligned<ChunkBytes>(
            total_bytes,
            chunks.begin,
            chunks.end)) {
        return;
    }

    run_chunk_range_tma_copy_fanout_16b_aligned_thread0<
        StageDepth,
        FillDepth,
        ChunkBytes,
        LoadFillDepth>(
            src_base,
            fanout_dsts,
            fanout_dst_count,
            total_bytes,
            chunks.begin,
            chunks.end,
            shared_raw,
            barriers);
}

template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes,
    int LoadFillDepth = FillDepth>
__device__ void reduce_window_range_tma_fanout(
    const void* src_base,
    void* const* fanout_dsts,
    const uint8_t* fanout_reduce_scope,
    int fanout_dst_count,
    size_t total_bytes,
    int begin_window,
    int end_window,
    int window_chunks,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    static_assert(StageDepth > 0, "StageDepth must be > 0");
    static_assert(FillDepth > 0, "FillDepth must be > 0");
    static_assert(LoadFillDepth > 0, "LoadFillDepth must be > 0");
    static_assert(LoadFillDepth + FillDepth <= StageDepth,
                  "LoadFillDepth + FillDepth must be <= StageDepth");
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");

    if (begin_window >= end_window ||
        window_chunks <= 0 ||
        total_bytes == 0 ||
        fanout_dst_count <= 0) {
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

    /*
     * First pass: only support pure bulk-aligned fanout.  Tail handling for
     * reduce fanout is more delicate because scalar tails are not TMA atomics.
     */
    if (!chunk_range_is_16b_bulk_aligned<ChunkBytes>(
            total_bytes,
            chunks.begin,
            chunks.end)) {
        return;
    }

    run_chunk_range_tma_reduce_fanout_16b_aligned_thread0<
        StageDepth,
        FillDepth,
        ChunkBytes,
        LoadFillDepth>(
            src_base,
            fanout_dsts,
            fanout_reduce_scope,
            fanout_dst_count,
            total_bytes,
            chunks.begin,
            chunks.end,
            shared_raw,
            barriers);
}


template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes,
    typename Apply,
    int LoadFillDepth = FillDepth,
    int SmallTaskBytes = 0>
__device__ void run_window_range(
    const void* src_base,
    void* dst_base,
    size_t total_bytes,
    int begin_window,
    int end_window,
    int window_chunks,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    static_assert(SmallTaskBytes >= 0, "SmallTaskBytes must be >= 0");

    if (SmallTaskBytes > 0) {
        const ByteRange byte_range =
            byte_range_for_window_range<ChunkBytes>(
                begin_window,
                end_window,
                total_bytes,
                window_chunks);

        if (byte_range.bytes > 0 &&
            byte_range.bytes <= static_cast<size_t>(SmallTaskBytes)) {
            run_byte_range_single_tma<Apply>(
                src_base,
                dst_base,
                byte_range.begin,
                byte_range.bytes,
                shared_raw,
                barriers);
            return;
        }
    }

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
        Apply,
        LoadFillDepth>(
            src_base,
            dst_base,
            total_bytes,
            chunks.begin,
            chunks.end,
            shared_raw,
            barriers);
}

template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes,
    typename ReduceApply,
    int LoadFillDepth = FillDepth,
    int SmallTaskBytes = 0>
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
    static_assert(LoadFillDepth > 0, "LoadFillDepth must be > 0");
    static_assert(LoadFillDepth + FillDepth <= StageDepth,
                  "LoadFillDepth + FillDepth must be <= StageDepth");
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");

    if (begin_window >= end_window ||
        window_chunks <= 0 ||
        total_bytes == 0) {
        return;
    }

    static_assert(SmallTaskBytes >= 0, "SmallTaskBytes must be >= 0");

    if (SmallTaskBytes > 0) {
        const ByteRange byte_range =
            byte_range_for_window_range<ChunkBytes>(
                begin_window,
                end_window,
                total_bytes,
                window_chunks);

        if (byte_range.bytes > 0 &&
            byte_range.bytes <= static_cast<size_t>(SmallTaskBytes)) {
            run_byte_range_single_tma<ReduceApply>(
                src_base,
                dst_base,
                byte_range.begin,
                byte_range.bytes,
                shared_raw,
                barriers);

            publish_window_range_ready(
                window_ready_flags,
                begin_window,
                end_window,
                ready_window_base);

            __syncthreads();
            return;
        }
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

    const int total_range_chunks =
        chunks.end - chunks.begin;

    WindowSignalCursor signal_cursor =
        make_window_signal_cursor(
            begin_window,
            end_window,
            total_chunks,
            window_chunks,
            window_ready_flags,
            ready_window_base);

    for (int warm = 0; warm < LoadFillDepth; ++warm) {
        if (warm >= total_range_chunks) {
            break;
        }

        const int abs_chunk =
            chunks.begin + warm;

        PipelineStage stage =
            make_stage_for_abs_chunk<ChunkBytes>(
                src_bytes,
                dst_bytes,
                total_bytes,
                abs_chunk,
                warm,
                shared_raw,
                barriers);

        if (threadIdx.x == 0) {
            load.issue(&stage);
        }

        __syncthreads();
    }

    for (int iter = 0; iter < total_range_chunks; ++iter) {
        const int abs_chunk =
            chunks.begin + iter;

        const int cur_slot =
            iter % StageDepth;

        PipelineStage cur_stage =
            make_stage_for_abs_chunk<ChunkBytes>(
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

        const int future_iter =
            iter + LoadFillDepth;

        if (future_iter < total_range_chunks) {
            const int future_abs_chunk =
                chunks.begin + future_iter;

            const int future_slot =
                future_iter % StageDepth;

            PipelineStage future_stage =
                make_stage_for_abs_chunk<ChunkBytes>(
                    src_bytes,
                    dst_bytes,
                    total_bytes,
                    future_abs_chunk,
                    future_slot,
                    shared_raw,
                    barriers);

            if (threadIdx.x == 0) {
                if (future_iter >= StageDepth) {
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

    const int flag_idx =
        window_idx - ready_window_base;

    const int* window_ready =
        flag_idx >= 0 ? window_ready_flags + flag_idx : nullptr;

    if (threadIdx.x == 0) {
        wait_window_ready(window_ready);
    }

    __syncthreads();
}

template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes,
    typename Apply,
    int LoadFillDepth = FillDepth,
    int SmallTaskBytes = 0>
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
    static_assert(LoadFillDepth > 0, "LoadFillDepth must be > 0");
    static_assert(LoadFillDepth + FillDepth <= StageDepth,
                  "LoadFillDepth + FillDepth must be <= StageDepth");
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");

    if (begin_window >= end_window ||
        window_chunks <= 0 ||
        total_bytes == 0) {
        return;
    }

    static_assert(SmallTaskBytes >= 0, "SmallTaskBytes must be >= 0");

    if (SmallTaskBytes > 0) {
        const ByteRange byte_range =
            byte_range_for_window_range<ChunkBytes>(
                begin_window,
                end_window,
                total_bytes,
                window_chunks);

        if (byte_range.bytes > 0 &&
            byte_range.bytes <= static_cast<size_t>(SmallTaskBytes)) {
            for (int window_idx = begin_window;
                 window_idx < end_window;
                 ++window_idx) {
                wait_window_signal_for_window(
                    window_ready_flags,
                    window_idx,
                    ready_window_base);
            }

            run_byte_range_single_tma<Apply>(
                src_base,
                dst_base,
                byte_range.begin,
                byte_range.bytes,
                shared_raw,
                barriers);
            return;
        }
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

    const int total_range_chunks =
        chunks.end - chunks.begin;

    int last_waited_window = -1;

    for (int warm = 0; warm < LoadFillDepth; ++warm) {
        if (warm >= total_range_chunks) {
            break;
        }

        const int abs_chunk =
            chunks.begin + warm;

        const int window_idx =
            abs_chunk / window_chunks;

        if (window_idx != last_waited_window) {
            wait_window_signal_for_window(
                window_ready_flags,
                window_idx,
                ready_window_base);

            last_waited_window = window_idx;
        }

        PipelineStage stage =
            make_stage_for_abs_chunk<ChunkBytes>(
                src_bytes,
                dst_bytes,
                total_bytes,
                abs_chunk,
                warm,
                shared_raw,
                barriers);

        if (threadIdx.x == 0) {
            load.issue(&stage);
        }

        __syncthreads();
    }

    for (int iter = 0; iter < total_range_chunks; ++iter) {
        const int abs_chunk =
            chunks.begin + iter;

        const int cur_slot =
            iter % StageDepth;

        PipelineStage cur_stage =
            make_stage_for_abs_chunk<ChunkBytes>(
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

        const int future_iter =
            iter + LoadFillDepth;

        if (future_iter < total_range_chunks) {
            const int future_abs_chunk =
                chunks.begin + future_iter;

            const int future_window_idx =
                future_abs_chunk / window_chunks;

            if (future_window_idx != last_waited_window) {
                wait_window_signal_for_window(
                    window_ready_flags,
                    future_window_idx,
                    ready_window_base);

                last_waited_window = future_window_idx;
            }

            const int future_slot =
                future_iter % StageDepth;

            PipelineStage future_stage =
                make_stage_for_abs_chunk<ChunkBytes>(
                    src_bytes,
                    dst_bytes,
                    total_bytes,
                    future_abs_chunk,
                    future_slot,
                    shared_raw,
                    barriers);

            if (threadIdx.x == 0) {
                if (future_iter >= StageDepth) {
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
    }

    __syncthreads();
}

template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes,
    int LoadFillDepth = FillDepth,
    int SmallTaskBytes = 0>
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
    using CopyApply =
        PipelineTMACopy<StageDepth, FillDepth>;

    run_window_range_signal<
        StageDepth,
        FillDepth,
        ChunkBytes,
        CopyApply,
        LoadFillDepth,
        SmallTaskBytes>(
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
    typename ReduceApply,
    int LoadFillDepth = FillDepth,
    int SmallTaskBytes = 0>
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
        ReduceApply,
        LoadFillDepth,
        SmallTaskBytes>(
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
    int LoadFillDepth = FillDepth,
    int SmallTaskBytes = 0>
__device__ void copy_window_range_tma(
    const void* src_base,
    void* dst_base,
    size_t total_bytes,
    int begin_window,
    int end_window,
    int window_chunks,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    using CopyApply =
        PipelineTMACopy<StageDepth, FillDepth>;

    run_window_range<
        StageDepth,
        FillDepth,
        ChunkBytes,
        CopyApply,
        LoadFillDepth,
        SmallTaskBytes>(
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

    comm::kernels::fast_copy::copy_byte_range<VecT, Unroll>(
        src_base,
        dst_base,
        begin_byte,
        byte_count,
        static_cast<size_t>(threadIdx.x),
        static_cast<size_t>(blockDim.x));
}

template <int Unroll>
__device__ __forceinline__ void add_gmem_range_no_fence(
    const void* __restrict__ src_base,
    void* __restrict__ dst_base,
    size_t begin_byte,
    size_t byte_count) {
    static_assert(Unroll > 0, "Unroll must be > 0");

    if (byte_count == 0) {
        return;
    }

    comm::kernels::fast_add::add_f16_u128_byte_range<Unroll>(
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

    return comm::utils::min_sz(begin, total_bytes);
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

    return comm::utils::min_sz(end, total_bytes);
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

    return begin < end ? end - begin : 0;
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
}

template <
    int Unroll,
    size_t ChunkBytes>
__device__ void add_window_range_gmem(
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

    add_gmem_range_no_fence<Unroll>(
        src_base,
        dst_base,
        begin,
        bytes);
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
        const int flag_idx =
            window_idx - ready_window_base;

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
}

template <
    int Unroll,
    size_t ChunkBytes>
__device__ void add_window_range_gmem_after_ready(
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
        const int flag_idx =
            window_idx - ready_window_base;

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

        add_gmem_range_no_fence<Unroll>(
            src_base,
            dst_base,
            begin,
            bytes);
    }
}

} // namespace pipeline
} // namespace comm
} // namespace ooverlap
