#pragma once

#include "comm/params.h"
#include "comm/task/window_task.cuh"
#include "comm/pipeline/window_pipeline.cuh"
#include "comm/plan/window_plan.cuh"
#include "comm/tma_variant_config.h"
#include "comm/kernels/multi_gpu_ready_signal.cuh"

namespace ooverlap {
namespace comm {
namespace kernels {

/*
 * Execute one task.
 *
 * The switch is intentionally outside the chunk pipeline hot loop. Each case
 * calls one existing window-level helper, so the actual TMA/copy pipeline stays
 * compile-time specialized.
 *
 * FillDepth is the store/reduce-side async depth.
 * LoadFillDepth is the TMA-load warm-ahead depth.
 */
template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes,
    typename ReduceApply,
    typename FastCopyVecT = uint4,
    int FastCopyUnroll = TMA_TWO_GPU_PEER_FAST_COPY_UNROLL,
    int LoadFillDepth = FillDepth,
    int SmallTaskBytes = 0>
__device__ __forceinline__ void execute_window_task(
    const task::WindowTask& task,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    static_assert(StageDepth > 0, "StageDepth must be > 0");
    static_assert(FillDepth > 0, "FillDepth must be > 0");
    static_assert(LoadFillDepth > 0, "LoadFillDepth must be > 0");
    static_assert(LoadFillDepth + FillDepth <= StageDepth,
                  "LoadFillDepth + FillDepth must be <= StageDepth");
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");
    static_assert(FastCopyUnroll > 0, "FastCopyUnroll must be > 0");
    static_assert(SmallTaskBytes >= 0, "SmallTaskBytes must be >= 0");

    if (!window_task_has_work(task)) {
        return;
    }

    switch (task.op) {
        case task::WindowTaskOp::ReduceTMA:
            pipeline::run_window_range<
                StageDepth,
                FillDepth,
                ChunkBytes,
                ReduceApply,
                LoadFillDepth,
                SmallTaskBytes>(
                    task.src,
                    task.dst,
                    task.total_bytes,
                    task.begin_window,
                    task.end_window,
                    task.window_chunks,
                    shared_raw,
                    barriers);
            return;

        case task::WindowTaskOp::ReduceTMASignal:
            pipeline::run_window_range_signal<
                StageDepth,
                FillDepth,
                ChunkBytes,
                ReduceApply,
                LoadFillDepth,
                SmallTaskBytes>(
                    task.src,
                    task.dst,
                    task.total_bytes,
                    task.begin_window,
                    task.end_window,
                    task.window_chunks,
                    task.signal_flags,
                    task.signal_base_window,
                    shared_raw,
                    barriers);
            return;

        case task::WindowTaskOp::CopyTMA:
            pipeline::copy_window_range_tma<
                StageDepth,
                FillDepth,
                ChunkBytes,
                LoadFillDepth,
                SmallTaskBytes>(
                    task.src,
                    task.dst,
                    task.total_bytes,
                    task.begin_window,
                    task.end_window,
                    task.window_chunks,
                    shared_raw,
                    barriers);
            return;

        case task::WindowTaskOp::CopyFast:
            pipeline::copy_window_range_gmem<
                FastCopyVecT,
                FastCopyUnroll,
                ChunkBytes>(
                    task.src,
                    task.dst,
                    task.total_bytes,
                    task.begin_window,
                    task.end_window,
                    task.window_chunks);
            return;

        case task::WindowTaskOp::CopyFastAfterSignal:
            pipeline::copy_window_range_gmem_after_ready<
                FastCopyVecT,
                FastCopyUnroll,
                ChunkBytes>(
                    task.src,
                    task.dst,
                    task.total_bytes,
                    task.begin_window,
                    task.end_window,
                    task.window_chunks,
                    task.signal_flags,
                    task.signal_base_window);
            return;

        case task::WindowTaskOp::CopyTMASignal:
            pipeline::copy_window_range_tma_signal<
                StageDepth,
                FillDepth,
                ChunkBytes,
                LoadFillDepth,
                SmallTaskBytes>(
                    task.src,
                    task.dst,
                    task.total_bytes,
                    task.begin_window,
                    task.end_window,
                    task.window_chunks,
                    task.signal_flags,
                    task.signal_base_window,
                    shared_raw,
                    barriers);
            return;

        case task::WindowTaskOp::ReduceTMAAfterSignal:
            pipeline::reduce_window_range_tma_after_ready<
                StageDepth,
                FillDepth,
                ChunkBytes,
                ReduceApply,
                LoadFillDepth,
                SmallTaskBytes>(
                    task.src,
                    task.dst,
                    task.total_bytes,
                    task.begin_window,
                    task.end_window,
                    task.window_chunks,
                    task.signal_flags,
                    task.signal_base_window,
                    shared_raw,
                    barriers);
            return;


        case task::WindowTaskOp::ReadyPublish:
            if (threadIdx.x == 0) {
                publish_ready_signal(
                    task.signal_flags,
                    task.ready_epoch,
                    static_cast<MultiGpuReadySignalProtocol>(
                        task.ready_protocol));
            }
            return;
        
        case task::WindowTaskOp::ReadyWait:
            if (threadIdx.x == 0) {
                wait_until_ready_signal_at_least(
                    task.signal_flags,
                    task.ready_epoch,
                    task.ready_poll_sleep_cycles);
            }
            return;

        case task::WindowTaskOp::None:
        default:
            return;
    }
}

/*
 * Execute one CTA's contiguous task stripe.
 *
 * Contract:
 *
 *   tasks[cta_idx * tasks_per_cta + 0]
 *   tasks[cta_idx * tasks_per_cta + 1]
 *   ...
 *
 * are ordered for that CTA.
 *
 * Tasks in different CTA stripes are assumed independent unless they use an
 * explicit signal pair:
 *
 *   ReduceTMASignal
 *   CopyFastAfterSignal
 *
 * This is deliberately not a work-stealing queue.
 */
template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes,
    typename ReduceApply,
    typename FastCopyVecT = uint4,
    int FastCopyUnroll = TMA_TWO_GPU_PEER_FAST_COPY_UNROLL,
    int LoadFillDepth = FillDepth,
    int SmallTaskBytes = 0>
__device__ __forceinline__ void execute_window_task_stripe(
    const task::WindowTask* tasks,
    int total_tasks,
    int tasks_per_cta,
    int cta_idx,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    static_assert(StageDepth > 0, "StageDepth must be > 0");
    static_assert(FillDepth > 0, "FillDepth must be > 0");
    static_assert(LoadFillDepth > 0, "LoadFillDepth must be > 0");
    static_assert(LoadFillDepth + FillDepth <= StageDepth,
                  "LoadFillDepth + FillDepth must be <= StageDepth");
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");
    static_assert(FastCopyUnroll > 0, "FastCopyUnroll must be > 0");
    static_assert(SmallTaskBytes >= 0, "SmallTaskBytes must be >= 0");

    if (tasks == nullptr ||
        total_tasks <= 0 ||
        tasks_per_cta <= 0 ||
        cta_idx < 0) {
        return;
    }

    const int base = cta_idx * tasks_per_cta;

    if (base >= total_tasks) {
        return;
    }

    for (int local_task = 0; local_task < tasks_per_cta; ++local_task) {
        const int task_idx = base + local_task;

        if (task_idx >= total_tasks) {
            return;
        }

        const task::WindowTask task = tasks[task_idx];

        execute_window_task<
            StageDepth,
            FillDepth,
            ChunkBytes,
            ReduceApply,
            FastCopyVecT,
            FastCopyUnroll,
            LoadFillDepth,
            SmallTaskBytes>(
                task,
                shared_raw,
                barriers);

        if (task.terminal) {
            return;
        }

        /*
         * The fast aligned TMA path in run_chunk_range is intentionally
         * thread0-only and has no internal __syncthreads().
         *
         * Keep one task-boundary barrier so nonzero threads do not start a
         * later full-CTA task, for example CopyFast, before thread 0 has
         * completed the previous TMA task.
         *
         * This replaces many per-chunk barriers with one barrier per task.
         */
        __syncthreads();
    }
}

} // namespace kernels
} // namespace comm
} // namespace ooverlap
