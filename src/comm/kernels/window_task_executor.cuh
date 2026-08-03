#pragma once

#include "comm/params.h"
#include "comm/task/window_task.cuh"
#include "comm/pipeline/window_pipeline.cuh"
#include "comm/plan/transfer_plan.h"
#include "comm/plan/window_plan.cuh"
#include "comm/tma_variant_config.h"
#include "comm/kernels/multi_gpu_ready_signal.cuh"

#include <cuda/atomic>

namespace ooverlap {
namespace comm {
namespace kernels {

__device__ __forceinline__ void wait_until_cta_barrier_counter_at_least(
    unsigned int* counter,
    unsigned int target) {
    if (counter == nullptr || target == 0) {
        return;
    }

    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> state(*counter);

    while (state.load(cuda::memory_order_acquire) < target) {
        __nanosleep(16);
    }
}

__device__ __forceinline__ void advance_cta_barrier_counter(
    unsigned int* counter,
    unsigned int increment) {

    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> state(*counter);
    state.fetch_add(increment, cuda::memory_order_release);
}

__device__ __forceinline__ void arrive_and_wait_cta_barrier(
    unsigned int* counter,
    unsigned int target) {

    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> state(*counter);
        state.fetch_add(1u, cuda::memory_order_acq_rel);

        while (state.load(cuda::memory_order_acquire) < target) {
            __nanosleep(16);
        }
    }
}

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
    int MaxPeers,
    typename FastCopyVecT = uint4,
    int FastCopyUnroll = TMA_TWO_GPU_PEER_FAST_COPY_UNROLL,
    int LoadFillDepth = FillDepth,
    int SmallTaskBytes = 0>
__device__ __forceinline__ void execute_window_task(
    const task::WindowTask& task,
    unsigned char* shared_raw,
    sync::semaphore* barriers,
    int* local_ready_signal,
    MultiGpuReadySignalPlan<MaxPeers> ready_plan,
    int collective_epoch,
    unsigned int* cta_barrier_counter) {
    static_assert(StageDepth > 0, "StageDepth must be > 0");
    static_assert(FillDepth > 0, "FillDepth must be > 0");
    static_assert(LoadFillDepth > 0, "LoadFillDepth must be > 0");
    static_assert(LoadFillDepth + FillDepth <= StageDepth,
                  "LoadFillDepth + FillDepth must be <= StageDepth");
    static_assert(ChunkBytes > 0, "ChunkBytes must be > 0");
    static_assert(FastCopyUnroll > 0, "FastCopyUnroll must be > 0");
    static_assert(SmallTaskBytes >= 0, "SmallTaskBytes must be >= 0");

    switch (task.op) {
        case task::WindowTaskOp::ReduceTMA:
            pipeline::run_window_range<
                StageDepth,
                FillDepth,
                ChunkBytes,
                ReduceApply,
                LoadFillDepth,
                SmallTaskBytes>(
                    task.payload.window.src,
                    task.payload.window.dst,
                    task.payload.window.total_bytes,
                    task.payload.window.begin_window,
                    task.payload.window.end_window,
                    task.payload.window.window_chunks,
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
                    task.payload.window.src,
                    task.payload.window.dst,
                    task.payload.window.total_bytes,
                    task.payload.window.begin_window,
                    task.payload.window.end_window,
                    task.payload.window.window_chunks,
                    shared_raw,
                    barriers);
            return;

        case task::WindowTaskOp::CopyTMAFanout:
            pipeline::copy_window_range_tma_fanout<
                StageDepth,
                FillDepth,
                ChunkBytes,
                LoadFillDepth,
                SmallTaskBytes>(
                    task.payload.fanout.src,
                    task.payload.fanout.fanout_dsts,
                    static_cast<int>(
                        task.payload.fanout.fanout_dst_count),
                    task.payload.fanout.total_bytes,
                    task.payload.fanout.begin_window,
                    task.payload.fanout.end_window,
                    task.payload.fanout.window_chunks,
                    shared_raw,
                    barriers);
            return;

        case task::WindowTaskOp::ReduceTMAFanout:
            pipeline::reduce_window_range_tma_fanout<
                StageDepth,
                FillDepth,
                ChunkBytes,
                ReduceApply,
                LoadFillDepth,
                SmallTaskBytes>(
                    task.payload.fanout.src,
                    task.payload.fanout.fanout_dsts,
                    task.payload.fanout.fanout_reduce_scope,
                    static_cast<int>(
                        task.payload.fanout.fanout_dst_count),
                    task.payload.fanout.total_bytes,
                    task.payload.fanout.begin_window,
                    task.payload.fanout.end_window,
                    task.payload.fanout.window_chunks,
                    shared_raw,
                    barriers);
            return;

        case task::WindowTaskOp::ReadyPublish:
            if (threadIdx.x == 0) {
                publish_ready_signal(
                    task.payload.ready.ready_signal,
                    task.payload.ready.ready_epoch,
                    static_cast<MultiGpuReadySignalProtocol>(
                        task.payload.ready.ready_protocol));
            }
            return;

        case task::WindowTaskOp::ReadyWait:
            if (threadIdx.x == 0) {
                wait_until_ready_signal_at_least(
                    task.payload.ready.ready_signal,
                    task.payload.ready.ready_epoch,
                    64);
            }
            return;

        case task::WindowTaskOp::ReadyPublishWait:
            if (threadIdx.x == 0) {
                publish_then_wait_ready_signal_for_cta(
                    static_cast<int>(blockIdx.x),
                    task.payload.ready.ready_owner_cta,
                    task.payload.ready.ready_signal,
                    task.payload.ready.ready_epoch,
                    static_cast<MultiGpuReadySignalProtocol>(
                        task.payload.ready.ready_protocol),
                    task.payload.ready.ready_wait_signal,
                    task.payload.ready.ready_wait_epoch,
                    64);
            }
            return;

        case task::WindowTaskOp::FinalPeerRendezvous:
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                const int ready_value =
                    collective_epoch *
                        comm::plan::kReadySignalPhaseStride +
                    static_cast<int>(task.payload.ready_phase);

                publish_ready_signal(
                    local_ready_signal,
                    ready_value,
                    ready_plan.protocol);

                for (int peer_idx = 0;
                     peer_idx < ready_plan.peer_count;
                     ++peer_idx) {
                    wait_until_ready_signal_at_least(
                        ready_plan.peer_ready_signals[peer_idx],
                        ready_value,
                        ready_plan.poll_sleep_cycles);
                }
            }
            return;

        case task::WindowTaskOp::Barrier:
            arrive_and_wait_cta_barrier(
                cta_barrier_counter,
                task.payload.barrier_target);
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
 * Tasks in different CTA stripes are independent unless the lowered plan adds
 * explicit ready-signal or barrier tasks.
 *
 * This is deliberately not a work-stealing queue.
 */
template <
    int StageDepth,
    int FillDepth,
    size_t ChunkBytes,
    typename ReduceApply,
    int MaxPeers,
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
    sync::semaphore* barriers,
    int* local_ready_signal,
    MultiGpuReadySignalPlan<MaxPeers> ready_plan,
    int collective_epoch,
    unsigned int* cta_barrier_counter) {
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

        if (!task::window_task_runs_on_cta(task, cta_idx)) {
            continue;
        }

        execute_window_task<
            StageDepth,
            FillDepth,
            ChunkBytes,
            ReduceApply,
            MaxPeers,
            FastCopyVecT,
            FastCopyUnroll,
            LoadFillDepth,
            SmallTaskBytes>(
                task,
                shared_raw,
                barriers,
                local_ready_signal,
                ready_plan,
                collective_epoch,
                cta_barrier_counter);

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
