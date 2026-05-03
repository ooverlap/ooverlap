#pragma once

#include "comm/launch_config.h"
#include "comm/params.h"
#include "comm/plan/window_plan.cuh"
#include "comm/task/window_task.cuh"
#include "comm/utils/utils.h"

#include <cstddef>

namespace ooverlap {
namespace comm {
namespace plan {

constexpr int kTmaMultiGpuReduceScatterMaxPeers = 15;
constexpr int kTmaMultiGpuReduceScatterMaxWindowTasks = 256;

/*
 * Naive multi-GPU reduce-scatter plan.
 *
 * This plan only handles the local rank's partition.
 *
 * The caller already slices local_in/local_buf/peer_bufs to the local rank's
 * partition. Therefore, all tasks operate over [0, slice_bytes) for this rank.
 *
 * In-place path:
 *
 *   local_buf already contains this rank's contribution.
 *
 *   reduce peer0 -> local_buf
 *   reduce peer1 -> local_buf
 *   ...
 *
 * Out-of-place path:
 *
 *   local_in is separate from local_buf, so local_buf must first be initialized.
 *
 *   copy   local_in -> local_buf
 *   reduce peer0    -> local_buf
 *   reduce peer1    -> local_buf
 *   ...
 *
 * Unlike allreduce, there is no final copy/broadcast phase.
 */

__host__ __device__ __forceinline__ int naive_multi_gpu_reduce_scatter_tasks_per_cta(
    int peer_count,
    bool out_of_place) {
    if (peer_count < 0) {
        return 0;
    }

    /*
     * out_of_place adds one local_in -> local_buf initialization task.
     * Every peer adds one reduce task.
     */
    return (out_of_place ? 1 : 0) + peer_count;
}

__host__ __device__ __forceinline__ bool multi_gpu_reduce_scatter_use_fast_copy(
    comm::AllreducePlanKind plan_kind) {
    return plan_kind == comm::AllreducePlanKind::SeqFastGmem ||
           plan_kind == comm::AllreducePlanKind::OverlapFastGmem;
}

__host__ __device__ __forceinline__ comm::task::WindowTask make_reduce_scatter_copy_task(
    comm::AllreducePlanKind plan_kind,
    const void* src,
    void* dst,
    size_t total_bytes,
    int begin_window,
    int end_window,
    int window_chunks,
    bool terminal = false) {
    if (multi_gpu_reduce_scatter_use_fast_copy(plan_kind)) {
        return comm::task::make_copy_fast_task(
            src,
            dst,
            total_bytes,
            begin_window,
            end_window,
            window_chunks,
            terminal);
    }

    return comm::task::make_copy_tma_task(
        src,
        dst,
        total_bytes,
        begin_window,
        end_window,
        window_chunks,
        terminal);
}

template <int MaxTasks>
bool build_tma_multi_gpu_reduce_scatter_naive_plan(
    WindowTaskExecutorPlan<MaxTasks>* plan,
    int* out_num_blocks,
    const void* local_in,
    void* local_buf,
    void* const* peer_bufs,
    int peer_count,
    size_t slice_bytes,
    int num_windows,
    const comm::LaunchConfig& launch_config,
    bool needs_rendezvous) {
    if (plan == nullptr || out_num_blocks == nullptr) {
        return false;
    }

    window_task_executor_plan_clear(plan);
    *out_num_blocks = 0;

    if (!comm::launch_config_valid(launch_config)) {
        return false;
    }

    if (peer_count < 0 || peer_count > kTmaMultiGpuReduceScatterMaxPeers) {
        return false;
    }

    if (peer_count > 0 && peer_bufs == nullptr) {
        return false;
    }

    if (local_in == nullptr || local_buf == nullptr) {
        return false;
    }

    const bool out_of_place = (local_in != local_buf);

    const int tasks_per_cta =
        naive_multi_gpu_reduce_scatter_tasks_per_cta(
            peer_count,
            out_of_place);

    /*
     * Single-rank in-place reduce-scatter has no data movement.
     * Optionally launch one rendezvous-only CTA so collective ordering can be
     * preserved.
     */
    if (tasks_per_cta == 0 || slice_bytes == 0 || num_windows <= 0) {
        if (needs_rendezvous) {
            *out_num_blocks = 1;
        }
        return true;
    }

    if (tasks_per_cta > MaxTasks) {
        return false;
    }

    const int max_ctas_by_plan = MaxTasks / tasks_per_cta;

    if (max_ctas_by_plan <= 0) {
        return false;
    }

    int cta_count =
        comm::utils::cta_count_for_windows(
            num_windows,
            launch_config.max_ctas);

    cta_count = comm::utils::min_int(cta_count, max_ctas_by_plan);

    if (cta_count <= 0) {
        if (needs_rendezvous) {
            *out_num_blocks = 1;
        }
        return true;
    }

    plan->tasks_per_cta = tasks_per_cta;
    plan->total_tasks = cta_count * tasks_per_cta;

    if (plan->total_tasks > MaxTasks) {
        return false;
    }

    comm::utils::WindowRange full_range{};
    full_range.begin = 0;
    full_range.end = num_windows;

    for (int cta_idx = 0; cta_idx < cta_count; ++cta_idx) {
        const comm::utils::WindowRange cta_range =
            comm::utils::cta_window_range(
                cta_idx,
                cta_count,
                full_range);

        const int base = cta_idx * tasks_per_cta;
        int task_idx = base;

        /*
         * Initialize out-of-place destination with this rank's local
         * contribution for the owned reduce-scatter partition.
         */
        if (out_of_place) {
            const bool terminal = (peer_count == 0);

            plan->tasks[task_idx++] =
                make_reduce_scatter_copy_task(
                    launch_config.plan_kind,
                    local_in,
                    local_buf,
                    slice_bytes,
                    cta_range.begin,
                    cta_range.end,
                    launch_config.window_chunks,
                    terminal);
        }

        /*
         * Reduce all peer partitions into this rank's local partition.
         * The last reduce is terminal because reduce-scatter has no broadcast
         * phase after this.
         */
        for (int peer_idx = 0; peer_idx < peer_count; ++peer_idx) {
            if (peer_bufs[peer_idx] == nullptr) {
                return false;
            }

            const bool terminal = (peer_idx == peer_count - 1);

            plan->tasks[task_idx++] =
                comm::task::make_reduce_tma_task(
                    peer_bufs[peer_idx],
                    local_buf,
                    slice_bytes,
                    cta_range.begin,
                    cta_range.end,
                    launch_config.window_chunks,
                    terminal);
        }
    }

    *out_num_blocks = cta_count;
    return true;
}

} // namespace plan
} // namespace comm
} // namespace ooverlap
