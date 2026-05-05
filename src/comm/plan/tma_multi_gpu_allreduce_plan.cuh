#pragma once

#include "comm/launch_config.h"
#include "comm/plan/plan_params.cuh"
#include "comm/plan/window_plan.cuh"
#include "comm/task/window_task.cuh"
#include "comm/utils/utils.h"

#include <cstddef>

namespace ooverlap {
namespace comm {
namespace plan {

__host__ __device__ __forceinline__ int naive_multi_gpu_allreduce_tasks_per_cta(
    int peer_count,
    bool out_of_place) {
    if (peer_count < 0) {
        return 0;
    }

    return (out_of_place ? 1 : 0) + 2 * peer_count;
}

__host__ __device__ __forceinline__ bool multi_gpu_allreduce_use_fast_copy(
    comm::AllReducePlanKind plan_kind) {
    return plan_kind == comm::AllReducePlanKind::SeqFastCopyGmem ||
           plan_kind == comm::AllReducePlanKind::OverlapFastCopyGmem;
}

__host__ __device__ __forceinline__ comm::task::WindowTask make_allreduce_copy_task(
    comm::AllReducePlanKind plan_kind,
    const void* src,
    void* dst,
    size_t total_bytes,
    int begin_window,
    int end_window,
    int window_chunks,
    bool terminal = false) {
    if (multi_gpu_allreduce_use_fast_copy(plan_kind)) {
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
bool build_tma_multi_gpu_allreduce_naive_plan(
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

    if (launch_config.plan_for != comm::CollectivePlanFor::AllReduce) {
        return false;
    }

    if (!comm::launch_config_valid(launch_config)) {
        return false;
    }

    if (peer_count < 0 || peer_count > kTmaMultiGpuAllReduceMaxPeers) {
        return false;
    }

    if (peer_count > 0 && peer_bufs == nullptr) {
        return false;
    }

    if (local_in == nullptr || local_buf == nullptr) {
        return false;
    }

    const comm::AllReducePlanKind plan_kind =
        comm::allreduce_plan(launch_config);

    const bool out_of_place = (local_in != local_buf);

    const int tasks_per_cta =
        naive_multi_gpu_allreduce_tasks_per_cta(
            peer_count,
            out_of_place);

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

        int task_idx = cta_idx * tasks_per_cta;

        if (out_of_place) {
            const bool terminal = (peer_count == 0);

            plan->tasks[task_idx++] =
                make_allreduce_copy_task(
                    plan_kind,
                    local_in,
                    local_buf,
                    slice_bytes,
                    cta_range.begin,
                    cta_range.end,
                    launch_config.window_chunks,
                    terminal);
        }

        for (int peer_idx = 0; peer_idx < peer_count; ++peer_idx) {
            if (peer_bufs[peer_idx] == nullptr) {
                return false;
            }

            plan->tasks[task_idx++] =
                comm::task::make_reduce_tma_task(
                    peer_bufs[peer_idx],
                    local_buf,
                    slice_bytes,
                    cta_range.begin,
                    cta_range.end,
                    launch_config.window_chunks,
                    false);
        }

        for (int peer_idx = 0; peer_idx < peer_count; ++peer_idx) {
            const bool terminal = (peer_idx == peer_count - 1);

            plan->tasks[task_idx++] =
                make_allreduce_copy_task(
                    plan_kind,
                    local_buf,
                    peer_bufs[peer_idx],
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
