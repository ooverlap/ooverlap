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

constexpr int kTmaTwoGpuPeerAllreduceSeqTasksPerCta = 2;
constexpr int kTmaTwoGpuPeerAllreduceOverlapTasksPerCta = 1;

/*
 * Enough for:
 *
 *   TmaCopy:
 *     max_ctas <= 32, two tasks per CTA
 *
 *   SeqFastGmem:
 *     max_ctas <= 32, two tasks per CTA
 *
 *   OverlapFastGmem:
 *     max_ctas <= 64, one task per CTA
 *
 * The plan is passed as a kernel parameter, so the measured stream work does
 * not include a per-launch cudaMemcpyAsync task upload.
 */
constexpr int kTmaTwoGpuPeerAllreduceMaxWindowTasks = 64;

__host__ __device__ __forceinline__ int overlap_pair_count_for_windows(
    int owned_windows,
    int max_ctas) {
    if (owned_windows <= 0 ||
        max_ctas < TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA) {
        return 0;
    }

    return comm::utils::min_int(
        owned_windows,
        max_ctas / TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA);
}

__host__ __device__ __forceinline__ int required_window_ready_flags_for_plan(
    comm::AllreducePlanKind plan_kind,
    bool out_of_place,
    int rank,
    int num_windows) {
    switch (plan_kind) {
        case comm::AllreducePlanKind::TmaCopy:
            /*
             * Different-buffer normal TMA uses the overlap topology over the
             * whole logical window range.
             */
            return out_of_place ? num_windows : 0;

        case comm::AllreducePlanKind::OverlapFastGmem:
            return comm::utils::rank_window_count(num_windows, rank);

        case comm::AllreducePlanKind::SeqFastGmem:
        default:
            return 0;
    }
}

template <int MaxTasks>
bool build_tma_copy_inplace_plan(
    WindowTaskExecutorPlan<MaxTasks>* plan,
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t total_bytes,
    int rank,
    int num_windows,
    int ctas_per_rank,
    int window_chunks) {
    (void)local_in;

    if (plan == nullptr) {
        return false;
    }

    window_task_executor_plan_clear(plan);

    plan->tasks_per_cta = kTmaTwoGpuPeerAllreduceSeqTasksPerCta;
    plan->total_tasks =
        ctas_per_rank * kTmaTwoGpuPeerAllreduceSeqTasksPerCta;

    if (plan->total_tasks > MaxTasks) {
        return false;
    }

    const comm::utils::WindowRange rank_range =
        comm::utils::rank_window_range(num_windows, rank);

    for (int cta_idx = 0; cta_idx < ctas_per_rank; ++cta_idx) {
        const comm::utils::WindowRange cta_range =
            comm::utils::cta_window_range(
                cta_idx,
                ctas_per_rank,
                rank_range);

        const int base =
            cta_idx * kTmaTwoGpuPeerAllreduceSeqTasksPerCta;

        /*
         * Normal TMA-copy in-place policy:
         *
         *   reduce peer -> local
         *   copy   local -> peer using TMA
         */
        plan->tasks[base + 0] =
            comm::task::make_reduce_tma_task(
                peer_buf,
                local_buf,
                total_bytes,
                cta_range.begin,
                cta_range.end,
                window_chunks,
                false);

        plan->tasks[base + 1] =
            comm::task::make_copy_tma_task(
                local_buf,
                peer_buf,
                total_bytes,
                cta_range.begin,
                cta_range.end,
                window_chunks,
                true);
    }

    return true;
}

template <int MaxTasks>
bool build_tma_copy_out_of_place_overlap_plan(
    WindowTaskExecutorPlan<MaxTasks>* plan,
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t total_bytes,
    int num_windows,
    int pair_count,
    int window_chunks,
    int* window_ready_flags) {
    if (plan == nullptr) {
        return false;
    }

    window_task_executor_plan_clear(plan);

    plan->tasks_per_cta = kTmaTwoGpuPeerAllreduceOverlapTasksPerCta;
    plan->total_tasks =
        pair_count * TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA;

    if (plan->total_tasks > MaxTasks) {
        return false;
    }

    comm::utils::WindowRange full_range{};
    full_range.begin = 0;
    full_range.end = num_windows;

    for (int pair_idx = 0; pair_idx < pair_count; ++pair_idx) {
        const comm::utils::WindowRange pair_range =
            comm::utils::cta_window_range(
                pair_idx,
                pair_count,
                full_range);

        const int producer_block =
            pair_idx * TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA + 0;

        const int consumer_block =
            pair_idx * TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA + 1;

        /*
         * Different-buffer normal-TMA overlap policy:
         *
         *   producer CTA:
         *     copy peer -> local output and signal completed windows
         *
         *   consumer CTA:
         *     wait for signal, then reduce local input -> local output
         */
        plan->tasks[producer_block] =
            comm::task::make_copy_tma_signal_task(
                peer_buf,
                local_buf,
                total_bytes,
                pair_range.begin,
                pair_range.end,
                window_chunks,
                window_ready_flags,
                0,
                true);

        plan->tasks[consumer_block] =
            comm::task::make_reduce_tma_after_signal_task(
                local_in,
                local_buf,
                total_bytes,
                pair_range.begin,
                pair_range.end,
                window_chunks,
                window_ready_flags,
                0,
                true);
    }

    return true;
}

template <int MaxTasks>
bool build_seq_fast_gmem_plan(
    WindowTaskExecutorPlan<MaxTasks>* plan,
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t total_bytes,
    int rank,
    int num_windows,
    int ctas_per_rank,
    int window_chunks) {
    (void)local_in;

    if (plan == nullptr) {
        return false;
    }

    window_task_executor_plan_clear(plan);

    plan->tasks_per_cta = kTmaTwoGpuPeerAllreduceSeqTasksPerCta;
    plan->total_tasks =
        ctas_per_rank * kTmaTwoGpuPeerAllreduceSeqTasksPerCta;

    if (plan->total_tasks > MaxTasks) {
        return false;
    }

    const comm::utils::WindowRange rank_range =
        comm::utils::rank_window_range(num_windows, rank);

    for (int cta_idx = 0; cta_idx < ctas_per_rank; ++cta_idx) {
        const comm::utils::WindowRange cta_range =
            comm::utils::cta_window_range(
                cta_idx,
                ctas_per_rank,
                rank_range);

        const int base =
            cta_idx * kTmaTwoGpuPeerAllreduceSeqTasksPerCta;

        /*
         * Sequential fast-gmem policy:
         *
         *   reduce peer -> local
         *   copy   local -> peer using fast global-memory vector copy
         */
        plan->tasks[base + 0] =
            comm::task::make_reduce_tma_task(
                peer_buf,
                local_buf,
                total_bytes,
                cta_range.begin,
                cta_range.end,
                window_chunks,
                false);

        plan->tasks[base + 1] =
            comm::task::make_copy_fast_task(
                local_buf,
                peer_buf,
                total_bytes,
                cta_range.begin,
                cta_range.end,
                window_chunks,
                true);
    }

    return true;
}

template <int MaxTasks>
bool build_overlap_fast_gmem_plan(
    WindowTaskExecutorPlan<MaxTasks>* plan,
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t total_bytes,
    int rank,
    int num_windows,
    int pair_count,
    int window_chunks,
    int* window_ready_flags) {
    (void)local_in;

    if (plan == nullptr) {
        return false;
    }

    window_task_executor_plan_clear(plan);

    plan->tasks_per_cta = kTmaTwoGpuPeerAllreduceOverlapTasksPerCta;
    plan->total_tasks =
        pair_count * TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA;

    if (plan->total_tasks > MaxTasks) {
        return false;
    }

    const comm::utils::WindowRange rank_range =
        comm::utils::rank_window_range(num_windows, rank);

    for (int pair_idx = 0; pair_idx < pair_count; ++pair_idx) {
        const comm::utils::WindowRange pair_range =
            comm::utils::cta_window_range(
                pair_idx,
                pair_count,
                rank_range);

        const int producer_block =
            pair_idx * TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA + 0;

        const int consumer_block =
            pair_idx * TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA + 1;

        /*
         * Overlap fast-gmem policy:
         *
         *   producer CTA:
         *     reduce peer -> local and signal completed windows
         *
         *   consumer CTA:
         *     wait for signal, then copy local -> peer using fast gmem copy
         */
        plan->tasks[producer_block] =
            comm::task::make_reduce_tma_signal_task(
                peer_buf,
                local_buf,
                total_bytes,
                pair_range.begin,
                pair_range.end,
                window_chunks,
                window_ready_flags,
                rank_range.begin,
                true);

        plan->tasks[consumer_block] =
            comm::task::make_copy_fast_after_signal_task(
                local_buf,
                peer_buf,
                total_bytes,
                pair_range.begin,
                pair_range.end,
                window_chunks,
                window_ready_flags,
                rank_range.begin,
                true);
    }

    return true;
}

template <int MaxTasks>
bool build_tma_two_gpu_peer_allreduce_plan(
    WindowTaskExecutorPlan<MaxTasks>* plan,
    int* out_num_blocks,
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t total_bytes,
    int rank,
    int num_windows,
    const comm::LaunchConfig& launch_config,
    bool needs_rendezvous,
    int* window_ready_flags) {
    if (plan == nullptr || out_num_blocks == nullptr) {
        return false;
    }

    *out_num_blocks = 0;

    if (!comm::launch_config_valid(launch_config)) {
        return false;
    }

    const bool out_of_place = (local_in != local_buf);

    switch (launch_config.plan_kind) {
        case comm::AllreducePlanKind::TmaCopy: {
            if (out_of_place) {
                /*
                 * Different-buffer normal TMA uses the TMA-overlap task
                 * topology:
                 *
                 *   CopyTMASignal
                 *   ReduceTMAAfterSignal
                 */
                if (!comm::launch_config_valid_for_overlap(launch_config)) {
                    return false;
                }

                const int owned_windows = num_windows;

                const int pair_count =
                    overlap_pair_count_for_windows(
                        owned_windows,
                        launch_config.max_ctas);

                int num_blocks =
                    pair_count * TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA;

                if (needs_rendezvous) {
                    num_blocks = comm::utils::max_int(1, num_blocks);
                }

                if (pair_count *
                        TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA *
                        kTmaTwoGpuPeerAllreduceOverlapTasksPerCta >
                    MaxTasks) {
                    return false;
                }

                const bool ok =
                    build_tma_copy_out_of_place_overlap_plan(
                        plan,
                        local_in,
                        local_buf,
                        peer_buf,
                        total_bytes,
                        num_windows,
                        pair_count,
                        launch_config.window_chunks,
                        window_ready_flags);

                if (!ok) {
                    return false;
                }

                *out_num_blocks = num_blocks;
                return true;
            }

            const int owned_windows =
                comm::utils::rank_window_count(num_windows, rank);

            const int ctas_per_rank =
                comm::utils::cta_count_for_windows(
                    owned_windows,
                    launch_config.max_ctas);

            int num_blocks =
                needs_rendezvous
                    ? comm::utils::max_int(1, ctas_per_rank)
                    : ctas_per_rank;

            if (ctas_per_rank *
                    kTmaTwoGpuPeerAllreduceSeqTasksPerCta >
                MaxTasks) {
                return false;
            }

            const bool ok =
                build_tma_copy_inplace_plan(
                    plan,
                    local_in,
                    local_buf,
                    peer_buf,
                    total_bytes,
                    rank,
                    num_windows,
                    ctas_per_rank,
                    launch_config.window_chunks);

            if (!ok) {
                return false;
            }

            *out_num_blocks = num_blocks;
            return true;
        }

        case comm::AllreducePlanKind::SeqFastGmem: {
            const int owned_windows =
                comm::utils::rank_window_count(num_windows, rank);

            const int ctas_per_rank =
                comm::utils::cta_count_for_windows(
                    owned_windows,
                    launch_config.max_ctas);

            int num_blocks =
                needs_rendezvous
                    ? comm::utils::max_int(1, ctas_per_rank)
                    : ctas_per_rank;

            if (ctas_per_rank *
                    kTmaTwoGpuPeerAllreduceSeqTasksPerCta >
                MaxTasks) {
                return false;
            }

            const bool ok =
                build_seq_fast_gmem_plan(
                    plan,
                    local_in,
                    local_buf,
                    peer_buf,
                    total_bytes,
                    rank,
                    num_windows,
                    ctas_per_rank,
                    launch_config.window_chunks);

            if (!ok) {
                return false;
            }

            *out_num_blocks = num_blocks;
            return true;
        }

        case comm::AllreducePlanKind::OverlapFastGmem: {
            if (!comm::launch_config_valid_for_overlap(launch_config)) {
                return false;
            }

            const int owned_windows =
                comm::utils::rank_window_count(num_windows, rank);

            const int pair_count =
                overlap_pair_count_for_windows(
                    owned_windows,
                    launch_config.max_ctas);

            int num_blocks =
                pair_count * TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA;

            if (needs_rendezvous) {
                num_blocks = comm::utils::max_int(1, num_blocks);
            }

            if (pair_count *
                    TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA *
                    kTmaTwoGpuPeerAllreduceOverlapTasksPerCta >
                MaxTasks) {
                return false;
            }

            const bool ok =
                build_overlap_fast_gmem_plan(
                    plan,
                    local_in,
                    local_buf,
                    peer_buf,
                    total_bytes,
                    rank,
                    num_windows,
                    pair_count,
                    launch_config.window_chunks,
                    window_ready_flags);

            if (!ok) {
                return false;
            }

            *out_num_blocks = num_blocks;
            return true;
        }

        default:
            return false;
    }
}

} // namespace plan
} // namespace comm
} // namespace ooverlap
