#pragma once

#include "comm/launch_config.h"
#include "comm/plan/transfer_plan.h"
#include "comm/plan/window_plan.cuh"
#include "comm/task/window_task.cuh"
#include "comm/utils/collective_utils.h"
#include "comm/utils/utils.h"

#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {
namespace plan {

/*
 * Rank-local pointer binding.
 *
 * Each process/rank fills this using its own pointer view. For multiprocess
 * CUDA IPC this means rank_buffer[peer] is the cudaIpcOpenMemHandle address as
 * seen in the current process, not a globally meaningful pointer value.
 */
template <int MaxRanks, int MaxStagingSlots>
struct RankPointerBinding {
    static_assert(MaxRanks > 0, "MaxRanks must be > 0");
    static_assert(MaxStagingSlots >= 0, "MaxStagingSlots must be >= 0");

    int current_rank = -1;
    int world_size = 0;

    void* rank_buffer[MaxRanks] = {};
    const void* rank_input[MaxRanks] = {};
    void* rank_output[MaxRanks] = {};

    void* shm_staging[MaxStagingSlots == 0 ? 1 : MaxStagingSlots] = {};
};

__host__ __device__ __forceinline__ const unsigned char* offset_const_bytes(
    const void* ptr,
    std::size_t byte_offset) {
    return reinterpret_cast<const unsigned char*>(ptr) + byte_offset;
}

__host__ __device__ __forceinline__ unsigned char* offset_bytes(
    void* ptr,
    std::size_t byte_offset) {
    return reinterpret_cast<unsigned char*>(ptr) + byte_offset;
}

template <int MaxRanks, int MaxStagingSlots>
__host__ __device__ __forceinline__ const void* resolve_logical_const_ptr(
    const LogicalBufferRef& ref,
    const RankPointerBinding<MaxRanks, MaxStagingSlots>& binding) {
    const void* base = nullptr;

    switch (ref.role) {
        case LogicalBufferRole::RankBuffer:
            if (ref.owner_rank < 0 || ref.owner_rank >= MaxRanks) {
                return nullptr;
            }
            base = binding.rank_buffer[ref.owner_rank];
            break;

        case LogicalBufferRole::RankInput:
            if (ref.owner_rank < 0 || ref.owner_rank >= MaxRanks) {
                return nullptr;
            }
            base = binding.rank_input[ref.owner_rank];
            break;

        case LogicalBufferRole::RankOutput:
            if (ref.owner_rank < 0 || ref.owner_rank >= MaxRanks) {
                return nullptr;
            }
            base = binding.rank_output[ref.owner_rank];
            break;

        case LogicalBufferRole::ShmStaging:
            if (ref.staging_slot < 0 || ref.staging_slot >= MaxStagingSlots) {
                return nullptr;
            }
            base = binding.shm_staging[ref.staging_slot];
            break;

        default:
            return nullptr;
    }

    if (base == nullptr) {
        return nullptr;
    }

    return offset_const_bytes(base, ref.byte_offset);
}

template <int MaxRanks, int MaxStagingSlots>
__host__ __device__ __forceinline__ void* resolve_logical_mut_ptr(
    const LogicalBufferRef& ref,
    const RankPointerBinding<MaxRanks, MaxStagingSlots>& binding) {
    void* base = nullptr;

    switch (ref.role) {
        case LogicalBufferRole::RankBuffer:
            if (ref.owner_rank < 0 || ref.owner_rank >= MaxRanks) {
                return nullptr;
            }
            base = binding.rank_buffer[ref.owner_rank];
            break;

        case LogicalBufferRole::RankInput:
            /*
             * RankInput is intentionally const for sources. Do not lower a
             * destination to RankInput.
             */
            return nullptr;

        case LogicalBufferRole::RankOutput:
            if (ref.owner_rank < 0 || ref.owner_rank >= MaxRanks) {
                return nullptr;
            }
            base = binding.rank_output[ref.owner_rank];
            break;

        case LogicalBufferRole::ShmStaging:
            if (ref.staging_slot < 0 || ref.staging_slot >= MaxStagingSlots) {
                return nullptr;
            }
            base = binding.shm_staging[ref.staging_slot];
            break;

        default:
            return nullptr;
    }

    if (base == nullptr) {
        return nullptr;
    }

    return offset_bytes(base, ref.byte_offset);
}

inline bool transfer_transport_direct(
    topology::TransportKind transport) {
    return transport == topology::TransportKind::DirectNvlink ||
           transport == topology::TransportKind::DirectPcie;
}

inline bool transfer_should_use_fast_copy(
    const TransferTask& task) {
    /*
     * First pass:
     * - SHM is not lowered to WindowTask yet.
     * - DirectPcie is lowered to fast global-memory copy for copy tasks.
     * - DirectNvlink copy can still use TMA copy.
     *
     * Later this should consult topology probe capabilities directly.
     */
    return task.op == TransferOp::Copy &&
           task.transport == topology::TransportKind::DirectPcie;
}

inline bool lower_transfer_task_to_window_task(
    const TransferTask& transfer,
    const void* src,
    void* dst,
    int begin_window,
    int end_window,
    task::WindowTask* out) {
    if (out == nullptr ||
        src == nullptr ||
        dst == nullptr ||
        begin_window >= end_window ||
        transfer.window_chunks <= 0 ||
        transfer.bytes == 0) {
        return false;
    }

    if (!transfer_transport_direct(transfer.transport)) {
        /*
         * SHM/staging transports need their own executor/lowering rules.
         */
        return false;
    }

    if (transfer.op == TransferOp::Reduce) {
        if (!transfer.requires_tma_reduce) {
            return false;
        }

        *out =
            task::make_reduce_tma_task(
                src,
                dst,
                transfer.bytes,
                begin_window,
                end_window,
                transfer.window_chunks,
                transfer.terminal);

        return true;
    }

    if (transfer.op == TransferOp::Copy) {
        if (transfer_should_use_fast_copy(transfer)) {
            *out =
                task::make_copy_fast_task(
                    src,
                    dst,
                    transfer.bytes,
                    begin_window,
                    end_window,
                    transfer.window_chunks,
                    transfer.terminal);
        } else {
            *out =
                task::make_copy_tma_task(
                    src,
                    dst,
                    transfer.bytes,
                    begin_window,
                    end_window,
                    transfer.window_chunks,
                    transfer.terminal);
        }

        return true;
    }

    return false;
}

template <
    int MaxTransferTasks,
    int MaxWindowTasks,
    int MaxRanks,
    int MaxStagingSlots>
bool lower_transfer_plan_for_rank(
    const TransferPlan<MaxTransferTasks>& transfer_plan,
    const RankPointerBinding<MaxRanks, MaxStagingSlots>& binding,
    const comm::LaunchConfig& launch_config,
    WindowTaskExecutorPlan<MaxWindowTasks>* out_window_plan,
    int* out_num_blocks,
    int reserved_prefix_tasks_per_cta = 0) {
    if (out_window_plan == nullptr ||
        out_num_blocks == nullptr ||
        binding.current_rank < 0 ||
        binding.current_rank >= binding.world_size ||
        binding.world_size <= 0 ||
        binding.world_size > MaxRanks ||
        launch_config.max_ctas < 0) {
        return false;
    }

    out_window_plan->total_tasks = 0;
    out_window_plan->tasks_per_cta = 0;
    *out_num_blocks = 0;

    int rank_transfer_count = 0;
    int max_end_window = 0;

    for (int i = 0; i < transfer_plan.total_tasks; ++i) {
        const TransferTask& transfer = transfer_plan.tasks[i];

        if (transfer.executor_rank != binding.current_rank) {
            continue;
        }

        if (!transfer_task_has_work(transfer)) {
            return false;
        }

        ++rank_transfer_count;
        max_end_window =
            comm::utils::max_int(
                max_end_window,
                transfer.end_window);
    }

    if (rank_transfer_count == 0) {
        return true;
    }

    if (rank_transfer_count > MaxWindowTasks) {
        return false;
    }

    const int tasks_per_cta_after_prefix =
        rank_transfer_count + reserved_prefix_tasks_per_cta;
    
    if (tasks_per_cta_after_prefix <= 0 ||
        tasks_per_cta_after_prefix > MaxWindowTasks) {
        return false;
    }
    
    const int max_ctas_by_plan =
        MaxWindowTasks / tasks_per_cta_after_prefix;

    if (max_ctas_by_plan <= 0) {
        return false;
    }

    int cta_count =
        comm::utils::cta_count_for_windows(
            max_end_window,
            launch_config.max_ctas);

    cta_count =
        comm::utils::min_int(
            cta_count,
            max_ctas_by_plan);

    if (cta_count <= 0) {
        return true;
    }

    const int total_window_tasks =
        cta_count * rank_transfer_count;

    if (total_window_tasks > MaxWindowTasks) {
        return false;
    }

    const comm::utils::WindowRange full_range{0, max_end_window};

    for (int cta_idx = 0; cta_idx < cta_count; ++cta_idx) {
        const comm::utils::WindowRange cta_range =
            comm::utils::cta_window_range(
                cta_idx,
                cta_count,
                full_range);

        int task_idx =
            cta_idx * rank_transfer_count;

        for (int i = 0; i < transfer_plan.total_tasks; ++i) {
            const TransferTask& transfer = transfer_plan.tasks[i];

            if (transfer.executor_rank != binding.current_rank) {
                continue;
            }

            const int begin_window =
                comm::utils::max_int(
                    transfer.begin_window,
                    cta_range.begin);

            const int end_window =
                comm::utils::min_int(
                    transfer.end_window,
                    cta_range.end);

            if (begin_window >= end_window) {
                /*
                 * Keep task striping rectangular. A no-op task preserves
                 * tasks_per_cta indexing.
                 */
                out_window_plan->tasks[task_idx++] = task::WindowTask{};
                continue;
            }

            const void* src =
                resolve_logical_const_ptr(
                    transfer.src,
                    binding);

            void* dst =
                resolve_logical_mut_ptr(
                    transfer.dst,
                    binding);

            task::WindowTask window_task{};

            if (!lower_transfer_task_to_window_task(
                    transfer,
                    src,
                    dst,
                    begin_window,
                    end_window,
                    &window_task)) {
                out_window_plan->total_tasks = 0;
                out_window_plan->tasks_per_cta = 0;
                *out_num_blocks = 0;
                return false;
            }

            out_window_plan->tasks[task_idx++] = window_task;
        }
    }

    out_window_plan->tasks_per_cta = rank_transfer_count;
    out_window_plan->total_tasks = total_window_tasks;
    *out_num_blocks = cta_count;

    return true;
}

} // namespace plan
} // namespace comm
} // namespace ooverlap
