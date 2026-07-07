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
 * Launch-time ready-signal binding.
 *
 * TransferPlan remains pointer-free/cacheable. ReadyPublish/ReadyWait
 * TransferTasks only describe logical ready ranks. The actual ready-signal
 * pointers, epoch, protocol, and polling policy are supplied here at lowering
 * time by the enqueue path.
 */
template <int MaxRanks>
struct ReadySignalBinding {
    /*
     * Channel-aware fields.  New code should fill these.
     */
    int* local_ready_signal_by_channel[kReadySignalChannelCount] = {};
    const int* ready_signal_by_rank_channel
        [MaxRanks][kReadySignalChannelCount] = {};
    int protocol_by_channel[kReadySignalChannelCount] = {};
    int poll_sleep_cycles_by_channel[kReadySignalChannelCount] = {};

    /*
     * Compatibility fields for old launchers.  These are interpreted as the
     * DeviceMemory channel when the channel-aware entries are null.
     */
    int* local_ready_signal = nullptr;
    const int* ready_signal_by_rank[MaxRanks] = {};
    int epoch = 0;
    int protocol = 0;
    int poll_sleep_cycles = 0;
};

__host__ __device__ __forceinline__ bool valid_ready_signal_channel(
    int channel) {
    return channel >= 0 && channel < kReadySignalChannelCount;
}

__host__ __device__ __forceinline__ bool valid_ready_signal_phase(
    int ready_phase) {
    return ready_phase >= 0 && ready_phase < kReadySignalPhaseStride;
}

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

inline bool transfer_task_uses_shm_staging(
    const TransferTask& task) {
    return task.src.role == LogicalBufferRole::ShmStaging ||
           task.dst.role == LogicalBufferRole::ShmStaging;
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
           (task.transport == topology::TransportKind::DirectPcie ||
            transfer_task_uses_shm_staging(task));
}

__host__ __device__ __forceinline__ bool transfer_task_is_ready(
    const TransferTask& task) {
    return task.op == TransferOp::ReadyPublish ||
           task.op == TransferOp::ReadyWait;
}

__host__ __device__ __forceinline__ bool transfer_task_is_windowed(
    const TransferTask& task) {
    return task.op == TransferOp::Copy ||
           task.op == TransferOp::Reduce;
}

inline bool lower_ready_transfer_task_to_window_task(
    const TransferTask& transfer,
    const ReadySignalBinding<16>* /* unused */) {
    /*
     * Placeholder overload intentionally not used.
     *
     * The real implementation is templated below.  This function only prevents
     * accidental non-templated declarations from being introduced elsewhere.
     */
    (void)transfer;
    return false;
}

template <int MaxRanks>
inline bool lower_ready_transfer_task_to_window_task(
    const TransferTask& transfer,
    const ReadySignalBinding<MaxRanks>& ready,
    task::WindowTask* out) {
    if (out == nullptr ||
        ready.epoch <= 0 ||
        transfer.executor_rank < 0 ||
        transfer.ready_rank < 0 ||
        transfer.ready_rank >= MaxRanks ||
        !valid_ready_signal_channel(transfer.ready_channel) ||
        !valid_ready_signal_phase(transfer.ready_phase)) {
        return false;
    }

    const int channel = transfer.ready_channel;
    const int ready_value =
        ready.epoch * kReadySignalPhaseStride + transfer.ready_phase;

    int* local_signal =
        ready.local_ready_signal_by_channel[channel];

    const int* peer_signal =
        ready.ready_signal_by_rank_channel[transfer.ready_rank][channel];

    int protocol =
        ready.protocol_by_channel[channel];

    int poll_sleep_cycles =
        ready.poll_sleep_cycles_by_channel[channel];

    /*
     * Backward compatibility for old single-channel launchers.
     */
    if (channel == static_cast<int>(ReadySignalChannel::DeviceMemory)) {
        if (local_signal == nullptr) {
            local_signal = ready.local_ready_signal;
        }

        if (peer_signal == nullptr) {
            peer_signal = ready.ready_signal_by_rank[transfer.ready_rank];
        }

        if (protocol == 0) {
            protocol = ready.protocol;
        }

        if (poll_sleep_cycles == 0) {
            poll_sleep_cycles = ready.poll_sleep_cycles;
        }
    }

    if (transfer.op == TransferOp::ReadyPublish) {
        if (local_signal == nullptr) {
            return false;
        }

        *out =
            task::make_ready_publish_task(
                local_signal,
                ready_value,
                protocol,
                transfer.terminal);

        return true;
    }

    if (transfer.op == TransferOp::ReadyWait) {
        if (peer_signal == nullptr) {
            return false;
        }

        *out =
            task::make_ready_wait_task(
                peer_signal,
                ready_value,
                poll_sleep_cycles,
                transfer.terminal);

        return true;
    }

    return false;
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

    const bool uses_shm_staging =
        transfer_task_uses_shm_staging(transfer);

    if (!transfer_transport_direct(transfer.transport) &&
        !(transfer.op == TransferOp::Copy && uses_shm_staging)) {
        /*
         * Non-direct non-staging transports still need their own executor rules.
         *
         * A Copy task that explicitly references ShmStaging is lowered to
         * CopyFast below.  This lets planners opt into staging by using
         * shm_staging_ref(slot, offset) without requiring a dedicated staging
         * WindowTaskOp yet.
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
    int reserved_prefix_tasks_per_cta = 0,
    const ReadySignalBinding<MaxRanks>* ready_binding = nullptr) {
    if (out_window_plan == nullptr ||
        out_num_blocks == nullptr ||
        binding.current_rank < 0 ||
        binding.current_rank >= binding.world_size ||
        binding.world_size <= 0 ||
        binding.world_size > MaxRanks ||
        launch_config.max_ctas < 0 ||
        reserved_prefix_tasks_per_cta < 0) {
        return false;
    }

    out_window_plan->total_tasks = 0;
    out_window_plan->tasks_per_cta = 0;
    *out_num_blocks = 0;

    /*
     * Transition behavior:
     *
     * - If ready_binding is supplied, ReadyPublish/ReadyWait TransferTasks are
     *   lowered into WindowTasks and replicated into every CTA stripe.
     *
     * - If ready_binding is nullptr, logical ready tasks are ignored.  This keeps
     *   old launcher-side prepend_ready_tasks_to_each_cta() code compiling while
     *   the launchers are migrated.
     */
    const bool lower_ready_tasks =
        ready_binding != nullptr &&
        ready_binding->epoch > 0;

    int rank_ready_task_count = 0;
    int rank_window_task_count = 0;
    int max_end_window = 0;

    for (int i = 0; i < transfer_plan.total_tasks; ++i) {
        const TransferTask& transfer = transfer_plan.tasks[i];

        if (transfer.executor_rank != binding.current_rank) {
            continue;
        }

        if (!transfer_task_has_work(transfer)) {
            return false;
        }

        if (transfer_task_is_ready(transfer)) {
            if (lower_ready_tasks) {
                ++rank_ready_task_count;
            }

            continue;
        }

        if (!transfer_task_is_windowed(transfer)) {
            return false;
        }

        ++rank_window_task_count;

        max_end_window =
            comm::utils::max_int(
                max_end_window,
                transfer.end_window);
    }

    if (rank_ready_task_count == 0 && rank_window_task_count == 0) {
        return true;
    }

    const int tasks_per_cta =
        rank_ready_task_count + rank_window_task_count;

    if (tasks_per_cta <= 0 || tasks_per_cta > MaxWindowTasks) {
        return false;
    }

    /*
     * reserved_prefix_tasks_per_cta is kept only for compatibility with the
     * temporary launcher-side ready prepend path.  New code should pass zero
     * here and use ready_binding.
     */
    const int capacity_tasks_per_cta =
        tasks_per_cta + reserved_prefix_tasks_per_cta;

    if (capacity_tasks_per_cta <= 0 ||
        capacity_tasks_per_cta > MaxWindowTasks) {
        return false;
    }

    const int max_ctas_by_plan =
        MaxWindowTasks / capacity_tasks_per_cta;

    if (max_ctas_by_plan <= 0) {
        return false;
    }

    int cta_count = 0;

    if (rank_window_task_count > 0) {
        cta_count =
            comm::utils::cta_count_for_windows(
                max_end_window,
                launch_config.max_ctas);
    } else {
        /*
         * Sync-only plan.  This is mostly useful for testing.  Real collectives
         * normally have at least one windowed transfer task.
         */
        cta_count = 1;
    }

    cta_count =
        comm::utils::min_int(
            cta_count,
            max_ctas_by_plan);

    if (cta_count <= 0) {
        return true;
    }

    const int total_window_tasks =
        cta_count * tasks_per_cta;

    if (total_window_tasks > MaxWindowTasks) {
        return false;
    }

    const comm::utils::WindowRange full_range{0, max_end_window};

    for (int cta_idx = 0; cta_idx < cta_count; ++cta_idx) {
        const comm::utils::WindowRange cta_range =
            rank_window_task_count > 0
                ? comm::utils::cta_window_range(
                      cta_idx,
                      cta_count,
                      full_range)
                : comm::utils::WindowRange{0, 0};

        int task_idx =
            cta_idx * tasks_per_cta;

        for (int i = 0; i < transfer_plan.total_tasks; ++i) {
            const TransferTask& transfer = transfer_plan.tasks[i];

            if (transfer.executor_rank != binding.current_rank) {
                continue;
            }

            if (transfer_task_is_ready(transfer)) {
                if (!lower_ready_tasks) {
                    continue;
                }

                task::WindowTask ready_task{};

                if (!lower_ready_transfer_task_to_window_task(
                        transfer,
                        *ready_binding,
                        &ready_task)) {
                    out_window_plan->total_tasks = 0;
                    out_window_plan->tasks_per_cta = 0;
                    *out_num_blocks = 0;
                    return false;
                }

                out_window_plan->tasks[task_idx++] = ready_task;
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

    out_window_plan->tasks_per_cta = tasks_per_cta;
    out_window_plan->total_tasks = total_window_tasks;
    *out_num_blocks = cta_count;

    return true;
}

} // namespace plan
} // namespace comm
} // namespace ooverlap
