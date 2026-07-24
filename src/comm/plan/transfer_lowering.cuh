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
 * pointers, epoch, and protocol are supplied here at lowering time by the
 * enqueue path. WindowTask ready waits use a fixed 64-cycle sleep.
 */
template <int MaxRanks>
struct ReadySignalBinding {
    /*
     * Channel-aware fields. New code should fill these.
     */
    int* local_ready_signal_by_channel[kReadySignalChannelCount] = {};
    const int* ready_signal_by_rank_channel
        [MaxRanks][kReadySignalChannelCount] = {};
    int protocol_by_channel[kReadySignalChannelCount] = {};

    /*
     * Compatibility fields for old launchers. These are interpreted as the
     * DeviceMemory channel when the channel-aware entries are null.
     */
    int* local_ready_signal = nullptr;
    const int* ready_signal_by_rank[MaxRanks] = {};
    int epoch = 0;
    int protocol = 0;
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
    /*return task.op == TransferOp::Copy &&*/
           /*(task.transport == topology::TransportKind::DirectPcie ||*/
            /*transfer_task_uses_shm_staging(task));*/

    return false;
}

__host__ __device__ __forceinline__ bool transfer_task_is_ready(
    const TransferTask& task) {
    return task.op == TransferOp::ReadyPublish ||
           task.op == TransferOp::ReadyWait;
}

__host__ __device__ __forceinline__ bool transfer_task_is_barrier(
    const TransferTask& task) {
    return task.op == TransferOp::Barrier;
}

__host__ __device__ __forceinline__ bool transfer_task_is_windowed(
    const TransferTask& task) {
    return task.op == TransferOp::Copy ||
           task.op == TransferOp::Reduce ||
           task.op == TransferOp::CopyFanout ||
           task.op == TransferOp::ReduceFanout;
}

inline bool lower_ready_transfer_task_to_window_task(
    const TransferTask& transfer,
    const ReadySignalBinding<16>* /* unused */) {
    /*
     * Placeholder overload intentionally not used.
     *
     * The real implementation is templated below. This function only prevents
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
                transfer.terminal);

        return true;
    }

    return false;
}


/* OOVERLAP_READY_PUBLISH_WAIT_MERGE_PATCH: lower consecutive ReadyPublish + ReadyWait into one task. */
template <int MaxRanks>
inline bool lower_ready_publish_wait_transfer_tasks_to_window_task(
    const TransferTask& publish,
    const TransferTask& wait,
    const ReadySignalBinding<MaxRanks>& ready,
    int owner_cta,
    task::WindowTask* out) {
    if (out == nullptr ||
        ready.epoch <= 0 ||
        publish.op != TransferOp::ReadyPublish ||
        wait.op != TransferOp::ReadyWait ||
        publish.executor_rank != wait.executor_rank ||
        publish.executor_rank < 0 ||
        publish.ready_rank < 0 ||
        publish.ready_rank >= MaxRanks ||
        wait.ready_rank < 0 ||
        wait.ready_rank >= MaxRanks ||
        !valid_ready_signal_channel(publish.ready_channel) ||
        !valid_ready_signal_channel(wait.ready_channel) ||
        !valid_ready_signal_phase(publish.ready_phase) ||
        !valid_ready_signal_phase(wait.ready_phase)) {
        return false;
    }

    const int publish_channel = publish.ready_channel;
    const int wait_channel = wait.ready_channel;

    int* publish_signal =
        ready.local_ready_signal_by_channel[publish_channel];

    int publish_protocol =
        ready.protocol_by_channel[publish_channel];

    const int* wait_signal =
        ready.ready_signal_by_rank_channel[wait.ready_rank][wait_channel];

    /*
     * Backward compatibility for old single-channel launchers.
     */
    if (publish_channel == static_cast<int>(ReadySignalChannel::DeviceMemory)) {
        if (publish_signal == nullptr) {
            publish_signal = ready.local_ready_signal;
        }

        if (publish_protocol == 0) {
            publish_protocol = ready.protocol;
        }
    }

    if (wait_channel == static_cast<int>(ReadySignalChannel::DeviceMemory)) {
        if (wait_signal == nullptr) {
            wait_signal = ready.ready_signal_by_rank[wait.ready_rank];
        }
    }

    if (publish_signal == nullptr || wait_signal == nullptr) {
        return false;
    }

    const int publish_value =
        ready.epoch * kReadySignalPhaseStride + publish.ready_phase;
    const int wait_value =
        ready.epoch * kReadySignalPhaseStride + wait.ready_phase;

    *out =
        task::make_ready_publish_wait_task(
            publish_signal,
            publish_value,
            publish_protocol,
            wait_signal,
            wait_value,
            owner_cta,
            wait.terminal);

    return true;
}

inline task::WindowTaskOp select_copy_window_op(
    const TransferTask& transfer) {
    if (transfer_should_use_fast_copy(transfer)) {
        return task::WindowTaskOp::CopyFast;
    }

    return task::WindowTaskOp::CopyTMA;
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
    
    const bool copy_uses_shm_staging =
        transfer.op == TransferOp::Copy &&
        uses_shm_staging;
    
    const bool reduce_from_shm_staging =
        transfer.op == TransferOp::Reduce &&
        transfer.src.role == LogicalBufferRole::ShmStaging &&
        transfer.dst.role != LogicalBufferRole::ShmStaging;
    
    if (!transfer_transport_direct(transfer.transport) &&
        !copy_uses_shm_staging &&
        !reduce_from_shm_staging) {
        /*
         * Non-direct non-staging transports still need their own executor rules.
         *
         * Staged collectives are allowed to use ShmStaging explicitly:
         *   - Copy RankBuffer -> ShmStaging
         *   - Copy ShmStaging -> RankBuffer
         *   - Reduce ShmStaging -> RankBuffer
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
        switch (select_copy_window_op(transfer)) {
            case task::WindowTaskOp::CopyFast:
                *out =
                    task::make_copy_fast_task(
                        src,
                        dst,
                        transfer.bytes,
                        begin_window,
                        end_window,
                        transfer.window_chunks,
                        transfer.terminal);
                return true;

            case task::WindowTaskOp::CopyTMA:
                *out =
                    task::make_copy_tma_task(
                        src,
                        dst,
                        transfer.bytes,
                        begin_window,
                        end_window,
                        transfer.window_chunks,
                        transfer.terminal);
                return true;

            default:
                return false;
        }
    }

    return false;
}


/*
 * OOVERLAP_FANOUT_TRANSFER_LOWERING_HELPER_PATCH:
 *
 * Lower a pointer-resolved TransferOp::{CopyFanout, ReduceFanout} into a
 * WindowTask. Pointer resolution still happens in emit_window_plan_from_rank_tasks
 * because it has access to RankPointerBinding.
 */
inline bool lower_fanout_transfer_task_to_window_task(
    const TransferTask& transfer,
    const void* src,
    void* const* fanout_dsts,
    int begin_window,
    int end_window,
    task::WindowTask* out) {
    if (out == nullptr ||
        src == nullptr ||
        fanout_dsts == nullptr ||
        begin_window >= end_window ||
        transfer.window_chunks <= 0 ||
        transfer.bytes == 0 ||
        transfer.fanout_dst_count <= 0 ||
        transfer.fanout_dst_count > TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS) {
        return false;
    }

    for (int i = 0; i < transfer.fanout_dst_count; ++i) {
        if (fanout_dsts[i] == nullptr) {
            return false;
        }
    }

    if (!transfer_transport_direct(transfer.transport)) {
        /*
         * First implementation only supports direct fanout. Shm/staged fanout
         * can be added later with explicit executor rules.
         */
        return false;
    }

    if (transfer.op == TransferOp::CopyFanout) {
        if (!transfer.requires_tma_load || !transfer.requires_tma_store) {
            return false;
        }

        *out =
            task::make_copy_tma_fanout_task(
                src,
                fanout_dsts,
                transfer.fanout_dst_count,
                transfer.bytes,
                begin_window,
                end_window,
                transfer.window_chunks,
                transfer.terminal);

        return true;
    }

    if (transfer.op == TransferOp::ReduceFanout) {
        if (!transfer.requires_tma_load || !transfer.requires_tma_reduce) {
            return false;
        }

        *out =
            task::make_reduce_tma_fanout_task(
                src,
                fanout_dsts,
                transfer.fanout_reduce_scope,
                transfer.fanout_dst_count,
                transfer.bytes,
                begin_window,
                end_window,
                transfer.window_chunks,
                transfer.terminal);

        return true;
    }

    return false;
}


namespace lowering_detail {

/*
 * Pass options are intentionally conservative. The default behavior should remain
 * equivalent to the old monolithic lower_transfer_plan_for_rank().
 *
 * Add real optimization toggles here as you implement them.
 */
struct LoweringPassOptions {
    bool enable_transfer_passes = true;
    bool enable_window_passes = true;

    /*
     * Merge consecutive ReadyPublish + ReadyWait into one WindowTask.
     *
     * Only one CTA performs the publish. All CTAs wait on the peer ready signal.
     * Multiple merged pairs are assigned owner CTAs round-robin to avoid putting
     * every publish on CTA 0.
     */
    bool enable_ready_publish_wait_merge = true;

    /*
     * Assign independent TransferOp::Reduce tasks to contiguous CTA groups.
     * The all-reduce launcher enables this and supplies the group-size limit.
     */
    bool enable_reduce_cta_groups = false;
    int max_ctas_per_reduce_task = 8;

    /* Counter value released by CTA 0 after entry synchronization. */
    unsigned int cta_barrier_start = 1u;

    /*
     * The current executor has historically had a temporary local-task limit in
     * execute_window_task_stripe(). Keep this disabled by default so the skeleton
     * preserves old lowering behavior. Enable while debugging if needed.
     */
    bool enforce_executor_task_limit = false;
    int max_executable_tasks_per_cta = 2;
};

template <int MaxRanks>
struct LoweringContext {
    int current_rank = -1;
    int world_size = 0;

    const comm::LaunchConfig* launch_config = nullptr;
    const ReadySignalBinding<MaxRanks>* ready_binding = nullptr;

    bool lower_ready_tasks = false;
    int reserved_prefix_tasks_per_cta = 0;

    LoweringPassOptions options{};
};

template <int MaxRanks, int MaxStagingSlots>
inline bool make_lowering_context(
    const RankPointerBinding<MaxRanks, MaxStagingSlots>& binding,
    const comm::LaunchConfig& launch_config,
    int reserved_prefix_tasks_per_cta,
    const ReadySignalBinding<MaxRanks>* ready_binding,
    const LoweringPassOptions& options,
    LoweringContext<MaxRanks>* out) {
    if (out == nullptr ||
        binding.current_rank < 0 ||
        binding.current_rank >= binding.world_size ||
        binding.world_size <= 0 ||
        binding.world_size > MaxRanks ||
        launch_config.max_ctas < 0 ||
        reserved_prefix_tasks_per_cta < 0 ||
        (options.enable_reduce_cta_groups &&
         (options.max_ctas_per_reduce_task <= 0 ||
          options.max_ctas_per_reduce_task > task::kWindowTaskMaxCtas))) {
        return false;
    }

    LoweringContext<MaxRanks> ctx{};
    ctx.current_rank = binding.current_rank;
    ctx.world_size = binding.world_size;
    ctx.launch_config = &launch_config;
    ctx.ready_binding = ready_binding;
    ctx.lower_ready_tasks =
        ready_binding != nullptr &&
        ready_binding->epoch > 0;
    ctx.reserved_prefix_tasks_per_cta = reserved_prefix_tasks_per_cta;
    ctx.options = options;

    *out = ctx;
    return true;
}

template <int MaxTransferTasks>
struct RankTransferTaskBuffer {
    static_assert(MaxTransferTasks > 0, "MaxTransferTasks must be > 0");

    int count;
    int ready_count;
    int ready_pair_count;
    int barrier_count;
    int window_count;
    int max_end_window;

    /*
     * Do not value-initialize this array. This buffer is hot in the lowering
     * path, and TransferTask is relatively large. The valid range is
     * tasks[0..count), and every entry in that range is assigned by
     * rank_transfer_task_buffer_push() before being read.
     *
     * OOVERLAP_NO_RANK_TASK_BUFFER_ZEROING_PATCH.
     */
    TransferTask tasks[MaxTransferTasks];
};

template <int MaxTransferTasks>
inline void rank_transfer_task_buffer_reset(
    RankTransferTaskBuffer<MaxTransferTasks>* buffer) {
    if (buffer == nullptr) {
        return;
    }

    /*
     * Reset only metadata. Leaving stale entries outside tasks[0..count)
     * untouched avoids a large memset/copy on every lower_transfer_plan_for_rank
     * call.
     */
    buffer->count = 0;
    buffer->ready_count = 0;
    buffer->ready_pair_count = 0;
    buffer->barrier_count = 0;
    buffer->window_count = 0;
    buffer->max_end_window = 0;
}

template <int MaxTransferTasks>
inline bool rank_transfer_task_buffer_push(
    RankTransferTaskBuffer<MaxTransferTasks>* buffer,
    const TransferTask& task) {
    if (buffer == nullptr ||
        buffer->count < 0 ||
        buffer->count >= MaxTransferTasks) {
        return false;
    }

    buffer->tasks[buffer->count++] = task;
    return true;
}


inline bool transfer_tasks_are_mergeable_ready_publish_wait_pair(
    const TransferTask& publish,
    const TransferTask& wait) {
    return publish.op == TransferOp::ReadyPublish &&
           wait.op == TransferOp::ReadyWait &&
           publish.executor_rank == wait.executor_rank &&
           !publish.terminal;
}

template <int MaxTransferTasks>
inline bool recompute_rank_transfer_task_stats(
    RankTransferTaskBuffer<MaxTransferTasks>* buffer) {
    if (buffer == nullptr ||
        buffer->count < 0 ||
        buffer->count > MaxTransferTasks) {
        return false;
    }

    buffer->ready_count = 0;
    buffer->ready_pair_count = 0;
    buffer->barrier_count = 0;
    buffer->window_count = 0;
    buffer->max_end_window = 0;

    for (int i = 0; i < buffer->count; ++i) {
        const TransferTask& transfer = buffer->tasks[i];

        if (transfer_task_is_barrier(transfer)) {
            ++buffer->barrier_count;
            continue;
        }

        if (transfer_task_is_ready(transfer)) {
            ++buffer->ready_count;

            if (i + 1 < buffer->count &&
                transfer_tasks_are_mergeable_ready_publish_wait_pair(
                    buffer->tasks[i],
                    buffer->tasks[i + 1])) {
                ++buffer->ready_count;
                ++buffer->ready_pair_count;
                ++i;
            }

            continue;
        }

        if (!transfer_task_is_windowed(transfer)) {
            return false;
        }

        ++buffer->window_count;
        buffer->max_end_window =
            comm::utils::max_int(
                buffer->max_end_window,
                transfer.end_window);
    }

    return true;
}

template <int MaxTransferTasks, int MaxRanks>
inline bool collect_rank_transfer_tasks(
    const TransferPlan<MaxTransferTasks>& transfer_plan,
    const LoweringContext<MaxRanks>& ctx,
    RankTransferTaskBuffer<MaxTransferTasks>* out) {
    if (out == nullptr ||
        transfer_plan.total_tasks < 0 ||
        transfer_plan.total_tasks > MaxTransferTasks) {
        return false;
    }

    rank_transfer_task_buffer_reset(out);

    for (int i = 0; i < transfer_plan.total_tasks; ++i) {
        const TransferTask& transfer = transfer_plan.tasks[i];

        if (transfer.executor_rank != ctx.current_rank) {
            continue;
        }

        if (!transfer_task_has_work(transfer)) {
            return false;
        }

        if (transfer_task_is_barrier(transfer)) {
            if (!rank_transfer_task_buffer_push(out, transfer)) {
                return false;
            }

            continue;
        }

        if (transfer_task_is_ready(transfer)) {
            if (!ctx.lower_ready_tasks) {
                continue;
            }

            if (!rank_transfer_task_buffer_push(out, transfer)) {
                return false;
            }

            continue;
        }

        if (!transfer_task_is_windowed(transfer)) {
            return false;
        }

        if (!rank_transfer_task_buffer_push(out, transfer)) {
            return false;
        }
    }

    return recompute_rank_transfer_task_stats(out);
}

/*
 * Transfer-task pass skeletons.
 *
 * These run before pointer binding. They are the right place for logical IR
 * rewrites: ready cleanup, adjacent task merging, safe reordering by phase,
 * topology/staging rewrites, etc.
 */
template <int MaxTransferTasks, int MaxRanks>
inline bool pass_validate_transfer_tasks(
    const LoweringContext<MaxRanks>& ctx,
    const RankTransferTaskBuffer<MaxTransferTasks>& tasks) {
    if (tasks.count < 0 || tasks.count > MaxTransferTasks) {
        return false;
    }

    for (int i = 0; i < tasks.count; ++i) {
        const TransferTask& transfer = tasks.tasks[i];

        if (transfer.executor_rank != ctx.current_rank ||
            !transfer_task_has_work(transfer)) {
            return false;
        }

        if (transfer_task_is_barrier(transfer)) {
            continue;
        }

        if (transfer_task_is_ready(transfer)) {
            if (!ctx.lower_ready_tasks) {
                return false;
            }
            continue;
        }

        if (!transfer_task_is_windowed(transfer)) {
            return false;
        }
    }

    return true;
}

template <int MaxTransferTasks, int MaxRanks>
inline bool pass_drop_disabled_ready_tasks(
    const LoweringContext<MaxRanks>& ctx,
    RankTransferTaskBuffer<MaxTransferTasks>* tasks) {
    if (tasks == nullptr) {
        return false;
    }

    if (ctx.lower_ready_tasks) {
        return true;
    }

    int write = 0;
    for (int read = 0; read < tasks->count; ++read) {
        const TransferTask& transfer = tasks->tasks[read];
        if (transfer_task_is_ready(transfer)) {
            continue;
        }
        tasks->tasks[write++] = transfer;
    }

    /*
     * Do not clear stale tail entries. tasks->count is the only valid length,
     * and downstream code never reads tasks[write..old_count).
     */
    tasks->count = write;
    return recompute_rank_transfer_task_stats(tasks);
}

template <int MaxTransferTasks, int MaxRanks>
inline bool pass_drop_hardcoded_entry_ready_tasks(
    const LoweringContext<MaxRanks>& ctx,
    RankTransferTaskBuffer<MaxTransferTasks>* tasks) {
    if (tasks == nullptr) {
        return false;
    }

    if (!ctx.lower_ready_tasks || tasks->count <= 0) {
        return true;
    }

    const int device_channel =
        static_cast<int>(ReadySignalChannel::DeviceMemory);

    /*
     * OOVERLAP_HARDCODED_ENTRY_READY_RENDEZVOUS_PATCH:
     *
     * The executor kernel now performs the complete phase-0 DeviceMemory entry
     * rendezvous before execute_window_task_stripe(): one local publish followed
     * by waits for every peer in ready_plan. Remove every matching task from the
     * initial contiguous ready-task prefix, not just the first publish/wait pair.
     *
     * Preserve HostMapped tasks, nonzero phases, and all ready tasks that occur
     * after the entry prefix. Those may express staged/island dependencies that
     * are not handled by the kernel's direct-device entry rendezvous.
     */
    int write = 0;
    bool in_entry_ready_prefix = true;

    for (int read = 0; read < tasks->count; ++read) {
        const TransferTask& transfer = tasks->tasks[read];

        if (in_entry_ready_prefix && transfer_task_is_ready(transfer)) {
            const bool handled_by_kernel_entry_rendezvous =
                transfer.executor_rank == ctx.current_rank &&
                transfer.ready_phase == 0 &&
                transfer.ready_channel == device_channel &&
                (transfer.op == TransferOp::ReadyPublish ||
                 transfer.op == TransferOp::ReadyWait);

            if (handled_by_kernel_entry_rendezvous) {
                continue;
            }
        } else {
            in_entry_ready_prefix = false;
        }

        tasks->tasks[write++] = transfer;
    }

    tasks->count = write;

    return recompute_rank_transfer_task_stats(tasks);
}

template <int MaxTransferTasks, int MaxRanks>
inline bool pass_placeholder_optimize_transfer_tasks(
    const LoweringContext<MaxRanks>& /* ctx */,
    RankTransferTaskBuffer<MaxTransferTasks>* tasks) {
    /*
     * TODO examples:
     * - remove redundant ReadyWait/ReadyPublish pairs
     * - coalesce adjacent Copy tasks with same src/dst/transport/caps
     * - coalesce adjacent Reduce tasks with same src/dst/transport/caps
     * - split or reshape staging tasks
     * - reorder independent tasks by phase after dependency proof
     */
    return tasks != nullptr;
}

template <int MaxTransferTasks, int MaxRanks>
inline bool run_transfer_task_lowering_passes(
    const LoweringContext<MaxRanks>& ctx,
    RankTransferTaskBuffer<MaxTransferTasks>* tasks) {
    if (tasks == nullptr) {
        return false;
    }

    if (!ctx.options.enable_transfer_passes) {
        return recompute_rank_transfer_task_stats(tasks);
    }

    if (!pass_validate_transfer_tasks(ctx, *tasks)) {
        return false;
    }

    if (!pass_drop_disabled_ready_tasks(ctx, tasks)) {
        return false;
    }

    if (!pass_drop_hardcoded_entry_ready_tasks(ctx, tasks)) {
        return false;
    }

    if (!pass_placeholder_optimize_transfer_tasks(ctx, tasks)) {
        return false;
    }

    if (!recompute_rank_transfer_task_stats(tasks)) {
        return false;
    }

    return pass_validate_transfer_tasks(ctx, *tasks);
}

struct LoweringShape {
    int tasks_per_cta = 0;
    int capacity_tasks_per_cta = 0;
    int max_ctas_by_plan = 0;
    int cta_count = 0;
    int total_window_tasks = 0;
    int max_end_window = 0;
};


struct ReduceCtaGroup {
    task::WindowTaskCtaMask mask = 0;
    int first_cta = 0;
    int cta_count = 0;
};

inline task::WindowTaskCtaMask make_contiguous_cta_mask(
    int first_cta,
    int cta_count) {
    if (first_cta < 0 ||
        cta_count <= 0 ||
        first_cta >= task::kWindowTaskMaxCtas) {
        return 0;
    }

    const int end_cta =
        comm::utils::min_int(
            first_cta + cta_count,
            task::kWindowTaskMaxCtas);

    task::WindowTaskCtaMask mask = 0;

    for (int cta = first_cta; cta < end_cta; ++cta) {
        mask |=
            task::WindowTaskCtaMask{1}
            << static_cast<unsigned int>(cta);
    }

    return mask;
}

inline ReduceCtaGroup reduce_cta_group_for_task(
    int reduce_task_index,
    int launched_ctas,
    int max_ctas_per_reduce_task) {
    ReduceCtaGroup group{};

    if (reduce_task_index < 0 ||
        launched_ctas <= 0 ||
        max_ctas_per_reduce_task <= 0) {
        return group;
    }

    const int usable_ctas =
        comm::utils::min_int(
            launched_ctas,
            task::kWindowTaskMaxCtas);

    const int group_size =
        comm::utils::min_int(
            max_ctas_per_reduce_task,
            usable_ctas);

    const int group_count =
        comm::utils::ceil_div_int(
            usable_ctas,
            group_size);

    const int group_index =
        reduce_task_index % group_count;

    group.first_cta = group_index * group_size;
    group.cta_count =
        comm::utils::min_int(
            group_size,
            usable_ctas - group.first_cta);
    group.mask =
        make_contiguous_cta_mask(
            group.first_cta,
            group.cta_count);

    return group;
}

template <int MaxTransferTasks, int MaxRanks>
inline bool barrier_follows_full_cta_reduce(
    const RankTransferTaskBuffer<MaxTransferTasks>& tasks,
    int barrier_task_index,
    const LoweringContext<MaxRanks>& ctx,
    int launched_ctas) {
    int producer_index = barrier_task_index - 1;

    while (producer_index >= 0 &&
           transfer_task_is_ready(tasks.tasks[producer_index])) {
        --producer_index;
    }

    if (producer_index < 0) {
        return false;
    }

    const TransferTask& producer = tasks.tasks[producer_index];

    if (producer.op == TransferOp::ReduceFanout) {
        return true;
    }

    if (producer.op != TransferOp::Reduce) {
        return false;
    }

    if (!ctx.options.enable_reduce_cta_groups) {
        return true;
    }

    int reduce_task_index = 0;
    for (int i = 0; i < producer_index; ++i) {
        if (tasks.tasks[i].op == TransferOp::Reduce) {
            ++reduce_task_index;
        }
    }

    const ReduceCtaGroup group =
        reduce_cta_group_for_task(
            reduce_task_index,
            launched_ctas,
            ctx.options.max_ctas_per_reduce_task);

    return group.first_cta == 0 && group.cta_count == launched_ctas;
}

template <int MaxTransferTasks, int MaxWindowTasks, int MaxRanks>
inline bool compute_lowering_shape(
    const LoweringContext<MaxRanks>& ctx,
    const RankTransferTaskBuffer<MaxTransferTasks>& tasks,
    LoweringShape* out) {
    if (out == nullptr || ctx.launch_config == nullptr) {
        return false;
    }

    LoweringShape shape{};
    shape.max_end_window = tasks.max_end_window;

    if (tasks.ready_count == 0 &&
        tasks.barrier_count == 0 &&
        tasks.window_count == 0) {
        *out = shape;
        return true;
    }

    const int merged_ready_task_savings =
        ctx.options.enable_ready_publish_wait_merge
            ? tasks.ready_pair_count
            : 0;

    shape.tasks_per_cta =
        tasks.ready_count +
        tasks.barrier_count +
        tasks.window_count -
        merged_ready_task_savings;

    if (shape.tasks_per_cta <= 0 || shape.tasks_per_cta > MaxWindowTasks) {
        return false;
    }

    /*
     * reserved_prefix_tasks_per_cta is kept only for compatibility with the
     * temporary launcher-side ready prepend path. New code should pass zero
     * here and use ready_binding.
     */
    shape.capacity_tasks_per_cta =
        shape.tasks_per_cta + ctx.reserved_prefix_tasks_per_cta;

    if (shape.capacity_tasks_per_cta <= 0 ||
        shape.capacity_tasks_per_cta > MaxWindowTasks) {
        return false;
    }

    shape.max_ctas_by_plan =
        MaxWindowTasks / shape.capacity_tasks_per_cta;

    if (shape.max_ctas_by_plan <= 0) {
        return false;
    }

    if (tasks.window_count > 0) {
        shape.cta_count =
            comm::utils::cta_count_for_windows(
                tasks.max_end_window,
                ctx.launch_config->max_ctas);
    } else {
        /* Sync-only plan. Mostly useful for tests. */
        shape.cta_count = 1;
    }

    shape.cta_count =
        comm::utils::min_int(
            shape.cta_count,
            shape.max_ctas_by_plan);

    if (ctx.options.enable_reduce_cta_groups) {
        shape.cta_count =
            comm::utils::min_int(
                shape.cta_count,
                task::kWindowTaskMaxCtas);
    }

    if (shape.cta_count <= 0) {
        *out = shape;
        return true;
    }

    shape.total_window_tasks = shape.cta_count * shape.tasks_per_cta;

    if (shape.total_window_tasks > MaxWindowTasks) {
        return false;
    }

    *out = shape;
    return true;
}

template <int MaxTransferTasks, int MaxWindowTasks, int MaxRanks, int MaxStagingSlots>
inline bool emit_window_plan_from_rank_tasks(
    const RankTransferTaskBuffer<MaxTransferTasks>& rank_tasks,
    const RankPointerBinding<MaxRanks, MaxStagingSlots>& binding,
    const LoweringContext<MaxRanks>& ctx,
    const LoweringShape& shape,
    WindowTaskExecutorPlan<MaxWindowTasks>* out_window_plan,
    int* out_num_blocks) {
    if (out_window_plan == nullptr || out_num_blocks == nullptr) {
        return false;
    }

    out_window_plan->total_tasks = 0;
    out_window_plan->tasks_per_cta = 0;
    *out_num_blocks = 0;

    if (shape.cta_count <= 0 || shape.tasks_per_cta <= 0) {
        return true;
    }

    if (shape.total_window_tasks > MaxWindowTasks) {
        return false;
    }

    const comm::utils::WindowRange full_range{0, shape.max_end_window};

    for (int cta_idx = 0; cta_idx < shape.cta_count; ++cta_idx) {
        const comm::utils::WindowRange cta_range =
            rank_tasks.window_count > 0
                ? comm::utils::cta_window_range(
                      cta_idx,
                      shape.cta_count,
                      full_range)
                : comm::utils::WindowRange{0, 0};

        int task_idx = cta_idx * shape.tasks_per_cta;
        int ready_pair_index = 0;
        int reduce_task_index = 0;
        int active_barrier_index = 0;

        for (int i = 0; i < rank_tasks.count; ++i) {
            const TransferTask& transfer = rank_tasks.tasks[i];

            if (transfer_task_is_barrier(transfer)) {
                if (barrier_follows_full_cta_reduce(
                        rank_tasks,
                        i,
                        ctx,
                        shape.cta_count)) {
                    out_window_plan->tasks[task_idx++] = task::WindowTask{};
                    continue;
                }

                const unsigned int barrier_target =
                    ctx.options.cta_barrier_start +
                    static_cast<unsigned int>(
                        (active_barrier_index + 1) * shape.cta_count);

                out_window_plan->tasks[task_idx++] =
                    task::make_barrier_task(barrier_target);
                ++active_barrier_index;
                continue;
            }

            if (transfer_task_is_ready(transfer)) {
                if (!ctx.lower_ready_tasks || ctx.ready_binding == nullptr) {
                    continue;
                }

                if (ctx.options.enable_ready_publish_wait_merge &&
                    i + 1 < rank_tasks.count &&
                    transfer_tasks_are_mergeable_ready_publish_wait_pair(
                        rank_tasks.tasks[i],
                        rank_tasks.tasks[i + 1])) {
                    const int owner_cta =
                        shape.cta_count > 0
                            ? (ready_pair_index % shape.cta_count)
                            : 0;
                
                    task::WindowTask ready_task{};
                
                    if (cta_idx == owner_cta) {
                        if (!lower_ready_publish_wait_transfer_tasks_to_window_task(
                                rank_tasks.tasks[i],
                                rank_tasks.tasks[i + 1],
                                *ctx.ready_binding,
                                owner_cta,
                                &ready_task)) {
                            out_window_plan->total_tasks = 0;
                            out_window_plan->tasks_per_cta = 0;
                            *out_num_blocks = 0;
                            return false;
                        }
                    } else {
                        /*
                         * Non-owner CTAs only wait on the peer signal. They must not publish
                         * to the same local ready slot.
                         */
                        if (!lower_ready_transfer_task_to_window_task(
                                rank_tasks.tasks[i + 1],
                                *ctx.ready_binding,
                                &ready_task)) {
                            out_window_plan->total_tasks = 0;
                            out_window_plan->tasks_per_cta = 0;
                            *out_num_blocks = 0;
                            return false;
                        }
                    }
                
                    out_window_plan->tasks[task_idx++] = ready_task;
                    ++ready_pair_index;
                    ++i;
                    continue;
                } 

                task::WindowTask ready_task{};

                if (!lower_ready_transfer_task_to_window_task(
                        transfer,
                        *ctx.ready_binding,
                        &ready_task)) {
                    out_window_plan->total_tasks = 0;
                    out_window_plan->tasks_per_cta = 0;
                    *out_num_blocks = 0;
                    return false;
                }

                out_window_plan->tasks[task_idx++] = ready_task;
                continue;
            }

            comm::utils::WindowRange task_cta_range = cta_range;
            task::WindowTaskCtaMask cta_mask = task::kWindowTaskAllCtas;

            if (ctx.options.enable_reduce_cta_groups &&
                transfer.op == TransferOp::Reduce) {
                const ReduceCtaGroup group =
                    reduce_cta_group_for_task(
                        reduce_task_index++,
                        shape.cta_count,
                        ctx.options.max_ctas_per_reduce_task);

                cta_mask = group.mask;

                if (cta_idx < group.first_cta ||
                    cta_idx >= group.first_cta + group.cta_count) {
                    task::WindowTask skipped_task{};
                    skipped_task.cta_mask = cta_mask;
                    out_window_plan->tasks[task_idx++] = skipped_task;
                    continue;
                }

                task_cta_range =
                    comm::utils::cta_window_range(
                        cta_idx - group.first_cta,
                        group.cta_count,
                        comm::utils::WindowRange{
                            transfer.begin_window,
                            transfer.end_window});
            }

            const int begin_window =
                comm::utils::max_int(
                    transfer.begin_window,
                    task_cta_range.begin);

            const int end_window =
                comm::utils::min_int(
                    transfer.end_window,
                    task_cta_range.end);

            if (begin_window >= end_window) {
                /*
                 * Keep task striping rectangular. A no-op task preserves
                 * tasks_per_cta indexing.
                 */
                task::WindowTask no_work_task{};
                no_work_task.cta_mask = cta_mask;
                out_window_plan->tasks[task_idx++] = no_work_task;
                continue;
            }

            const void* src =
                resolve_logical_const_ptr(
                    transfer.src,
                    binding);

            /*
             * OOVERLAP_FANOUT_EMIT_WINDOW_PLAN_PATCH:
             *
             * Fanout tasks resolve one source and then resolve every logical
             * fanout destination into this rank's pointer view.
             */
            if (transfer.op == TransferOp::CopyFanout ||
                transfer.op == TransferOp::ReduceFanout) {
                if (src == nullptr ||
                    transfer.fanout_dst_count <= 0 ||
                    transfer.fanout_dst_count > TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS) {
                    out_window_plan->total_tasks = 0;
                    out_window_plan->tasks_per_cta = 0;
                    *out_num_blocks = 0;
                    return false;
                }

                void* fanout_dsts[TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS] = {};

                for (int fanout_idx = 0;
                     fanout_idx < transfer.fanout_dst_count;
                     ++fanout_idx) {
                    fanout_dsts[fanout_idx] =
                        resolve_logical_mut_ptr(
                            transfer.fanout_dsts[fanout_idx],
                            binding);

                    if (fanout_dsts[fanout_idx] == nullptr) {
                        out_window_plan->total_tasks = 0;
                        out_window_plan->tasks_per_cta = 0;
                        *out_num_blocks = 0;
                        return false;
                    }
                }

                task::WindowTask window_task{};

                if (!lower_fanout_transfer_task_to_window_task(
                        transfer,
                        src,
                        fanout_dsts,
                        begin_window,
                        end_window,
                        &window_task)) {
                    out_window_plan->total_tasks = 0;
                    out_window_plan->tasks_per_cta = 0;
                    *out_num_blocks = 0;
                    return false;
                }

                window_task.cta_mask = cta_mask;
                out_window_plan->tasks[task_idx++] = window_task;
                continue;
            }

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

            window_task.cta_mask = cta_mask;
            out_window_plan->tasks[task_idx++] = window_task;
        }
    }

    out_window_plan->tasks_per_cta = shape.tasks_per_cta;
    out_window_plan->total_tasks = shape.total_window_tasks;
    *out_num_blocks = shape.cta_count;

    return true;
}

/*
 * Window-plan pass skeletons.
 *
 * These run after pointer binding and final WindowTask selection. They are the
 * right place for low-level peephole passes, stripe-shape checks, and future
 * signal-variant selection.
 */
template <int MaxWindowTasks, int MaxRanks>
inline bool pass_validate_window_plan_shape(
    const LoweringContext<MaxRanks>& /* ctx */,
    const WindowTaskExecutorPlan<MaxWindowTasks>& plan,
    int num_blocks) {
    if (num_blocks < 0 ||
        plan.total_tasks < 0 ||
        plan.total_tasks > MaxWindowTasks ||
        plan.tasks_per_cta < 0) {
        return false;
    }

    if (num_blocks == 0) {
        return plan.total_tasks == 0 && plan.tasks_per_cta == 0;
    }

    if (plan.tasks_per_cta <= 0) {
        return false;
    }

    return plan.total_tasks == num_blocks * plan.tasks_per_cta;
}

template <int MaxWindowTasks, int MaxRanks>
inline bool pass_validate_executor_task_limit(
    const LoweringContext<MaxRanks>& ctx,
    const WindowTaskExecutorPlan<MaxWindowTasks>& plan) {
    if (!ctx.options.enforce_executor_task_limit) {
        return true;
    }

    return plan.tasks_per_cta <= ctx.options.max_executable_tasks_per_cta;
}

template <int MaxWindowTasks, int MaxRanks>
inline bool pass_placeholder_optimize_window_plan(
    const LoweringContext<MaxRanks>& /* ctx */,
    WindowTaskExecutorPlan<MaxWindowTasks>* plan,
    int* num_blocks) {
    /*
     * TODO examples:
     * - select CopyTMASignal / CopyFastAfterSignal / ReduceTMAAfterSignal
     * - merge adjacent WindowTasks inside each CTA stripe
     * - remove redundant ReadyWait/ReadyPublish after lowering
     * - validate terminal task position
     * - compact stripes only if executor contract changes accordingly
     */
    return plan != nullptr && num_blocks != nullptr;
}

template <int MaxWindowTasks, int MaxRanks>
inline bool run_window_plan_lowering_passes(
    const LoweringContext<MaxRanks>& ctx,
    WindowTaskExecutorPlan<MaxWindowTasks>* plan,
    int* num_blocks) {
    if (plan == nullptr || num_blocks == nullptr) {
        return false;
    }

    if (!ctx.options.enable_window_passes) {
        return pass_validate_window_plan_shape(ctx, *plan, *num_blocks);
    }

    if (!pass_validate_window_plan_shape(ctx, *plan, *num_blocks)) {
        return false;
    }

    if (!pass_validate_executor_task_limit(ctx, *plan)) {
        return false;
    }

    if (!pass_placeholder_optimize_window_plan(ctx, plan, num_blocks)) {
        return false;
    }

    return pass_validate_window_plan_shape(ctx, *plan, *num_blocks);
}

/* OOVERLAP_READY_PUBLISH_WAIT_MERGE_PATCH: lowering changes active. */
} // namespace lowering_detail

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
    const ReadySignalBinding<MaxRanks>* ready_binding = nullptr,
    const lowering_detail::LoweringPassOptions& pass_options =
        lowering_detail::LoweringPassOptions{}) {
    if (out_window_plan == nullptr || out_num_blocks == nullptr) {
        return false;
    }

    out_window_plan->total_tasks = 0;
    out_window_plan->tasks_per_cta = 0;
    *out_num_blocks = 0;

    lowering_detail::LoweringContext<MaxRanks> ctx{};

    if (!lowering_detail::make_lowering_context(
            binding,
            launch_config,
            reserved_prefix_tasks_per_cta,
            ready_binding,
            pass_options,
            &ctx)) {
        return false;
    }

    thread_local static lowering_detail::RankTransferTaskBuffer<MaxTransferTasks> rank_tasks;
    lowering_detail::rank_transfer_task_buffer_reset(&rank_tasks);

    if (!lowering_detail::collect_rank_transfer_tasks(
            transfer_plan,
            ctx,
            &rank_tasks)) {
        return false;
    }

    if (!lowering_detail::run_transfer_task_lowering_passes(
            ctx,
            &rank_tasks)) {
        return false;
    }

    lowering_detail::LoweringShape shape{};

    if (!lowering_detail::compute_lowering_shape<
            MaxTransferTasks,
            MaxWindowTasks,
            MaxRanks>(
                ctx,
                rank_tasks,
                &shape)) {
        return false;
    }

    if (shape.cta_count <= 0 || shape.tasks_per_cta <= 0) {
        return true;
    }

    if (!lowering_detail::emit_window_plan_from_rank_tasks<
            MaxTransferTasks,
            MaxWindowTasks,
            MaxRanks,
            MaxStagingSlots>(
                rank_tasks,
                binding,
                ctx,
                shape,
                out_window_plan,
                out_num_blocks)) {
        return false;
    }

    if (!lowering_detail::run_window_plan_lowering_passes(
            ctx,
            out_window_plan,
            out_num_blocks)) {
        out_window_plan->total_tasks = 0;
        out_window_plan->tasks_per_cta = 0;
        *out_num_blocks = 0;
        return false;
    }

    return true;
}

} // namespace plan
} // namespace comm
} // namespace ooverlap
