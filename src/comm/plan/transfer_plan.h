#pragma once

#include "comm/launch_config.h"
#include "topology/topology.h"

#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {
namespace plan {

/*
 * Logical transfer task layer.
 *
 * This layer intentionally does not store raw CUDA pointers. A TransferTask
 * describes rank-level data movement/reduction intent. It is lowered later,
 * inside each rank/process, into WindowTask objects using that rank's local
 * pointer view.
 */

enum class TransferOp : std::uint8_t {
    None = 0,
    Copy = 1,
    Reduce = 2,

    ReadyPublish = 3,
    ReadyWait = 4,
};

/*
 * Logical ready-signal channel.
 *
 * TransferPlan stays pointer-free.  A channel only says which class of signal
 * must be used for a ReadyPublish/ReadyWait edge.  Lowering resolves the
 * channel to this rank's concrete pointer/protocol.
 */
enum class ReadySignalChannel : std::uint8_t {
    DeviceMemory = 0,
    HostMapped = 1,
};

constexpr int kReadySignalChannelCount = 2;

/*
 * ReadyPublish/ReadyWait tasks can be used multiple times inside one collective
 * epoch.  Lowering converts (collective_epoch, ready_phase) into a monotonically
 * increasing signal value:
 *
 *   ready_value = collective_epoch * kReadySignalPhaseStride + ready_phase
 *
 * Keep this larger than the number of synchronization phases a planner can emit
 * inside one collective.
 */
constexpr int kReadySignalPhaseStride = 1024;

enum class LogicalBufferRole : std::uint8_t {
    RankBuffer = 0,
    RankInput = 1,
    RankOutput = 2,
    ShmStaging = 3,
};

struct LogicalBufferRef {
    LogicalBufferRole role = LogicalBufferRole::RankBuffer;

    /*
     * For RankBuffer/RankInput/RankOutput this is the rank that owns the
     * logical buffer.
     *
     * For ShmStaging this can be left at -1; staging_slot selects the staging
     * allocation.
     */
    int owner_rank = -1;

    /*
     * Used only for LogicalBufferRole::ShmStaging.
     */
    int staging_slot = -1;

    /*
     * Byte offset into the logical buffer. This offset is interpreted only
     * during rank-local lowering, after the pointer for the buffer is resolved.
     */
    std::size_t byte_offset = 0;
};

__host__ __device__ __forceinline__ LogicalBufferRef rank_buffer_ref(
    int rank,
    std::size_t byte_offset = 0) {
    LogicalBufferRef ref{};
    ref.role = LogicalBufferRole::RankBuffer;
    ref.owner_rank = rank;
    ref.byte_offset = byte_offset;
    return ref;
}

__host__ __device__ __forceinline__ LogicalBufferRef rank_input_ref(
    int rank,
    std::size_t byte_offset = 0) {
    LogicalBufferRef ref{};
    ref.role = LogicalBufferRole::RankInput;
    ref.owner_rank = rank;
    ref.byte_offset = byte_offset;
    return ref;
}

__host__ __device__ __forceinline__ LogicalBufferRef rank_output_ref(
    int rank,
    std::size_t byte_offset = 0) {
    LogicalBufferRef ref{};
    ref.role = LogicalBufferRole::RankOutput;
    ref.owner_rank = rank;
    ref.byte_offset = byte_offset;
    return ref;
}

__host__ __device__ __forceinline__ LogicalBufferRef shm_staging_ref(
    int staging_slot,
    std::size_t byte_offset = 0) {
    LogicalBufferRef ref{};
    ref.role = LogicalBufferRole::ShmStaging;
    ref.owner_rank = -1;
    ref.staging_slot = staging_slot;
    ref.byte_offset = byte_offset;
    return ref;
}

struct TransferTask {
    TransferOp op = TransferOp::None;

    /*
     * The rank whose CUDA kernel will execute this transfer.
     */
    int executor_rank = -1;

    /*
     * Semantic source/destination ranks. These are used for topology decisions
     * and debugging. The actual buffer references are src/dst below.
     */
    int src_rank = -1;
    int dst_rank = -1;

    LogicalBufferRef src{};
    LogicalBufferRef dst{};

    /*
     * Number of bytes in the logical transfer after src/dst byte_offset.
     */
    std::size_t bytes = 0;

    /*
     * Windowing parameters for the logical transfer. Most planners should emit
     * begin_window=0 and end_window=num_windows for the whole transfer; lowering
     * splits this range over CTAs.
     */
    int begin_window = 0;
    int end_window = 0;
    int window_chunks = 0;

    /*
     * Transport selected by the logical topology planner.
     */
    topology::TransportKind transport = topology::TransportKind::DirectNvlink;

    /*
     * Capability requirements. The topology planner sets these; the lowerer
     * uses them to choose WindowTaskOp or reject unsupported lowering.
     */
    bool requires_tma_load = false;
    bool requires_tma_store = false;
    bool requires_tma_reduce = false;
    bool requires_native_atomic = false;

    /*
     * Preserved for the current executor model. This means the same thing as
     * WindowTask::terminal after lowering.
     */
    bool terminal = false;

    /*
     * Optional ordering/algorithm phase. The current lowering preserves task
     * order as emitted. This field is for later cross-transport scheduling.
     */
    int phase = 0;

    /*
     * Ready task fields.  ready_rank is the owner rank of the signal.  For
     * ReadyPublish it is normally executor_rank. ready_channel is a
     * ReadySignalChannel integer.
     */
    int ready_rank = -1;
    int ready_channel = static_cast<int>(ReadySignalChannel::DeviceMemory);

    /*
     * Synchronization phase within the collective epoch.
     *
     * ready_phase=0 is the entry rendezvous. Later planner phases can use
     * ready_phase=1,2,... for island-complete, staging-complete, etc. This is
     * separate from task.phase, which is only local task ordering.
     */
    int ready_phase = 0;
};

template <int MaxTransferTasks>
struct TransferPlan {
    static_assert(MaxTransferTasks > 0, "MaxTransferTasks must be > 0");

    int world_size = 0;
    int total_tasks = 0;
    TransferTask tasks[MaxTransferTasks] = {};
};

template <int MaxTransferTasks>
__host__ __device__ __forceinline__ void transfer_plan_clear(
    TransferPlan<MaxTransferTasks>* plan) {
    if (plan == nullptr) {
        return;
    }

    plan->world_size = 0;
    plan->total_tasks = 0;

    for (int i = 0; i < MaxTransferTasks; ++i) {
        plan->tasks[i] = TransferTask{};
    }
}

template <int MaxTransferTasks>
__host__ __device__ __forceinline__ bool transfer_plan_push(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferTask& task) {
    if (plan == nullptr ||
        plan->total_tasks < 0 ||
        plan->total_tasks >= MaxTransferTasks) {
        return false;
    }

    plan->tasks[plan->total_tasks++] = task;
    return true;
}

__host__ __device__ __forceinline__ bool transfer_task_has_work(
    const TransferTask& task) {
    if (task.op == TransferOp::ReadyPublish ||
        task.op == TransferOp::ReadyWait) {
        return task.executor_rank >= 0 &&
               task.ready_rank >= 0 &&
               task.ready_channel >= 0 &&
               task.ready_channel < kReadySignalChannelCount &&
               task.ready_phase >= 0 &&
               task.ready_phase < kReadySignalPhaseStride;
    }

    return task.op != TransferOp::None &&
           task.executor_rank >= 0 &&
           task.bytes != 0 &&
           task.window_chunks > 0 &&
           task.begin_window < task.end_window;
}

} // namespace plan
} // namespace comm
} // namespace ooverlap
