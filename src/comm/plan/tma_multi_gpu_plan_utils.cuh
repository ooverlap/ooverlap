#pragma once

#include "comm/plan/transfer_plan.h"
#include "comm/utils/utils.h"

#include <cstddef>
#include <cstdio>
#include <set>

//#define OOVERLAP_DEBUG_TRANSFER_PLAN 1

namespace ooverlap {
namespace comm {
namespace plan {

/*
 * Common logical-transfer planner utilities.
 *
 * Collective-specific transfer planners live in:
 *
 *   tma_multi_gpu_allreduce_plan.cuh
 *   tma_multi_gpu_reduce_scatter_plan.cuh
 *   tma_multi_gpu_all_gather_plan.cuh
 *
 * This file contains only the shared pointer-free planning helpers.
 */

constexpr int kPlannerMaxRanks = 16;
constexpr int kPlannerMaxStagingSlots = 4;

struct RankTopologyView {
    const topology::Topology* topology = nullptr;

    /*
     * rank_devices[rank] is the CUDA device ordinal for that rank.
     */
    const int* rank_devices = nullptr;
    int world_size = 0;
};

struct TransferPlanBuildInput {
    RankTopologyView topo{};

    comm::CollectivePlanFor collective = comm::CollectivePlanFor::AllReduce;
    comm::LaunchConfig launch_config{};

    int world_size = 0;

    /*
     * Element count for the collective call, before dtype_size scaling.
     */
    std::size_t count = 0;
    std::size_t dtype_size = 0;

    /*
     * Current public collectives are in-place over full logical buffers.
     */
    bool out_of_place = false;

    int staging_slot_count = 0;
    std::size_t staging_bytes[kPlannerMaxStagingSlots] = {};
    int staging_numa_nodes[kPlannerMaxStagingSlots] = {};
};

__host__ __device__ __forceinline__ bool valid_rank(
    int rank,
    int world_size) {
    return rank >= 0 && rank < world_size;
}

template <int MaxTransferTasks>
__host__ __device__ __forceinline__ void transfer_plan_reset_metadata(
    TransferPlan<MaxTransferTasks>* plan,
    int world_size) {
    plan->world_size = world_size;
    plan->total_tasks = 0;
}

template <int MaxTransferTasks>
__host__ __device__ __forceinline__ void transfer_plan_abort_build(
    TransferPlan<MaxTransferTasks>* plan) {
    if (plan != nullptr) {
        plan->world_size = 0;
        plan->total_tasks = 0;
    }
}

template <int MaxTransferTasks>
__host__ __device__ __forceinline__ bool transfer_plan_push_fast(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferTask& task) {
    if (plan->total_tasks >= MaxTransferTasks) {
        return false;
    }

    plan->tasks[plan->total_tasks++] = task;
    return true;
}

inline const topology::Link* find_topology_link_by_devices(
    const topology::Topology* topology,
    int src_device,
    int dst_device) {
    if (topology == nullptr) {
        return nullptr;
    }

    for (const topology::Link& link : topology->links) {
        if (link.src_device == src_device &&
            link.dst_device == dst_device) {
            return &link;
        }
    }

    return nullptr;
}

inline const topology::TransportInfo* find_transport_info(
    const topology::Link* link,
    topology::TransportKind kind) {
    if (link == nullptr) {
        return nullptr;
    }

    for (const topology::TransportInfo& transport : link->transports) {
        if (transport.kind == kind) {
            return &transport;
        }
    }

    return nullptr;
}

inline topology::TransportKind choose_direct_or_fallback_transport(
    const RankTopologyView& topo,
    int executor_rank,
    int target_rank) {
    if (!valid_rank(executor_rank, topo.world_size) ||
        !valid_rank(target_rank, topo.world_size) ||
        topo.rank_devices == nullptr) {
        return topology::TransportKind::DirectPcie;
    }

    if (executor_rank == target_rank) {
        return topology::TransportKind::DirectNvlink;
    }

    const int executor_device = topo.rank_devices[executor_rank];
    const int target_device = topo.rank_devices[target_rank];

    const topology::Link* link =
        find_topology_link_by_devices(
            topo.topology,
            executor_device,
            target_device);

    const topology::TransportInfo* nvlink =
        find_transport_info(
            link,
            topology::TransportKind::DirectNvlink);

    if (nvlink != nullptr && nvlink->available) {
        return topology::TransportKind::DirectNvlink;
    }

    const topology::TransportInfo* pcie =
        find_transport_info(
            link,
            topology::TransportKind::DirectPcie);

    if (pcie != nullptr && pcie->available) {
        return topology::TransportKind::DirectPcie;
    }

    const topology::TransportInfo* shm =
        find_transport_info(
            link,
            topology::TransportKind::Shm);

    if (shm != nullptr && shm->available) {
        return topology::TransportKind::Shm;
    }

    /*
     * Preserve old behavior when no topology is supplied: treat peers as direct.
     */
    return topology::TransportKind::DirectPcie;
}

inline ReadySignalChannel choose_ready_signal_channel(
    const RankTopologyView& topo,
    int waiter_rank,
    int owner_rank) {
    if (!valid_rank(waiter_rank, topo.world_size) ||
        !valid_rank(owner_rank, topo.world_size)) {
        return ReadySignalChannel::DeviceMemory;
    }

    if (waiter_rank == owner_rank) {
        return ReadySignalChannel::DeviceMemory;
    }

    const topology::TransportKind direct_or_fallback =
        choose_direct_or_fallback_transport(
            topo,
            waiter_rank,
            owner_rank);

    return direct_or_fallback == topology::TransportKind::DirectNvlink ||
           direct_or_fallback == topology::TransportKind::DirectPcie
               ? ReadySignalChannel::DeviceMemory
               : ReadySignalChannel::HostMapped;
}

struct TransportMatrix {
    topology::TransportKind kind[kPlannerMaxRanks][kPlannerMaxRanks] = {};
};

inline bool build_transport_matrix(
    const RankTopologyView& topo,
    int world_size,
    TransportMatrix* out) {
    if (out == nullptr ||
        world_size <= 0 ||
        world_size > kPlannerMaxRanks) {
        return false;
    }

    RankTopologyView fixed_topo = topo;
    fixed_topo.world_size = world_size;

    for (int src = 0; src < world_size; ++src) {
        for (int dst = 0; dst < world_size; ++dst) {
            out->kind[src][dst] =
                choose_direct_or_fallback_transport(
                    fixed_topo,
                    src,
                    dst);
        }
    }

    return true;
}

__host__ __device__ __forceinline__ bool compute_rank_slice_bytes_fast(
    std::size_t count,
    std::size_t dtype_size,
    int rank,
    int world_size,
    std::size_t* out_begin_bytes,
    std::size_t* out_slice_bytes) {
    const std::size_t world =
        static_cast<std::size_t>(world_size);
    const std::size_t r =
        static_cast<std::size_t>(rank);

    const std::size_t base =
        count / world;
    const std::size_t rem =
        count - base * world;

    const std::size_t begin_elems =
        r * base + ((r < rem) ? r : rem);
    const std::size_t slice_elems =
        base + ((r < rem) ? 1 : 0);

    *out_begin_bytes = begin_elems * dtype_size;
    *out_slice_bytes = slice_elems * dtype_size;
    return true;
}

inline int window_count_for_transfer_bytes(
    std::size_t bytes,
    const comm::LaunchConfig& launch_config) {
    return comm::utils::window_count_for_bytes(
        bytes,
        static_cast<std::size_t>(launch_config.chunk_bytes),
        launch_config.window_chunks);
}

__host__ __device__ __forceinline__ bool transport_uses_direct_tma(
    topology::TransportKind transport) {
    return transport == topology::TransportKind::DirectNvlink ||
           transport == topology::TransportKind::DirectPcie;
}

inline TransferTask make_copy_transfer_task(
    int executor_rank,
    int src_rank,
    int dst_rank,
    LogicalBufferRef src,
    LogicalBufferRef dst,
    std::size_t bytes,
    int num_windows,
    int window_chunks,
    topology::TransportKind transport,
    bool terminal,
    int phase) {
    const bool direct_tma =
        transport_uses_direct_tma(transport);

    TransferTask task{};
    task.op = TransferOp::Copy;
    task.executor_rank = executor_rank;
    task.src_rank = src_rank;
    task.dst_rank = dst_rank;
    task.src = src;
    task.dst = dst;
    task.bytes = bytes;
    task.begin_window = 0;
    task.end_window = num_windows;
    task.window_chunks = window_chunks;
    task.transport = transport;
    task.requires_tma_load = direct_tma;
    task.requires_tma_store = direct_tma;
    task.requires_tma_reduce = false;
    task.requires_native_atomic = false;
    task.terminal = terminal;
    task.phase = phase;
    return task;
}

inline TransferTask make_reduce_transfer_task(
    int executor_rank,
    int src_rank,
    int dst_rank,
    LogicalBufferRef src,
    LogicalBufferRef dst,
    std::size_t bytes,
    int num_windows,
    int window_chunks,
    topology::TransportKind transport,
    bool terminal,
    int phase) {
    const bool direct_tma =
        transport_uses_direct_tma(transport);

    TransferTask task{};
    task.op = TransferOp::Reduce;
    task.executor_rank = executor_rank;
    task.src_rank = src_rank;
    task.dst_rank = dst_rank;
    task.src = src;
    task.dst = dst;
    task.bytes = bytes;
    task.begin_window = 0;
    task.end_window = num_windows;
    task.window_chunks = window_chunks;
    task.transport = transport;
    task.requires_tma_load = direct_tma;
    task.requires_tma_store = false;
    task.requires_tma_reduce = direct_tma;
    task.requires_native_atomic = direct_tma;
    task.terminal = terminal;
    task.phase = phase;
    return task;
}



/*
 * OOVERLAP_ALLGATHER_COPY_FANOUT_PLANNER_V1
 *
 * Logical helper for one source rank that TMA-loads one input buffer and
 * copy-fanouts the tile into multiple destination buffers.
 */
inline TransferTask make_copy_fanout_transfer_task(
    int executor_rank,
    int src_rank,
    const int* dst_ranks,
    LogicalBufferRef src,
    const LogicalBufferRef* fanout_dsts,
    int fanout_dst_count,
    std::size_t bytes,
    int num_windows,
    int window_chunks,
    topology::TransportKind transport,
    bool terminal,
    int phase) {
    const bool direct_tma =
        transport_uses_direct_tma(transport);

    TransferTask task{};
    task.op = TransferOp::CopyFanout;
    task.executor_rank = executor_rank;
    task.src_rank = src_rank;
    task.dst_rank =
        (dst_ranks != nullptr && fanout_dst_count > 0)
            ? dst_ranks[0]
            : -1;
    task.src = src;
    task.bytes = bytes;
    task.begin_window = 0;
    task.end_window = num_windows;
    task.window_chunks = window_chunks;
    task.transport = transport;
    task.requires_tma_load = direct_tma;
    task.requires_tma_store = direct_tma;
    task.requires_tma_reduce = false;
    task.requires_native_atomic = false;
    task.terminal = terminal;
    task.phase = phase;
    task.fanout_dst_count = fanout_dst_count;

    const int clamped =
        fanout_dst_count < TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS
            ? fanout_dst_count
            : TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS;

    for (int i = 0; i < clamped; ++i) {
        task.fanout_dsts[i] = fanout_dsts[i];
        task.fanout_dst_rank[i] =
            dst_ranks != nullptr ? dst_ranks[i] : -1;
        task.fanout_reduce_scope[i] = 0;
    }

    return task;
}

/*
 * OOVERLAP_OUT_OF_PLACE_ALLREDUCE_REDUCE_FANOUT_PATCH
 *
 * Logical helper for one source rank that TMA-loads one input buffer and
 * reduce-fanouts the tile into multiple output buffers.
 */
inline TransferTask make_reduce_fanout_transfer_task(
    int executor_rank,
    int src_rank,
    const int* dst_ranks,
    LogicalBufferRef src,
    const LogicalBufferRef* fanout_dsts,
    int fanout_dst_count,
    std::size_t bytes,
    int num_windows,
    int window_chunks,
    topology::TransportKind transport,
    bool terminal,
    int phase) {
    const bool direct_tma =
        transport_uses_direct_tma(transport);

    TransferTask task{};
    task.op = TransferOp::ReduceFanout;
    task.executor_rank = executor_rank;
    task.src_rank = src_rank;
    task.dst_rank =
        (dst_ranks != nullptr && fanout_dst_count > 0)
            ? dst_ranks[0]
            : -1;
    task.src = src;
    task.bytes = bytes;
    task.begin_window = 0;
    task.end_window = num_windows;
    task.window_chunks = window_chunks;
    task.transport = transport;
    task.requires_tma_load = direct_tma;
    task.requires_tma_store = false;
    task.requires_tma_reduce = direct_tma;
    task.requires_native_atomic = direct_tma;
    task.terminal = terminal;
    task.phase = phase;
    task.fanout_dst_count = fanout_dst_count;

    const int clamped =
        fanout_dst_count < TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS
            ? fanout_dst_count
            : TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS;

    for (int i = 0; i < clamped; ++i) {
        task.fanout_dsts[i] = fanout_dsts[i];
        task.fanout_dst_rank[i] =
            dst_ranks != nullptr ? dst_ranks[i] : -1;
        task.fanout_reduce_scope[i] = 0;
    }

    return task;
}

inline TransferTask make_ready_publish_transfer_task(
    int executor_rank,
    ReadySignalChannel channel,
    int phase,
    int ready_phase) {
    TransferTask task{};
    task.op = TransferOp::ReadyPublish;
    task.executor_rank = executor_rank;
    task.ready_rank = executor_rank;
    task.ready_channel = static_cast<int>(channel);
    task.ready_phase = ready_phase;
    task.phase = phase;
    return task;
}

inline TransferTask make_ready_publish_transfer_task(
    int executor_rank,
    ReadySignalChannel channel,
    int phase) {
    return make_ready_publish_transfer_task(
        executor_rank,
        channel,
        phase,
        0);
}

inline TransferTask make_ready_publish_transfer_task(
    int executor_rank,
    int phase) {
    return make_ready_publish_transfer_task(
        executor_rank,
        ReadySignalChannel::DeviceMemory,
        phase,
        0);
}

inline TransferTask make_ready_wait_transfer_task(
    int executor_rank,
    int ready_rank,
    ReadySignalChannel channel,
    int phase,
    int ready_phase) {
    TransferTask task{};
    task.op = TransferOp::ReadyWait;
    task.executor_rank = executor_rank;
    task.ready_rank = ready_rank;
    task.ready_channel = static_cast<int>(channel);
    task.ready_phase = ready_phase;
    task.phase = phase;
    return task;
}

inline TransferTask make_ready_wait_transfer_task(
    int executor_rank,
    int ready_rank,
    ReadySignalChannel channel,
    int phase) {
    return make_ready_wait_transfer_task(
        executor_rank,
        ready_rank,
        channel,
        phase,
        0);
}

inline TransferTask make_ready_wait_transfer_task(
    int executor_rank,
    int ready_rank,
    int phase) {
    return make_ready_wait_transfer_task(
        executor_rank,
        ready_rank,
        ReadySignalChannel::DeviceMemory,
        phase,
        0);
}

template <int MaxTransferTasks>
inline bool append_ready_rendezvous_tasks(
    TransferPlan<MaxTransferTasks>* plan,
    const RankTopologyView& topo,
    int rank,
    int world_size,
    int* phase) {
    if (plan == nullptr ||
        phase == nullptr ||
        rank < 0 ||
        rank >= world_size ||
        world_size <= 1) {
        return true;
    }

    bool publish_channel_used[kReadySignalChannelCount] = {};

    for (int peer = 0; peer < world_size; ++peer) {
        if (peer == rank) {
            continue;
        }

        /*
         * Which channel will peer use when waiting on this rank?
         * This is directional and may differ from the channel this rank uses
         * when waiting on peer in asymmetric topologies.
         */
        const ReadySignalChannel publish_channel =
            choose_ready_signal_channel(
                topo,
                peer,
                rank);

        publish_channel_used[static_cast<int>(publish_channel)] = true;
    }

    for (int channel = 0; channel < kReadySignalChannelCount; ++channel) {
        if (!publish_channel_used[channel]) {
            continue;
        }

        const TransferTask publish =
            make_ready_publish_transfer_task(
                rank,
                static_cast<ReadySignalChannel>(channel),
                (*phase)++);

        if (!transfer_plan_push_fast(plan, publish)) {
            transfer_plan_abort_build(plan);
            return false;
        }
    }

    for (int peer = 0; peer < world_size; ++peer) {
        if (peer == rank) {
            continue;
        }

        const ReadySignalChannel wait_channel =
            choose_ready_signal_channel(
                topo,
                rank,
                peer);

        const TransferTask wait =
            make_ready_wait_transfer_task(
                rank,
                peer,
                wait_channel,
                (*phase)++);

        if (!transfer_plan_push_fast(plan, wait)) {
            transfer_plan_abort_build(plan);
            return false;
        }
    }

    return true;
}

template <int MaxTransferTasks>
inline bool append_ready_rendezvous_tasks(
    TransferPlan<MaxTransferTasks>* plan,
    int rank,
    int world_size,
    int* phase) {
    return append_ready_rendezvous_tasks(
        plan,
        RankTopologyView{},
        rank,
        world_size,
        phase);
}

inline bool valid_build_input(
    const TransferPlanBuildInput& input) {
    return input.world_size > 0 &&
           input.world_size <= kPlannerMaxRanks &&
           input.dtype_size != 0 &&
           input.launch_config.window_chunks > 0 &&
           input.launch_config.chunk_bytes > 0 &&
           input.staging_slot_count >= 0 &&
           input.staging_slot_count <= kPlannerMaxStagingSlots;
}

__host__ __device__ __forceinline__ bool staging_slice_valid(
    const TransferPlanBuildInput& input,
    int staging_slot,
    std::size_t byte_offset,
    std::size_t bytes) {
    if (staging_slot < 0 ||
        staging_slot >= input.staging_slot_count ||
        staging_slot >= kPlannerMaxStagingSlots ||
        bytes == 0) {
        return false;
    }

    const std::size_t slot_bytes =
        input.staging_bytes[staging_slot];

    if (slot_bytes == 0 ||
        byte_offset > slot_bytes) {
        return false;
    }

    return bytes <= slot_bytes - byte_offset;
}

__host__ __device__ __forceinline__ int staging_slot_numa_node(
    const TransferPlanBuildInput& input,
    int staging_slot) {
    if (staging_slot < 0 ||
        staging_slot >= input.staging_slot_count ||
        staging_slot >= kPlannerMaxStagingSlots) {
        return -1;
    }

    return input.staging_numa_nodes[staging_slot];
}

__host__ __device__ __forceinline__ int choose_staging_slot_for_numa(
    const TransferPlanBuildInput& input,
    int preferred_numa_node,
    std::size_t required_bytes) {
    if (input.staging_slot_count <= 0 ||
        required_bytes == 0) {
        return -1;
    }

    if (preferred_numa_node >= 0) {
        for (int slot = 0; slot < input.staging_slot_count; ++slot) {
            if (staging_slot_numa_node(input, slot) == preferred_numa_node &&
                staging_slice_valid(input, slot, 0, required_bytes)) {
                return slot;
            }
        }
    }

    for (int slot = 0; slot < input.staging_slot_count; ++slot) {
        if (staging_slice_valid(input, slot, 0, required_bytes)) {
            return slot;
        }
    }

    return -1;
}

inline const char* debug_transfer_op_name(
    TransferOp op) {
    switch (op) {
        case TransferOp::None:
            return "None";
        case TransferOp::Copy:
            return "Copy";
        case TransferOp::Reduce:
            return "Reduce";
        case TransferOp::ReadyPublish:
            return "ReadyPublish";
        case TransferOp::ReadyWait:
            return "ReadyWait";
        case TransferOp::CopyFanout:
            return "CopyFanout";
        case TransferOp::ReduceFanout:
            return "ReduceFanout";
        case TransferOp::Barrier:
            return "Barrier";
        default:
            return "UnknownTransferOp";
    }
}

inline const char* debug_logical_buffer_role_name(
    LogicalBufferRole role) {
    switch (role) {
        case LogicalBufferRole::RankBuffer:
            return "RankBuffer";
        case LogicalBufferRole::RankInput:
            return "RankInput";
        case LogicalBufferRole::RankOutput:
            return "RankOutput";
        case LogicalBufferRole::ShmStaging:
            return "ShmStaging";
        default:
            return "UnknownLogicalBufferRole";
    }
}

inline const char* debug_transport_kind_name(
    topology::TransportKind transport) {
    switch (transport) {
        case topology::TransportKind::DirectNvlink:
            return "DirectNvlink";
        case topology::TransportKind::DirectPcie:
            return "DirectPcie";
        case topology::TransportKind::Shm:
            return "Shm";
        default:
            return "UnknownTransport";
    }
}

inline const char* debug_ready_signal_channel_name(
    ReadySignalChannel channel) {
    switch (channel) {
        case ReadySignalChannel::DeviceMemory:
            return "DeviceMemory";
        case ReadySignalChannel::HostMapped:
            return "HostMapped";
        default:
            return "UnknownReadyChannel";
    }
}

inline const char* debug_ready_signal_channel_name(
    int channel) {
    if (channel < 0 || channel >= kReadySignalChannelCount) {
        return "InvalidReadyChannel";
    }

    return debug_ready_signal_channel_name(
        static_cast<ReadySignalChannel>(channel));
}

inline void debug_print_logical_buffer_ref(
    const char* label,
    const LogicalBufferRef& ref) {
    std::fprintf(
        stderr,
        "%s={role=%s(%d) owner_rank=%d staging_slot=%d byte_offset=%zu}",
        label != nullptr ? label : "ref",
        debug_logical_buffer_role_name(ref.role),
        static_cast<int>(ref.role),
        ref.owner_rank,
        ref.staging_slot,
        ref.byte_offset);
}

inline void debug_print_transfer_task(
    int task_idx,
    const TransferTask& task) {
    std::fprintf(
        stderr,
        "  task[%d]: op=%s(%d) executor=%d src_rank=%d dst_rank=%d "
        "bytes=%zu windows=[%d,%d) window_chunks=%d "
        "transport=%s(%d) caps={load=%d store=%d reduce=%d atomic=%d} "
        "terminal=%d phase=%d ready={rank=%d channel=%s(%d) phase=%d} ",
        task_idx,
        debug_transfer_op_name(task.op),
        static_cast<int>(task.op),
        task.executor_rank,
        task.src_rank,
        task.dst_rank,
        task.bytes,
        task.begin_window,
        task.end_window,
        task.window_chunks,
        debug_transport_kind_name(task.transport),
        static_cast<int>(task.transport),
        static_cast<int>(task.requires_tma_load),
        static_cast<int>(task.requires_tma_store),
        static_cast<int>(task.requires_tma_reduce),
        static_cast<int>(task.requires_native_atomic),
        static_cast<int>(task.terminal),
        task.phase,
        task.ready_rank,
        debug_ready_signal_channel_name(task.ready_channel),
        task.ready_channel,
        task.ready_phase);

    debug_print_logical_buffer_ref("src", task.src);
    std::fprintf(stderr, " ");
    debug_print_logical_buffer_ref("dst", task.dst);
    std::fprintf(stderr, "\n");
}

template <int MaxTransferTasks>
inline void debug_print_transfer_plan_for_rank(
    const char* tag,
    const TransferPlan<MaxTransferTasks>& plan,
    int rank) {
    std::fprintf(
        stderr,
        "\n[%s rank=%d] TransferPlan: world_size=%d total_tasks=%d max_tasks=%d\n",
        tag != nullptr ? tag : "transfer_plan",
        rank,
        plan.world_size,
        plan.total_tasks,
        MaxTransferTasks);

    if (plan.total_tasks < 0 || plan.total_tasks > MaxTransferTasks) {
        std::fprintf(
            stderr,
            "  invalid total_tasks=%d for max_tasks=%d\n\n",
            plan.total_tasks,
            MaxTransferTasks);
        return;
    }

    for (int i = 0; i < plan.total_tasks; ++i) {
        if (plan.tasks[i].executor_rank == rank) {
            debug_print_transfer_task(i, plan.tasks[i]);
        }
    }

    std::fprintf(stderr, "\n");
}

template <int MaxTransferTasks>
inline void debug_print_transfer_plan(
    const char* tag,
    const TransferPlan<MaxTransferTasks>& plan) {
    std::fprintf(
        stderr,
        "\n[%s] TransferPlan: world_size=%d total_tasks=%d max_tasks=%d\n",
        tag != nullptr ? tag : "transfer_plan",
        plan.world_size,
        plan.total_tasks,
        MaxTransferTasks);

    if (plan.total_tasks < 0 || plan.total_tasks > MaxTransferTasks) {
        std::fprintf(
            stderr,
            "  invalid total_tasks=%d for max_tasks=%d\n\n",
            plan.total_tasks,
            MaxTransferTasks);
        return;
    }

    std::set<int> unique_ranks;

    for (int i = 0; i < plan.total_tasks; ++i) {
        unique_ranks.insert(plan.tasks[i].executor_rank);
    }

    for (auto& it : unique_ranks) {
        debug_print_transfer_plan_for_rank(tag, plan, it);
    }

    std::fprintf(stderr, "\n");
}

template <int MaxTransferTasks>
inline void debug_print_transfer_plan_if_enabled(
    const char* tag,
    const TransferPlan<MaxTransferTasks>& plan) {
#if defined(OOVERLAP_DEBUG_TRANSFER_PLAN)
    debug_print_transfer_plan(tag, plan);
#else
    (void)tag;
    (void)plan;
#endif
}


} // namespace plan
} // namespace comm
} // namespace ooverlap
