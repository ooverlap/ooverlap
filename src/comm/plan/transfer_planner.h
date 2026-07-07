#pragma once

#include "comm/plan/transfer_plan.h"
#include "comm/utils/utils.h"

#include <cstddef>

namespace ooverlap {
namespace comm {
namespace plan {

/*
 * Topology/logical planning layer.
 *
 * This is kept header-only because the concrete TransferPlan size is a template
 * parameter.  The build path is optimized for the benchmark/research case:
 *
 *   - no full TransferTask array clear on build
 *   - no rank_partition helper call per rank
 *   - no repeated topology link scan per emitted task
 *   - no heavy validation inside the inner loops
 */

constexpr int kPlannerMaxRanks = 16;

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

inline TransferTask make_ready_publish_transfer_task(
    int executor_rank,
    int phase) {
    TransferTask task{};
    task.op = TransferOp::ReadyPublish;
    task.executor_rank = executor_rank;
    task.ready_rank = executor_rank;
    task.phase = phase;
    return task;
}

inline TransferTask make_ready_wait_transfer_task(
    int executor_rank,
    int ready_rank,
    int phase) {
    TransferTask task{};
    task.op = TransferOp::ReadyWait;
    task.executor_rank = executor_rank;
    task.ready_rank = ready_rank;
    task.phase = phase;
    return task;
}

template <int MaxTransferTasks>
inline bool append_ready_rendezvous_tasks(
    TransferPlan<MaxTransferTasks>* plan,
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

    const TransferTask publish =
        make_ready_publish_transfer_task(
            rank,
            (*phase)++);

    if (!transfer_plan_push_fast(plan, publish)) {
        transfer_plan_abort_build(plan);
        return false;
    }

    for (int peer = 0; peer < world_size; ++peer) {
        if (peer == rank) {
            continue;
        }

        const TransferTask wait =
            make_ready_wait_transfer_task(
                rank,
                peer,
                (*phase)++);

        if (!transfer_plan_push_fast(plan, wait)) {
            transfer_plan_abort_build(plan);
            return false;
        }
    }

    return true;
}

inline bool valid_build_input(
    const TransferPlanBuildInput& input) {
    return input.world_size > 0 &&
           input.world_size <= kPlannerMaxRanks &&
           input.dtype_size != 0 &&
           input.launch_config.window_chunks > 0 &&
           input.launch_config.chunk_bytes > 0;
}

/*
 * First-pass logical allreduce planner.
 *
 * Logical shape:
 *   optional local input copy
 *   reduce every peer's rank slice into executor rank's buffer
 *   copy executor rank's result slice back to every peer
 */
template <int MaxTransferTasks>
bool build_allreduce_transfer_plan(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input) {
    if (plan == nullptr || !valid_build_input(input)) {
        return false;
    }

    transfer_plan_reset_metadata(plan, input.world_size);

    TransportMatrix transports{};
    if (!build_transport_matrix(
            input.topo,
            input.world_size,
            &transports)) {
        transfer_plan_abort_build(plan);
        return false;
    }

    for (int rank = 0; rank < input.world_size; ++rank) {
        std::size_t slice_begin_bytes = 0;
        std::size_t slice_bytes = 0;

        compute_rank_slice_bytes_fast(
            input.count,
            input.dtype_size,
            rank,
            input.world_size,
            &slice_begin_bytes,
            &slice_bytes);

        const int num_windows =
            window_count_for_transfer_bytes(
                slice_bytes,
                input.launch_config);

        if (slice_bytes == 0 || num_windows <= 0) {
            continue;
        }

        int phase = 0;

        if (input.out_of_place) {
            const bool terminal =
                input.world_size == 1;

            const TransferTask local_copy =
                make_copy_transfer_task(
                    rank,
                    rank,
                    rank,
                    rank_input_ref(rank, slice_begin_bytes),
                    rank_buffer_ref(rank, slice_begin_bytes),
                    slice_bytes,
                    num_windows,
                    input.launch_config.window_chunks,
                    topology::TransportKind::DirectNvlink,
                    terminal,
                    phase++);

            if (!transfer_plan_push_fast(plan, local_copy)) {
                transfer_plan_abort_build(plan);
                return false;
            }
        }

        if (!append_ready_rendezvous_tasks(
                plan,
                rank,
                input.world_size,
                &phase)) {
            return false;
        }

        for (int peer = 0; peer < input.world_size; ++peer) {
            if (peer == rank) {
                continue;
            }

            const TransferTask reduce =
                make_reduce_transfer_task(
                    rank,
                    peer,
                    rank,
                    rank_buffer_ref(peer, slice_begin_bytes),
                    rank_buffer_ref(rank, slice_begin_bytes),
                    slice_bytes,
                    num_windows,
                    input.launch_config.window_chunks,
                    transports.kind[rank][peer],
                    false,
                    phase++);

            if (!transfer_plan_push_fast(plan, reduce)) {
                transfer_plan_abort_build(plan);
                return false;
            }
        }

        int remaining_peers =
            input.world_size - 1;

        for (int peer = 0; peer < input.world_size; ++peer) {
            if (peer == rank) {
                continue;
            }

            --remaining_peers;

            const TransferTask copy =
                make_copy_transfer_task(
                    rank,
                    rank,
                    peer,
                    rank_buffer_ref(rank, slice_begin_bytes),
                    rank_buffer_ref(peer, slice_begin_bytes),
                    slice_bytes,
                    num_windows,
                    input.launch_config.window_chunks,
                    transports.kind[rank][peer],
                    remaining_peers == 0,
                    phase++);

            if (!transfer_plan_push_fast(plan, copy)) {
                transfer_plan_abort_build(plan);
                return false;
            }
        }
    }

    return true;
}

/*
 * First-pass reduce-scatter logical planner.
 */
template <int MaxTransferTasks>
bool build_reduce_scatter_transfer_plan(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input) {
    if (plan == nullptr || !valid_build_input(input)) {
        return false;
    }

    transfer_plan_reset_metadata(plan, input.world_size);

    TransportMatrix transports{};
    if (!build_transport_matrix(
            input.topo,
            input.world_size,
            &transports)) {
        transfer_plan_abort_build(plan);
        return false;
    }

    for (int rank = 0; rank < input.world_size; ++rank) {
        std::size_t slice_begin_bytes = 0;
        std::size_t slice_bytes = 0;

        compute_rank_slice_bytes_fast(
            input.count,
            input.dtype_size,
            rank,
            input.world_size,
            &slice_begin_bytes,
            &slice_bytes);

        const int num_windows =
            window_count_for_transfer_bytes(
                slice_bytes,
                input.launch_config);

        if (slice_bytes == 0 || num_windows <= 0) {
            continue;
        }

        int phase = 0;

        if (input.out_of_place) {
            const TransferTask local_copy =
                make_copy_transfer_task(
                    rank,
                    rank,
                    rank,
                    rank_input_ref(rank, slice_begin_bytes),
                    rank_buffer_ref(rank, slice_begin_bytes),
                    slice_bytes,
                    num_windows,
                    input.launch_config.window_chunks,
                    topology::TransportKind::DirectNvlink,
                    input.world_size == 1,
                    phase++);

            if (!transfer_plan_push_fast(plan, local_copy)) {
                transfer_plan_abort_build(plan);
                return false;
            }
        }
        
        if (!append_ready_rendezvous_tasks(
                plan,
                rank,
                input.world_size,
                &phase)) {
            return false;
        }

        int remaining_peers =
            input.world_size - 1;

        for (int peer = 0; peer < input.world_size; ++peer) {
            if (peer == rank) {
                continue;
            }

            --remaining_peers;

            const TransferTask reduce =
                make_reduce_transfer_task(
                    rank,
                    peer,
                    rank,
                    rank_buffer_ref(peer, slice_begin_bytes),
                    rank_buffer_ref(rank, slice_begin_bytes),
                    slice_bytes,
                    num_windows,
                    input.launch_config.window_chunks,
                    transports.kind[rank][peer],
                    remaining_peers == 0,
                    phase++);

            if (!transfer_plan_push_fast(plan, reduce)) {
                transfer_plan_abort_build(plan);
                return false;
            }
        }
    }

    return true;
}

/*
 * First-pass all-gather logical planner.
 *
 * Semantics: each executor rank sends/copies its partition to every peer rank.
 */
template <int MaxTransferTasks>
bool build_all_gather_transfer_plan(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input) {
    if (plan == nullptr || !valid_build_input(input)) {
        return false;
    }

    transfer_plan_reset_metadata(plan, input.world_size);

    TransportMatrix transports{};
    if (!build_transport_matrix(
            input.topo,
            input.world_size,
            &transports)) {
        transfer_plan_abort_build(plan);
        return false;
    }

    for (int rank = 0; rank < input.world_size; ++rank) {
        std::size_t slice_begin_bytes = 0;
        std::size_t slice_bytes = 0;

        compute_rank_slice_bytes_fast(
            input.count,
            input.dtype_size,
            rank,
            input.world_size,
            &slice_begin_bytes,
            &slice_bytes);

        const int num_windows =
            window_count_for_transfer_bytes(
                slice_bytes,
                input.launch_config);

        if (slice_bytes == 0 || num_windows <= 0) {
            continue;
        }

        int phase = 0;

        if (input.out_of_place) {
            const TransferTask local_copy =
                make_copy_transfer_task(
                    rank,
                    rank,
                    rank,
                    rank_input_ref(rank, slice_begin_bytes),
                    rank_buffer_ref(rank, slice_begin_bytes),
                    slice_bytes,
                    num_windows,
                    input.launch_config.window_chunks,
                    topology::TransportKind::DirectNvlink,
                    input.world_size == 1,
                    phase++);

            if (!transfer_plan_push_fast(plan, local_copy)) {
                transfer_plan_abort_build(plan);
                return false;
            }
        }

        if (!append_ready_rendezvous_tasks(
                plan,
                rank,
                input.world_size,
                &phase)) {
            return false;
        }

        int remaining_peers =
            input.world_size - 1;

        for (int peer = 0; peer < input.world_size; ++peer) {
            if (peer == rank) {
                continue;
            }

            --remaining_peers;

            const TransferTask copy =
                make_copy_transfer_task(
                    rank,
                    rank,
                    peer,
                    rank_buffer_ref(rank, slice_begin_bytes),
                    rank_buffer_ref(peer, slice_begin_bytes),
                    slice_bytes,
                    num_windows,
                    input.launch_config.window_chunks,
                    transports.kind[rank][peer],
                    remaining_peers == 0,
                    phase++);

            if (!transfer_plan_push_fast(plan, copy)) {
                transfer_plan_abort_build(plan);
                return false;
            }
        }
    }

    return true;
}

} // namespace plan
} // namespace comm
} // namespace ooverlap
