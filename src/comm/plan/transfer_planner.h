#pragma once

#include "comm/plan/transfer_plan.h"
#include "comm/utils/collective_utils.h"
#include "comm/utils/utils.h"

#include <cstddef>

namespace ooverlap {
namespace comm {
namespace plan {

/*
 * Topology/logical planning layer.
 *
 * These helpers build rank-level TransferTask plans. They do not touch raw
 * pointers. They are safe for same-process and multiprocess usage as long as
 * every rank/process passes the same rank->device map and topology view.
 */

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
     * For out-of-place collectives, set this true so the logical planner emits
     * RankInput -> RankBuffer local-copy tasks. Current public allreduce path
     * is in-place, so this defaults false.
     */
    bool out_of_place = false;
};

__host__ __device__ __forceinline__ bool valid_rank(
    int rank,
    int world_size) {
    return rank >= 0 && rank < world_size;
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
     * Later routing can tighten this to "unsupported".
     */
    return topology::TransportKind::DirectPcie;
}

inline bool compute_rank_slice_bytes(
    std::size_t count,
    std::size_t dtype_size,
    int rank,
    int world_size,
    std::size_t* out_begin_bytes,
    std::size_t* out_slice_bytes) {
    if (out_begin_bytes == nullptr ||
        out_slice_bytes == nullptr ||
        dtype_size == 0) {
        return false;
    }

    *out_begin_bytes = 0;
    *out_slice_bytes = 0;

    std::size_t begin_elems = 0;
    std::size_t slice_elems = 0;

    if (!comm::utils::rank_partition(
            count,
            rank,
            world_size,
            &begin_elems,
            &slice_elems)) {
        return false;
    }

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
    task.requires_tma_load =
        transport == topology::TransportKind::DirectNvlink ||
        transport == topology::TransportKind::DirectPcie;
    task.requires_tma_store =
        transport == topology::TransportKind::DirectNvlink ||
        transport == topology::TransportKind::DirectPcie;
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
    task.requires_tma_load =
        transport == topology::TransportKind::DirectNvlink ||
        transport == topology::TransportKind::DirectPcie;
    task.requires_tma_store = false;
    task.requires_tma_reduce =
        transport == topology::TransportKind::DirectNvlink ||
        transport == topology::TransportKind::DirectPcie;
    task.requires_native_atomic =
        transport == topology::TransportKind::DirectNvlink ||
        transport == topology::TransportKind::DirectPcie;
    task.terminal = terminal;
    task.phase = phase;
    return task;
}

/*
 * First-pass logical allreduce planner.
 *
 * This emits the same logical shape as the current naive allreduce builder:
 *   optional local input copy
 *   reduce every peer's rank slice into executor rank's buffer
 *   copy executor rank's result slice back to every peer
 *
 * It is topology-aware only at the level of annotating each transfer with the
 * selected transport. The lowering layer still decides whether it can lower
 * that transport to current WindowTask operations.
 */
template <int MaxTransferTasks>
bool build_allreduce_transfer_plan(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input) {
    if (plan == nullptr ||
        input.world_size <= 0 ||
        input.dtype_size == 0 ||
        input.launch_config.window_chunks <= 0 ||
        input.launch_config.chunk_bytes <= 0) {
        return false;
    }

    // TODO: this is slow as we saw before
    transfer_plan_clear(plan);
    plan->world_size = input.world_size;

    for (int rank = 0; rank < input.world_size; ++rank) {
        std::size_t slice_begin_bytes = 0;
        std::size_t slice_bytes = 0;

        if (!compute_rank_slice_bytes(
                input.count,
                input.dtype_size,
                rank,
                input.world_size,
                &slice_begin_bytes,
                &slice_bytes)) {
            transfer_plan_clear(plan);
            return false;
        }

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

            TransferTask local_copy =
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

            if (!transfer_plan_push(plan, local_copy)) {
                transfer_plan_clear(plan);
                return false;
            }
        }

        for (int peer = 0; peer < input.world_size; ++peer) {
            if (peer == rank) {
                continue;
            }

            const topology::TransportKind transport =
                choose_direct_or_fallback_transport(
                    input.topo,
                    rank,
                    peer);

            TransferTask reduce =
                make_reduce_transfer_task(
                    rank,
                    peer,
                    rank,
                    rank_buffer_ref(peer, slice_begin_bytes),
                    rank_buffer_ref(rank, slice_begin_bytes),
                    slice_bytes,
                    num_windows,
                    input.launch_config.window_chunks,
                    transport,
                    false,
                    phase++);

            if (!transfer_plan_push(plan, reduce)) {
                transfer_plan_clear(plan);
                return false;
            }
        }

        int remaining_peers = input.world_size - 1;

        for (int peer = 0; peer < input.world_size; ++peer) {
            if (peer == rank) {
                continue;
            }

            --remaining_peers;

            const topology::TransportKind transport =
                choose_direct_or_fallback_transport(
                    input.topo,
                    rank,
                    peer);

            TransferTask copy =
                make_copy_transfer_task(
                    rank,
                    rank,
                    peer,
                    rank_buffer_ref(rank, slice_begin_bytes),
                    rank_buffer_ref(peer, slice_begin_bytes),
                    slice_bytes,
                    num_windows,
                    input.launch_config.window_chunks,
                    transport,
                    remaining_peers == 0,
                    phase++);

            if (!transfer_plan_push(plan, copy)) {
                transfer_plan_clear(plan);
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
    if (plan == nullptr ||
        input.world_size <= 0 ||
        input.dtype_size == 0 ||
        input.launch_config.window_chunks <= 0 ||
        input.launch_config.chunk_bytes <= 0) {
        return false;
    }

    transfer_plan_clear(plan);
    plan->world_size = input.world_size;

    for (int rank = 0; rank < input.world_size; ++rank) {
        std::size_t slice_begin_bytes = 0;
        std::size_t slice_bytes = 0;

        if (!compute_rank_slice_bytes(
                input.count,
                input.dtype_size,
                rank,
                input.world_size,
                &slice_begin_bytes,
                &slice_bytes)) {
            transfer_plan_clear(plan);
            return false;
        }

        const int num_windows =
            window_count_for_transfer_bytes(
                slice_bytes,
                input.launch_config);

        if (slice_bytes == 0 || num_windows <= 0) {
            continue;
        }

        int phase = 0;

        if (input.out_of_place) {
            TransferTask local_copy =
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

            if (!transfer_plan_push(plan, local_copy)) {
                transfer_plan_clear(plan);
                return false;
            }
        }

        int remaining_peers = input.world_size - 1;

        for (int peer = 0; peer < input.world_size; ++peer) {
            if (peer == rank) {
                continue;
            }

            --remaining_peers;

            const topology::TransportKind transport =
                choose_direct_or_fallback_transport(
                    input.topo,
                    rank,
                    peer);

            TransferTask reduce =
                make_reduce_transfer_task(
                    rank,
                    peer,
                    rank,
                    rank_buffer_ref(peer, slice_begin_bytes),
                    rank_buffer_ref(rank, slice_begin_bytes),
                    slice_bytes,
                    num_windows,
                    input.launch_config.window_chunks,
                    transport,
                    remaining_peers == 0,
                    phase++);

            if (!transfer_plan_push(plan, reduce)) {
                transfer_plan_clear(plan);
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
 * Lowering may later choose push or pull physical direction depending on
 * transport and WindowTask capabilities.
 */
template <int MaxTransferTasks>
bool build_all_gather_transfer_plan(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input) {
    if (plan == nullptr ||
        input.world_size <= 0 ||
        input.dtype_size == 0 ||
        input.launch_config.window_chunks <= 0 ||
        input.launch_config.chunk_bytes <= 0) {
        return false;
    }

    transfer_plan_clear(plan);
    plan->world_size = input.world_size;

    for (int rank = 0; rank < input.world_size; ++rank) {
        std::size_t slice_begin_bytes = 0;
        std::size_t slice_bytes = 0;

        if (!compute_rank_slice_bytes(
                input.count,
                input.dtype_size,
                rank,
                input.world_size,
                &slice_begin_bytes,
                &slice_bytes)) {
            transfer_plan_clear(plan);
            return false;
        }

        const int num_windows =
            window_count_for_transfer_bytes(
                slice_bytes,
                input.launch_config);

        if (slice_bytes == 0 || num_windows <= 0) {
            continue;
        }

        int phase = 0;

        if (input.out_of_place) {
            TransferTask local_copy =
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

            if (!transfer_plan_push(plan, local_copy)) {
                transfer_plan_clear(plan);
                return false;
            }
        }

        int remaining_peers = input.world_size - 1;

        for (int peer = 0; peer < input.world_size; ++peer) {
            if (peer == rank) {
                continue;
            }

            --remaining_peers;

            const topology::TransportKind transport =
                choose_direct_or_fallback_transport(
                    input.topo,
                    rank,
                    peer);

            TransferTask copy =
                make_copy_transfer_task(
                    rank,
                    rank,
                    peer,
                    rank_buffer_ref(rank, slice_begin_bytes),
                    rank_buffer_ref(peer, slice_begin_bytes),
                    slice_bytes,
                    num_windows,
                    input.launch_config.window_chunks,
                    transport,
                    remaining_peers == 0,
                    phase++);

            if (!transfer_plan_push(plan, copy)) {
                transfer_plan_clear(plan);
                return false;
            }
        }
    }

    return true;
}

} // namespace plan
} // namespace comm
} // namespace ooverlap
