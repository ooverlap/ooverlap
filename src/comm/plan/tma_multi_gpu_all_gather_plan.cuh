#pragma once

#include "comm/plan/tma_multi_gpu_plan_utils.cuh"

namespace ooverlap {
namespace comm {
namespace plan {

namespace all_gather_detail {

constexpr int kReadyPhaseEntry = 0;
constexpr int kReadyPhaseIslandGatherDone = 1;
constexpr int kReadyPhaseIslandStagedBase = 2;
constexpr int kReadyPhaseRemoteLeaderHasBase =
    kReadyPhaseIslandStagedBase + kPlannerMaxRanks;

struct IslandPlan {
    int island_count = 0;

    int island_of_rank[kPlannerMaxRanks] = {};
    int island_size[kPlannerMaxRanks] = {};
    int island_ranks[kPlannerMaxRanks][kPlannerMaxRanks] = {};

    int leader_rank[kPlannerMaxRanks] = {};
    int first_rank[kPlannerMaxRanks] = {};
    int last_rank[kPlannerMaxRanks] = {};

    int preferred_numa_node[kPlannerMaxRanks] = {};
    int staging_slot[kPlannerMaxRanks] = {};

    std::size_t block_begin_bytes[kPlannerMaxRanks] = {};
    std::size_t block_bytes[kPlannerMaxRanks] = {};
};

__host__ __device__ __forceinline__ bool all_gather_transport_is_direct(
    topology::TransportKind transport) {
    return transport == topology::TransportKind::DirectNvlink ||
           transport == topology::TransportKind::DirectPcie;
}

inline int rank_numa_node(
    const TransferPlanBuildInput& input,
    int rank) {
    if (!valid_rank(rank, input.world_size) ||
        input.topo.topology == nullptr ||
        input.topo.rank_devices == nullptr) {
        return -1;
    }

    const int device =
        input.topo.rank_devices[rank];

    for (const topology::Node& node : input.topo.topology->nodes) {
        if (node.device == device) {
            return node.numa_node;
        }
    }

    return -1;
}

inline int staged_ready_phase_for_island(
    int island) {
    return kReadyPhaseIslandStagedBase + island;
}

inline int remote_leader_ready_phase(
    int src_island,
    int dst_island) {
    return kReadyPhaseRemoteLeaderHasBase +
           src_island * kPlannerMaxRanks +
           dst_island;
}

template <int MaxTransferTasks>
bool push_task_or_abort(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferTask& task) {
    if (!transfer_plan_push_fast(plan, task)) {
        transfer_plan_abort_build(plan);
        return false;
    }

    return true;
}

template <int MaxTransferTasks>
bool push_copy_task(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    int executor_rank,
    int src_rank,
    int dst_rank,
    LogicalBufferRef src,
    LogicalBufferRef dst,
    std::size_t bytes,
    int num_windows,
    topology::TransportKind transport,
    int* task_phase) {
    if (task_phase == nullptr ||
        bytes == 0 ||
        num_windows <= 0) {
        transfer_plan_abort_build(plan);
        return false;
    }

    const TransferTask copy =
        make_copy_transfer_task(
            executor_rank,
            src_rank,
            dst_rank,
            src,
            dst,
            bytes,
            num_windows,
            input.launch_config.window_chunks,
            transport,
            false,
            (*task_phase)++);

    return push_task_or_abort(plan, copy);
}


/*
 * OOVERLAP_ALLGATHER_COPY_FANOUT_PLANNER_V1
 *
 * The current low-level TMA fanout path handles only fully 16-byte-aligned
 * bulk ranges. Keep ordinary Copy tasks as the correctness fallback for every
 * other slice.
 */
inline bool all_gather_copy_fanout_compatible(
    std::size_t byte_offset,
    std::size_t bytes) {
    constexpr std::size_t kTmaBulkAlignment = 16;

    return (byte_offset % kTmaBulkAlignment) == 0 &&
           (bytes % kTmaBulkAlignment) == 0;
}

template <int MaxTransferTasks>
bool push_copy_fanout_task(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    int executor_rank,
    int src_rank,
    const int* dst_ranks,
    LogicalBufferRef src,
    const LogicalBufferRef* fanout_dsts,
    int fanout_dst_count,
    std::size_t bytes,
    int num_windows,
    topology::TransportKind transport,
    int* task_phase) {
    if (task_phase == nullptr ||
        dst_ranks == nullptr ||
        fanout_dsts == nullptr ||
        fanout_dst_count <= 0 ||
        fanout_dst_count > TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS ||
        bytes == 0 ||
        num_windows <= 0 ||
        !all_gather_transport_is_direct(transport)) {
        transfer_plan_abort_build(plan);
        return false;
    }

    const TransferTask copy =
        make_copy_fanout_transfer_task(
            executor_rank,
            src_rank,
            dst_ranks,
            src,
            fanout_dsts,
            fanout_dst_count,
            bytes,
            num_windows,
            input.launch_config.window_chunks,
            transport,
            false,
            (*task_phase)++);

    return push_task_or_abort(plan, copy);
}

template <int MaxTransferTasks>
bool push_copy_batch_task(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    int executor_rank,
    int src_rank,
    const int* dst_ranks,
    LogicalBufferRef src,
    const LogicalBufferRef* fanout_dsts,
    int fanout_dst_count,
    std::size_t bytes,
    int num_windows,
    topology::TransportKind transport,
    int* task_phase) {
    if (fanout_dst_count == 1) {
        return push_copy_task(
            plan,
            input,
            executor_rank,
            src_rank,
            dst_ranks[0],
            src,
            fanout_dsts[0],
            bytes,
            num_windows,
            transport,
            task_phase);
    }

    return push_copy_fanout_task(
        plan,
        input,
        executor_rank,
        src_rank,
        dst_ranks,
        src,
        fanout_dsts,
        fanout_dst_count,
        bytes,
        num_windows,
        transport,
        task_phase);
}

template <int MaxTransferTasks>
bool emit_all_gather_source_copies(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    const TransportMatrix& transports,
    int src_rank,
    const int* candidate_ranks,
    int candidate_count,
    std::size_t slice_begin_bytes,
    std::size_t slice_bytes,
    int slice_windows,
    int* task_phase) {
    if (plan == nullptr ||
        candidate_ranks == nullptr ||
        candidate_count < 0 ||
        candidate_count > input.world_size ||
        !valid_rank(src_rank, input.world_size) ||
        task_phase == nullptr ||
        slice_bytes == 0 ||
        slice_windows <= 0) {
        transfer_plan_abort_build(plan);
        return false;
    }

    const LogicalBufferRef src =
        rank_buffer_ref(src_rank, slice_begin_bytes);

    if (!all_gather_copy_fanout_compatible(
            slice_begin_bytes,
            slice_bytes)) {
        for (int i = 0; i < candidate_count; ++i) {
            const int dst_rank = candidate_ranks[i];

            if (dst_rank == src_rank) {
                continue;
            }

            if (!valid_rank(dst_rank, input.world_size) ||
                !all_gather_transport_is_direct(
                    transports.kind[src_rank][dst_rank])) {
                transfer_plan_abort_build(plan);
                return false;
            }

            if (!push_copy_task(
                    plan,
                    input,
                    src_rank,
                    src_rank,
                    dst_rank,
                    src,
                    rank_buffer_ref(dst_rank, slice_begin_bytes),
                    slice_bytes,
                    slice_windows,
                    transports.kind[src_rank][dst_rank],
                    task_phase)) {
                return false;
            }
        }

        return true;
    }

    int batch_dst_ranks[TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS] = {};
    LogicalBufferRef batch_dsts[TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS] = {};
    int batch_count = 0;
    topology::TransportKind batch_transport =
        topology::TransportKind::DirectNvlink;

    for (int i = 0; i < candidate_count; ++i) {
        const int dst_rank = candidate_ranks[i];

        if (dst_rank == src_rank) {
            continue;
        }

        if (!valid_rank(dst_rank, input.world_size)) {
            transfer_plan_abort_build(plan);
            return false;
        }

        const topology::TransportKind transport =
            transports.kind[src_rank][dst_rank];

        if (!all_gather_transport_is_direct(transport)) {
            transfer_plan_abort_build(plan);
            return false;
        }

        const bool flush_batch =
            batch_count > 0 &&
            (batch_count == TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS ||
             transport != batch_transport);

        if (flush_batch) {
            if (!push_copy_batch_task(
                    plan,
                    input,
                    src_rank,
                    src_rank,
                    batch_dst_ranks,
                    src,
                    batch_dsts,
                    batch_count,
                    slice_bytes,
                    slice_windows,
                    batch_transport,
                    task_phase)) {
                return false;
            }

            batch_count = 0;
        }

        if (batch_count == 0) {
            batch_transport = transport;
        }

        batch_dst_ranks[batch_count] = dst_rank;
        batch_dsts[batch_count] =
            rank_buffer_ref(dst_rank, slice_begin_bytes);
        ++batch_count;
    }

    if (batch_count > 0 &&
        !push_copy_batch_task(
            plan,
            input,
            src_rank,
            src_rank,
            batch_dst_ranks,
            src,
            batch_dsts,
            batch_count,
            slice_bytes,
            slice_windows,
            batch_transport,
            task_phase)) {
        return false;
    }

    return true;
}

template <int MaxTransferTasks>
bool push_ready_publish_task(
    TransferPlan<MaxTransferTasks>* plan,
    int publisher_rank,
    ReadySignalChannel channel,
    int ready_phase,
    int* task_phase) {
    if (task_phase == nullptr) {
        transfer_plan_abort_build(plan);
        return false;
    }

    const TransferTask publish =
        make_ready_publish_transfer_task(
            publisher_rank,
            channel,
            (*task_phase)++,
            ready_phase);

    return push_task_or_abort(plan, publish);
}

template <int MaxTransferTasks>
bool push_ready_wait_task(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    int waiter_rank,
    int owner_rank,
    int ready_phase,
    int* task_phase) {
    if (task_phase == nullptr) {
        transfer_plan_abort_build(plan);
        return false;
    }

    const ReadySignalChannel channel =
        choose_ready_signal_channel(
            input.topo,
            waiter_rank,
            owner_rank);

    const TransferTask wait =
        make_ready_wait_transfer_task(
            waiter_rank,
            owner_rank,
            channel,
            (*task_phase)++,
            ready_phase);

    return push_task_or_abort(plan, wait);
}

template <int MaxTransferTasks>
bool push_ready_publish_for_waiters(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    int publisher_rank,
    const int* waiters,
    int waiter_count,
    int ready_phase,
    int* task_phase) {
    if (waiters == nullptr || waiter_count <= 0) {
        return true;
    }

    bool channel_used[kReadySignalChannelCount] = {};

    for (int i = 0; i < waiter_count; ++i) {
        const int waiter_rank = waiters[i];

        if (!valid_rank(waiter_rank, input.world_size) ||
            waiter_rank == publisher_rank) {
            continue;
        }

        const ReadySignalChannel channel =
            choose_ready_signal_channel(
                input.topo,
                waiter_rank,
                publisher_rank);

        channel_used[static_cast<int>(channel)] = true;
    }

    for (int channel = 0; channel < kReadySignalChannelCount; ++channel) {
        if (!channel_used[channel]) {
            continue;
        }

        if (!push_ready_publish_task(
                plan,
                publisher_rank,
                static_cast<ReadySignalChannel>(channel),
                ready_phase,
                task_phase)) {
            return false;
        }
    }

    return true;
}

template <int MaxTransferTasks>
void mark_last_task_per_rank_terminal(
    TransferPlan<MaxTransferTasks>* plan,
    int world_size) {
    if (plan == nullptr) {
        return;
    }

    for (int i = 0; i < plan->total_tasks; ++i) {
        plan->tasks[i].terminal = false;
    }

    for (int rank = 0; rank < world_size; ++rank) {
        for (int i = plan->total_tasks - 1; i >= 0; --i) {
            if (plan->tasks[i].executor_rank == rank) {
                plan->tasks[i].terminal = true;
                break;
            }
        }
    }
}

inline bool build_all_gather_islands(
    const TransferPlanBuildInput& input,
    const TransportMatrix& transports,
    IslandPlan* out) {
    if (out == nullptr ||
        input.world_size <= 0 ||
        input.world_size > kPlannerMaxRanks) {
        return false;
    }

    *out = IslandPlan{};

    bool visited[kPlannerMaxRanks] = {};
    int queue[kPlannerMaxRanks] = {};

    for (int seed = 0; seed < input.world_size; ++seed) {
        if (visited[seed]) {
            continue;
        }

        const int island = out->island_count++;

        int head = 0;
        int tail = 0;

        visited[seed] = true;
        queue[tail++] = seed;
        out->island_of_rank[seed] = island;

        while (head < tail) {
            const int rank = queue[head++];

            for (int peer = 0; peer < input.world_size; ++peer) {
                if (visited[peer]) {
                    continue;
                }

                const bool direct_both_ways =
                    all_gather_transport_is_direct(
                        transports.kind[rank][peer]) &&
                    all_gather_transport_is_direct(
                        transports.kind[peer][rank]);

                if (!direct_both_ways) {
                    continue;
                }

                visited[peer] = true;
                out->island_of_rank[peer] = island;
                queue[tail++] = peer;
            }
        }
    }

    for (int island = 0; island < out->island_count; ++island) {
        out->leader_rank[island] = -1;
        out->first_rank[island] = input.world_size;
        out->last_rank[island] = -1;
        out->preferred_numa_node[island] = -1;
        out->staging_slot[island] = -1;
    }

    /*
     * Store each island's ranks sorted by logical rank.  This is important
     * because staged island blocks are copied as one contiguous rank-order byte
     * range.
     */
    for (int rank = 0; rank < input.world_size; ++rank) {
        const int island = out->island_of_rank[rank];
        const int idx = out->island_size[island]++;

        out->island_ranks[island][idx] = rank;

        if (rank < out->first_rank[island]) {
            out->first_rank[island] = rank;
        }

        if (rank > out->last_rank[island]) {
            out->last_rank[island] = rank;
        }

        if (out->leader_rank[island] < 0 ||
            rank < out->leader_rank[island]) {
            out->leader_rank[island] = rank;
        }
    }

    for (int island = 0; island < out->island_count; ++island) {
        /*
         * The first staged implementation requires each island to occupy a
         * contiguous logical rank interval.  Otherwise that island's gathered
         * output is not one contiguous block in all-gather rank-order layout.
         */
        const int expected_size =
            out->last_rank[island] - out->first_rank[island] + 1;

        if (expected_size != out->island_size[island]) {
            return false;
        }

        out->preferred_numa_node[island] =
            rank_numa_node(
                input,
                out->leader_rank[island]);
    }

    return true;
}

inline bool fill_all_gather_slice_metadata(
    const TransferPlanBuildInput& input,
    const IslandPlan& islands,
    std::size_t* slice_begin_bytes,
    std::size_t* slice_bytes,
    int* slice_windows,
    IslandPlan* out_islands) {
    if (slice_begin_bytes == nullptr ||
        slice_bytes == nullptr ||
        slice_windows == nullptr ||
        out_islands == nullptr) {
        return false;
    }

    *out_islands = islands;

    for (int rank = 0; rank < input.world_size; ++rank) {
        compute_rank_slice_bytes_fast(
            input.count,
            input.dtype_size,
            rank,
            input.world_size,
            &slice_begin_bytes[rank],
            &slice_bytes[rank]);

        slice_windows[rank] =
            window_count_for_transfer_bytes(
                slice_bytes[rank],
                input.launch_config);

        if (slice_bytes[rank] == 0 || slice_windows[rank] <= 0) {
            return false;
        }
    }

    for (int island = 0; island < out_islands->island_count; ++island) {
        const int first = out_islands->first_rank[island];
        const int last = out_islands->last_rank[island];

        const std::size_t begin =
            slice_begin_bytes[first];
        const std::size_t end =
            slice_begin_bytes[last] + slice_bytes[last];

        if (end < begin) {
            return false;
        }

        out_islands->block_begin_bytes[island] = begin;
        out_islands->block_bytes[island] = end - begin;
    }

    return true;
}

inline bool assign_staging_slots(
    const TransferPlanBuildInput& input,
    IslandPlan* islands) {
    if (islands == nullptr) {
        return false;
    }

    bool used[kPlannerMaxStagingSlots] = {};

    for (int island = 0; island < islands->island_count; ++island) {
        const std::size_t required_bytes =
            islands->block_bytes[island];

        int selected = -1;

        const int preferred_numa =
            islands->preferred_numa_node[island];

        if (preferred_numa >= 0) {
            for (int slot = 0; slot < input.staging_slot_count; ++slot) {
                if (!used[slot] &&
                    staging_slot_numa_node(input, slot) == preferred_numa &&
                    staging_slice_valid(input, slot, 0, required_bytes)) {
                    selected = slot;
                    break;
                }
            }
        }

        if (selected < 0) {
            for (int slot = 0; slot < input.staging_slot_count; ++slot) {
                if (!used[slot] &&
                    staging_slice_valid(input, slot, 0, required_bytes)) {
                    selected = slot;
                    break;
                }
            }
        }

        if (selected < 0) {
            return false;
        }

        used[selected] = true;
        islands->staging_slot[island] = selected;
    }

    return true;
}

template <int MaxTransferTasks>
bool emit_all_gather_optional_local_copies(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    const std::size_t* slice_begin_bytes,
    const std::size_t* slice_bytes,
    const int* slice_windows,
    int* task_phase) {
    if (!input.out_of_place) {
        return true;
    }

    for (int rank = 0; rank < input.world_size; ++rank) {
        if (!push_copy_task(
                plan,
                input,
                rank,
                rank,
                rank,
                rank_input_ref(rank, slice_begin_bytes[rank]),
                rank_buffer_ref(rank, slice_begin_bytes[rank]),
                slice_bytes[rank],
                slice_windows[rank],
                topology::TransportKind::DirectNvlink,
                &task_phase[rank])) {
            return false;
        }
    }

    return true;
}

template <int MaxTransferTasks>
bool emit_all_gather_entry_rendezvous(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    int* task_phase) {
    for (int rank = 0; rank < input.world_size; ++rank) {
        if (!append_ready_rendezvous_tasks(
                plan,
                input.topo,
                rank,
                input.world_size,
                &task_phase[rank])) {
            return false;
        }
    }

    return true;
}

template <int MaxTransferTasks>
bool build_direct_all_gather_plan(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    const TransportMatrix& transports) {
    int task_phase[kPlannerMaxRanks] = {};
    std::size_t slice_begin_bytes[kPlannerMaxRanks] = {};
    std::size_t slice_bytes[kPlannerMaxRanks] = {};
    int slice_windows[kPlannerMaxRanks] = {};

    IslandPlan one_rank_per_slice{};
    one_rank_per_slice.island_count = input.world_size;

    for (int rank = 0; rank < input.world_size; ++rank) {
        one_rank_per_slice.first_rank[rank] = rank;
        one_rank_per_slice.last_rank[rank] = rank;
    }

    if (!fill_all_gather_slice_metadata(
            input,
            one_rank_per_slice,
            slice_begin_bytes,
            slice_bytes,
            slice_windows,
            &one_rank_per_slice)) {
        transfer_plan_abort_build(plan);
        return false;
    }

    if (!emit_all_gather_optional_local_copies(
            plan,
            input,
            slice_begin_bytes,
            slice_bytes,
            slice_windows,
            task_phase)) {
        return false;
    }

    if (!emit_all_gather_entry_rendezvous(
            plan,
            input,
            task_phase)) {
        return false;
    }

    int all_ranks[kPlannerMaxRanks] = {};
    for (int rank = 0; rank < input.world_size; ++rank) {
        all_ranks[rank] = rank;
    }

    for (int src_rank = 0; src_rank < input.world_size; ++src_rank) {
        if (!emit_all_gather_source_copies(
                plan,
                input,
                transports,
                src_rank,
                all_ranks,
                input.world_size,
                slice_begin_bytes[src_rank],
                slice_bytes[src_rank],
                slice_windows[src_rank],
                &task_phase[src_rank])) {
            return false;
        }
    }

    mark_last_task_per_rank_terminal(plan, input.world_size);
    return true;
}

template <int MaxTransferTasks>
bool emit_intra_island_all_gather(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    const TransportMatrix& transports,
    const IslandPlan& islands,
    const std::size_t* slice_begin_bytes,
    const std::size_t* slice_bytes,
    const int* slice_windows,
    int* task_phase) {
    for (int island = 0; island < islands.island_count; ++island) {
        for (int src_idx = 0;
             src_idx < islands.island_size[island];
             ++src_idx) {
            const int src_rank =
                islands.island_ranks[island][src_idx];

            if (!emit_all_gather_source_copies(
                    plan,
                    input,
                    transports,
                    src_rank,
                    islands.island_ranks[island],
                    islands.island_size[island],
                    slice_begin_bytes[src_rank],
                    slice_bytes[src_rank],
                    slice_windows[src_rank],
                    &task_phase[src_rank])) {
                return false;
            }
        }
    }

    return true;
}

template <int MaxTransferTasks>
bool emit_island_gather_done_sync(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    const IslandPlan& islands,
    int* task_phase) {
    for (int island = 0; island < islands.island_count; ++island) {
        const int leader =
            islands.leader_rank[island];

        int waiters[kPlannerMaxRanks] = {};
        int waiter_count = 0;

        for (int idx = 0; idx < islands.island_size[island]; ++idx) {
            const int rank =
                islands.island_ranks[island][idx];

            if (rank == leader) {
                continue;
            }

            waiters[waiter_count++] = leader;

            if (!push_ready_publish_for_waiters(
                    plan,
                    input,
                    rank,
                    waiters,
                    1,
                    kReadyPhaseIslandGatherDone,
                    &task_phase[rank])) {
                return false;
            }

            if (!push_ready_wait_task(
                    plan,
                    input,
                    leader,
                    rank,
                    kReadyPhaseIslandGatherDone,
                    &task_phase[leader])) {
                return false;
            }
        }
    }

    return true;
}

template <int MaxTransferTasks>
bool emit_island_blocks_to_staging(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    const IslandPlan& islands,
    int* task_phase) {
    for (int src_island = 0;
         src_island < islands.island_count;
         ++src_island) {
        const int leader =
            islands.leader_rank[src_island];
        const int staging_slot =
            islands.staging_slot[src_island];
        const std::size_t block_begin =
            islands.block_begin_bytes[src_island];
        const std::size_t block_bytes =
            islands.block_bytes[src_island];

        const int num_windows =
            window_count_for_transfer_bytes(
                block_bytes,
                input.launch_config);

        if (!staging_slice_valid(
                input,
                staging_slot,
                0,
                block_bytes) ||
            num_windows <= 0) {
            transfer_plan_abort_build(plan);
            return false;
        }

        if (!push_copy_task(
                plan,
                input,
                leader,
                leader,
                leader,
                rank_buffer_ref(leader, block_begin),
                shm_staging_ref(staging_slot, 0),
                block_bytes,
                num_windows,
                topology::TransportKind::Shm,
                &task_phase[leader])) {
            return false;
        }

        int remote_leaders[kPlannerMaxRanks] = {};
        int remote_leader_count = 0;

        for (int dst_island = 0;
             dst_island < islands.island_count;
             ++dst_island) {
            if (dst_island == src_island) {
                continue;
            }

            remote_leaders[remote_leader_count++] =
                islands.leader_rank[dst_island];
        }

        if (!push_ready_publish_for_waiters(
                plan,
                input,
                leader,
                remote_leaders,
                remote_leader_count,
                staged_ready_phase_for_island(src_island),
                &task_phase[leader])) {
            return false;
        }
    }

    return true;
}

template <int MaxTransferTasks>
bool emit_remote_staging_to_destination_leaders(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    const IslandPlan& islands,
    int* task_phase) {
    for (int dst_island = 0;
         dst_island < islands.island_count;
         ++dst_island) {
        const int dst_leader =
            islands.leader_rank[dst_island];

        for (int src_island = 0;
             src_island < islands.island_count;
             ++src_island) {
            if (src_island == dst_island) {
                continue;
            }

            const int src_leader =
                islands.leader_rank[src_island];
            const int staging_slot =
                islands.staging_slot[src_island];
            const std::size_t block_begin =
                islands.block_begin_bytes[src_island];
            const std::size_t block_bytes =
                islands.block_bytes[src_island];

            const int num_windows =
                window_count_for_transfer_bytes(
                    block_bytes,
                    input.launch_config);

            if (!push_ready_wait_task(
                    plan,
                    input,
                    dst_leader,
                    src_leader,
                    staged_ready_phase_for_island(src_island),
                    &task_phase[dst_leader])) {
                return false;
            }

            if (!push_copy_task(
                    plan,
                    input,
                    dst_leader,
                    src_leader,
                    dst_leader,
                    shm_staging_ref(staging_slot, 0),
                    rank_buffer_ref(dst_leader, block_begin),
                    block_bytes,
                    num_windows,
                    topology::TransportKind::Shm,
                    &task_phase[dst_leader])) {
                return false;
            }

            int local_waiters[kPlannerMaxRanks] = {};
            int local_waiter_count = 0;

            for (int idx = 0;
                 idx < islands.island_size[dst_island];
                 ++idx) {
                const int rank =
                    islands.island_ranks[dst_island][idx];

                if (rank == dst_leader) {
                    continue;
                }

                local_waiters[local_waiter_count++] = rank;
            }

            if (!push_ready_publish_for_waiters(
                    plan,
                    input,
                    dst_leader,
                    local_waiters,
                    local_waiter_count,
                    remote_leader_ready_phase(src_island, dst_island),
                    &task_phase[dst_leader])) {
                return false;
            }
        }
    }

    return true;
}

template <int MaxTransferTasks>
bool emit_destination_leader_to_island_ranks(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    const TransportMatrix& transports,
    const IslandPlan& islands,
    int* task_phase) {
    for (int dst_island = 0;
         dst_island < islands.island_count;
         ++dst_island) {
        const int dst_leader =
            islands.leader_rank[dst_island];

        for (int src_island = 0;
             src_island < islands.island_count;
             ++src_island) {
            if (src_island == dst_island) {
                continue;
            }

            const std::size_t block_begin =
                islands.block_begin_bytes[src_island];
            const std::size_t block_bytes =
                islands.block_bytes[src_island];

            const int num_windows =
                window_count_for_transfer_bytes(
                    block_bytes,
                    input.launch_config);

            for (int idx = 0;
                 idx < islands.island_size[dst_island];
                 ++idx) {
                const int dst_rank =
                    islands.island_ranks[dst_island][idx];

                if (dst_rank == dst_leader) {
                    continue;
                }

                if (!push_ready_wait_task(
                        plan,
                        input,
                        dst_rank,
                        dst_leader,
                        remote_leader_ready_phase(src_island, dst_island),
                        &task_phase[dst_rank])) {
                    return false;
                }

                if (!all_gather_transport_is_direct(
                        transports.kind[dst_rank][dst_leader])) {
                    transfer_plan_abort_build(plan);
                    return false;
                }

                if (!push_copy_task(
                        plan,
                        input,
                        dst_rank,
                        dst_leader,
                        dst_rank,
                        rank_buffer_ref(dst_leader, block_begin),
                        rank_buffer_ref(dst_rank, block_begin),
                        block_bytes,
                        num_windows,
                        transports.kind[dst_rank][dst_leader],
                        &task_phase[dst_rank])) {
                    return false;
                }
            }
        }
    }

    return true;
}

template <int MaxTransferTasks>
bool build_staged_island_all_gather_plan(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    const TransportMatrix& transports,
    const IslandPlan& discovered_islands) {
    int task_phase[kPlannerMaxRanks] = {};
    std::size_t slice_begin_bytes[kPlannerMaxRanks] = {};
    std::size_t slice_bytes[kPlannerMaxRanks] = {};
    int slice_windows[kPlannerMaxRanks] = {};

    IslandPlan islands{};

    if (!fill_all_gather_slice_metadata(
            input,
            discovered_islands,
            slice_begin_bytes,
            slice_bytes,
            slice_windows,
            &islands)) {
        transfer_plan_abort_build(plan);
        return false;
    }

    if (!assign_staging_slots(input, &islands)) {
        transfer_plan_abort_build(plan);
        return false;
    }

    if (!emit_all_gather_optional_local_copies(
            plan,
            input,
            slice_begin_bytes,
            slice_bytes,
            slice_windows,
            task_phase)) {
        return false;
    }

    if (!emit_all_gather_entry_rendezvous(
            plan,
            input,
            task_phase)) {
        return false;
    }

    if (!emit_intra_island_all_gather(
            plan,
            input,
            transports,
            islands,
            slice_begin_bytes,
            slice_bytes,
            slice_windows,
            task_phase)) {
        return false;
    }

    if (!emit_island_gather_done_sync(
            plan,
            input,
            islands,
            task_phase)) {
        return false;
    }

    if (!emit_island_blocks_to_staging(
            plan,
            input,
            islands,
            task_phase)) {
        return false;
    }

    if (!emit_remote_staging_to_destination_leaders(
            plan,
            input,
            islands,
            task_phase)) {
        return false;
    }

    if (!emit_destination_leader_to_island_ranks(
            plan,
            input,
            transports,
            islands,
            task_phase)) {
        return false;
    }

    mark_last_task_per_rank_terminal(plan, input.world_size);
    return true;
}

} // namespace all_gather_detail

/*
 * Logical all-gather planner.
 *
 * Direct case:
 *   each source rank copies its slice directly to every destination rank.
 *
 * SYS/staged case:
 *   1. Build direct-access islands.
 *   2. All-gather inside each island with direct rank-buffer copies.
 *   3. Island leader waits for local island gather completion.
 *   4. Island leader copies the whole contiguous island block to one staging slot.
 *   5. Destination island leader reads each remote island block from staging.
 *   6. Destination island ranks wait on their leader, then receive that remote
 *      block from the leader over direct island transport.
 *
 * The staged path requires each island to be a contiguous logical rank interval,
 * e.g. [0,1] and [2,3].  If an island is non-contiguous, e.g. [0,2], the island
 * data is not one contiguous all-gather output block, so this first version
 * rejects the plan instead of generating an incorrect staged transfer.
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

    all_gather_detail::IslandPlan islands{};
    if (!all_gather_detail::build_all_gather_islands(
            input,
            transports,
            &islands)) {
        transfer_plan_abort_build(plan);
        return false;
    }

    if (islands.island_count <= 1) {
        const bool ok =
            all_gather_detail::build_direct_all_gather_plan(
                plan,
                input,
                transports);

        if (ok) {
            debug_print_transfer_plan_if_enabled("all_gather", *plan);
        }

        return ok;
    }

    const bool ok =
        all_gather_detail::build_staged_island_all_gather_plan(
            plan,
            input,
            transports,
            islands);

    if (ok) {
        debug_print_transfer_plan_if_enabled("all_gather", *plan);
    }

    return ok;
}

} // namespace plan
} // namespace comm
} // namespace ooverlap
