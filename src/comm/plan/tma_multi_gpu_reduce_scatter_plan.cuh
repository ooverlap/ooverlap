#pragma once

#include "comm/plan/tma_multi_gpu_plan_utils.cuh"

namespace ooverlap {
namespace comm {
namespace plan {

namespace reduce_scatter_detail {

constexpr int kReadyPhaseEntry = 0;
constexpr int kReadyPhaseIslandLocalPartialBase = 1;
constexpr int kReadyPhaseIslandStagedBase =
    kReadyPhaseIslandLocalPartialBase + kPlannerMaxRanks;

struct IslandPlan {
    int island_count = 0;

    int island_of_rank[kPlannerMaxRanks] = {};
    int island_size[kPlannerMaxRanks] = {};
    int island_ranks[kPlannerMaxRanks][kPlannerMaxRanks] = {};

    int leader_rank[kPlannerMaxRanks] = {};
    int preferred_numa_node[kPlannerMaxRanks] = {};
    int staging_slot[kPlannerMaxRanks] = {};
};

__host__ __device__ __forceinline__ bool reduce_scatter_transport_is_direct(
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

inline int local_partial_ready_phase_for_island(
    int island) {
    return kReadyPhaseIslandLocalPartialBase + island;
}

inline int staged_partial_ready_phase_for_island(
    int island) {
    return kReadyPhaseIslandStagedBase + island;
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

template <int MaxTransferTasks>
bool push_reduce_task(
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

    TransferTask reduce =
        make_reduce_transfer_task(
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

    /*
     * Staged reduce-scatter reduces remote island partials from mapped staging
     * into the destination rank's output slice. Keep the same staged all-reduce
     * convention: the planner opts this task into the existing ReduceTMA lowering
     * path when the source is explicitly ShmStaging. The lowering path must allow
     * Reduce tasks whose source role is LogicalBufferRole::ShmStaging.
     */
    if (src.role == LogicalBufferRole::ShmStaging) {
        reduce.requires_tma_load = true;
        reduce.requires_tma_store = false;
        reduce.requires_tma_reduce = true;
        reduce.requires_native_atomic = true;
    }

    return push_task_or_abort(plan, reduce);
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

inline bool build_reduce_scatter_islands(
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
                    reduce_scatter_transport_is_direct(
                        transports.kind[rank][peer]) &&
                    reduce_scatter_transport_is_direct(
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
        out->preferred_numa_node[island] = -1;
        out->staging_slot[island] = -1;
    }

    for (int rank = 0; rank < input.world_size; ++rank) {
        const int island = out->island_of_rank[rank];
        const int idx = out->island_size[island]++;

        out->island_ranks[island][idx] = rank;

        if (out->leader_rank[island] < 0 ||
            rank < out->leader_rank[island]) {
            out->leader_rank[island] = rank;
        }
    }

    for (int island = 0; island < out->island_count; ++island) {
        out->preferred_numa_node[island] =
            rank_numa_node(
                input,
                out->leader_rank[island]);
    }

    return true;
}

inline bool fill_reduce_scatter_slice_metadata(
    const TransferPlanBuildInput& input,
    std::size_t* slice_begin_bytes,
    std::size_t* slice_bytes,
    int* slice_windows,
    std::size_t* out_total_bytes,
    int* out_total_windows) {
    if (slice_begin_bytes == nullptr ||
        slice_bytes == nullptr ||
        slice_windows == nullptr ||
        out_total_bytes == nullptr ||
        out_total_windows == nullptr) {
        return false;
    }

    const std::size_t total_bytes =
        input.count * input.dtype_size;

    const int total_windows =
        window_count_for_transfer_bytes(
            total_bytes,
            input.launch_config);

    if (total_bytes == 0 || total_windows <= 0) {
        return false;
    }

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

    *out_total_bytes = total_bytes;
    *out_total_windows = total_windows;
    return true;
}

inline bool assign_staging_slots(
    const TransferPlanBuildInput& input,
    IslandPlan* islands,
    std::size_t required_bytes) {
    if (islands == nullptr || required_bytes == 0) {
        return false;
    }

    bool used[kPlannerMaxStagingSlots] = {};

    for (int island = 0; island < islands->island_count; ++island) {
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
bool emit_reduce_scatter_optional_slice_copies(
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
bool emit_reduce_scatter_optional_full_copies_for_staged_path(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    std::size_t total_bytes,
    int total_windows,
    int* task_phase) {
    if (!input.out_of_place) {
        return true;
    }

    /*
     * The staged path builds full-tensor island partials on each island leader.
     * Therefore, if the input/output API is out-of-place, each rank buffer must
     * first receive the full logical input, not only the local output slice.
     */
    for (int rank = 0; rank < input.world_size; ++rank) {
        if (!push_copy_task(
                plan,
                input,
                rank,
                rank,
                rank,
                rank_input_ref(rank, 0),
                rank_buffer_ref(rank, 0),
                total_bytes,
                total_windows,
                topology::TransportKind::DirectNvlink,
                &task_phase[rank])) {
            return false;
        }
    }

    return true;
}

template <int MaxTransferTasks>
bool emit_reduce_scatter_entry_rendezvous(
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
bool build_direct_reduce_scatter_plan(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    const TransportMatrix& transports) {
    int task_phase[kPlannerMaxRanks] = {};
    std::size_t slice_begin_bytes[kPlannerMaxRanks] = {};
    std::size_t slice_bytes[kPlannerMaxRanks] = {};
    int slice_windows[kPlannerMaxRanks] = {};
    std::size_t total_bytes = 0;
    int total_windows = 0;

    if (!fill_reduce_scatter_slice_metadata(
            input,
            slice_begin_bytes,
            slice_bytes,
            slice_windows,
            &total_bytes,
            &total_windows)) {
        transfer_plan_abort_build(plan);
        return false;
    }

    if (!emit_reduce_scatter_optional_slice_copies(
            plan,
            input,
            slice_begin_bytes,
            slice_bytes,
            slice_windows,
            task_phase)) {
        return false;
    }

    if (!emit_reduce_scatter_entry_rendezvous(
            plan,
            input,
            task_phase)) {
        return false;
    }

    for (int rank = 0; rank < input.world_size; ++rank) {
        for (int peer = 0; peer < input.world_size; ++peer) {
            if (peer == rank) {
                continue;
            }

            if (!reduce_scatter_transport_is_direct(
                    transports.kind[rank][peer])) {
                transfer_plan_abort_build(plan);
                return false;
            }

            if (!push_reduce_task(
                    plan,
                    input,
                    rank,
                    peer,
                    rank,
                    rank_buffer_ref(peer, slice_begin_bytes[rank]),
                    rank_buffer_ref(rank, slice_begin_bytes[rank]),
                    slice_bytes[rank],
                    slice_windows[rank],
                    transports.kind[rank][peer],
                    &task_phase[rank])) {
                return false;
            }
        }
    }

    mark_last_task_per_rank_terminal(plan, input.world_size);
    return true;
}

template <int MaxTransferTasks>
bool emit_intra_island_reductions_to_leaders(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    const TransportMatrix& transports,
    const IslandPlan& islands,
    std::size_t total_bytes,
    int total_windows,
    int* task_phase) {
    for (int island = 0; island < islands.island_count; ++island) {
        const int leader = islands.leader_rank[island];

        for (int idx = 0; idx < islands.island_size[island]; ++idx) {
            const int rank = islands.island_ranks[island][idx];

            if (rank == leader) {
                continue;
            }

            if (!reduce_scatter_transport_is_direct(
                    transports.kind[leader][rank])) {
                transfer_plan_abort_build(plan);
                return false;
            }

            if (!push_reduce_task(
                    plan,
                    input,
                    leader,
                    rank,
                    leader,
                    rank_buffer_ref(rank, 0),
                    rank_buffer_ref(leader, 0),
                    total_bytes,
                    total_windows,
                    transports.kind[leader][rank],
                    &task_phase[leader])) {
                return false;
            }
        }
    }

    return true;
}

template <int MaxTransferTasks>
bool emit_local_leader_partials_to_island_ranks(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    const TransportMatrix& transports,
    const IslandPlan& islands,
    const std::size_t* slice_begin_bytes,
    const std::size_t* slice_bytes,
    const int* slice_windows,
    int* task_phase) {
    for (int island = 0; island < islands.island_count; ++island) {
        const int leader = islands.leader_rank[island];

        int local_waiters[kPlannerMaxRanks] = {};
        int local_waiter_count = 0;

        for (int idx = 0; idx < islands.island_size[island]; ++idx) {
            const int rank = islands.island_ranks[island][idx];

            if (rank == leader) {
                continue;
            }

            local_waiters[local_waiter_count++] = rank;
        }

        if (!push_ready_publish_for_waiters(
                plan,
                input,
                leader,
                local_waiters,
                local_waiter_count,
                local_partial_ready_phase_for_island(island),
                &task_phase[leader])) {
            return false;
        }

        for (int i = 0; i < local_waiter_count; ++i) {
            const int rank = local_waiters[i];

            if (!push_ready_wait_task(
                    plan,
                    input,
                    rank,
                    leader,
                    local_partial_ready_phase_for_island(island),
                    &task_phase[rank])) {
                return false;
            }

            if (!reduce_scatter_transport_is_direct(
                    transports.kind[rank][leader])) {
                transfer_plan_abort_build(plan);
                return false;
            }

            /*
             * The leader now owns the local-island partial for every output
             * slice. Non-leader ranks overwrite their local slice with that
             * partial before reducing remote island partials into it.
             */
            if (!push_copy_task(
                    plan,
                    input,
                    rank,
                    leader,
                    rank,
                    rank_buffer_ref(leader, slice_begin_bytes[rank]),
                    rank_buffer_ref(rank, slice_begin_bytes[rank]),
                    slice_bytes[rank],
                    slice_windows[rank],
                    transports.kind[rank][leader],
                    &task_phase[rank])) {
                return false;
            }
        }
    }

    return true;
}

template <int MaxTransferTasks>
bool emit_leader_partials_to_staging(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    const IslandPlan& islands,
    std::size_t total_bytes,
    int total_windows,
    int* task_phase) {
    for (int island = 0; island < islands.island_count; ++island) {
        const int leader = islands.leader_rank[island];
        const int staging_slot = islands.staging_slot[island];

        if (!staging_slice_valid(
                input,
                staging_slot,
                0,
                total_bytes)) {
            transfer_plan_abort_build(plan);
            return false;
        }

        if (!push_copy_task(
                plan,
                input,
                leader,
                leader,
                leader,
                rank_buffer_ref(leader, 0),
                shm_staging_ref(staging_slot, 0),
                total_bytes,
                total_windows,
                topology::TransportKind::Shm,
                &task_phase[leader])) {
            return false;
        }

        int remote_waiters[kPlannerMaxRanks] = {};
        int remote_waiter_count = 0;

        for (int dst_island = 0;
             dst_island < islands.island_count;
             ++dst_island) {
            if (dst_island == island) {
                continue;
            }

            for (int idx = 0;
                 idx < islands.island_size[dst_island];
                 ++idx) {
                remote_waiters[remote_waiter_count++] =
                    islands.island_ranks[dst_island][idx];
            }
        }

        if (!push_ready_publish_for_waiters(
                plan,
                input,
                leader,
                remote_waiters,
                remote_waiter_count,
                staged_partial_ready_phase_for_island(island),
                &task_phase[leader])) {
            return false;
        }
    }

    return true;
}

template <int MaxTransferTasks>
bool emit_remote_partials_to_destination_ranks(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    const IslandPlan& islands,
    const std::size_t* slice_begin_bytes,
    const std::size_t* slice_bytes,
    const int* slice_windows,
    int* task_phase) {
    for (int dst_island = 0;
         dst_island < islands.island_count;
         ++dst_island) {
        for (int idx = 0;
             idx < islands.island_size[dst_island];
             ++idx) {
            const int dst_rank =
                islands.island_ranks[dst_island][idx];

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

                if (!push_ready_wait_task(
                        plan,
                        input,
                        dst_rank,
                        src_leader,
                        staged_partial_ready_phase_for_island(src_island),
                        &task_phase[dst_rank])) {
                    return false;
                }

                if (!push_reduce_task(
                        plan,
                        input,
                        dst_rank,
                        src_leader,
                        dst_rank,
                        shm_staging_ref(
                            staging_slot,
                            slice_begin_bytes[dst_rank]),
                        rank_buffer_ref(
                            dst_rank,
                            slice_begin_bytes[dst_rank]),
                        slice_bytes[dst_rank],
                        slice_windows[dst_rank],
                        topology::TransportKind::Shm,
                        &task_phase[dst_rank])) {
                    return false;
                }
            }
        }
    }

    return true;
}

template <int MaxTransferTasks>
bool build_staged_island_reduce_scatter_plan(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    const TransportMatrix& transports,
    IslandPlan islands) {
    int task_phase[kPlannerMaxRanks] = {};
    std::size_t slice_begin_bytes[kPlannerMaxRanks] = {};
    std::size_t slice_bytes[kPlannerMaxRanks] = {};
    int slice_windows[kPlannerMaxRanks] = {};
    std::size_t total_bytes = 0;
    int total_windows = 0;

    if (!fill_reduce_scatter_slice_metadata(
            input,
            slice_begin_bytes,
            slice_bytes,
            slice_windows,
            &total_bytes,
            &total_windows)) {
        transfer_plan_abort_build(plan);
        return false;
    }

    if (!assign_staging_slots(
            input,
            &islands,
            total_bytes)) {
        transfer_plan_abort_build(plan);
        return false;
    }

    if (!emit_reduce_scatter_optional_full_copies_for_staged_path(
            plan,
            input,
            total_bytes,
            total_windows,
            task_phase)) {
        return false;
    }

    if (!emit_reduce_scatter_entry_rendezvous(
            plan,
            input,
            task_phase)) {
        return false;
    }

    if (!emit_intra_island_reductions_to_leaders(
            plan,
            input,
            transports,
            islands,
            total_bytes,
            total_windows,
            task_phase)) {
        return false;
    }

    if (!emit_local_leader_partials_to_island_ranks(
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

    if (!emit_leader_partials_to_staging(
            plan,
            input,
            islands,
            total_bytes,
            total_windows,
            task_phase)) {
        return false;
    }

    if (!emit_remote_partials_to_destination_ranks(
            plan,
            input,
            islands,
            slice_begin_bytes,
            slice_bytes,
            slice_windows,
            task_phase)) {
        return false;
    }

    mark_last_task_per_rank_terminal(plan, input.world_size);
    return true;
}

} // namespace reduce_scatter_detail

/*
 * Logical reduce-scatter planner.
 *
 * Direct case:
 *   Preserve the original sharded reduce-scatter plan:
 *     each rank owns one output slice and directly reduces every peer's matching
 *     slice into that rank's result buffer.
 *
 * SYS/staged island case:
 *   1. Build direct-access islands.
 *   2. Each island leader reduces all local island ranks' full tensors into the
 *      leader buffer. This creates one full-tensor partial per island.
 *   3. Non-leader island ranks wait for the local leader and copy their own
 *      output slice from the leader partial.
 *   4. Each island leader copies its full partial to a host-mapped staging slot
 *      and publishes staged-ready to all remote destination ranks.
 *   5. Every destination rank waits on each remote island staged-ready signal,
 *      then reduces only its own output slice from that remote staged partial
 *      into its rank buffer.
 *
 * This intentionally mirrors the staged all-reduce planner's island/staging
 * structure, but it only imports remote slices needed by each destination rank.
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

    reduce_scatter_detail::IslandPlan islands{};
    if (!reduce_scatter_detail::build_reduce_scatter_islands(
            input,
            transports,
            &islands)) {
        transfer_plan_abort_build(plan);
        return false;
    }

    if (islands.island_count <= 1) {
        const bool ok =
            reduce_scatter_detail::build_direct_reduce_scatter_plan(
                plan,
                input,
                transports);

        if (ok) {
            debug_print_transfer_plan_if_enabled("reduce_scatter", *plan);
        }

        return ok;
    }

    const bool ok =
        reduce_scatter_detail::build_staged_island_reduce_scatter_plan(
            plan,
            input,
            transports,
            islands);

    if (ok) {
        debug_print_transfer_plan_if_enabled("reduce_scatter", *plan);
    }

    return ok;
}

} // namespace plan
} // namespace comm
} // namespace ooverlap
