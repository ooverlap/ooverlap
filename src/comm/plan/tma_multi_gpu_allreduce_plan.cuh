#pragma once

#include "comm/plan/tma_multi_gpu_plan_utils.cuh"

namespace ooverlap {
namespace comm {
namespace plan {

namespace allreduce_detail {

constexpr int kReadyPhaseEntry = 0;
constexpr int kReadyPhaseIslandPartialStagedBase = 1;
constexpr int kReadyPhaseIslandFinalBase =
    kReadyPhaseIslandPartialStagedBase + kPlannerMaxRanks;

struct IslandPlan {
    int island_count = 0;

    int island_of_rank[kPlannerMaxRanks] = {};
    int island_size[kPlannerMaxRanks] = {};
    int island_ranks[kPlannerMaxRanks][kPlannerMaxRanks] = {};

    int leader_rank[kPlannerMaxRanks] = {};
    int preferred_numa_node[kPlannerMaxRanks] = {};
    int staging_slot[kPlannerMaxRanks] = {};
};

__host__ __device__ __forceinline__ bool allreduce_transport_is_direct(
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

inline int staged_partial_ready_phase_for_island(
    int island) {
    return kReadyPhaseIslandPartialStagedBase + island;
}

inline int final_ready_phase_for_island(
    int island) {
    return kReadyPhaseIslandFinalBase + island;
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
     * Staged all-reduce needs to reduce a partial tensor loaded from a mapped
     * staging slot into the leader's rank buffer. The generic helper marks
     * Shm as non-TMA-capable. For this first staged planner, opt the task into
     * the existing ReduceTMA lowering path when the source is explicitly
     * ShmStaging. The lowering file must also allow this case.
     */
    if (src.role == LogicalBufferRole::ShmStaging) {
        reduce.requires_tma_load = true;
        reduce.requires_tma_store = false;
        reduce.requires_tma_reduce = true;
        reduce.requires_native_atomic = true;
    }

    return push_task_or_abort(plan, reduce);
}


/*
 * OOVERLAP_OUT_OF_PLACE_ALLREDUCE_REDUCE_FANOUT_PATCH
 *
 * Push a ReduceFanout task for out-of-place allreduce.
 */
template <int MaxTransferTasks>
bool push_reduce_fanout_task(
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
        num_windows <= 0) {
        transfer_plan_abort_build(plan);
        return false;
    }

    const TransferTask reduce_fanout =
        make_reduce_fanout_transfer_task(
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

    return push_task_or_abort(plan, reduce_fanout);
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

__host__ __device__ __forceinline__ bool logical_buffer_ref_equal(
    const LogicalBufferRef& lhs,
    const LogicalBufferRef& rhs) {
    return lhs.role == rhs.role &&
           lhs.owner_rank == rhs.owner_rank &&
           lhs.staging_slot == rhs.staging_slot &&
           lhs.byte_offset == rhs.byte_offset;
}

__host__ __device__ __forceinline__ bool logical_buffer_is_local_device_buffer(
    const LogicalBufferRef& ref,
    int executor_rank) {
    return executor_rank >= 0 &&
           ref.owner_rank == executor_rank &&
           (ref.role == LogicalBufferRole::RankBuffer ||
            ref.role == LogicalBufferRole::RankOutput);
}

inline bool reduce_task_writes_range(
    const TransferTask& task,
    int executor_rank,
    const LogicalBufferRef& range,
    std::size_t bytes) {
    if (task.executor_rank != executor_rank ||
        task.bytes != bytes) {
        return false;
    }

    if (task.op == TransferOp::Reduce) {
        return logical_buffer_ref_equal(task.dst, range);
    }

    if (task.op == TransferOp::ReduceFanout) {
        for (int i = 0; i < task.fanout_dst_count; ++i) {
            if (logical_buffer_ref_equal(task.fanout_dsts[i], range)) {
                return true;
            }
        }
    }

    return false;
}

inline bool barrier_matches_range(
    const TransferTask& task,
    int executor_rank,
    const LogicalBufferRef& range,
    std::size_t bytes) {
    return task.op == TransferOp::Barrier &&
           task.executor_rank == executor_rank &&
           task.bytes == bytes &&
           logical_buffer_ref_equal(task.src, range);
}

template <int MaxTransferTasks>
bool insert_local_reduce_copy_barriers(
    TransferPlan<MaxTransferTasks>* plan) {
    if (plan == nullptr ||
        plan->total_tasks < 0 ||
        plan->total_tasks > MaxTransferTasks) {
        return false;
    }

    for (int consumer_idx = 0;
         consumer_idx < plan->total_tasks;
         ++consumer_idx) {
        const TransferTask consumer = plan->tasks[consumer_idx];

        if (consumer.op != TransferOp::Copy &&
            consumer.op != TransferOp::CopyFanout) {
            continue;
        }

        const LogicalBufferRef range = consumer.src;

        if (consumer.bytes == 0 ||
            !logical_buffer_is_local_device_buffer(
                range,
                consumer.executor_rank)) {
            continue;
        }

        bool has_producer = false;
        bool has_barrier = false;

        for (int i = 0; i < consumer_idx; ++i) {
            const TransferTask& previous = plan->tasks[i];

            if (barrier_matches_range(
                    previous,
                    consumer.executor_rank,
                    range,
                    consumer.bytes)) {
                has_barrier = true;
                break;
            }

            if (reduce_task_writes_range(
                    previous,
                    consumer.executor_rank,
                    range,
                    consumer.bytes)) {
                has_producer = true;
            }
        }

        if (!has_producer || has_barrier) {
            continue;
        }

        if (plan->total_tasks >= MaxTransferTasks) {
            return false;
        }

        for (int i = plan->total_tasks; i > consumer_idx; --i) {
            plan->tasks[i] = plan->tasks[i - 1];
        }

        TransferTask barrier{};
        barrier.op = TransferOp::Barrier;
        barrier.executor_rank = consumer.executor_rank;
        barrier.src_rank = consumer.executor_rank;
        barrier.dst_rank = consumer.executor_rank;
        barrier.src = range;
        barrier.dst = range;
        barrier.bytes = consumer.bytes;
        barrier.phase = consumer.phase;

        plan->tasks[consumer_idx] = barrier;
        ++plan->total_tasks;

        /* Skip the consumer that was shifted one slot to the right. */
        ++consumer_idx;
    }

    int phase_by_rank[kPlannerMaxRanks] = {};

    for (int i = 0; i < plan->total_tasks; ++i) {
        TransferTask& task = plan->tasks[i];

        if (task.executor_rank >= 0 &&
            task.executor_rank < kPlannerMaxRanks) {
            task.phase = phase_by_rank[task.executor_rank]++;
        }
    }

    return true;
}

inline bool build_allreduce_islands(
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
                    allreduce_transport_is_direct(
                        transports.kind[rank][peer]) &&
                    allreduce_transport_is_direct(
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
bool emit_allreduce_optional_local_copies(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    std::size_t total_bytes,
    int num_windows,
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
                rank_input_ref(rank, 0),
                rank_buffer_ref(rank, 0),
                total_bytes,
                num_windows,
                topology::TransportKind::DirectNvlink,
                &task_phase[rank])) {
            return false;
        }
    }

    return true;
}

template <int MaxTransferTasks>
bool emit_allreduce_entry_rendezvous(
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


/*
 * OOVERLAP_OUT_OF_PLACE_ALLREDUCE_REDUCE_FANOUT_PATCH
 *
 * Direct out-of-place allreduce fast path:
 *   for each source rank S:
 *       TMA load rank_input[S]
 *       TMA reduce-fanout into rank_output[0..world_size)
 *
 * Destination output buffers must already contain the reduction identity.
 * For the current f16 add/sum fanout implementation, that means zeroed output.
 */
template <int MaxTransferTasks>
bool build_out_of_place_direct_allreduce_fanout_plan(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    const TransportMatrix& transports) {
    if (plan == nullptr || !input.out_of_place) {
        return false;
    }

    if (input.world_size <= 0 ||
        input.world_size > TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS ||
        input.dtype_size == 0 ||
        input.count > static_cast<std::size_t>(-1) / input.dtype_size) {
        transfer_plan_abort_build(plan);
        return false;
    }

    const std::size_t total_bytes = input.count * input.dtype_size;
    const int num_windows =
        window_count_for_transfer_bytes(
            total_bytes,
            input.launch_config);

    if (total_bytes == 0 || num_windows <= 0) {
        transfer_plan_abort_build(plan);
        return false;
    }

    int task_phase[kPlannerMaxRanks] = {};

    if (!emit_allreduce_entry_rendezvous(
            plan,
            input,
            task_phase)) {
        return false;
    }

    for (int src_rank = 0; src_rank < input.world_size; ++src_rank) {
        int dst_ranks[TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS] = {};
        LogicalBufferRef dst_refs[TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS] = {};

        topology::TransportKind selected_transport =
            topology::TransportKind::DirectNvlink;

        for (int dst_rank = 0; dst_rank < input.world_size; ++dst_rank) {
            if (!allreduce_transport_is_direct(
                    transports.kind[src_rank][dst_rank])) {
                transfer_plan_abort_build(plan);
                return false;
            }

            if (src_rank != dst_rank) {
                selected_transport = transports.kind[src_rank][dst_rank];
            }

            dst_ranks[dst_rank] = dst_rank;
            dst_refs[dst_rank] = rank_output_ref(dst_rank, 0);
        }

        if (!push_reduce_fanout_task(
                plan,
                input,
                src_rank,
                src_rank,
                dst_ranks,
                rank_input_ref(src_rank, 0),
                dst_refs,
                input.world_size,
                total_bytes,
                num_windows,
                selected_transport,
                &task_phase[src_rank])) {
            return false;
        }
    }

    mark_last_task_per_rank_terminal(plan, input.world_size);
    return true;
}

template <int MaxTransferTasks>
bool build_direct_allreduce_plan(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    const TransportMatrix& transports) {
    int task_phase[kPlannerMaxRanks] = {};

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

        int& phase = task_phase[rank];

        if (input.out_of_place) {
            if (!push_copy_task(
                    plan,
                    input,
                    rank,
                    rank,
                    rank,
                    rank_input_ref(rank, slice_begin_bytes),
                    rank_buffer_ref(rank, slice_begin_bytes),
                    slice_bytes,
                    num_windows,
                    topology::TransportKind::DirectNvlink,
                    &phase)) {
                return false;
            }
        }

        if (!append_ready_rendezvous_tasks(
                plan,
                input.topo,
                rank,
                input.world_size,
                &phase)) {
            return false;
        }

        for (int peer = 0; peer < input.world_size; ++peer) {
            if (peer == rank) {
                continue;
            }

            if (!allreduce_transport_is_direct(
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
                    rank_buffer_ref(peer, slice_begin_bytes),
                    rank_buffer_ref(rank, slice_begin_bytes),
                    slice_bytes,
                    num_windows,
                    transports.kind[rank][peer],
                    &phase)) {
                return false;
            }
        }

        for (int peer = 0; peer < input.world_size; ++peer) {
            if (peer == rank) {
                continue;
            }

            if (!allreduce_transport_is_direct(
                    transports.kind[rank][peer])) {
                transfer_plan_abort_build(plan);
                return false;
            }

            if (!push_copy_task(
                    plan,
                    input,
                    rank,
                    rank,
                    peer,
                    rank_buffer_ref(rank, slice_begin_bytes),
                    rank_buffer_ref(peer, slice_begin_bytes),
                    slice_bytes,
                    num_windows,
                    transports.kind[rank][peer],
                    &phase)) {
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
    int num_windows,
    int* task_phase) {
    for (int island = 0; island < islands.island_count; ++island) {
        const int leader = islands.leader_rank[island];

        for (int idx = 0; idx < islands.island_size[island]; ++idx) {
            const int rank = islands.island_ranks[island][idx];

            if (rank == leader) {
                continue;
            }

            if (!allreduce_transport_is_direct(
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
                    num_windows,
                    transports.kind[leader][rank],
                    &task_phase[leader])) {
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
    int num_windows,
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
            if (dst_island == island) {
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
                staged_partial_ready_phase_for_island(island),
                &task_phase[leader])) {
            return false;
        }
    }

    return true;
}

template <int MaxTransferTasks>
bool emit_remote_partials_to_destination_leaders(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    const IslandPlan& islands,
    std::size_t total_bytes,
    int num_windows,
    int* task_phase) {
    for (int dst_island = 0;
         dst_island < islands.island_count;
         ++dst_island) {
        const int dst_leader = islands.leader_rank[dst_island];

        for (int src_island = 0;
             src_island < islands.island_count;
             ++src_island) {
            if (src_island == dst_island) {
                continue;
            }

            const int src_leader = islands.leader_rank[src_island];
            const int staging_slot = islands.staging_slot[src_island];

            if (!push_ready_wait_task(
                    plan,
                    input,
                    dst_leader,
                    src_leader,
                    staged_partial_ready_phase_for_island(src_island),
                    &task_phase[dst_leader])) {
                return false;
            }

            if (!push_reduce_task(
                    plan,
                    input,
                    dst_leader,
                    src_leader,
                    dst_leader,
                    shm_staging_ref(staging_slot, 0),
                    rank_buffer_ref(dst_leader, 0),
                    total_bytes,
                    num_windows,
                    topology::TransportKind::Shm,
                    &task_phase[dst_leader])) {
                return false;
            }
        }
    }

    return true;
}

template <int MaxTransferTasks>
bool emit_leader_final_to_island_ranks(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    const TransportMatrix& transports,
    const IslandPlan& islands,
    std::size_t total_bytes,
    int num_windows,
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
                final_ready_phase_for_island(island),
                &task_phase[leader])) {
            return false;
        }

        for (int idx = 0; idx < islands.island_size[island]; ++idx) {
            const int rank = islands.island_ranks[island][idx];

            if (rank == leader) {
                continue;
            }

            if (!push_ready_wait_task(
                    plan,
                    input,
                    rank,
                    leader,
                    final_ready_phase_for_island(island),
                    &task_phase[rank])) {
                return false;
            }

            if (!allreduce_transport_is_direct(
                    transports.kind[rank][leader])) {
                transfer_plan_abort_build(plan);
                return false;
            }

            if (!push_copy_task(
                    plan,
                    input,
                    rank,
                    leader,
                    rank,
                    rank_buffer_ref(leader, 0),
                    rank_buffer_ref(rank, 0),
                    total_bytes,
                    num_windows,
                    transports.kind[rank][leader],
                    &task_phase[rank])) {
                return false;
            }
        }
    }

    return true;
}

template <int MaxTransferTasks>
bool build_staged_island_allreduce_plan(
    TransferPlan<MaxTransferTasks>* plan,
    const TransferPlanBuildInput& input,
    const TransportMatrix& transports,
    IslandPlan islands) {
    const std::size_t total_bytes = input.count * input.dtype_size;
    const int num_windows =
        window_count_for_transfer_bytes(
            total_bytes,
            input.launch_config);

    if (total_bytes == 0 || num_windows <= 0) {
        transfer_plan_abort_build(plan);
        return false;
    }

    int task_phase[kPlannerMaxRanks] = {};

    if (!assign_staging_slots(
            input,
            &islands,
            total_bytes)) {
        transfer_plan_abort_build(plan);
        return false;
    }

    if (!emit_allreduce_optional_local_copies(
            plan,
            input,
            total_bytes,
            num_windows,
            task_phase)) {
        return false;
    }

    if (!emit_allreduce_entry_rendezvous(
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
            num_windows,
            task_phase)) {
        return false;
    }

    if (!emit_leader_partials_to_staging(
            plan,
            input,
            islands,
            total_bytes,
            num_windows,
            task_phase)) {
        return false;
    }

    if (!emit_remote_partials_to_destination_leaders(
            plan,
            input,
            islands,
            total_bytes,
            num_windows,
            task_phase)) {
        return false;
    }

    if (!emit_leader_final_to_island_ranks(
            plan,
            input,
            transports,
            islands,
            total_bytes,
            num_windows,
            task_phase)) {
        return false;
    }

    mark_last_task_per_rank_terminal(plan, input.world_size);
    return true;
}

} // namespace allreduce_detail

/*
 * Logical all-reduce planner.
 *
 * Direct case:
 *   Preserve the existing sharded all-reduce plan:
 *     each rank reduces one rank slice, then broadcasts that result slice.
 *
 * SYS/staged island case:
 *   1. Build direct-access islands.
 *   2. Each island leader reduces all local island ranks' full buffers into the
 *      leader buffer. This creates one full-tensor partial per island.
 *   3. Each island leader copies its full partial to a host-mapped staging slot.
 *   4. Every destination island leader waits on each remote island staged-ready
 *      signal, then reduces that remote staged partial into its own leader
 *      buffer.
 *   5. The island leader publishes final-ready to local island ranks.
 *   6. Local island ranks copy the full final result from the leader over direct
 *      island transport.
 *
 * This first staged version intentionally uses one full tensor staging slot per
 * island. That is simple and avoids staging slot reuse hazards. It requires the
 * lowering path to allow Reduce tasks whose source is LogicalBufferRole::ShmStaging.
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

    allreduce_detail::IslandPlan islands{};
    if (!allreduce_detail::build_allreduce_islands(
            input,
            transports,
            &islands)) {
        transfer_plan_abort_build(plan);
        return false;
    }

    if (islands.island_count <= 1) {
        const bool ok =
            input.out_of_place
                ? allreduce_detail::build_out_of_place_direct_allreduce_fanout_plan(
                    plan,
                    input,
                    transports)
                : allreduce_detail::build_direct_allreduce_plan(
                    plan,
                    input,
                    transports);

        if (!ok) {
            return false;
        }

        if (!allreduce_detail::insert_local_reduce_copy_barriers(plan)) {
            transfer_plan_abort_build(plan);
            return false;
        }

        debug_print_transfer_plan_if_enabled("allreduce", *plan);
        return true;
    }

    if (input.out_of_place) {
        transfer_plan_abort_build(plan);
        return false;
    }

    const bool ok =
        allreduce_detail::build_staged_island_allreduce_plan(
            plan,
            input,
            transports,
            islands);

    if (!ok) {
        return false;
    }

    if (!allreduce_detail::insert_local_reduce_copy_barriers(plan)) {
        transfer_plan_abort_build(plan);
        return false;
    }

    debug_print_transfer_plan_if_enabled("allreduce", *plan);
    return true;
}

} // namespace plan
} // namespace comm
} // namespace ooverlap
