#pragma once

#include "comm/plan/tma_multi_gpu_plan_utils.cuh"

namespace ooverlap {
namespace comm {
namespace plan {

/*
 * Logical reduce-scatter planner.
 *
 * Logical shape per executor rank:
 *   optional local input copy
 *   ready rendezvous
 *   reduce every peer's rank slice into executor rank's result slice
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
                input.topo,
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

} // namespace plan
} // namespace comm
} // namespace ooverlap
