#pragma once

#include "comm/ooverlap_comm_private.h"
#include "comm/params.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ooverlap {

constexpr size_t kFastReduceScatterMaxBytes =
    static_cast<size_t>(TMA_TWO_GPU_PEER_SMALL_TASK_BYTES) *
    static_cast<size_t>(6);

static_assert(
    TMA_TWO_GPU_PEER_SMALL_TASK_BYTES > 0,
    "fast reduce-scatter CTA bytes must be positive");
static_assert(
    (TMA_TWO_GPU_PEER_SMALL_TASK_BYTES % 16) == 0,
    "fast reduce-scatter CTA bytes must be 16-byte aligned");
static_assert(
    TMA_TWO_GPU_PEER_MAX_CTAS > 0,
    "fast reduce-scatter must allow at least one CTA");

inline bool fast_reduce_scatter_eligible(
    const oo_group_t* group,
    const comm::api::CollectiveLaunchState& launch,
    size_t count) {
    if (group == nullptr ||
        !group->is_all_peer_to_peer ||
        launch.out_of_place ||
        launch.local_ptr == nullptr ||
        launch.rank < 0 ||
        launch.world_size < 2 ||
        launch.world_size != group->num_devices ||
        launch.peer_count != launch.world_size - 1 ||
        launch.peer_count > TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS ||
        launch.dtype_size == 0 ||
        launch.bytes == 0 ||
        launch.bytes > kFastReduceScatterMaxBytes ||
        count == 0 ||
        count > static_cast<size_t>(-1) / launch.dtype_size ||
        count * launch.dtype_size != launch.bytes ||
        launch.collective_epoch <= 0) {
        return false;
    }

    const size_t world = static_cast<size_t>(launch.world_size);
    const size_t base = count / world;
    const size_t rem = count - base * world;

    int total_ctas = 0;

    for (int dst_rank = 0;
         dst_rank < launch.world_size;
         ++dst_rank) {
        const size_t rank = static_cast<size_t>(dst_rank);
        const size_t shard_offset_elements =
            rank * base + ((rank < rem) ? rank : rem);
        const size_t shard_count =
            base + ((rank < rem) ? 1u : 0u);

        if (shard_count == 0 ||
            shard_offset_elements > count ||
            shard_count > count - shard_offset_elements ||
            shard_offset_elements >
                static_cast<size_t>(-1) / launch.dtype_size ||
            shard_count >
                static_cast<size_t>(-1) / launch.dtype_size) {
            return false;
        }

        const size_t shard_offset_bytes =
            shard_offset_elements * launch.dtype_size;
        const size_t shard_bytes =
            shard_count * launch.dtype_size;

        if (shard_offset_bytes > launch.bytes ||
            shard_bytes > launch.bytes - shard_offset_bytes ||
            (shard_offset_bytes % 16) != 0 ||
            (shard_bytes % 16) != 0) {
            return false;
        }

        const std::uintptr_t local_address =
            reinterpret_cast<std::uintptr_t>(launch.local_ptr) +
            shard_offset_bytes;

        if ((local_address & static_cast<std::uintptr_t>(15)) != 0) {
            return false;
        }

        if (dst_rank == launch.rank) {
            continue;
        }

        int peer_idx = -1;
        for (int candidate = 0;
             candidate < launch.peer_count;
             ++candidate) {
            if (launch.peer_ranks[candidate] == dst_rank) {
                peer_idx = candidate;
                break;
            }
        }

        if (peer_idx < 0 || launch.peer_ptrs[peer_idx] == nullptr) {
            return false;
        }

        const std::uintptr_t peer_address =
            reinterpret_cast<std::uintptr_t>(launch.peer_ptrs[peer_idx]) +
            shard_offset_bytes;

        if ((peer_address & static_cast<std::uintptr_t>(15)) != 0) {
            return false;
        }

        const size_t ctas =
            (shard_bytes +
             static_cast<size_t>(TMA_TWO_GPU_PEER_SMALL_TASK_BYTES) - 1) /
            static_cast<size_t>(TMA_TWO_GPU_PEER_SMALL_TASK_BYTES);

        if (ctas == 0 ||
            ctas > static_cast<size_t>(TMA_TWO_GPU_PEER_MAX_CTAS) ||
            total_ctas > TMA_TWO_GPU_PEER_MAX_CTAS -
                static_cast<int>(ctas)) {
            return false;
        }

        total_ctas += static_cast<int>(ctas);
    }

    return total_ctas > 0 &&
           total_ctas <= TMA_TWO_GPU_PEER_MAX_CTAS;
}

cudaError_t enqueue_fast_multi_gpu_reduce_scatter_rank_sm90(
    const comm::api::CollectiveLaunchState& launch,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    cudaStream_t stream,
    int scratch_index);

} // namespace ooverlap
