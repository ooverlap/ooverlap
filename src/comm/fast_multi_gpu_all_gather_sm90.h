#pragma once

#include "comm/ooverlap_comm_private.h"
#include "comm/params.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ooverlap {

/*
 * One CTA loads one chunk of this rank's partition and fanout-stores it to all
 * peers. The budget therefore applies to the local partition, not to the full
 * gathered tensor.
 */
constexpr int kFastAllGatherMaxCtas = 3;

static_assert(
    TMA_TWO_GPU_PEER_SMALL_TASK_BYTES > 0,
    "fast all-gather CTA bytes must be positive");
static_assert(
    (TMA_TWO_GPU_PEER_SMALL_TASK_BYTES % 16) == 0,
    "fast all-gather CTA bytes must be 16-byte aligned");
static_assert(
    kFastAllGatherMaxCtas > 0,
    "fast all-gather CTA budget must be positive");
static_assert(
    kFastAllGatherMaxCtas <= TMA_TWO_GPU_PEER_MAX_CTAS,
    "fast all-gather CTA budget exceeds the global limit");

inline bool fast_all_gather_eligible(
    const oo_group_t* group,
    const comm::api::CollectiveLaunchState& launch,
    size_t count) {
    if (group == nullptr ||
        !group->is_all_peer_to_peer ||
        launch.out_of_place ||
        launch.local_ptr == nullptr ||
        launch.rank < 0 ||
        launch.rank >= launch.world_size ||
        launch.world_size < 2 ||
        launch.world_size != group->num_devices ||
        launch.peer_count != launch.world_size - 1 ||
        launch.peer_count <= 0 ||
        launch.peer_count > TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS ||
        launch.dtype_size == 0 ||
        launch.bytes == 0 ||
        count == 0 ||
        count > static_cast<size_t>(-1) / launch.dtype_size ||
        count * launch.dtype_size != launch.bytes ||
        launch.collective_epoch <= 0) {
        return false;
    }

    const void* rank_ptrs[kOoMaxLocalDevices] = {};
    bool seen_rank[kOoMaxLocalDevices] = {};

    rank_ptrs[launch.rank] = launch.local_ptr;
    seen_rank[launch.rank] = true;

    for (int peer_idx = 0;
         peer_idx < launch.peer_count;
         ++peer_idx) {
        const int peer_rank = launch.peer_ranks[peer_idx];

        if (peer_rank < 0 ||
            peer_rank >= launch.world_size ||
            peer_rank >= kOoMaxLocalDevices ||
            seen_rank[peer_rank] ||
            launch.peer_ptrs[peer_idx] == nullptr ||
            launch.peer_publish_signals[peer_idx] == nullptr ||
            launch.local_wait_signals[peer_idx] == nullptr) {
            return false;
        }

        rank_ptrs[peer_rank] = launch.peer_ptrs[peer_idx];
        seen_rank[peer_rank] = true;
    }

    const size_t world = static_cast<size_t>(launch.world_size);
    const size_t base = count / world;
    const size_t rem = count - base * world;

    /*
     * Make the fast/fallback decision collective-wide. Every rank validates
     * every partition so uneven counts cannot send some ranks down the fast
     * path while other ranks enter the generic rendezvous protocol.
     */
    for (int src_rank = 0;
         src_rank < launch.world_size;
         ++src_rank) {
        if (!seen_rank[src_rank] || rank_ptrs[src_rank] == nullptr) {
            return false;
        }

        const size_t rank = static_cast<size_t>(src_rank);
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
            shard_bytes > static_cast<size_t>(UINT32_MAX) ||
            (shard_offset_bytes % 16) != 0 ||
            (shard_bytes % 16) != 0) {
            return false;
        }

        const std::uintptr_t source_address =
            reinterpret_cast<std::uintptr_t>(rank_ptrs[src_rank]) +
            shard_offset_bytes;

        if ((source_address & static_cast<std::uintptr_t>(15)) != 0) {
            return false;
        }

        const size_t ctas =
            (shard_bytes +
             static_cast<size_t>(TMA_TWO_GPU_PEER_SMALL_TASK_BYTES) - 1) /
            static_cast<size_t>(TMA_TWO_GPU_PEER_SMALL_TASK_BYTES);

        if (ctas == 0 ||
            ctas > static_cast<size_t>(kFastAllGatherMaxCtas)) {
            return false;
        }
    }

    return true;
}

cudaError_t enqueue_fast_multi_gpu_all_gather_rank_sm90(
    const comm::api::CollectiveLaunchState& launch,
    size_t count,
    oo_dtype_t dtype,
    cudaStream_t stream,
    int scratch_index);

} // namespace ooverlap
