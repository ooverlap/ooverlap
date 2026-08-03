#pragma once

#include "comm/ooverlap_comm_private.h"
#include "comm/params.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ooverlap {

constexpr size_t kFastAllreduceMaxBytes =
    static_cast<size_t>(TMA_TWO_GPU_PEER_SMALL_TASK_BYTES) *
    static_cast<size_t>(TMA_TWO_GPU_PEER_MAX_CTAS);

static_assert(
    TMA_TWO_GPU_PEER_SMALL_TASK_BYTES > 0,
    "fast allreduce CTA bytes must be positive");
static_assert(
    (TMA_TWO_GPU_PEER_SMALL_TASK_BYTES % 16) == 0,
    "fast allreduce CTA bytes must be 16-byte aligned");
static_assert(
    TMA_TWO_GPU_PEER_MAX_CTAS > 0,
    "fast allreduce must allow at least one CTA");

inline bool fast_allreduce_eligible(
    const oo_group_t* group,
    size_t offset_bytes,
    size_t bytes) {
    return group != nullptr &&
           group->is_all_peer_to_peer &&
           group->num_devices >= 2 &&
           group->num_devices - 1 <= TMA_TWO_GPU_PEER_MAX_FANOUT_DSTS &&
           bytes != 0 &&
           bytes <= kFastAllreduceMaxBytes &&
           (offset_bytes % 16) == 0 &&
           (bytes % 16) == 0;
}

cudaError_t enqueue_fast_multi_gpu_allreduce_rank_sm90(
    const comm::api::FastAllreduceLaunchState& launch,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    cudaStream_t stream,
    int scratch_index);

} // namespace ooverlap
