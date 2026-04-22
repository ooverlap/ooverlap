#pragma once

#include "comm/exec/chunk.h"

#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {
namespace exec {

struct RangeSchedulerSubmission {
    const unsigned char* src = nullptr;
    unsigned char* dst = nullptr;
    size_t bytes = 0;

    ChunkOpKind op = ChunkOpKind::kInvalid;

    // Generic metadata carried into emitted chunks.
    uint32_t op_id = 0;
    int dst_rank = -1;
    uint64_t user_tag = 0;
};

__host__ __device__ __forceinline__ void range_scheduler_submission_clear(
    RangeSchedulerSubmission* sub) {
    sub->src = nullptr;
    sub->dst = nullptr;
    sub->bytes = 0;
    sub->op = ChunkOpKind::kInvalid;
    sub->op_id = 0;
    sub->dst_rank = -1;
    sub->user_tag = 0;
}

__host__ __device__ __forceinline__ bool range_scheduler_submission_is_valid(
    const RangeSchedulerSubmission* sub) {
    return sub != nullptr &&
           sub->src != nullptr &&
           sub->dst != nullptr &&
           sub->bytes > 0 &&
           sub->op != ChunkOpKind::kInvalid;
}

// Kept only so old template sites still compile cleanly.
template <int WatchThreads>
struct ChunkSchedulerScratch {
    int unused = 0;
};

template <int WatchThreads = 1>
struct ChunkScheduler {
    const RangeSchedulerSubmission* submission = nullptr;
    size_t chunk_bytes = 0;

    // Offset of the next chunk to activate.
    size_t next_offset_bytes = 0;

    bool has_active_chunk = false;
    size_t active_offset_bytes = 0;
    int active_chunk_idx = -1;

    Chunk current{};

    ChunkSchedulerScratch<WatchThreads>* scratch = nullptr;
};

template <int WatchThreads>
__host__ __device__ __forceinline__ void chunk_scheduler_reset(
    ChunkScheduler<WatchThreads>* sched) {
    sched->submission = nullptr;
    sched->chunk_bytes = 0;
    sched->next_offset_bytes = 0;
    sched->has_active_chunk = false;
    sched->active_offset_bytes = 0;
    sched->active_chunk_idx = -1;
    chunk_clear(&sched->current);
    sched->scratch = nullptr;
}

template <int WatchThreads>
__host__ __device__ __forceinline__ void chunk_scheduler_bind_scratch(
    ChunkScheduler<WatchThreads>* sched,
    ChunkSchedulerScratch<WatchThreads>* scratch) {
    sched->scratch = scratch;
}

template <int WatchThreads>
__host__ __device__ __forceinline__ void chunk_scheduler_init(
    ChunkScheduler<WatchThreads>* sched,
    const RangeSchedulerSubmission* submission,
    size_t chunk_bytes,
    ChunkSchedulerScratch<WatchThreads>* scratch = nullptr) {
    sched->submission = submission;
    sched->chunk_bytes = chunk_bytes;
    sched->next_offset_bytes = 0;
    sched->has_active_chunk = false;
    sched->active_offset_bytes = 0;
    sched->active_chunk_idx = -1;
    chunk_clear(&sched->current);
    sched->scratch = scratch;
}

template <int WatchThreads>
__host__ __device__ __forceinline__ bool chunk_scheduler_has_current(
    const ChunkScheduler<WatchThreads>* sched) {
    return chunk_is_valid(&sched->current);
}

template <int WatchThreads>
__host__ __device__ __forceinline__ const Chunk* chunk_scheduler_current(
    const ChunkScheduler<WatchThreads>* sched) {
    return &sched->current;
}

template <int WatchThreads>
__host__ __device__ __forceinline__ uint32_t chunk_scheduler_active_queue_id(
    const ChunkScheduler<WatchThreads>* sched) {
    if (sched == nullptr ||
        !sched->has_active_chunk ||
        sched->submission == nullptr) {
        return 0;
    }
    return sched->submission->op_id;
}

template <int WatchThreads>
__host__ __device__ __forceinline__ int chunk_scheduler_active_dst_rank(
    const ChunkScheduler<WatchThreads>* sched) {
    if (sched == nullptr ||
        !sched->has_active_chunk ||
        sched->submission == nullptr) {
        return -1;
    }
    return sched->submission->dst_rank;
}

template <int WatchThreads>
__host__ __device__ __forceinline__ bool chunk_scheduler_build_chunk_at_offset(
    const ChunkScheduler<WatchThreads>* sched,
    size_t offset_bytes,
    Chunk* out) {
    if (out == nullptr) {
        return false;
    }
    chunk_clear(out);

    if (sched == nullptr ||
        sched->submission == nullptr ||
        !range_scheduler_submission_is_valid(sched->submission) ||
        sched->chunk_bytes == 0) {
        return false;
    }

    const RangeSchedulerSubmission& sub = *sched->submission;

    if (offset_bytes >= sub.bytes) {
        return false;
    }

    const size_t remaining = sub.bytes - offset_bytes;
    const size_t this_chunk_bytes =
        (remaining < sched->chunk_bytes) ? remaining : sched->chunk_bytes;
    const int chunk_idx =
        static_cast<int>(offset_bytes / sched->chunk_bytes);

    out->src = sub.src + offset_bytes;
    out->dst = sub.dst + offset_bytes;
    out->bytes = this_chunk_bytes;
    out->op_id = sub.op_id;
    out->dst_rank = sub.dst_rank;
    out->user_tag = sub.user_tag;
    out->chunk_idx = chunk_idx;
    out->range_offset_bytes = offset_bytes;
    out->op = sub.op;

    return true;
}

template <int WatchThreads>
__host__ __device__ __forceinline__ bool chunk_scheduler_try_activate_next_chunk(
    ChunkScheduler<WatchThreads>* sched) {
    if (sched == nullptr) {
        return false;
    }

    if (sched->has_active_chunk) {
        return true;
    }

    if (!chunk_scheduler_build_chunk_at_offset(
            sched,
            sched->next_offset_bytes,
            &sched->current)) {
        return false;
    }

    sched->has_active_chunk = true;
    sched->active_offset_bytes = sched->next_offset_bytes;
    sched->active_chunk_idx = sched->current.chunk_idx;
    return true;
}

template <int WatchThreads>
__host__ __device__ __forceinline__ bool chunk_scheduler_try_prime_current(
    ChunkScheduler<WatchThreads>* sched) {
    if (sched == nullptr) {
        return false;
    }

    if (chunk_is_valid(&sched->current)) {
        return true;
    }

    return chunk_scheduler_try_activate_next_chunk(sched);
}

template <int WatchThreads>
__host__ __device__ __forceinline__ bool chunk_scheduler_peek_next(
    const ChunkScheduler<WatchThreads>* sched,
    Chunk* out) {
    if (sched == nullptr || out == nullptr) {
        return false;
    }

    const size_t next_offset =
        sched->has_active_chunk
            ? (sched->active_offset_bytes + sched->current.bytes)
            : sched->next_offset_bytes;

    return chunk_scheduler_build_chunk_at_offset(sched, next_offset, out);
}

template <int WatchThreads>
__host__ __device__ __forceinline__ void chunk_scheduler_advance(
    ChunkScheduler<WatchThreads>* sched) {
    if (sched == nullptr) {
        return;
    }

    if (!sched->has_active_chunk) {
        chunk_clear(&sched->current);
        return;
    }

    sched->next_offset_bytes = sched->active_offset_bytes + sched->current.bytes;
    sched->has_active_chunk = false;
    sched->active_offset_bytes = 0;
    sched->active_chunk_idx = -1;
    chunk_clear(&sched->current);

    chunk_scheduler_try_activate_next_chunk(sched);
}

} // namespace exec
} // namespace comm
} // namespace ooverlap
