#pragma once

#include "ooverlap/sync/sync.cuh"

#include <cstddef>

namespace ooverlap {
namespace comm {

template <int ChunkBytes, int StageDepth>
struct TmaPipelineVariant {
    static_assert(ChunkBytes >= 16, "ChunkBytes must be >= 16");
    static_assert((ChunkBytes % 16) == 0, "ChunkBytes must be 16-byte aligned");
    static_assert(StageDepth >= 2, "StageDepth must be >= 2");
    static_assert((StageDepth % 2) == 0, "StageDepth must be even");

    static constexpr int chunk_bytes = ChunkBytes;
    static constexpr int stage_depth = StageDepth;
    static constexpr int stage_gap = StageDepth / 2;
    static constexpr int barrier_count = StageDepth;

    static constexpr size_t reduce_shared_bytes =
        static_cast<size_t>(StageDepth) * static_cast<size_t>(ChunkBytes);

    static constexpr size_t copy_shared_bytes =
        static_cast<size_t>(StageDepth) * static_cast<size_t>(ChunkBytes);

    static constexpr size_t dynamic_shared_bytes =
        (reduce_shared_bytes > copy_shared_bytes)
            ? reduce_shared_bytes
            : copy_shared_bytes;

    static constexpr size_t static_shared_bytes =
        static_cast<size_t>(barrier_count) *
        sizeof(::ooverlap::sync::semaphore);

    static constexpr size_t total_shared_bytes =
        dynamic_shared_bytes + static_shared_bytes;
};

} // namespace comm
} // namespace ooverlap

/*
 * Supported runtime-selectable pipeline variants.
 *
 * Add/remove pairs here.
 *
 * Current candidates:
 *   16 KiB x depth 8  = current default shape
 *   32 KiB x depth 4
 *   64 KiB x depth 2
 *   100 KiB x depth 2
 */
#define OOVERLAP_TMA_TWO_GPU_PEER_FOR_EACH_VARIANT(M) \
    M(4 * 1024, 32)                                   \
    M(8 * 1024, 16)                                   \
    M(16 * 1024, 8)                                   \
    M(32 * 1024, 4)                                   \
    M(64 * 1024, 2)                                   \
    M(100 * 1024, 2)
