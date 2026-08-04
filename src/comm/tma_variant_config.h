#pragma once

#include "ooverlap/sync/sync.cuh"

#include <cstddef>

namespace ooverlap {
namespace comm {

/* OOVERLAP_CHUNK_ONLY_THREAD0_WINDOW_PIPELINE_V1 */


template <
    int ChunkBytes,
    int StageDepth,
    int FillDepth = StageDepth / 2,
    int LoadFillDepth = FillDepth>
struct TmaPipelineVariant {
    static_assert(ChunkBytes >= 16, "ChunkBytes must be >= 16");
    static_assert((ChunkBytes % 16) == 0, "ChunkBytes must be 16-byte aligned");
    static_assert(StageDepth >= 2, "StageDepth must be >= 2");
    static_assert((StageDepth % 2) == 0, "StageDepth must be even");
    static_assert(FillDepth > 0, "FillDepth must be > 0");
    static_assert(LoadFillDepth > 0, "LoadFillDepth must be > 0");
    static_assert(FillDepth + LoadFillDepth <= StageDepth,
                  "FillDepth + LoadFillDepth must be <= StageDepth");

    static constexpr int chunk_bytes = ChunkBytes;
    static constexpr int stage_depth = StageDepth;

    /*
     * stage_gap is kept as the old name for compatibility. It is now the
     * apply-side depth used by store/reduce waits.
     */
    static constexpr int stage_gap = FillDepth;
    static constexpr int fill_depth = FillDepth;
    static constexpr int load_fill_depth = LoadFillDepth;

    static constexpr int barrier_count = StageDepth;

    static constexpr size_t reduce_shared_bytes =
        static_cast<size_t>(StageDepth) * static_cast<size_t>(ChunkBytes);

    static constexpr size_t copy_shared_bytes =
        static_cast<size_t>(StageDepth) * static_cast<size_t>(ChunkBytes);

    static constexpr size_t pipeline_shared_bytes =
        (reduce_shared_bytes > copy_shared_bytes)
            ? reduce_shared_bytes
            : copy_shared_bytes;

    static constexpr size_t dynamic_shared_bytes =
        pipeline_shared_bytes;

    static constexpr size_t static_shared_bytes =
        static_cast<size_t>(barrier_count) *
        sizeof(::ooverlap::sync::semaphore);

    static constexpr size_t total_shared_bytes =
        dynamic_shared_bytes + static_shared_bytes;
};

} // namespace comm
} // namespace ooverlap

/*
 * Old two-argument macro kept for compatibility with any code that only cares
 * about chunk/stage pairs.
 */
#define OOVERLAP_TMA_TWO_GPU_PEER_FOR_EACH_VARIANT(M) \
    M(2 * 1024, 64)                                   \
    M(4 * 1024, 32)                                   \
    M(8 * 1024, 16)                                   \
    M(8 * 1024, 24)                                   \
    M(16 * 1024, 8)                                   \
    M(32 * 1024, 4)                                   \
    M(64 * 1024, 2)                                   \
    M(100 * 1024, 2)

/*
 * New four-argument macro for the asymmetric pipeline.
 * Arguments are: chunk bytes, stage depth, store/reduce fill depth, load fill depth.
 */
#define OOVERLAP_TMA_TWO_GPU_PEER_FOR_EACH_VARIANT_WITH_DEPTH(M) \
    M(2 * 1024, 64, 32, 32)                                      \
    M(4 * 1024, 32, 16, 16)                                      \
    M(8 * 1024, 16, 8, 8)                                        \
    M(16 * 1024, 8, 4, 4)                                        \
    M(32 * 1024, 4, 2, 2)                                        \
    M(64 * 1024, 2, 1, 1)                                        \
    M(100 * 1024, 2, 1, 1)
