#pragma once

#include "ooverlap/sync/sync.cuh"

#include <cstddef>

// -----------------------------------------------------------------------------
// Tunables
// -----------------------------------------------------------------------------
//
// Keep these as preprocessor defines so they can be used uniformly in host code,
// device code, template arguments, shared-memory array sizes, and launch config.
// -----------------------------------------------------------------------------

#define TMA_TWO_GPU_PEER_THREADS 256
#define TMA_TWO_GPU_PEER_MAX_WINDOWS 16
#define TMA_TWO_GPU_PEER_CHUNK_BYTES (16 * 1024)

// Phase 1: owner rank reduces its local window into peer buffer.
#define TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH 8
#define TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP \
    (TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH / 2)

// Phase 2: non-owner rank copies finalized local window back to peer buffer.
#define TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH 8
#define TMA_TWO_GPU_PEER_COPY_STAGE_GAP \
    (TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH / 2)

static_assert(TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH % 2 == 0,
              "TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH must be even");
static_assert(TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH % 2 == 0,
              "TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH must be even");
static_assert(TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP >= 1,
              "TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP must be >= 1");
static_assert(TMA_TWO_GPU_PEER_COPY_STAGE_GAP >= 1,
              "TMA_TWO_GPU_PEER_COPY_STAGE_GAP must be >= 1");

#define TMA_TWO_GPU_PEER_REDUCE_SHARED_BYTES \
    (static_cast<size_t>(TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH) * \
     TMA_TWO_GPU_PEER_CHUNK_BYTES)

#define TMA_TWO_GPU_PEER_COPY_SHARED_BYTES \
    (static_cast<size_t>(TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH) * \
     TMA_TWO_GPU_PEER_CHUNK_BYTES)

#define TMA_TWO_GPU_PEER_DYNAMIC_SHARED_BYTES \
    ((TMA_TWO_GPU_PEER_REDUCE_SHARED_BYTES > \
      TMA_TWO_GPU_PEER_COPY_SHARED_BYTES) \
         ? TMA_TWO_GPU_PEER_REDUCE_SHARED_BYTES \
         : TMA_TWO_GPU_PEER_COPY_SHARED_BYTES)

#define TMA_TWO_GPU_PEER_BARRIER_COUNT \
    ((TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH > \
      TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH) \
         ? TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH \
         : TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH)

#define TMA_TWO_GPU_PEER_STATIC_SHARED_BYTES \
    (static_cast<size_t>(TMA_TWO_GPU_PEER_BARRIER_COUNT) * \
     sizeof(::ooverlap::sync::semaphore))

// progress0 / progress1 layout:
//   [0 .. TMA_TWO_GPU_PEER_MAX_WINDOWS-1] = reduce-done flags
//   [TMA_TWO_GPU_PEER_MAX_WINDOWS .. 2*TMA_TWO_GPU_PEER_MAX_WINDOWS-1]
//       = copy-done flags
#define TMA_TWO_GPU_PEER_PROGRESS_BYTES \
    (static_cast<size_t>(2 * TMA_TWO_GPU_PEER_MAX_WINDOWS) * sizeof(int))
