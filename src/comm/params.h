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

#define TMA_TWO_GPU_PEER_THREADS 512
#define TMA_TWO_GPU_PEER_MAX_WINDOWS 8
#define TMA_TWO_GPU_PEER_CHUNK_BYTES (16 * 1024)

// Phase 1: owner rank reduces its local window into peer buffer.
#define TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH 8
#define TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP \
    (TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH / 2)

// Phase 2: owner rank copies finalized peer window back to local buffer.
#define TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH 8
#define TMA_TWO_GPU_PEER_COPY_STAGE_GAP \
    (TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH / 2)

// Fast global-memory copy path.
#define TMA_TWO_GPU_PEER_FAST_COPY_UNROLL 8

// Overlapped TMA-reduce + fast-copy path.
//
// role 0 CTA: TMA reduce producer
// role 1 CTA: fast-copy consumer
#define TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_WINDOW 2

// Producer publishes progress every N completed chunks.
// Larger = less signal overhead, more lag.
// Smaller = better overlap, more fences/atomics.
#define TMA_TWO_GPU_PEER_OVERLAP_SIGNAL_BATCH_CHUNKS 16

static_assert(TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH % 2 == 0,
              "TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH must be even");
static_assert(TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH % 2 == 0,
              "TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH must be even");
static_assert(TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP >= 1,
              "TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP must be >= 1");
static_assert(TMA_TWO_GPU_PEER_COPY_STAGE_GAP >= 1,
              "TMA_TWO_GPU_PEER_COPY_STAGE_GAP must be >= 1");

static_assert(TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_WINDOW == 2,
              "overlap path expects exactly producer+consumer CTAs");
static_assert(TMA_TWO_GPU_PEER_OVERLAP_SIGNAL_BATCH_CHUNKS >= 1,
              "overlap signal batch must be >= 1");

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
