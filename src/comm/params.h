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

#define TMA_TWO_GPU_PEER_THREADS 1024

// Maximum CTAs per rank. Each active CTA owns one contiguous range of windows.
// The actual CTA count is:
//
//     min(TMA_TWO_GPU_PEER_MAX_CTAS, rank_owned_window_count)
//
// except rendezvous-only launches may still launch one CTA with no work.
#define TMA_TWO_GPU_PEER_MAX_CTAS 16

// Chunk = smem pipeline granularity.
#define TMA_TWO_GPU_PEER_CHUNK_BYTES (16 * 1024)

// Window = signaling / work-assignment granularity.
// Each window contains this many chunks, except the final tail window.
#define TMA_TWO_GPU_PEER_WINDOW_CHUNKS 16

#define TMA_TWO_GPU_PEER_MAX_WINDOW_BYTES \
    (static_cast<size_t>(TMA_TWO_GPU_PEER_WINDOW_CHUNKS) * \
     static_cast<size_t>(TMA_TWO_GPU_PEER_CHUNK_BYTES))

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
//
// This is now roles per assigned CTA-range, not blocks per window.
#define TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA 2

static_assert(TMA_TWO_GPU_PEER_MAX_CTAS >= 1,
              "TMA_TWO_GPU_PEER_MAX_CTAS must be >= 1");
static_assert(TMA_TWO_GPU_PEER_CHUNK_BYTES >= 16,
              "TMA_TWO_GPU_PEER_CHUNK_BYTES must be >= 16");
static_assert(TMA_TWO_GPU_PEER_WINDOW_CHUNKS >= 1,
              "TMA_TWO_GPU_PEER_WINDOW_CHUNKS must be >= 1");
static_assert(TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH % 2 == 0,
              "TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH must be even");
static_assert(TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH % 2 == 0,
              "TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH must be even");
static_assert(TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP >= 1,
              "TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP must be >= 1");
static_assert(TMA_TWO_GPU_PEER_COPY_STAGE_GAP >= 1,
              "TMA_TWO_GPU_PEER_COPY_STAGE_GAP must be >= 1");
static_assert(TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP <=
                  TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH,
              "TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP must be <= depth");
static_assert(TMA_TWO_GPU_PEER_COPY_STAGE_GAP <=
                  TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH,
              "TMA_TWO_GPU_PEER_COPY_STAGE_GAP must be <= depth");

static_assert(TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA == 2,
              "overlap path expects exactly producer+consumer CTA roles");

#define TMA_TWO_GPU_PEER_REDUCE_SHARED_BYTES \
    (static_cast<size_t>(TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH) * \
     static_cast<size_t>(TMA_TWO_GPU_PEER_CHUNK_BYTES))

#define TMA_TWO_GPU_PEER_COPY_SHARED_BYTES \
    (static_cast<size_t>(TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH) * \
     static_cast<size_t>(TMA_TWO_GPU_PEER_CHUNK_BYTES))

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
