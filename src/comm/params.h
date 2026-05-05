#pragma once

#include "ooverlap/sync/sync.cuh"

#include <cstddef>

// -----------------------------------------------------------------------------
// Runtime defaults
// -----------------------------------------------------------------------------
//
// These are defaults only. Runtime launch paths should use comm::LaunchConfig.
// -----------------------------------------------------------------------------

#define TMA_TWO_GPU_PEER_DEFAULT_THREADS 1024
#define TMA_TWO_GPU_PEER_DEFAULT_MAX_CTAS 8
#define TMA_TWO_GPU_PEER_DEFAULT_WINDOW_CHUNKS 64

#define TMA_TWO_GPU_PEER_DEFAULT_CHUNK_BYTES (8 * 1024)
#define TMA_TWO_GPU_PEER_DEFAULT_STAGE_DEPTH 16

// Fast global-memory copy path remains compile-time for now.
#define TMA_TWO_GPU_PEER_FAST_COPY_UNROLL 16

// Overlap path still uses producer/consumer CTA pairs.
#define TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA 2

static_assert(TMA_TWO_GPU_PEER_DEFAULT_THREADS >= 32,
              "default thread count must be >= 32");
static_assert(TMA_TWO_GPU_PEER_DEFAULT_THREADS <= 1024,
              "default thread count must be <= 1024");
static_assert((TMA_TWO_GPU_PEER_DEFAULT_THREADS % 32) == 0,
              "default thread count must be warp-aligned");

static_assert(TMA_TWO_GPU_PEER_DEFAULT_MAX_CTAS >= 1,
              "default max CTAs must be >= 1");
static_assert(TMA_TWO_GPU_PEER_DEFAULT_WINDOW_CHUNKS >= 1,
              "default window chunks must be >= 1");

static_assert(TMA_TWO_GPU_PEER_DEFAULT_CHUNK_BYTES >= 16,
              "default chunk bytes must be >= 16");
static_assert((TMA_TWO_GPU_PEER_DEFAULT_CHUNK_BYTES % 16) == 0,
              "default chunk bytes must be 16-byte aligned");
static_assert(TMA_TWO_GPU_PEER_DEFAULT_STAGE_DEPTH >= 2,
              "default stage depth must be >= 2");
static_assert((TMA_TWO_GPU_PEER_DEFAULT_STAGE_DEPTH % 2) == 0,
              "default stage depth must be even");

static_assert(TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA == 2,
              "overlap path expects exactly producer+consumer CTA roles");

/*
 * Compatibility names for places that still report compile-time defaults.
 * New launch code should use LaunchConfig + TmaPipelineVariant instead.
 */
#define TMA_TWO_GPU_PEER_CHUNK_BYTES TMA_TWO_GPU_PEER_DEFAULT_CHUNK_BYTES

#define TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH TMA_TWO_GPU_PEER_DEFAULT_STAGE_DEPTH
#define TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP \
    (TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH / 2)

#define TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH TMA_TWO_GPU_PEER_DEFAULT_STAGE_DEPTH
#define TMA_TWO_GPU_PEER_COPY_STAGE_GAP \
    (TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH / 2)

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
