#pragma once

#include "ooverlap/sync/sync.cuh"

#include <cstddef>

// -----------------------------------------------------------------------------
// Runtime defaults
// -----------------------------------------------------------------------------
//
// These are defaults only. They should not be used directly by kernel launch
// code after this refactor. Runtime launch paths should use comm::LaunchConfig.
// -----------------------------------------------------------------------------

#define TMA_TWO_GPU_PEER_DEFAULT_THREADS 1024
#define TMA_TWO_GPU_PEER_DEFAULT_MAX_CTAS 8
#define TMA_TWO_GPU_PEER_DEFAULT_WINDOW_CHUNKS 32

// -----------------------------------------------------------------------------
// Compile-time tunables
// -----------------------------------------------------------------------------
//
// These remain compile-time because they affect template instantiation, TMA wait
// templates, shared-memory layout, barrier counts, or fast-copy code generation.
// -----------------------------------------------------------------------------

// Chunk = smem pipeline granularity.
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

static_assert(TMA_TWO_GPU_PEER_CHUNK_BYTES >= 16,
              "TMA_TWO_GPU_PEER_CHUNK_BYTES must be >= 16");
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
