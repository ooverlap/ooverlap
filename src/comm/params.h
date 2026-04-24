#pragma once

// -----------------------------------------------------------------------------
// Tunables
// -----------------------------------------------------------------------------

constexpr int kTwoGpuPeerThreads = 16;
constexpr int kTwoGpuPeerMaxWindows = 16;
constexpr size_t kTwoGpuPeerChunkBytes = 32 * 1024;

// phase 1: owner rank reduces its local window into peer buffer
constexpr int kTwoGpuPeerReduceStageDepth = 4;
constexpr int kTwoGpuPeerReduceStageGap = kTwoGpuPeerReduceStageDepth / 2;

// phase 2: non-owner rank copies finalized local window back to peer buffer
constexpr int kTwoGpuPeerCopyStageDepth = 4;
constexpr int kTwoGpuPeerCopyStageGap = kTwoGpuPeerCopyStageDepth / 2;

static_assert(kTwoGpuPeerReduceStageDepth % 2 == 0,
              "kTwoGpuPeerReduceStageDepth must be even");
static_assert(kTwoGpuPeerCopyStageDepth % 2 == 0,
              "kTwoGpuPeerCopyStageDepth must be even");
static_assert(kTwoGpuPeerReduceStageGap >= 1,
              "kTwoGpuPeerReduceStageGap must be >= 1");
static_assert(kTwoGpuPeerCopyStageGap >= 1,
              "kTwoGpuPeerCopyStageGap must be >= 1");

constexpr size_t kTwoGpuPeerReduceSharedBytes =
    static_cast<size_t>(kTwoGpuPeerReduceStageDepth) * kTwoGpuPeerChunkBytes;

constexpr size_t kTwoGpuPeerCopySharedBytes =
    static_cast<size_t>(kTwoGpuPeerCopyStageDepth) * kTwoGpuPeerChunkBytes;

constexpr size_t kTwoGpuPeerDynamicSharedBytes =
    (kTwoGpuPeerReduceSharedBytes > kTwoGpuPeerCopySharedBytes)
        ? kTwoGpuPeerReduceSharedBytes
        : kTwoGpuPeerCopySharedBytes;

constexpr int kTwoGpuPeerBarrierCount =
    (kTwoGpuPeerReduceStageDepth > kTwoGpuPeerCopyStageDepth)
        ? kTwoGpuPeerReduceStageDepth
        : kTwoGpuPeerCopyStageDepth;

constexpr size_t kTwoGpuPeerStaticSharedBytes =
    static_cast<size_t>(kTwoGpuPeerBarrierCount) * sizeof(sync::semaphore);


