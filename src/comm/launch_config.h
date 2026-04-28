#pragma once

#include "comm/params.h"

namespace ooverlap {
namespace comm {

struct LaunchConfig {
    int threads = TMA_TWO_GPU_PEER_DEFAULT_THREADS;
    int max_ctas = TMA_TWO_GPU_PEER_DEFAULT_MAX_CTAS;
    int window_chunks = TMA_TWO_GPU_PEER_DEFAULT_WINDOW_CHUNKS;

    int chunk_bytes = TMA_TWO_GPU_PEER_DEFAULT_CHUNK_BYTES;
    int stage_depth = TMA_TWO_GPU_PEER_DEFAULT_STAGE_DEPTH;
};

__host__ __device__ __forceinline__ LaunchConfig default_launch_config() {
    return LaunchConfig{};
}

__host__ __device__ __forceinline__ bool launch_config_valid(
    LaunchConfig config) {
    if (config.threads <= 0 || config.threads > 1024) {
        return false;
    }

    if ((config.threads % 32) != 0) {
        return false;
    }

    if (config.max_ctas <= 0) {
        return false;
    }

    if (config.window_chunks <= 0) {
        return false;
    }

    if (config.chunk_bytes < 16) {
        return false;
    }

    if ((config.chunk_bytes % 16) != 0) {
        return false;
    }

    if (config.stage_depth < 2) {
        return false;
    }

    if ((config.stage_depth % 2) != 0) {
        return false;
    }

    return true;
}

__host__ __device__ __forceinline__ bool launch_config_valid_for_overlap(
    LaunchConfig config) {
    if (!launch_config_valid(config)) {
        return false;
    }

    return config.max_ctas >= TMA_TWO_GPU_PEER_OVERLAP_BLOCKS_PER_CTA;
}

} // namespace comm
} // namespace ooverlap
