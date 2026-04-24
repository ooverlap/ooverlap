#pragma once

#include "comm/params.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ooverlap {
namespace comm {
namespace utils {

struct Window {
    int index;
    int start_chunk;
    int chunk_count;
    int owner_rank;
};

__host__ __device__ __forceinline__ size_t min_sz(size_t a, size_t b) {
    return (a < b) ? a : b;
}

__host__ __device__ __forceinline__ int ceil_div_int64_to_int(
    size_t num,
    size_t den) {
    return static_cast<int>((num + den - 1) / den);
}

__host__ __device__ __forceinline__ int min_int(int a, int b) {
    return (a < b) ? a : b;
}

__host__ __device__ __forceinline__ int window_num_chunks(int num_chunks) {
    return min_int(num_chunks, TMA_TWO_GPU_PEER_MAX_WINDOWS);
}

__host__ __device__ __forceinline__ int window_chunk_start(
    int window_idx,
    int num_chunks,
    int num_windows) {
    const int base = num_chunks / num_windows;
    const int rem = num_chunks % num_windows;
    return window_idx * base + ((window_idx < rem) ? window_idx : rem);
}

__host__ __device__ __forceinline__ int window_chunk_count(
    int window_idx,
    int num_chunks,
    int num_windows) {
    const int base = num_chunks / num_windows;
    const int rem = num_chunks % num_windows;
    return base + ((window_idx < rem) ? 1 : 0);
}

__host__ __device__ __forceinline__ Window make_window(
    int window_idx,
    int num_chunks,
    int num_windows) {
    Window window{};
    window.index = window_idx;
    window.start_chunk = window_chunk_start(window_idx, num_chunks, num_windows);
    window.chunk_count = window_chunk_count(window_idx, num_chunks, num_windows);
    window.owner_rank = window_idx & 1;
    return window;
}

} // namespace utils
} // namespace comm
} // namespace ooverlap
