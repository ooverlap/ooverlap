#pragma once

#include "comm/params.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ooverlap {
namespace comm {
namespace utils {

struct Window {
    int index = 0;
    int start_chunk = 0;
    int chunk_count = 0;
    int owner_rank = -1;
};

struct WindowRange {
    int begin = 0;
    int end = 0;
};

__host__ __device__ __forceinline__ size_t min_sz(size_t a, size_t b) {
    return (a < b) ? a : b;
}

__host__ __device__ __forceinline__ int min_int(int a, int b) {
    return (a < b) ? a : b;
}

__host__ __device__ __forceinline__ int max_int(int a, int b) {
    return (a > b) ? a : b;
}

__host__ __device__ __forceinline__ int ceil_div_int(
    int num,
    int den) {
    if (num <= 0 || den <= 0) {
        return 0;
    }

    return (num + den - 1) / den;
}

__host__ __device__ __forceinline__ int ceil_div_int64_to_int(
    size_t num,
    size_t den) {
    if (num == 0 || den == 0) {
        return 0;
    }

    return static_cast<int>((num + den - 1) / den);
}

__host__ __device__ __forceinline__ int window_count_for_chunks(
    int num_chunks,
    int chunks_per_window) {
    return ceil_div_int(num_chunks, chunks_per_window);
}

__host__ __device__ __forceinline__ int window_count_for_bytes(
    size_t total_bytes,
    size_t chunk_bytes,
    int chunks_per_window) {
    if (total_bytes == 0 || chunk_bytes == 0 || chunks_per_window <= 0) {
        return 0;
    }

    const int num_chunks = ceil_div_int64_to_int(total_bytes, chunk_bytes);

    return window_count_for_chunks(num_chunks, chunks_per_window);
}

__host__ __device__ __forceinline__ int rank0_window_count(
    int num_windows) {
    return (num_windows + 1) / 2;
}

__host__ __device__ __forceinline__ int window_owner_rank(
    int window_idx,
    int num_windows) {
    if (window_idx < 0 || window_idx >= num_windows) {
        return -1;
    }

    return (window_idx < rank0_window_count(num_windows)) ? 0 : 1;
}

__host__ __device__ __forceinline__ WindowRange rank_window_range(
    int num_windows,
    int rank) {
    WindowRange range{};

    if (num_windows <= 0 || (rank != 0 && rank != 1)) {
        return range;
    }

    const int split = rank0_window_count(num_windows);

    if (rank == 0) {
        range.begin = 0;
        range.end = split;
    } else {
        range.begin = split;
        range.end = num_windows;
    }

    return range;
}

__host__ __device__ __forceinline__ int window_range_count(
    WindowRange range) {
    return (range.begin < range.end) ? (range.end - range.begin) : 0;
}

__host__ __device__ __forceinline__ int rank_window_count(
    int num_windows,
    int rank) {
    return window_range_count(rank_window_range(num_windows, rank));
}

__host__ __device__ __forceinline__ int cta_count_for_windows(
    int window_count,
    int max_ctas) {
    if (window_count <= 0 || max_ctas <= 0) {
        return 0;
    }

    return min_int(window_count, max_ctas);
}

__host__ __device__ __forceinline__ WindowRange cta_window_range(
    int cta_idx,
    int cta_count,
    WindowRange full_range) {
    WindowRange out{};

    const int total = window_range_count(full_range);

    if (total <= 0 || cta_count <= 0 || cta_idx < 0 || cta_idx >= cta_count) {
        return out;
    }

    const int base = total / cta_count;
    const int rem = total % cta_count;

    const int local_begin =
        cta_idx * base + ((cta_idx < rem) ? cta_idx : rem);

    const int local_count = base + ((cta_idx < rem) ? 1 : 0);

    out.begin = full_range.begin + local_begin;
    out.end = out.begin + local_count;

    return out;
}

__host__ __device__ __forceinline__ int window_chunk_start(
    int window_idx,
    int chunks_per_window) {
    if (window_idx < 0 || chunks_per_window <= 0) {
        return 0;
    }

    return window_idx * chunks_per_window;
}

__host__ __device__ __forceinline__ int window_chunk_count(
    int window_idx,
    int num_chunks,
    int chunks_per_window) {
    if (window_idx < 0 || num_chunks <= 0 || chunks_per_window <= 0) {
        return 0;
    }

    const int start = window_chunk_start(window_idx, chunks_per_window);

    if (start >= num_chunks) {
        return 0;
    }

    return min_int(chunks_per_window, num_chunks - start);
}

__host__ __device__ __forceinline__ Window make_window(
    int window_idx,
    int num_chunks,
    int chunks_per_window) {
    const int num_windows =
        window_count_for_chunks(num_chunks, chunks_per_window);

    Window window{};
    window.index = window_idx;
    window.start_chunk = window_chunk_start(window_idx, chunks_per_window);
    window.chunk_count =
        window_chunk_count(window_idx, num_chunks, chunks_per_window);
    window.owner_rank = window_owner_rank(window_idx, num_windows);

    return window;
}

__host__ __device__ __forceinline__ size_t window_begin_byte(
    Window window,
    size_t chunk_bytes) {
    return static_cast<size_t>(window.start_chunk) * chunk_bytes;
}

__host__ __device__ __forceinline__ size_t window_end_byte(
    Window window,
    size_t total_bytes,
    size_t chunk_bytes) {
    const size_t raw_end =
        static_cast<size_t>(window.start_chunk + window.chunk_count) *
        chunk_bytes;

    return min_sz(raw_end, total_bytes);
}

__host__ __device__ __forceinline__ size_t window_size_bytes(
    Window window,
    size_t total_bytes,
    size_t chunk_bytes) {
    const size_t begin = window_begin_byte(window, chunk_bytes);
    const size_t end = window_end_byte(window, total_bytes, chunk_bytes);

    return (begin < end) ? (end - begin) : 0;
}

} // namespace utils
} // namespace comm
} // namespace ooverlap
