#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace ooverlap {

/*
  RE semantics used here:

  After your existing SM90 signal GEMM, each communication segment is contiguous
  in memory as a flat block of:
      seg_tiles * TileM * TileN  elements

  We reinterpret that block as:
      [ seg_tiles * TileM , TileN ]

  i.e. "micro-rows" of width TileN.

  RE is indexed in these micro-row units, not in full original GEMM rows.
  That matches the FlashOverlap scatter mapping style where the remap length is
  M * N / TileN, not just M.

  For a segment that begins at flat element offset acc_addr:
      row_begin = acc_addr / TileN
      row_count = seg_tiles * TileM

  Then:
      dst_local_row = RE[row_begin + src_local_row] - row_begin

  must land in [0, row_count).
*/

__global__ void scatter_row_remap_sm90_kernel(
    const half* __restrict__ src,
    half* __restrict__ dst,
    const int32_t* __restrict__ RE,
    int row_begin,
    int row_count,
    int row_width) {
  int idx = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  int total = row_count * row_width;
  if (idx >= total) {
    return;
  }

  int src_local_row = idx / row_width;
  int col           = idx - src_local_row * row_width;

  int src_global_row = row_begin + src_local_row;
  int dst_global_row = RE[src_global_row];
  int dst_local_row  = dst_global_row - row_begin;

  // Keep this safe for bring-up. A correct RE should always map inside segment.
  if (unsigned(dst_local_row) < unsigned(row_count)) {
    dst[dst_local_row * row_width + col] =
        src[src_local_row * row_width + col];
  }
}

inline cudaError_t launch_scatter_row_remap_sm90(
    const half* src,
    half* dst,
    const int32_t* RE,
    int row_begin,
    int row_count,
    int row_width,
    cudaStream_t stream) {
  if (row_count <= 0 || row_width <= 0) {
    return cudaSuccess;
  }

  int total = row_count * row_width;
  constexpr int kThreads = 256;
  int blocks = (total + kThreads - 1) / kThreads;

  scatter_row_remap_sm90_kernel<<<blocks, kThreads, 0, stream>>>(
      src, dst, RE, row_begin, row_count, row_width);

  return cudaGetLastError();
}

} // namespace ooverlap
