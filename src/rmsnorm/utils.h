#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define OOVERLAP_WARP_SIZE 32
#define OOVERLAP_MAX_THREADS_PER_BLOCK 1024
#define OOVERLAP_DIV_UP(x, y) (((x) + (y) - 1) / (y))

__device__ __forceinline__ float warpReduceSumFloat(float v) {
  v += __shfl_down_sync(0xffffffff, v, 16);
  v += __shfl_down_sync(0xffffffff, v, 8);
  v += __shfl_down_sync(0xffffffff, v, 4);
  v += __shfl_down_sync(0xffffffff, v, 2);
  v += __shfl_down_sync(0xffffffff, v, 1);
  return v;
}

__device__ __forceinline__ float blockReduceSumFloat(float v, float* shared_mem) {
  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int warp_count = (blockDim.x + 31) >> 5;

  for (int mask = 16; mask >= 1; mask >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, mask);
  }

  if (lane_id == 0) {
    shared_mem[warp_id] = v;
  }
  __syncthreads();

  v = (lane_id < warp_count) ? shared_mem[lane_id] : 0.0f;

  if (warp_id == 0) {
    for (int mask = 16; mask >= 1; mask >>= 1) {
      v += __shfl_xor_sync(0xffffffff, v, mask);
    }
  }
  v = __shfl_sync(0xffffffff, v, 0);
  return v;
}
