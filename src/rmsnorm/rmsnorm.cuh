#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "utils.h"

namespace ooverlap {

__global__ __forceinline__ void rmsnorm_kernel(
    const half* __restrict__ x,
    const half* __restrict__ rw,
    half* __restrict__ o,
    int bs,
    int dim) {

  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int j = tid << 4;  // 16 elems / thread

  if (bid >= bs || j >= dim) {
    return;
  }

  half2 x_val[8];
  half2 w_val[8];
  float pow_sum = 0.0f;

  *(float4*)(&x_val[0]) = *(const float4*)(&x[bid * dim + j]);
  *(float4*)(&x_val[4]) = *(const float4*)(&x[bid * dim + j + 8]);
  *(float4*)(&w_val[0]) = *(const float4*)(&rw[j]);
  *(float4*)(&w_val[4]) = *(const float4*)(&rw[j + 8]);

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    pow_sum += __half2float(x_val[i].x) * __half2float(x_val[i].x);
    pow_sum += __half2float(x_val[i].y) * __half2float(x_val[i].y);
  }

  __shared__ float warpLevelSums[OOVERLAP_WARP_SIZE];
  pow_sum = blockReduceSumFloat(pow_sum, warpLevelSums);

  if (tid == 0) {
    warpLevelSums[0] = rsqrtf(pow_sum / static_cast<float>(dim) + 1e-5f);
  }
  __syncthreads();

  const float scaling = warpLevelSums[0];

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    x_val[i].x = __float2half(__half2float(x_val[i].x) * scaling);
    x_val[i].y = __float2half(__half2float(x_val[i].y) * scaling);
    x_val[i] = __hmul2(x_val[i], w_val[i]);
  }

  *(float4*)(&o[bid * dim + j]) = *(float4*)(&x_val[0]);
  *(float4*)(&o[bid * dim + j + 8]) = *(float4*)(&x_val[4]);
}

__global__ __forceinline__ void reorder_rmsnorm_kernel(
    const half* __restrict__ x,
    const half* __restrict__ rw,
    half* __restrict__ o,
    int bs,
    int dim,
    int64_t BM,
    int64_t BN,
    int64_t ldn,
    int64_t rldn,
    const int* __restrict__ RA) {

  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int j = tid << 4;  // 16 elems / thread

  if (bid >= bs || j >= dim) {
    return;
  }

  half2 x_val[8];
  half2 w_val[8];
  float pow_sum = 0.0f;

  const int old_index = static_cast<int>(bid / BM * ldn + j / BN);
  const int new_index = RA[old_index];
  const int new_row   = static_cast<int>(new_index / rldn * BM + (bid % BM));
  const int new_col   = static_cast<int>(new_index % rldn * BN + (j % BN));

  *(float4*)(&x_val[0]) = *(const float4*)(&x[new_row * (rldn * BN) + new_col]);
  *(float4*)(&x_val[4]) = *(const float4*)(&x[new_row * (rldn * BN) + new_col + 8]);
  *(float4*)(&w_val[0]) = *(const float4*)(&rw[j]);
  *(float4*)(&w_val[4]) = *(const float4*)(&rw[j + 8]);

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    pow_sum += __half2float(x_val[i].x) * __half2float(x_val[i].x);
    pow_sum += __half2float(x_val[i].y) * __half2float(x_val[i].y);
  }

  __shared__ float warpLevelSums[OOVERLAP_WARP_SIZE];
  pow_sum = blockReduceSumFloat(pow_sum, warpLevelSums);

  if (tid == 0) {
    warpLevelSums[0] = rsqrtf(pow_sum / static_cast<float>(dim) + 1e-5f);
  }
  __syncthreads();

  const float scaling = warpLevelSums[0];

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    x_val[i].x = __float2half(__half2float(x_val[i].x) * scaling);
    x_val[i].y = __float2half(__half2float(x_val[i].y) * scaling);
    x_val[i] = __hmul2(x_val[i], w_val[i]);
  }

  *(float4*)(&o[bid * dim + j]) = *(float4*)(&x_val[0]);
  *(float4*)(&o[bid * dim + j + 8]) = *(float4*)(&x_val[4]);
}

} // namespace ooverlap
