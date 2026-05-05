#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace ooverlap {

// Plain SM90 GEMM for baseline.
//
// Layout contract:
//   A      : logical row-major [M, K], physical torch shape (M, K), contiguous
//   B_col  : logical column-major [K, N], physical torch shape (N, K), contiguous
//            i.e. B_col[n, k] stores logical B[k, n]
//   D_col  : logical column-major [M, N], physical torch shape (N, M), contiguous
//            i.e. D_col[n, m] stores logical D[m, n]
//
// This matches CUTLASS profiler rows with:
//   A=f16:row, B=f16:column, C=f16:column, D=f16:column
struct GemmPlainSm90AlgoMeta {
  int tile_m;
  int tile_n;
  int tile_k;
  int cluster_m;
  int cluster_n;
  int cluster_k;
  int stages;              // -1 means StageCountAuto
  const char* mainloop;    // ws | pingpong | cooperative
  const char* epilogue;    // auto
  const char* scheduler;   // normal | stream_k
};

int gemm_plain_sm90_algo_count();

bool gemm_plain_sm90_get_algo_meta(
    int algo,
    GemmPlainSm90AlgoMeta* meta);

bool gemm_plain_sm90_dispatch(
    int algo,
    int M,
    int N,
    int K,
    void* A,
    void* B_col,
    void* D_col,
    cudaStream_t stream);

}  // namespace ooverlap
