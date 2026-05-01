#pragma once

#include <cuda_runtime.h>

namespace ooverlap {

// Plain SM90 GEMM, no signal, no reorder, no NCCL.
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
