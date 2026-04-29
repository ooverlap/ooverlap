#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace ooverlap {

struct GemmSignalSm90AlgoMeta {
  int tile_m;
  int tile_n;
  int tile_k;
  int cluster_m;
  int cluster_n;
  int cluster_k;
  const char* mainloop;
  const char* epilogue;
};

// Returns number of generated SM90 signal GEMM algos.
int gemm_signal_sm90_algo_count();

// Returns false if algo is out of range.
bool gemm_signal_sm90_get_algo_meta(
    int algo,
    GemmSignalSm90AlgoMeta* meta);

// Minimal dispatch for tests.
// Pointers are device pointers.
// - A, B, D: fp16 device buffers
// - MM, RA, CommThr: int32 device buffers
//
// Returns true if algo is supported, false otherwise.
bool gemm_signal_sm90_dispatch(
    int algo,
    int M, int N, int K,
    int ReLDN,
    int num_segments,
    int32_t* CommThr,
    void* A, void* B, void* D,
    int32_t* MM, int32_t* RA,
    bool Monitor,
    cudaStream_t stream);

} // namespace ooverlap
