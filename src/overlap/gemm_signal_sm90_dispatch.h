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
  int stages;              // -1 means StageCountAuto
  const char* mainloop;    // ws | pingpong | cooperative
  const char* epilogue;    // auto
  const char* scheduler;   // normal | stream_k
};

int gemm_signal_sm90_algo_count();

bool gemm_signal_sm90_get_algo_meta(
    int algo,
    GemmSignalSm90AlgoMeta* meta);

bool gemm_signal_sm90_dispatch(
    int algo,
    int M, int N, int K,
    int ReLDN,
    int num_segments,
    int32_t* CommThr,
    void* A, void* B, void* D,
    int32_t* MM, int32_t* RA,
    int active_sm_count,
    bool Monitor,
    cudaStream_t stream);

} // namespace ooverlap
