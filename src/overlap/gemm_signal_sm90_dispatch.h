#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace ooverlap {

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
    int32_t* CommThr,
    void* A, void* B, void* D,
    int32_t* MM, int32_t* RA,
    bool Monitor,
    cudaStream_t stream);

} // namespace ooverlap
