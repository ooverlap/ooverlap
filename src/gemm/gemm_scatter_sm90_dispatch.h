#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace ooverlap {

// Scatter-path GEMM stage for SM90.
//
// IMPORTANT:
// This is intentionally the same GEMM stage as gemm_signal_sm90_dispatch():
//   - it writes the full reordered/segmented send buffer D
//   - it performs the same per-segment signaling through MM
//
// The scatter-specific row remap (RE) is applied later, segment-by-segment,
// before ncclReduceScatter() on the communication stream.
bool gemm_scatter_sm90_dispatch(
    int algo,
    int M, int N, int K,
    int ReLDN,
    int num_segments,
    int32_t* CommThr,
    void* A, void* B, void* D,
    int32_t* MM, int32_t* RA, int32_t* RE,
    bool Monitor,
    cudaStream_t stream);

} // namespace ooverlap
