#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace ooverlap {

// These typedefs match your SM90 dispatcher-style entrypoints.
// Unlike FlashOverlap, we are not storing hundreds of template instantiations
// here yet. The runtime "algo" is handled inside the dispatcher.

typedef bool (*SignalDispatchPtr)(
    int algo,
    int M, int N, int K,
    int ReLDN,
    int32_t* CommThr,
    void* A, void* B, void* D,
    int32_t* MM, int32_t* RA,
    bool Monitor,
    cudaStream_t stream);

typedef bool (*ScatterDispatchPtr)(
    int algo,
    int M, int N, int K,
    int ReLDN,
    int32_t* CommThr,
    void* A, void* B, void* D,
    int32_t* MM, int32_t* RA, int32_t* RE,
    bool Monitor,
    cudaStream_t stream);

} // namespace ooverlap
