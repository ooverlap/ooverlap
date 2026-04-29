/***************************************************************************************************
 * SM90 scatter-path GEMM dispatch
 *
 * This intentionally reuses the existing fused reorder+signal GEMM kernel.
 * The only scatter-specific addition is RE, which is NOT consumed here.
 * RE is consumed later by the segment-local row-remap kernel before NCCL RS.
 **************************************************************************************************/

#include "gemm_scatter_sm90_dispatch.h"
#include "gemm_signal_sm90_dispatch.h"

namespace ooverlap {

bool gemm_scatter_sm90_dispatch(
    int algo,
    int M, int N, int K,
    int ReLDN,
    int num_segments,
    int32_t* CommThr,
    void* A, void* B, void* D,
    int32_t* MM, int32_t* RA, int32_t* RE,
    bool Monitor,
    cudaStream_t stream) {
  (void)RE;  // scatter row-remap is applied after segment-ready, not inside GEMM

  return gemm_signal_sm90_dispatch(
      algo,
      M, N, K,
      ReLDN,
      num_segments,
      CommThr,
      A, B, D,
      MM, RA,
      Monitor,
      stream);
}

} // namespace ooverlap
