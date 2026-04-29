/***************************************************************************************************
 * SM90 port of gemm_signal.cu  (Route A: signal + reorder inside epilogue)
 **************************************************************************************************/

#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/fast_math.h"

#include "gemm_with_signal_sm90.h"   // Route-A header below
#include "gemm_signal_sm90_dispatch.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int ThreadblockM, int ThreadblockN, int ThreadblockK,
  int WarpM, int WarpN, int WarpK,
  int InstructionM, int InstructionN, int InstructionK,
  int NumStages, int SwizzleSize, int SplitK
>
void cutlass_gemm_signal_sm90(
  int M, int N, int K,
  int ReLDN, int* CommThr,
  half* A, half* B, half* D,
  int* MM, int* RA,
  bool Monitor,
  cudaStream_t stream = nullptr
) {
  using ThreadblockShape = cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>;
  using WarpShape        = cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>;
  using InstructionShape = cutlass::gemm::GemmShape<InstructionM, InstructionN, InstructionK>;

  cutlass::gemm::GemmCoord problem_size(M, N, K);

  using ElementA           = cutlass::half_t;
  using LayoutA            = cutlass::layout::RowMajor;
  using ElementB           = cutlass::half_t;
  using LayoutB            = cutlass::layout::ColumnMajor;
  using ElementC           = cutlass::half_t;
  using LayoutC            = cutlass::layout::RowMajor;

  // Use float accumulation/compute on SM90.
  using ElementAccumulator = float;

  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC, AlignmentC, ElementAccumulator, ElementAccumulator>;

  using GemmSignal = cutlass::GemmSignalSm90<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    EpilogueOp,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    NumStages,
    SwizzleSize
  >;

  // FlashOverlap-style packed output:
  //
  //   ReLDN = packed tile columns
  //
  //   packed_N = ReLDN * ThreadblockN
  //
  // For true per-tile contiguous communication, use:
  //
  //   ReLDN = 1
  //
  // Then D is interpreted as:
  //
  //   [num_tiles * ThreadblockM, ThreadblockN]
  //
  // and every GEMM tile is a single contiguous slice.
  int64_t ld_D_reshaped = int64_t(ReLDN) * int64_t(ThreadblockN);

  typename GemmSignal::Arguments arguments(
    problem_size,
    reinterpret_cast<cutlass::half_t*>(A),
    reinterpret_cast<cutlass::half_t*>(B),

    // C is unused because beta == 0. However, pass the same packed pointer and
    // packed leading dimension to avoid any mismatched source-layout behavior.
    reinterpret_cast<cutlass::half_t*>(D),     // C
    reinterpret_cast<cutlass::half_t*>(D),     // D

    (int64_t)K,                                // ldm_A, RowMajor A: ld = K
    (int64_t)K,                                // ldm_B, ColumnMajor B represented as [N,K]
    ld_D_reshaped,                             // ldm_C, packed/reshaped
    ld_D_reshaped,                             // ldm_D, packed/reshaped
    {
      ElementAccumulator(1.0f),
      ElementAccumulator(0.0f)
    },
    MM,
    RA,
    (N / ThreadblockN),                        // kMonitoredColumn = original tile-cols
    ReLDN,                                     // kReorderedColumn = packed tile-cols
    CommThr,
    Monitor
  );

  GemmSignal gemm_op;
  CUTLASS_CHECK_SM90(gemm_op.initialize(arguments));
  CUTLASS_CHECK_SM90(gemm_op(stream));
}

/////////////////////////////////////////////////////////////////////////////////////////////////

#define CUTLASS_GEMM_SIGNAL_SM90_INIT(ThreadblockM, ThreadblockN, ThreadblockK, WarpM,      \
                                       WarpN, WarpK, InstructionM, InstructionN,           \
                                       InstructionK, NumStages, SwizzleSize, SplitK)       \
    template void                                                                          \
    cutlass_gemm_signal_sm90<ThreadblockM, ThreadblockN, ThreadblockK, WarpM, WarpN,      \
                              WarpK, InstructionM, InstructionN, InstructionK,             \
                              NumStages, SwizzleSize, SplitK>(                             \
        int M, int N, int K, int ReLDN, int* CommThr, half* A, half* B, half* D,          \
        int* MM, int* RA, bool Monitor, cudaStream_t stream)

#include "../inc/signal_instances_sm90.inc"

#undef CUTLASS_GEMM_SIGNAL_SM90_INIT

// -------------------------------------------------------------------------------------------------
// Simple runtime dispatch for testing (1-GPU)
// -------------------------------------------------------------------------------------------------
namespace ooverlap {

bool gemm_signal_sm90_dispatch(
    int algo,
    int M, int N, int K,
    int ReLDN,
    int32_t* CommThr,
    void* A, void* B, void* D,
    int32_t* MM, int32_t* RA,
    bool Monitor,
    cudaStream_t stream) {

  // NOTE: for now we support one known-good instance:
  //   (128,128,32) TB, (64,64,32) Warp, (16,8,16) Inst, stages=3, swizzle=1, splitk=1
  switch (algo) {
    case 0:
      cutlass_gemm_signal_sm90<
          128, 128, 32,
          64,  64,  32,
          16,  8,   16,
          3,   1,   1>(
          M, N, K, ReLDN, reinterpret_cast<int*>(CommThr),
          reinterpret_cast<half*>(A), reinterpret_cast<half*>(B),
          reinterpret_cast<half*>(D), reinterpret_cast<int*>(MM),
          reinterpret_cast<int*>(RA), Monitor, stream);
      return true;

    case 1:
      cutlass_gemm_signal_sm90<
          128, 128, 64,
          64,  64,  32,
          16,  8,   16,
          3,   1,   1>(
          M, N, K, ReLDN, reinterpret_cast<int*>(CommThr),
          reinterpret_cast<half*>(A), reinterpret_cast<half*>(B),
          reinterpret_cast<half*>(D), reinterpret_cast<int*>(MM),
          reinterpret_cast<int*>(RA), Monitor, stream);
      return true;

    case 2:
      cutlass_gemm_signal_sm90<
          128, 128, 64,
          64,  64,  64,
          16,  8,   16,
          3,   1,   1>(
          M, N, K, ReLDN, reinterpret_cast<int*>(CommThr),
          reinterpret_cast<half*>(A), reinterpret_cast<half*>(B),
          reinterpret_cast<half*>(D), reinterpret_cast<int*>(MM),
          reinterpret_cast<int*>(RA), Monitor, stream);
      return true;

    case 3:
      cutlass_gemm_signal_sm90<
          128, 256, 32,
          64,  64,  32,
          16,  8,   16,
          3,   1,   1>(
          M, N, K, ReLDN, reinterpret_cast<int*>(CommThr),
          reinterpret_cast<half*>(A), reinterpret_cast<half*>(B),
          reinterpret_cast<half*>(D), reinterpret_cast<int*>(MM),
          reinterpret_cast<int*>(RA), Monitor, stream);
      return true;

    case 4:
      cutlass_gemm_signal_sm90<
          128, 256, 64,
          64,  64,  64,
          16,  8,   16,
          3,   1,   1>(
          M, N, K, ReLDN, reinterpret_cast<int*>(CommThr),
          reinterpret_cast<half*>(A), reinterpret_cast<half*>(B),
          reinterpret_cast<half*>(D), reinterpret_cast<int*>(MM),
          reinterpret_cast<int*>(RA), Monitor, stream);
      return true;

    case 5:
      cutlass_gemm_signal_sm90<
          256, 128, 32,
          64,  64,  32,
          16,  8,   16,
          3,   1,   1>(
          M, N, K, ReLDN, reinterpret_cast<int*>(CommThr),
          reinterpret_cast<half*>(A), reinterpret_cast<half*>(B),
          reinterpret_cast<half*>(D), reinterpret_cast<int*>(MM),
          reinterpret_cast<int*>(RA), Monitor, stream);
      return true;

    case 6:
      cutlass_gemm_signal_sm90<
          256, 128, 64,
          64,  64,  64,
          16,  8,   16,
          3,   1,   1>(
          M, N, K, ReLDN, reinterpret_cast<int*>(CommThr),
          reinterpret_cast<half*>(A), reinterpret_cast<half*>(B),
          reinterpret_cast<half*>(D), reinterpret_cast<int*>(MM),
          reinterpret_cast<int*>(RA), Monitor, stream);
      return true;

    default:
      return false;
  } 
}

} // namespace ooverlap
