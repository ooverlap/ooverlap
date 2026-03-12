/***************************************************************************************************
 * SM90 port of gemm_signal.cu  (Route A: signal + reorder inside epilogue)
 **************************************************************************************************/

#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/fast_math.h"

#include "gemm_with_signal_sm90.h"   // Route-A header below

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

  // Use float accumulation/compute on SM90
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

  // In FlashOverlap-style use, beta==0 so C is not read.
  // D is the output buffer that is interpreted as a "reshaped" matrix:
  //   new_N = ReLDN * ThreadblockN
  //   new_M = (M*N) / new_N
  //
  // This is the same idea as FlashOverlap SM80: reshape + tile-permutation to make segments contiguous.
  int64_t ld_D_reshaped = int64_t(ReLDN) * int64_t(ThreadblockN);

  typename GemmSignal::Arguments arguments(
    problem_size,
    reinterpret_cast<cutlass::half_t*>(A),
    reinterpret_cast<cutlass::half_t*>(B),
    reinterpret_cast<cutlass::half_t*>(D),     // C (unused if beta=0)
    reinterpret_cast<cutlass::half_t*>(D),     // D (FINAL, reordered + reshaped in-place)
    (int64_t)K,                                // ldm_A (RowMajor A: ld = K)
    (int64_t)K,                                // ldm_B (ColumnMajor B: ld = K)
    (int64_t)N,                                // ldm_C (RowMajor C: ld = N)
    ld_D_reshaped,                             // ldm_D (RowMajor reshaped D: ld = ReLDN*TileN)
    {
      ElementAccumulator(1.0f),
      ElementAccumulator(0.0f)
    },
    MM,
    RA,
    (N / ThreadblockN),                        // kMonitoredColumn = original tile-cols
    ReLDN,                                     // kReorderedColumn = reordered tile-cols
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
