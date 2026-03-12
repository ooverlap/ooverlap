/***************************************************************************************************
 * SM90 port of gemm_with_signal_sm90.h
 * Route A:
 *   - No temp buffer
 *   - No post-kernel reorder
 *   - Reorder + store + atomic signaling happen inside GEMM epilogue so overlap is possible
 **************************************************************************************************/
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/numeric_conversion.h"

// GEMM (CUTLASS 3.x)
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

// Epilogue (CUTLASS 3.x)
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

#include "cute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

#define CUTLASS_CHECK_SM90(status)                                                               \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error)                         \
                << " at line " << __LINE__ << std::endl;                                         \
      std::exit(EXIT_FAILURE);                                                                   \
    }                                                                                            \
  }

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Parameters needed for reorder + signaling inside epilogue.
struct SignalingEpilogueParams {
  int  *ptr_Monitored_Matrix;   // Atomic counter array (size >= num_segments + ... optional monitor space)
  int  *ptr_Reorder_Array;      // logical tile idx -> reordered tile idx
  int   kMonitoredColumn;       // original tile-cols (N / TileN)
  int   kReorderedColumn;       // reordered tile-cols (ReLDN)
  int  *kCommu_Seg_Array;       // segment sizes (sum == total tiles)
  bool  if_monitor;

  int   ThreadblockM;
  int   ThreadblockN;

  void *ptr_D;                  // base ptr of FINAL output buffer (interpreted as reshaped row-major)
  int   ld_D;                   // leading dim (elements) of reshaped output: kReorderedColumn * ThreadblockN

  CUTLASS_HOST_DEVICE
  SignalingEpilogueParams() :
    ptr_Monitored_Matrix(nullptr),
    ptr_Reorder_Array(nullptr),
    kMonitoredColumn(0),
    kReorderedColumn(0),
    kCommu_Seg_Array(nullptr),
    if_monitor(false),
    ThreadblockM(0),
    ThreadblockN(0),
    ptr_D(nullptr),
    ld_D(0)
  {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// Helper: CuTe stride for RowMajor / ColumnMajor (A/B)
template <typename LayoutTag>
struct CuteStride2D;

template <>
struct CuteStride2D<cutlass::layout::RowMajor> {
  CUTLASS_HOST_DEVICE
  static auto make(int64_t ld) {
    return cute::make_stride(ld, cute::Int<1>{}, int64_t{0});
  }
};

template <>
struct CuteStride2D<cutlass::layout::ColumnMajor> {
  CUTLASS_HOST_DEVICE
  static auto make(int64_t ld) {
    return cute::make_stride(cute::Int<1>{}, ld, int64_t{0});
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Reorder + signal epilogue wrapper.
/// This wraps a Base SM90 collective epilogue and *remaps the CTA tile coordinate* to the
/// reordered location (in the reshaped output space), then emits atomic signals.
template <class BaseEpilogue, class ThreadblockShape>
struct ReorderSignalEpilogue {

  // Base types expected by GemmUniversal
  using SharedStorage = typename BaseEpilogue::SharedStorage;

  // --- Arguments/Params ---
  // BaseEpilogue::Arguments usually contains { {alpha,beta}, ptr_C, stride_C, ptr_D, stride_D }.
  struct Arguments {
    typename BaseEpilogue::Arguments base;
    SignalingEpilogueParams signal;
  };

  struct Params {
    typename BaseEpilogue::Params base;
    SignalingEpilogueParams signal;

    CUTLASS_HOST_DEVICE
    Params() {}

    CUTLASS_HOST_DEVICE
    Params(Arguments const &args) : base(args.base), signal(args.signal) {}
  };

  // --- Device call operator ---
  // IMPORTANT: this signature matches the common CUTLASS 3.x collective epilogue call pattern:
  //   operator()(Params, SharedStorage, problem_shape, cta_coord, thread_idx, ...rest...)
  //
  // If your CUTLASS version uses a slightly different signature, this is the one place that may
  // need a tiny adjustment.
  template <class ProblemShape, class CtaCoord, class... Rest>
  CUTLASS_DEVICE
  void operator()(
      Params const &params,
      SharedStorage &shared_storage,
      ProblemShape const &problem_shape,
      CtaCoord const &cta_coord,
      int thread_idx,
      Rest&&... rest) const {

    // Extract original CTA tile coordinate (m,n). In SM90 kernels this is typically a cute coord.
    int cta_m = int(cute::get<0>(cta_coord));
    int cta_n = int(cute::get<1>(cta_coord));
    int cta_l = int(cute::get<3>(problem_shape)) == 0 ? 0 : 0; // L is usually 1; keep simple.

    // Original problem dimensions
    int M = int(cute::get<0>(problem_shape));
    int N = int(cute::get<1>(problem_shape));
    int K = int(cute::get<2>(problem_shape));
    int L = int(cute::get<3>(problem_shape));

    // Logical tile index in the ORIGINAL tile grid
    // tile_cols_original = kMonitoredColumn = N / TileN
    int tile_cols_original = params.signal.kMonitoredColumn;
    int logical_tile = cta_m * tile_cols_original + cta_n;

    // Look up reordered linear tile index
    int reordered_tile = params.signal.ptr_Reorder_Array[logical_tile];

    // Map to destination tile (row,col) in the RESHAPED layout
    int reordered_cols = params.signal.kReorderedColumn;
    int dst_tile_row = reordered_tile / reordered_cols;
    int dst_tile_col = reordered_tile % reordered_cols;

    // Build the reshaped output problem (same total elements, different 2D shape)
    // new_N = reordered_cols * TileN
    // new_M = (M*N) / new_N
    int TileN = params.signal.ThreadblockN;
    int new_N = reordered_cols * TileN;

    // Use 64-bit to avoid overflow in M*N
    int64_t total_elems = int64_t(M) * int64_t(N);
    int new_M = int(total_elems / int64_t(new_N));

    // Construct the mapped CTA coord in the reshaped space
    auto mapped_cta = cute::make_coord(dst_tile_row, dst_tile_col, cute::Int<0>{});

    // Construct the reshaped problem shape <M,N,K,L>
    auto reshaped_problem = cute::make_shape(new_M, new_N, K, L);

    // Call the base epilogue to do the real store/convert.
    // The base epilogue will store into D using mapped_cta + reshaped_problem.
    BaseEpilogue base_epilogue;
    base_epilogue(params.base,
                 shared_storage,
                 reshaped_problem,
                 mapped_cta,
                 thread_idx,
                 cutlass::forward<Rest>(rest)...);

    // Ensure the output tile is visible before we signal (extra safety for cross-stream consumer)
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence();

      // Segment classification (segment sizes array; sum == total tiles)
      int idx_bound = params.signal.kCommu_Seg_Array[0];
      int seg = 0;
      while (idx_bound <= reordered_tile) {
        seg += 1;
        idx_bound += params.signal.kCommu_Seg_Array[seg];
      }

      atomicAdd(&params.signal.ptr_Monitored_Matrix[seg], 1);

      if (params.signal.if_monitor) {
        // Optional global ordering info (same spirit as FlashOverlap)
        int global_order = atomicAdd(&params.signal.ptr_Monitored_Matrix[params.signal.kMonitoredColumn - 1], 1);

        cutlass::arch::global_store<int, sizeof(int)>(
          global_order,
          (void *)(params.signal.ptr_Monitored_Matrix + params.signal.kMonitoredColumn + reordered_tile),
          true
        );
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GemmSignalSm90: GEMM with fused reorder+signal epilogue
template <
  typename ElementInputA_,
  typename LayoutInputA_,
  typename ElementInputB_,
  typename LayoutInputB_,
  typename ElementOutput_,
  typename LayoutOutput_,
  typename ElementCompute_,
  typename EpilogueFunctorOp_,
  typename ThreadblockShape_,
  typename WarpShape_,
  typename InstructionShape_,
  int Stages,
  int SwizzleSize
>
class GemmSignalSm90 {
public:
  using ElementInputA  = ElementInputA_;
  using LayoutInputA   = LayoutInputA_;
  using ElementInputB  = ElementInputB_;
  using LayoutInputB   = LayoutInputB_;
  using ElementOutput  = ElementOutput_;
  using LayoutOutput   = LayoutOutput_;
  using ElementCompute = ElementCompute_;

  using ThreadblockShape = ThreadblockShape_;
  using WarpShape        = WarpShape_;
  using InstructionShape = InstructionShape_;

  static int const kStages  = Stages;
  static int const kSwizzle = SwizzleSize;

  // Output buffer is treated as RowMajor for reshaped storage.
  static_assert(cutlass::platform::is_same<LayoutOutput, cutlass::layout::RowMajor>::value,
                "Route-A fused reorder expects RowMajor output buffer interpretation.");

  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag       = cutlass::arch::Sm90;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementInputA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementInputB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  // Mainloop (TMA warp-specialized)
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ElementInputA, LayoutInputA, AlignmentA,
    ElementInputB, LayoutInputB, AlignmentB,
    ElementCompute,
    cute::Shape<cute::Int<ThreadblockShape::kM>,
                cute::Int<ThreadblockShape::kN>,
                cute::Int<ThreadblockShape::kK>>,
    cute::Shape<cute::_1, cute::_1, cute::_1>, // ClusterShape 1x1x1
    cutlass::gemm::collective::StageCountAutoWithMinStages<kStages>,
    cutlass::gemm::KernelTmaWarpSpecialized
  >::CollectiveOp;

  // Base epilogue (linear combination)
  using BaseCollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    cute::Shape<cute::Int<ThreadblockShape::kM>,
                cute::Int<ThreadblockShape::kN>,
                cute::Int<ThreadblockShape::kK>>,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementCompute,   // accumulator
    ElementCompute,   // compute
    ElementOutput,    // C
    LayoutOutput,     // C layout
    AlignmentC,
    ElementOutput,    // D
    LayoutOutput,     // D layout
    AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

  // Our wrapped epilogue that reorders + signals inside GEMM
  using CollectiveEpilogue =
    ReorderSignalEpilogue<BaseCollectiveEpilogue, ThreadblockShape>;

  // Kernel type
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,   // <M, N, K, L>
    CollectiveMainloop,
    CollectiveEpilogue
  >;

  using GemmDevice = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

public:
  struct Arguments {
    cutlass::gemm::GemmCoord   problem_size;
    ElementInputA             *ptr_A;
    ElementInputB             *ptr_B;
    ElementOutput             *ptr_C;       // (only used if beta != 0)
    ElementOutput             *ptr_D;       // FINAL reordered/reshaped output buffer
    int64_t                    ldm_A;
    int64_t                    ldm_B;
    int64_t                    ldm_C;
    int64_t                    ldm_D;       // reshaped leading dim = ReLDN*TileN
    ElementCompute             alpha;
    ElementCompute             beta;
    SignalingEpilogueParams    signal_params;

    Arguments() {}

    Arguments(
      cutlass::gemm::GemmCoord problem_size_,
      ElementInputA *ptr_A_,
      ElementInputB *ptr_B_,
      ElementOutput *ptr_C_,
      ElementOutput *ptr_D_,
      int64_t ldm_A_, int64_t ldm_B_,
      int64_t ldm_C_, int64_t ldm_D_,
      typename EpilogueFunctorOp_::Params linear_scaling,
      int *ptr_MM, int *ptr_RA,
      int  kMonitoredColumn,
      int  kReorderedColumn,
      int *kCommuSegArray,
      bool Monitor
    ) :
      problem_size(problem_size_),
      ptr_A(ptr_A_), ptr_B(ptr_B_),
      ptr_C(ptr_C_), ptr_D(ptr_D_),
      ldm_A(ldm_A_), ldm_B(ldm_B_),
      ldm_C(ldm_C_), ldm_D(ldm_D_),
      alpha(linear_scaling.alpha),
      beta(linear_scaling.beta)
    {
      signal_params.ptr_Monitored_Matrix = ptr_MM;
      signal_params.ptr_Reorder_Array    = ptr_RA;
      signal_params.kMonitoredColumn     = kMonitoredColumn;
      signal_params.kReorderedColumn     = kReorderedColumn;
      signal_params.kCommu_Seg_Array     = kCommuSegArray;
      signal_params.if_monitor           = Monitor;
      signal_params.ThreadblockM         = ThreadblockShape::kM;
      signal_params.ThreadblockN         = ThreadblockShape::kN;
      signal_params.ptr_D                = (void*)ptr_D_;
      signal_params.ld_D                 = int(ldm_D_);
    }
  };

private:
  Arguments  args_;
  GemmDevice gemm_device_;

public:
  GemmSignalSm90() {}

  Status initialize(Arguments const &args) {
    args_ = args;
    return cutlass::Status::kSuccess;
  }

  Status run(cudaStream_t stream) {
    int M = args_.problem_size.m();
    int N = args_.problem_size.n();
    int K = args_.problem_size.k();

    // Build epilogue args:
    // - ptr_D points to FINAL buffer
    // - stride_D uses reshaped leading dimension (ldm_D)
    typename CollectiveEpilogue::Arguments epi_args;
    epi_args.base = typename BaseCollectiveEpilogue::Arguments{
      {args_.alpha, args_.beta},
      reinterpret_cast<ElementOutput const*>(args_.ptr_C),
      CuteStride2D<LayoutOutput>::make(args_.ldm_C),
      reinterpret_cast<ElementOutput*>(args_.ptr_D),
      CuteStride2D<LayoutOutput>::make(args_.ldm_D)
    };
    epi_args.signal = args_.signal_params;

    typename GemmDevice::Arguments gemm_args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {
        reinterpret_cast<ElementInputA const*>(args_.ptr_A),
        CuteStride2D<LayoutInputA>::make(args_.ldm_A),
        reinterpret_cast<ElementInputB const*>(args_.ptr_B),
        CuteStride2D<LayoutInputB>::make(args_.ldm_B)
      },
      epi_args
    };

    Status status = gemm_device_.can_implement(gemm_args);
    if (status != Status::kSuccess) {
      return status;
    }

    status = gemm_device_.initialize(gemm_args, nullptr, stream);
    if (status != Status::kSuccess) {
      return status;
    }

    status = gemm_device_.run(stream);
    if (status != Status::kSuccess) {
      return status;
    }

    return cutlass::Status::kSuccess;
  }

  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass
