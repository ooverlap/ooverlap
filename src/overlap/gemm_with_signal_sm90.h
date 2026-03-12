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
#include <utility>

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/kernel_hardware_info.h"

// (Optional) forward-decls for collective builder
#include "cutlass/gemm/collective/collective_builder_decl.hpp"

// GEMM (CUTLASS 3.x)
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

// Epilogue (CUTLASS 3.x)
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

// CuTe
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
  int  *ptr_Monitored_Matrix;
  int  *ptr_Reorder_Array;      // logical tile idx -> reordered tile idx
  int   kMonitoredColumn;       // original tile-cols (N / TileN)
  int   kReorderedColumn;       // reordered tile-cols (ReLDN)
  int  *kCommu_Seg_Array;       // segment sizes (sum == total tiles)
  bool  if_monitor;

  int   ThreadblockM;
  int   ThreadblockN;

  void *ptr_D;                  // base ptr of FINAL output buffer (reshaped row-major)
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
// Helper: CuTe stride for RowMajor / ColumnMajor (A/B/C/D)
//
// IMPORTANT NOTE (why ColumnMajor looks "weird"):
// The SM90 CollectiveBuilder / MainloopArguments in your CUTLASS version expects the stride type
// for B to be cute::tuple<int64_t, C<1>, int64_t> (ld in slot 0), even when LayoutB is ColumnMajor.
// So we must produce that exact type to compile.
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
    // NOTE: This returns tuple<int64_t, C<1>, int64_t> (matches what the SM90 mainloop expects).
    // If you later switch to a CUTLASS version where ColumnMajor expects (C<1>, ld, 0),
    // this is the one place to change.
    return cute::make_stride(ld, cute::Int<1>{}, int64_t{0});
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Reorder + signal epilogue wrapper.
/// Must satisfy SM90 GemmUniversal "CollectiveEpilogue" interface.
template <class BaseEpilogue, class ThreadblockShape>
struct ReorderSignalEpilogue {

  // -----------------------------
  // Required nested aliases (forward from BaseEpilogue)
  // -----------------------------
  using ElementC          = typename BaseEpilogue::ElementC;
  using ElementD          = typename BaseEpilogue::ElementD;
  using StrideC           = typename BaseEpilogue::StrideC;
  using StrideD           = typename BaseEpilogue::StrideD;
  using ThreadEpilogueOp  = typename BaseEpilogue::ThreadEpilogueOp;
  using DispatchPolicy    = typename BaseEpilogue::DispatchPolicy;

  // These are required by sm90_gemm_tma_warpspecialized.hpp in your error log
  using TensorStorage     = typename BaseEpilogue::TensorStorage;
  using PipelineStorage   = typename BaseEpilogue::PipelineStorage;

  using SharedStorage     = typename BaseEpilogue::SharedStorage;

  // -----------------------------
  // Arguments / Params
  // -----------------------------
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

    CUTLASS_HOST_DEVICE
    Params(typename BaseEpilogue::Params const& base_, SignalingEpilogueParams const& sig_)
      : base(base_), signal(sig_) {}
  };

  // -----------------------------
  // Required static hooks (forward / adapt)
  // -----------------------------
  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return BaseEpilogue::can_implement(problem_shape, args.base);
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return BaseEpilogue::get_workspace_size(problem_shape, args.base);
  }

  static size_t get_workspace_alignment() {
    // Forward if you have it; otherwise keep a safe default.
    return 16;
  }

  // REQUIRED by sm90_gemm_tma_warpspecialized.hpp(220) in your log:
  //   CollectiveEpilogue::to_underlying_arguments(transformed_problem_shape, args.epilogue, workspace)
  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    auto base_params = BaseEpilogue::to_underlying_arguments(problem_shape, args.base, workspace);
    return Params{base_params, args.signal};
  }

  // -----------------------------
  // Device call operator (wrap + remap + signal)
  // -----------------------------
  template <class ProblemShape, class CtaCoord, class... Rest>
  CUTLASS_DEVICE
  void operator()(
      Params const &params,
      SharedStorage &shared_storage,
      ProblemShape const &problem_shape,
      CtaCoord const &cta_coord,
      int thread_idx,
      Rest&&... rest) const
  {
    // CTA tile coordinate is typically (m, n, l)
    int cta_m = int(cute::get<0>(cta_coord));
    int cta_n = int(cute::get<1>(cta_coord));
    int cta_l = int(cute::get<2>(cta_coord));

    int M = int(cute::get<0>(problem_shape));
    int N = int(cute::get<1>(problem_shape));
    int K = int(cute::get<2>(problem_shape));
    int L = int(cute::get<3>(problem_shape));

    // Logical tile in original layout
    int tile_cols_original = params.signal.kMonitoredColumn; // N / TileN
    int logical_tile = cta_m * tile_cols_original + cta_n;

    // Map to reordered tile id
    int reordered_tile = params.signal.ptr_Reorder_Array[logical_tile];

    int reordered_cols = params.signal.kReorderedColumn;
    int dst_tile_row = reordered_tile / reordered_cols;
    int dst_tile_col = reordered_tile % reordered_cols;

    // Reshape output: new_N = reordered_cols * TileN, new_M = (M*N)/new_N
    int TileN = params.signal.ThreadblockN;
    int new_N = reordered_cols * TileN;

    int64_t total_elems = int64_t(M) * int64_t(N);
    int new_M = int(total_elems / int64_t(new_N));

    auto mapped_cta = cute::make_coord(dst_tile_row, dst_tile_col, cta_l);
    auto reshaped_problem = cute::make_shape(new_M, new_N, K, L);

    // Call base epilogue to store into FINAL D at mapped location
    BaseEpilogue base_epilogue;
    base_epilogue(
      params.base,
      shared_storage,
      reshaped_problem,
      mapped_cta,
      thread_idx,
      std::forward<Rest>(rest)...
    );

    // Signal after store
    __syncthreads();
    if (thread_idx == 0) {
      __threadfence();

      int idx_bound = params.signal.kCommu_Seg_Array[0];
      int seg = 0;
      while (idx_bound <= reordered_tile) {
        seg += 1;
        idx_bound += params.signal.kCommu_Seg_Array[seg];
      }

      atomicAdd(&params.signal.ptr_Monitored_Matrix[seg], 1);

      if (params.signal.if_monitor) {
        int global_order = atomicAdd(
          &params.signal.ptr_Monitored_Matrix[params.signal.kMonitoredColumn - 1], 1);

        cutlass::arch::global_store<int, sizeof(int)>(
          global_order,
          (void *)(params.signal.ptr_Monitored_Matrix +
                   params.signal.kMonitoredColumn +
                   reordered_tile),
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

  static_assert(cutlass::platform::is_same<LayoutOutput, cutlass::layout::RowMajor>::value,
                "Route-A fused reorder expects RowMajor output buffer interpretation.");

  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag       = cutlass::arch::Sm90;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementInputA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementInputB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  // Mainloop (SM90 TMA warp-specialized)
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ElementInputA, LayoutInputA, AlignmentA,
    ElementInputB, LayoutInputB, AlignmentB,
    ElementCompute,
    cute::Shape<cute::Int<ThreadblockShape::kM>,
                cute::Int<ThreadblockShape::kN>,
                cute::Int<ThreadblockShape::kK>>,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::KernelTmaWarpSpecialized
  >::CollectiveOp;

  // Base epilogue
  using BaseCollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    cute::Shape<cute::Int<ThreadblockShape::kM>,
                cute::Int<ThreadblockShape::kN>,
                cute::Int<ThreadblockShape::kK>>,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementCompute,
    ElementCompute,
    ElementOutput,
    LayoutOutput,
    AlignmentC,
    ElementOutput,
    LayoutOutput,
    AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

  using CollectiveEpilogue = ReorderSignalEpilogue<BaseCollectiveEpilogue, ThreadblockShape>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
  >;

  using GemmDevice = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

public:
  struct Arguments {
    cutlass::gemm::GemmCoord   problem_size;
    ElementInputA             *ptr_A;
    ElementInputB             *ptr_B;
    ElementOutput             *ptr_C;
    ElementOutput             *ptr_D;  // FINAL reordered/reshaped output buffer
    int64_t                    ldm_A;
    int64_t                    ldm_B;
    int64_t                    ldm_C;
    int64_t                    ldm_D;
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

    using Kernel              = GemmKernel;
    using KernelArguments     = typename Kernel::Arguments;
    using ProblemShape        = typename Kernel::ProblemShape;
    using MainloopArguments   = typename Kernel::MainloopArguments;
    using EpilogueArguments   = typename Kernel::EpilogueArguments;
    using TileSchedArguments  = typename Kernel::TileSchedulerArguments;

    ProblemShape problem_shape = cute::make_shape(M, N, K, 1);

    MainloopArguments mainloop_args{
      reinterpret_cast<ElementInputA const*>(args_.ptr_A),
      CuteStride2D<LayoutInputA>::make(args_.ldm_A),
      reinterpret_cast<ElementInputB const*>(args_.ptr_B),
      CuteStride2D<LayoutInputB>::make(args_.ldm_B)
    };

    EpilogueArguments epilogue_args;
    epilogue_args.base = typename BaseCollectiveEpilogue::Arguments{
      {args_.alpha, args_.beta},
      reinterpret_cast<ElementOutput const*>(args_.ptr_C),
      CuteStride2D<LayoutOutput>::make(args_.ldm_C),
      reinterpret_cast<ElementOutput*>(args_.ptr_D),
      CuteStride2D<LayoutOutput>::make(args_.ldm_D)
    };
    epilogue_args.signal = args_.signal_params;

    cutlass::KernelHardwareInfo hw_info;
    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device_id);
    hw_info.device_id = device_id;
    hw_info.sm_count  = prop.multiProcessorCount;

    TileSchedArguments sched_args{};

    KernelArguments gemm_args(
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      mainloop_args,
      epilogue_args,
      hw_info,
      sched_args
    );

    Status status = gemm_device_.can_implement(gemm_args);
    if (status != Status::kSuccess) return status;

    status = gemm_device_.initialize(gemm_args, nullptr, stream);
    if (status != Status::kSuccess) return status;

    status = gemm_device_.run(stream);
    if (status != Status::kSuccess) return status;

    return cutlass::Status::kSuccess;
  }

  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass
