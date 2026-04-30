/***************************************************************************************************
 * SM90 CUTLASS 3.x GEMM wrapper.
 *
 * This version adds TileScheduler as an explicit template parameter so generated algos can
 * instantiate both normal CUTLASS scheduling and Stream-K scheduling:
 *
 *   TileM, TileN, TileK,
 *   StageCount,
 *   ClusterM, ClusterN, ClusterK,
 *   MainloopSchedule,
 *   EpilogueSchedule,
 *   TileScheduler
 *
 * OOVERLAP_USE_BASE_EPILOGUE_ONLY:
 *   0: use ReorderSignalEpilogue
 *   1: use CUTLASS base epilogue only
 **************************************************************************************************/
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <utility>

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/float8.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/kernel_hardware_info.h"

#include "cutlass/gemm/collective/collective_builder_decl.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

#include "cute/tensor.hpp"

#include "epilogue/reorder_epilogue.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef OOVERLAP_USE_BASE_EPILOGUE_ONLY
#define OOVERLAP_USE_BASE_EPILOGUE_ONLY 1
#endif

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

namespace detail {

template <class T, class... Args>
struct is_brace_constructible {
private:
  template <class U, class... A>
  static auto test(int) -> decltype(U{std::declval<A>()...}, std::true_type{});

  template <class, class...>
  static auto test(...) -> std::false_type;

public:
  static constexpr bool value = decltype(test<T, Args...>(0))::value;
};

template <
  class KernelArguments,
  class ProblemShape,
  class MainloopArguments,
  class EpilogueArguments,
  class TileSchedArguments
>
CUTLASS_HOST
KernelArguments make_kernel_arguments(
    ProblemShape const& problem_shape,
    MainloopArguments const& mainloop_args,
    EpilogueArguments const& epilogue_args,
    cutlass::KernelHardwareInfo const& hw_info,
    TileSchedArguments const& sched_args) {

  if constexpr (std::is_constructible<
      KernelArguments,
      cutlass::gemm::GemmUniversalMode,
      ProblemShape,
      MainloopArguments,
      EpilogueArguments,
      cutlass::KernelHardwareInfo,
      TileSchedArguments>::value) {

    return KernelArguments(
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      mainloop_args,
      epilogue_args,
      hw_info,
      sched_args
    );

  } else if constexpr (is_brace_constructible<
      KernelArguments,
      cutlass::gemm::GemmUniversalMode,
      ProblemShape,
      MainloopArguments,
      EpilogueArguments,
      cutlass::KernelHardwareInfo,
      TileSchedArguments>::value) {

    return KernelArguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      mainloop_args,
      epilogue_args,
      hw_info,
      sched_args
    };

  } else if constexpr (std::is_constructible<
      KernelArguments,
      cutlass::gemm::GemmUniversalMode,
      ProblemShape,
      MainloopArguments,
      EpilogueArguments,
      TileSchedArguments>::value) {

    return KernelArguments(
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      mainloop_args,
      epilogue_args,
      sched_args
    );

  } else if constexpr (is_brace_constructible<
      KernelArguments,
      cutlass::gemm::GemmUniversalMode,
      ProblemShape,
      MainloopArguments,
      EpilogueArguments,
      TileSchedArguments>::value) {

    return KernelArguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      mainloop_args,
      epilogue_args,
      sched_args
    };

  } else if constexpr (std::is_constructible<
      KernelArguments,
      cutlass::gemm::GemmUniversalMode,
      ProblemShape,
      MainloopArguments,
      EpilogueArguments,
      cutlass::KernelHardwareInfo>::value) {

    return KernelArguments(
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      mainloop_args,
      epilogue_args,
      hw_info
    );

  } else if constexpr (is_brace_constructible<
      KernelArguments,
      cutlass::gemm::GemmUniversalMode,
      ProblemShape,
      MainloopArguments,
      EpilogueArguments,
      cutlass::KernelHardwareInfo>::value) {

    return KernelArguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      mainloop_args,
      epilogue_args,
      hw_info
    };

  } else if constexpr (std::is_constructible<
      KernelArguments,
      cutlass::gemm::GemmUniversalMode,
      ProblemShape,
      MainloopArguments,
      EpilogueArguments>::value) {

    return KernelArguments(
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      mainloop_args,
      epilogue_args
    );

  } else if constexpr (is_brace_constructible<
      KernelArguments,
      cutlass::gemm::GemmUniversalMode,
      ProblemShape,
      MainloopArguments,
      EpilogueArguments>::value) {

    return KernelArguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      mainloop_args,
      epilogue_args
    };

  } else if constexpr (std::is_constructible<
      KernelArguments,
      ProblemShape,
      MainloopArguments,
      EpilogueArguments,
      cutlass::KernelHardwareInfo,
      TileSchedArguments>::value) {

    return KernelArguments(
      problem_shape,
      mainloop_args,
      epilogue_args,
      hw_info,
      sched_args
    );

  } else if constexpr (is_brace_constructible<
      KernelArguments,
      ProblemShape,
      MainloopArguments,
      EpilogueArguments,
      cutlass::KernelHardwareInfo,
      TileSchedArguments>::value) {

    return KernelArguments{
      problem_shape,
      mainloop_args,
      epilogue_args,
      hw_info,
      sched_args
    };

  } else if constexpr (std::is_constructible<
      KernelArguments,
      ProblemShape,
      MainloopArguments,
      EpilogueArguments,
      TileSchedArguments>::value) {

    return KernelArguments(
      problem_shape,
      mainloop_args,
      epilogue_args,
      sched_args
    );

  } else if constexpr (is_brace_constructible<
      KernelArguments,
      ProblemShape,
      MainloopArguments,
      EpilogueArguments,
      TileSchedArguments>::value) {

    return KernelArguments{
      problem_shape,
      mainloop_args,
      epilogue_args,
      sched_args
    };

  } else if constexpr (std::is_constructible<
      KernelArguments,
      ProblemShape,
      MainloopArguments,
      EpilogueArguments,
      cutlass::KernelHardwareInfo>::value) {

    return KernelArguments(
      problem_shape,
      mainloop_args,
      epilogue_args,
      hw_info
    );

  } else if constexpr (is_brace_constructible<
      KernelArguments,
      ProblemShape,
      MainloopArguments,
      EpilogueArguments,
      cutlass::KernelHardwareInfo>::value) {

    return KernelArguments{
      problem_shape,
      mainloop_args,
      epilogue_args,
      hw_info
    };

  } else if constexpr (std::is_constructible<
      KernelArguments,
      ProblemShape,
      MainloopArguments,
      EpilogueArguments>::value) {

    return KernelArguments(
      problem_shape,
      mainloop_args,
      epilogue_args
    );

  } else if constexpr (is_brace_constructible<
      KernelArguments,
      ProblemShape,
      MainloopArguments,
      EpilogueArguments>::value) {

    return KernelArguments{
      problem_shape,
      mainloop_args,
      epilogue_args
    };

  } else {
    static_assert(
      is_brace_constructible<
        KernelArguments,
        cutlass::gemm::GemmUniversalMode,
        ProblemShape,
        MainloopArguments,
        EpilogueArguments,
        cutlass::KernelHardwareInfo,
        TileSchedArguments>::value,
      "Unsupported CUTLASS SM90 KernelArguments constructor/aggregate layout for this schedule."
    );
  }
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

// Mainloop stride helper.
// For this repo, B is physically [N, K] contiguous and is passed as logical B^T.
// Your CUTLASS rev expects the mainloop stride type for LayoutB=ColumnMajor to be
// cute::tuple<int64_t, C<1>, int64_t>, so ColumnMajor intentionally uses the same
// physical row-major stride here.
template <typename LayoutTag>
struct CuteMainloopStride2D;

template <>
struct CuteMainloopStride2D<cutlass::layout::RowMajor> {
  CUTLASS_HOST_DEVICE
  static auto make(int64_t ld) {
    return cute::make_stride(ld, cute::Int<1>{}, int64_t{0});
  }
};

template <>
struct CuteMainloopStride2D<cutlass::layout::ColumnMajor> {
  CUTLASS_HOST_DEVICE
  static auto make(int64_t ld) {
    return cute::make_stride(ld, cute::Int<1>{}, int64_t{0});
  }
};

// Epilogue stride helper. This one uses true output layout semantics.
template <typename LayoutTag>
struct CuteEpilogueStride2D;

template <>
struct CuteEpilogueStride2D<cutlass::layout::RowMajor> {
  CUTLASS_HOST_DEVICE
  static auto make(int64_t ld) {
    return cute::make_stride(ld, cute::Int<1>{}, int64_t{0});
  }
};

template <>
struct CuteEpilogueStride2D<cutlass::layout::ColumnMajor> {
  CUTLASS_HOST_DEVICE
  static auto make(int64_t ld) {
    return cute::make_stride(cute::Int<1>{}, ld, int64_t{0});
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementInputA_,
  typename LayoutInputA_,
  typename ElementInputB_,
  typename LayoutInputB_,
  typename ElementOutput_,
  typename LayoutOutput_,
  typename ElementCompute_,
  int TileM_,
  int TileN_,
  int TileK_,
  typename StageCountType_,
  typename ClusterShape_,
  typename MainloopSchedule_,
  typename EpilogueSchedule_,
  typename TileScheduler_ = void
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

  static constexpr int TileM = TileM_;
  static constexpr int TileN = TileN_;
  static constexpr int TileK = TileK_;

  using ThreadblockShape = cutlass::gemm::GemmShape<TileM, TileN, TileK>;

  using TileShape = cute::Shape<
    cute::Int<TileM>,
    cute::Int<TileN>,
    cute::Int<TileK>
  >;

  using StageCountType   = StageCountType_;
  using ClusterShape     = ClusterShape_;
  using MainloopSchedule = MainloopSchedule_;
  using EpilogueSchedule = EpilogueSchedule_;
  using TileScheduler    = TileScheduler_;

  static_assert(
    cutlass::platform::is_same<LayoutOutput, cutlass::layout::ColumnMajor>::value,
    "Temporary GEMM-only test expects ColumnMajor output."
  );

  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag       = cutlass::arch::Sm90;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementInputA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementInputB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ElementInputA, LayoutInputA, AlignmentA,
    ElementInputB, LayoutInputB, AlignmentB,
    ElementCompute,
    TileShape,
    ClusterShape,
    StageCountType,
    MainloopSchedule
  >::CollectiveOp;

  using BaseCollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementCompute,
    ElementCompute,
    ElementOutput,
    LayoutOutput,
    AlignmentC,
    ElementOutput,
    LayoutOutput,
    AlignmentD,
    EpilogueSchedule
  >::CollectiveOp;

#if defined(OOVERLAP_USE_BASE_EPILOGUE_ONLY) && OOVERLAP_USE_BASE_EPILOGUE_ONLY
  using CollectiveEpilogue = BaseCollectiveEpilogue;
#else
  using CollectiveEpilogue = ReorderSignalEpilogue<BaseCollectiveEpilogue, ThreadblockShape>;
#endif

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    TileScheduler
  >;

  using GemmDevice = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

public:
  struct Arguments {
    cutlass::gemm::GemmCoord   problem_size;
    ElementInputA             *ptr_A;
    ElementInputB             *ptr_B;
    ElementOutput             *ptr_C;
    ElementOutput             *ptr_D;
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
      ElementCompute alpha_,
      ElementCompute beta_,
      int *ptr_MM, int *ptr_RA,
      int  kMonitoredColumn,
      int  kReorderedColumn,
      int *kCommuSegArray,
      int  numSegments,
      bool Monitor
    ) :
      problem_size(problem_size_),
      ptr_A(ptr_A_),
      ptr_B(ptr_B_),
      ptr_C(ptr_C_),
      ptr_D(ptr_D_),
      ldm_A(ldm_A_),
      ldm_B(ldm_B_),
      ldm_C(ldm_C_),
      ldm_D(ldm_D_),
      alpha(alpha_),
      beta(beta_)
    {
      signal_params.ptr_Monitored_Matrix = ptr_MM;
      signal_params.ptr_Reorder_Array    = ptr_RA;
      signal_params.kMonitoredColumn     = kMonitoredColumn;
      signal_params.kReorderedColumn     = kReorderedColumn;
      signal_params.kCommu_Seg_Array     = kCommuSegArray;
      signal_params.if_monitor           = Monitor;
      signal_params.ThreadblockM         = ThreadblockShape::kM;
      signal_params.ThreadblockN         = ThreadblockShape::kN;
      signal_params.ptr_D                = static_cast<void*>(ptr_D_);
      signal_params.ld_D                 = int(ldm_D_);
      signal_params.kEpilogueArrivalsPerTile = 0;
      signal_params.ptr_Debug_Arrivals   = nullptr;
      signal_params.num_segments         = numSegments;
    }
  };

private:
  Arguments  args_;
  GemmDevice gemm_device_;
  bool       initialized_;

public:
  GemmSignalSm90() : initialized_(false) {}

  Status initialize(Arguments const &args, cudaStream_t stream = nullptr) {
    args_ = args;

    int M = args_.problem_size.m();
    int N = args_.problem_size.n();
    int K = args_.problem_size.k();

    using Kernel             = GemmKernel;
    using KernelArguments    = typename Kernel::Arguments;
    using ProblemShape       = typename Kernel::ProblemShape;
    using MainloopArguments  = typename Kernel::MainloopArguments;
    using EpilogueArguments  = typename Kernel::EpilogueArguments;
    using TileSchedArguments = typename Kernel::TileSchedulerArguments;

    ProblemShape problem_shape = cute::make_shape(M, N, K, 1);

    MainloopArguments mainloop_args{
      reinterpret_cast<ElementInputA const*>(args_.ptr_A),
      CuteMainloopStride2D<LayoutInputA>::make(args_.ldm_A),
      reinterpret_cast<ElementInputB const*>(args_.ptr_B),
      CuteMainloopStride2D<LayoutInputB>::make(args_.ldm_B)
    };

#if defined(OOVERLAP_USE_BASE_EPILOGUE_ONLY) && OOVERLAP_USE_BASE_EPILOGUE_ONLY

    EpilogueArguments epilogue_args{
      {args_.alpha, args_.beta},
      reinterpret_cast<ElementOutput const*>(args_.ptr_C),
      CuteEpilogueStride2D<LayoutOutput>::make(args_.ldm_C),
      reinterpret_cast<ElementOutput*>(args_.ptr_D),
      CuteEpilogueStride2D<LayoutOutput>::make(args_.ldm_D)
    };

#else

    EpilogueArguments epilogue_args;
    epilogue_args.base = typename BaseCollectiveEpilogue::Arguments{
      {args_.alpha, args_.beta},
      reinterpret_cast<ElementOutput const*>(args_.ptr_C),
      CuteEpilogueStride2D<LayoutOutput>::make(args_.ldm_C),
      reinterpret_cast<ElementOutput*>(args_.ptr_D),
      CuteEpilogueStride2D<LayoutOutput>::make(args_.ldm_D)
    };
    epilogue_args.signal = args_.signal_params;

#endif

    cutlass::KernelHardwareInfo hw_info;

    int device_id = 0;
    cudaGetDevice(&device_id);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device_id);

    hw_info.device_id = device_id;
    hw_info.sm_count  = prop.multiProcessorCount;

    TileSchedArguments sched_args{};

    KernelArguments gemm_args =
      detail::make_kernel_arguments<KernelArguments>(
        problem_shape,
        mainloop_args,
        epilogue_args,
        hw_info,
        sched_args
      );

    Status status = gemm_device_.can_implement(gemm_args);
    if (status != Status::kSuccess) {
      initialized_ = false;
      return status;
    }

    status = gemm_device_.initialize(gemm_args, nullptr, stream);
    if (status != Status::kSuccess) {
      initialized_ = false;
      return status;
    }

    initialized_ = true;
    return cutlass::Status::kSuccess;
  }

  Status run(cudaStream_t stream) {
    if (!initialized_) {
      Status status = initialize(args_, stream);
      if (status != Status::kSuccess) {
        return status;
      }
    }

    return gemm_device_.run(stream);
  }

  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  void reset() {
    initialized_ = false;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass
