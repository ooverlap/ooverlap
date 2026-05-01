/***************************************************************************************************
 * SM90 CUTLASS 3.x GEMM-with-signal wrapper.
 *
 * This version is intentionally aligned with gemm_plain_sm90.cu:
 *
 *   A      : logical row-major [M, K], physical torch shape (M, K), contiguous
 *   B_col  : logical column-major [K, N], physical torch shape (N, K), contiguous
 *   D_col  : logical column-major [M, N], physical torch shape (N, M), contiguous
 *
 * For now, OOVERLAP_USE_BASE_EPILOGUE_ONLY defaults to 1, so reorder/signal epilogue
 * is compiled out and the kernel behaves like plain CUTLASS GEMM while keeping the
 * same signal API shape.
 *
 * Later, set OOVERLAP_USE_BASE_EPILOGUE_ONLY=0 to re-enable ReorderSignalEpilogue.
 **************************************************************************************************/
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cute/tensor.hpp"

#include "epilogue/reorder_epilogue.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef OOVERLAP_USE_BASE_EPILOGUE_ONLY
#define OOVERLAP_USE_BASE_EPILOGUE_ONLY 0
#endif

#define CUTLASS_CHECK_SM90(status)                                                 \
  {                                                                                \
    cutlass::Status error = status;                                                \
    if (error != cutlass::Status::kSuccess) {                                      \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error)          \
                << " at line " << __LINE__ << std::endl;                          \
      std::exit(EXIT_FAILURE);                                                     \
    }                                                                              \
  }

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace ooverlap_sm90_detail {

template <
    typename CollectiveMainloop,
    typename CollectiveEpilogue,
    typename TileScheduler>
struct GemmKernelSelector {
  using type = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileScheduler>;
};

template <
    typename CollectiveMainloop,
    typename CollectiveEpilogue>
struct GemmKernelSelector<CollectiveMainloop, CollectiveEpilogue, void> {
  using type = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;
};

template <typename Scheduler>
struct IsStreamK : std::false_type {};

template <>
struct IsStreamK<cutlass::gemm::StreamKScheduler> : std::true_type {};

}  // namespace ooverlap_sm90_detail

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
    cutlass::platform::is_same<LayoutInputA, cutlass::layout::RowMajor>::value,
    "This wrapper currently expects A row-major."
  );

  static_assert(
    cutlass::platform::is_same<LayoutInputB, cutlass::layout::ColumnMajor>::value,
    "This wrapper currently expects B column-major, physical shape [N, K]."
  );

  static_assert(
    cutlass::platform::is_same<LayoutOutput, cutlass::layout::RowMajor>::value,
    "This wrapper currently expects D row-major, physical shape [N, M]."
  );

  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag       = cutlass::arch::Sm90;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementInputA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementInputB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementInputA,
          LayoutInputA,
          AlignmentA,
          ElementInputB,
          LayoutInputB,
          AlignmentB,
          ElementCompute,
          TileShape,
          ClusterShape,
          StageCountType,
          MainloopSchedule>::CollectiveOp;

  using BaseCollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
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
          EpilogueSchedule>::CollectiveOp;

#if defined(OOVERLAP_USE_BASE_EPILOGUE_ONLY) && OOVERLAP_USE_BASE_EPILOGUE_ONLY
  using CollectiveEpilogue = BaseCollectiveEpilogue;
#else
  using CollectiveEpilogue =
      ReorderSignalEpilogue<BaseCollectiveEpilogue, ThreadblockShape>;
#endif

  using GemmKernel =
      typename cutlass::ooverlap_sm90_detail::GemmKernelSelector<
          CollectiveMainloop,
          CollectiveEpilogue,
          TileScheduler>::type;

  using GemmDevice = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

public:
  struct Arguments {
    cutlass::gemm::GemmCoord   problem_size;
    ElementInputA             *ptr_A;
    ElementInputB             *ptr_B;
    ElementOutput             *ptr_C;
    ElementOutput             *ptr_D;

    // Kept for API compatibility with the old signal path.
    // The base-epilogue path now uses CUTLASS packed strides instead.
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
      int64_t ldm_A_,
      int64_t ldm_B_,
      int64_t ldm_C_,
      int64_t ldm_D_,
      ElementCompute alpha_,
      ElementCompute beta_,
      int *ptr_MM,
      int *ptr_RA,
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
  static constexpr bool kIsStreamK =
      cutlass::ooverlap_sm90_detail::IsStreamK<TileScheduler>::value;

  Arguments  args_;
  GemmDevice gemm_device_;
  bool       initialized_;
  void*      workspace_;
  size_t     workspace_size_;

public:
  GemmSignalSm90()
      : initialized_(false),
        workspace_(nullptr),
        workspace_size_(0) {}

  ~GemmSignalSm90() {
    if (workspace_ != nullptr) {
      cudaFree(workspace_);
      workspace_ = nullptr;
      workspace_size_ = 0;
    }
  }

  Status initialize(Arguments const &args, cudaStream_t stream = nullptr) {
    args_ = args;

    int M = args_.problem_size.m();
    int N = args_.problem_size.n();
    int K = args_.problem_size.k();

    using MainloopArguments  = typename GemmKernel::MainloopArguments;
    using EpilogueArguments  = typename GemmKernel::EpilogueArguments;
    using StrideA            = typename GemmKernel::StrideA;
    using StrideB            = typename GemmKernel::StrideB;
    using StrideC            = typename GemmKernel::StrideC;
    using StrideD            = typename GemmKernel::StrideD;

    auto problem_shape = cute::make_shape(M, N, K);

    auto stride_A = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape(M, K, 1));

    auto stride_B = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape(N, K, 1));

    int original_tile_rows = (M + ThreadblockShape::kM - 1) / ThreadblockShape::kM;
    int original_tile_cols = (N + ThreadblockShape::kN - 1) / ThreadblockShape::kN;
    int original_tile_num  = original_tile_rows * original_tile_cols;
    
    int packed_tile_cols = original_tile_cols;
    
    #if !(defined(OOVERLAP_USE_BASE_EPILOGUE_ONLY) && OOVERLAP_USE_BASE_EPILOGUE_ONLY)
    if (args_.signal_params.kReorderedColumn > 0) {
      packed_tile_cols = args_.signal_params.kReorderedColumn;
    }
    #endif
    
    int packed_tile_rows = (original_tile_num + packed_tile_cols - 1) / packed_tile_cols;
    
    int out_rows = packed_tile_rows * ThreadblockShape::kM;
    int out_cols = packed_tile_cols * ThreadblockShape::kN;
    
    auto stride_C = cutlass::make_cute_packed_stride(
        StrideC{}, cute::make_shape(out_rows, out_cols, 1));
    
    auto stride_D = cutlass::make_cute_packed_stride(
        StrideD{}, cute::make_shape(out_rows, out_cols, 1));

    MainloopArguments mainloop_args{
      reinterpret_cast<ElementInputA const*>(args_.ptr_A),
      stride_A,
      reinterpret_cast<ElementInputB const*>(args_.ptr_B),
      stride_B
    };

#if defined(OOVERLAP_USE_BASE_EPILOGUE_ONLY) && OOVERLAP_USE_BASE_EPILOGUE_ONLY

    EpilogueArguments epilogue_args{
      {args_.alpha, args_.beta},
      reinterpret_cast<ElementOutput const*>(args_.ptr_C),
      stride_C,
      reinterpret_cast<ElementOutput*>(args_.ptr_D),
      stride_D
    };

#else

    EpilogueArguments epilogue_args;
    epilogue_args.base = typename BaseCollectiveEpilogue::Arguments{
      {args_.alpha, args_.beta},
      reinterpret_cast<ElementOutput const*>(args_.ptr_C),
      stride_C,
      reinterpret_cast<ElementOutput*>(args_.ptr_D),
      stride_D
    };
    epilogue_args.signal = args_.signal_params;

#endif

    cutlass::KernelHardwareInfo hw_info;

    int device_id = 0;
    cudaError_t dev_err = cudaGetDevice(&device_id);
    if (dev_err != cudaSuccess) {
      initialized_ = false;
      return cutlass::Status::kErrorInternal;
    }

    hw_info.device_id = device_id;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

    typename GemmDevice::Arguments gemm_args = [&]() {
      if constexpr (kIsStreamK) {
        using DecompositionMode =
            typename cutlass::gemm::kernel::detail::
                PersistentTileSchedulerSm90StreamKParams::DecompositionMode;

        typename GemmKernel::TileScheduler::Arguments scheduler_args{
            1,
            static_cast<int>(
                cutlass::gemm::kernel::detail::
                    PersistentTileSchedulerSm90::RasterOrder::AlongN),
            cutlass::gemm::kernel::detail::
                PersistentTileSchedulerSm90::RasterOrderOptions::Heuristic,
            DecompositionMode::StreamK};

        return typename GemmDevice::Arguments{
          cutlass::gemm::GemmUniversalMode::kGemm,
          problem_shape,
          mainloop_args,
          epilogue_args,
          hw_info,
          scheduler_args
        };
      } else {
        return typename GemmDevice::Arguments{
          cutlass::gemm::GemmUniversalMode::kGemm,
          problem_shape,
          mainloop_args,
          epilogue_args,
          hw_info
        };
      }
    }();

    Status status = gemm_device_.can_implement(gemm_args);
    if (status != Status::kSuccess) {
      initialized_ = false;
      return status;
    }

    size_t needed_workspace = GemmDevice::get_workspace_size(gemm_args);

    if (needed_workspace > workspace_size_) {
      if (workspace_ != nullptr) {
        cudaError_t free_err = cudaFree(workspace_);
        workspace_ = nullptr;
        workspace_size_ = 0;

        if (free_err != cudaSuccess) {
          initialized_ = false;
          return cutlass::Status::kErrorInternal;
        }
      }

      if (needed_workspace > 0) {
        cudaError_t malloc_err = cudaMalloc(&workspace_, needed_workspace);
        if (malloc_err != cudaSuccess) {
          initialized_ = false;
          workspace_ = nullptr;
          workspace_size_ = 0;
          return cutlass::Status::kErrorWorkspaceNull;
        }

        workspace_size_ = needed_workspace;
      }
    }

    status = gemm_device_.initialize(gemm_args, workspace_, stream);
    if (status != Status::kSuccess) {
      initialized_ = false;
      return status;
    }

    initialized_ = true;
    return cutlass::Status::kSuccess;
  }

  Status run(cudaStream_t stream) {
    if constexpr (kIsStreamK) {
      // Stream-K uses workspace/counters. Keep correctness first by resetting
      // CUTLASS scheduler state each launch. Use eager timing for Stream-K.
      Status status = initialize(args_, stream);
      if (status != Status::kSuccess) {
        return status;
      }

      return gemm_device_.run(stream);
    } else {
      if (!initialized_) {
        Status status = initialize(args_, stream);
        if (status != Status::kSuccess) {
          return status;
        }
      }

      return gemm_device_.run(stream);
    }
  }

  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  void reset() {
    initialized_ = false;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass
