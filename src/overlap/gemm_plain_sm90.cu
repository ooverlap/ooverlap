#include "overlap/gemm_plain_sm90_dispatch.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/kernel_hardware_info.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cute/tensor.hpp"

namespace ooverlap {
namespace detail {

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

inline bool check_status(cutlass::Status status, const char* where) {
  if (status == cutlass::Status::kSuccess) {
    return true;
  }

  std::cerr
      << "CUTLASS error in " << where << ": "
      << cutlassGetStatusString(status)
      << std::endl;

  return false;
}

inline bool check_cuda(cudaError_t err, const char* where) {
  if (err == cudaSuccess) {
    return true;
  }

  std::cerr
      << "CUDA error in " << where << ": "
      << cudaGetErrorString(err)
      << std::endl;

  return false;
}

template <
    int TileM,
    int TileN,
    int TileK,
    typename StageCountType,
    typename ClusterShape,
    typename MainloopSchedule,
    typename EpilogueSchedule,
    typename TileScheduler>
bool launch_plain_tnn_colmajor(
    int M,
    int N,
    int K,
    half* A,
    half* B_col,
    half* D_col,
    cudaStream_t stream) {
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::ColumnMajor;

  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using TileShape = cute::Shape<
      cute::Int<TileM>,
      cute::Int<TileN>,
      cute::Int<TileK>>;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementA,
          LayoutA,
          AlignmentA,
          ElementB,
          LayoutB,
          AlignmentB,
          ElementAccumulator,
          TileShape,
          ClusterShape,
          StageCountType,
          MainloopSchedule>::CollectiveOp;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          ElementCompute,
          ElementC,
          LayoutC,
          AlignmentC,
          ElementD,
          LayoutD,
          AlignmentD,
          EpilogueSchedule>::CollectiveOp;

  using GemmKernel =
      typename GemmKernelSelector<
          CollectiveMainloop,
          CollectiveEpilogue,
          TileScheduler>::type;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;

  auto problem_shape = cute::make_shape(M, N, K);

  // CUTLASS 3.x GEMM stride convention:
  //   A shape is (M, K, L)
  //   B shape is (N, K, L)
  //   C/D shape is (M, N, L)
  //
  // With LayoutB=ColumnMajor, StrideB should become (K, 1, ...), matching
  // physical B_col shape (N, K) contiguous.
  //
  // With LayoutD=ColumnMajor, StrideD should become (1, M, ...), matching
  // physical D_col shape (N, M) contiguous.
  auto stride_A = cutlass::make_cute_packed_stride(
      StrideA{}, cute::make_shape(M, K, 1));
  auto stride_B = cutlass::make_cute_packed_stride(
      StrideB{}, cute::make_shape(N, K, 1));
  auto stride_C = cutlass::make_cute_packed_stride(
      StrideC{}, cute::make_shape(M, N, 1));
  auto stride_D = cutlass::make_cute_packed_stride(
      StrideD{}, cute::make_shape(M, N, 1));

  cutlass::KernelHardwareInfo hw_info;

  int device_id = 0;
  if (!check_cuda(cudaGetDevice(&device_id), "cudaGetDevice")) {
    return false;
  }

  hw_info.device_id = device_id;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  float alpha = 1.0f;
  float beta = 0.0f;

  typename Gemm::Arguments arguments = [&]() {
    if constexpr (IsStreamK<TileScheduler>::value) {
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

      return typename Gemm::Arguments{
          cutlass::gemm::GemmUniversalMode::kGemm,
          problem_shape,
          {
              reinterpret_cast<ElementA const*>(A),
              stride_A,
              reinterpret_cast<ElementB const*>(B_col),
              stride_B,
          },
          {
              {alpha, beta},
              reinterpret_cast<ElementC const*>(D_col),
              stride_C,
              reinterpret_cast<ElementD*>(D_col),
              stride_D,
          },
          hw_info,
          scheduler_args};
    } else {
      return typename Gemm::Arguments{
          cutlass::gemm::GemmUniversalMode::kGemm,
          problem_shape,
          {
              reinterpret_cast<ElementA const*>(A),
              stride_A,
              reinterpret_cast<ElementB const*>(B_col),
              stride_B,
          },
          {
              {alpha, beta},
              reinterpret_cast<ElementC const*>(D_col),
              stride_C,
              reinterpret_cast<ElementD*>(D_col),
              stride_D,
          },
          hw_info};
    }
  }();

  Gemm gemm;

  cutlass::Status status = gemm.can_implement(arguments);
  if (!check_status(status, "can_implement")) {
    return false;
  }

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  void* workspace = nullptr;

  if (workspace_size > 0) {
    if (!check_cuda(cudaMalloc(&workspace, workspace_size), "cudaMalloc(workspace)")) {
      return false;
    }
  }

  status = gemm.initialize(arguments, workspace, stream);
  if (!check_status(status, "initialize")) {
    if (workspace != nullptr) {
      cudaFree(workspace);
    }
    return false;
  }

  status = gemm.run(stream);
  if (!check_status(status, "run")) {
    if (workspace != nullptr) {
      cudaFree(workspace);
    }
    return false;
  }

  if (workspace != nullptr) {
    if (!check_cuda(cudaFree(workspace), "cudaFree(workspace)")) {
      return false;
    }
  }

  return check_cuda(cudaGetLastError(), "cudaGetLastError");
}

}  // namespace detail

bool gemm_plain_sm90_dispatch(
    int algo,
    int M,
    int N,
    int K,
    void* A,
    void* B_col,
    void* D_col,
    cudaStream_t stream) {
  using Stage4 = cutlass::gemm::collective::StageCount<4>;
  using Stage5 = cutlass::gemm::collective::StageCount<5>;
  using Stage6 = cutlass::gemm::collective::StageCount<6>;
  using Stage7 = cutlass::gemm::collective::StageCount<7>;

  using Cluster1x1x1 = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using Cluster1x2x1 = cute::Shape<cute::_1, cute::_2, cute::_1>;
  using Cluster2x1x1 = cute::Shape<cute::_2, cute::_1, cute::_1>;

  using Coop = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using Pingpong = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using WS = cutlass::gemm::KernelTmaWarpSpecialized;

  using EpiAuto = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using StreamK = cutlass::gemm::StreamKScheduler;

  half* a = reinterpret_cast<half*>(A);
  half* b = reinterpret_cast<half*>(B_col);
  half* d = reinterpret_cast<half*>(D_col);

  switch (algo) {
    // Normal cooperative configs matching the fast non-Stream-K CSV candidates.
    case 0:
      return detail::launch_plain_tnn_colmajor<
          128, 256, 64,
          Stage4,
          Cluster2x1x1,
          Coop,
          EpiAuto,
          void>(M, N, K, a, b, d, stream);

    case 1:
      return detail::launch_plain_tnn_colmajor<
          256, 128, 64,
          Stage4,
          Cluster2x1x1,
          Coop,
          EpiAuto,
          void>(M, N, K, a, b, d, stream);

    case 2:
      return detail::launch_plain_tnn_colmajor<
          128, 256, 64,
          Stage4,
          Cluster1x2x1,
          Coop,
          EpiAuto,
          void>(M, N, K, a, b, d, stream);

    case 3:
      return detail::launch_plain_tnn_colmajor<
          256, 128, 64,
          Stage4,
          Cluster1x2x1,
          Coop,
          EpiAuto,
          void>(M, N, K, a, b, d, stream);

    // A couple of non-cooperative sanity configs.
    case 4:
      return detail::launch_plain_tnn_colmajor<
          64, 256, 64,
          Stage5,
          Cluster2x1x1,
          Pingpong,
          EpiAuto,
          void>(M, N, K, a, b, d, stream);

    case 5:
      return detail::launch_plain_tnn_colmajor<
          128, 128, 64,
          Stage7,
          Cluster1x1x1,
          WS,
          EpiAuto,
          void>(M, N, K, a, b, d, stream);

    // Stream-K versions of the two main candidates.
    case 10:
      return detail::launch_plain_tnn_colmajor<
          128, 256, 64,
          Stage4,
          Cluster2x1x1,
          Coop,
          EpiAuto,
          StreamK>(M, N, K, a, b, d, stream);

    case 11:
      return detail::launch_plain_tnn_colmajor<
          256, 128, 64,
          Stage4,
          Cluster2x1x1,
          Coop,
          EpiAuto,
          StreamK>(M, N, K, a, b, d, stream);

    case 12:
      return detail::launch_plain_tnn_colmajor<
          128, 256, 64,
          Stage4,
          Cluster1x2x1,
          Coop,
          EpiAuto,
          StreamK>(M, N, K, a, b, d, stream);

    case 13:
      return detail::launch_plain_tnn_colmajor<
          256, 128, 64,
          Stage4,
          Cluster1x2x1,
          Coop,
          EpiAuto,
          StreamK>(M, N, K, a, b, d, stream);

    // Backup 128x128 cooperative configs.
    case 20:
      return detail::launch_plain_tnn_colmajor<
          128, 128, 64,
          Stage6,
          Cluster2x1x1,
          Coop,
          EpiAuto,
          void>(M, N, K, a, b, d, stream);

    case 21:
      return detail::launch_plain_tnn_colmajor<
          128, 128, 64,
          Stage6,
          Cluster1x2x1,
          Coop,
          EpiAuto,
          void>(M, N, K, a, b, d, stream);

    default:
      std::cerr << "Unsupported plain SM90 algo=" << algo << std::endl;
      return false;
  }
}

}  // namespace ooverlap
