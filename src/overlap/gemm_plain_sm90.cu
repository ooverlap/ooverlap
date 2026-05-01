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

struct PlainCacheKey {
  bool valid;
  int device;
  int M;
  int N;
  int K;
  void* A;
  void* B;
  void* D;

  PlainCacheKey()
      : valid(false),
        device(-1),
        M(0),
        N(0),
        K(0),
        A(nullptr),
        B(nullptr),
        D(nullptr) {}

  bool same_as(PlainCacheKey const& other) const {
    return valid &&
           other.valid &&
           device == other.device &&
           M == other.M &&
           N == other.N &&
           K == other.K &&
           A == other.A &&
           B == other.B &&
           D == other.D;
  }
};

}  // namespace detail
}  // namespace ooverlap

template <
    int TileM,
    int TileN,
    int TileK,
    typename StageCountType,
    typename ClusterShape,
    typename MainloopSchedule,
    typename EpilogueSchedule,
    typename TileScheduler>
bool cutlass_gemm_plain_sm90(
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
      typename ooverlap::detail::GemmKernelSelector<
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
  // With LayoutB=ColumnMajor, StrideB becomes (K, 1, ...), matching physical
  // B_col shape (N, K) contiguous.
  //
  // With LayoutD=ColumnMajor, StrideD becomes (1, M, ...), matching physical
  // D_col shape (N, M) contiguous.
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
  if (!ooverlap::detail::check_cuda(cudaGetDevice(&device_id), "cudaGetDevice")) {
    return false;
  }

  hw_info.device_id = device_id;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

  float alpha = 1.0f;
  float beta = 0.0f;

  typename Gemm::Arguments arguments = [&]() {
    if constexpr (ooverlap::detail::IsStreamK<TileScheduler>::value) {
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

  static Gemm gemm;
  static ooverlap::detail::PlainCacheKey cached_key;
  static void* workspace = nullptr;
  static size_t workspace_size = 0;

  ooverlap::detail::PlainCacheKey new_key;
  new_key.valid = true;
  new_key.device = device_id;
  new_key.M = M;
  new_key.N = N;
  new_key.K = K;
  new_key.A = reinterpret_cast<void*>(A);
  new_key.B = reinterpret_cast<void*>(B_col);
  new_key.D = reinterpret_cast<void*>(D_col);

  // Normal schedulers can safely reuse the initialized GEMM object. This also
  // makes the normal path CUDA-graph-capturable after the first warmup call.
  //
  // Stream-K scheduler state is more delicate; keep it correct first by
  // reinitializing each launch. Use eager timing for Stream-K experiments.
  bool must_initialize =
      !cached_key.same_as(new_key) ||
      ooverlap::detail::IsStreamK<TileScheduler>::value;

  if (must_initialize) {
    cutlass::Status status = gemm.can_implement(arguments);
    if (!ooverlap::detail::check_status(status, "can_implement")) {
      return false;
    }

    size_t needed_workspace = Gemm::get_workspace_size(arguments);
    if (needed_workspace > workspace_size) {
      if (workspace != nullptr) {
        if (!ooverlap::detail::check_cuda(cudaFree(workspace), "cudaFree(old workspace)")) {
          return false;
        }
        workspace = nullptr;
        workspace_size = 0;
      }

      if (needed_workspace > 0) {
        if (!ooverlap::detail::check_cuda(
                cudaMalloc(&workspace, needed_workspace),
                "cudaMalloc(workspace)")) {
          return false;
        }
        workspace_size = needed_workspace;
      }
    }

    status = gemm.initialize(arguments, workspace, stream);
    if (!ooverlap::detail::check_status(status, "initialize")) {
      return false;
    }

    cached_key = new_key;
  }

  cutlass::Status status = gemm.run(stream);
  if (!ooverlap::detail::check_status(status, "run")) {
    return false;
  }

  return ooverlap::detail::check_cuda(cudaGetLastError(), "cudaGetLastError");
}

// explicit instantiations
#include "inc/plain_instances_sm90.inc"

// function pointer table
#include "tiling/plain_tiling_sm90.cuh"

namespace ooverlap {

int gemm_plain_sm90_algo_count() {
  return plain_sm90_func_count;
}

bool gemm_plain_sm90_get_algo_meta(
    int algo,
    GemmPlainSm90AlgoMeta* out) {
  if (out == nullptr) {
    return false;
  }

  if (algo < 0 || algo >= plain_sm90_func_count) {
    return false;
  }

  auto const& src = plain_sm90_algo_meta[algo];

  out->tile_m = src.tile_m;
  out->tile_n = src.tile_n;
  out->tile_k = src.tile_k;
  out->cluster_m = src.cluster_m;
  out->cluster_n = src.cluster_n;
  out->cluster_k = src.cluster_k;
  out->stages = src.stages;
  out->mainloop = src.mainloop;
  out->epilogue = src.epilogue;
  out->scheduler = src.scheduler;

  return true;
}

bool gemm_plain_sm90_dispatch(
    int algo,
    int M,
    int N,
    int K,
    void* A,
    void* B_col,
    void* D_col,
    cudaStream_t stream) {
  if (algo < 0 || algo >= plain_sm90_func_count) {
    std::cerr << "Unsupported plain SM90 algo=" << algo
              << " count=" << plain_sm90_func_count << std::endl;
    return false;
  }

  return plain_sm90_func_table[algo](
      M,
      N,
      K,
      reinterpret_cast<half*>(A),
      reinterpret_cast<half*>(B_col),
      reinterpret_cast<half*>(D_col),
      stream);
}

}  // namespace ooverlap
