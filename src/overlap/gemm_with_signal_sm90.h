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
#include "cutlass/float8.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/kernel_hardware_info.h"

// Optional forward-decls for collective builder
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
// IMPORTANT:
// For your CUTLASS rev (as evidenced by the tuple type mismatch you hit),
// B's stride type is expected to be cute::tuple<int64_t, C<1>, int64_t> even when LayoutB is ColumnMajor.
// This matches the common trick: treat B as (N,K) row-major "packed" and interpret it as ColumnMajor(K,N).
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
    // NOTE: must match expected type in your SM90 builder instantiation
    return cute::make_stride(ld, cute::Int<1>{}, int64_t{0});
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Reorder + signal epilogue wrapper.
///
/// This wrapper MUST match the SM90 GemmUniversal warp-specialized epilogue interface.
/// The SM90 kernel calls:
///   - CollectiveEpilogue::prefetch_tma_descriptors(...)
///   - collective_epilogue.is_producer_load_needed()
///   - collective_epilogue.load(...)
///   - collective_epilogue.load_tail(...)
///   - collective_epilogue.store(...)
///   - collective_epilogue.store_tail(...)
///
/// We forward everything to BaseEpilogue, but remap CTA coords into the reordered/reshaped output space.
/// Signaling is done after store_tail() (closest point to "stores are flushed").
/////////////////////////////////////////////////////////////////////////////////////////////////

template <class BaseEpilogue, class ThreadblockShape>
struct ReorderSignalEpilogue {

  // -----------------------------
  // Required type aliases (forward)
  // -----------------------------
  using ElementC         = typename BaseEpilogue::ElementC;
  using ElementD         = typename BaseEpilogue::ElementD;
  using StrideC          = typename BaseEpilogue::StrideC;
  using StrideD          = typename BaseEpilogue::StrideD;
  using ThreadEpilogueOp = typename BaseEpilogue::ThreadEpilogueOp;
  using DispatchPolicy   = typename BaseEpilogue::DispatchPolicy;

  using SharedStorage    = typename BaseEpilogue::SharedStorage;

  // SM90 kernel expects these
  using TensorStorage    = typename BaseEpilogue::TensorStorage;
  using PipelineStorage  = typename BaseEpilogue::PipelineStorage;

  using LoadPipeline        = typename BaseEpilogue::LoadPipeline;
  using StorePipeline       = typename BaseEpilogue::StorePipeline;
  using LoadPipelineState   = typename BaseEpilogue::LoadPipelineState;
  using StorePipelineState  = typename BaseEpilogue::StorePipelineState;

  static constexpr bool RequiresTransactionBytes = BaseEpilogue::RequiresTransactionBytes;

  // Detect NumEpilogueWarpGroups if present; otherwise assume 4 (common for SM90)
  template <class T, class = void>
  struct HasNumEpiWGs : std::false_type {};
  template <class T>
  struct HasNumEpiWGs<T, std::void_t<decltype(T::NumEpilogueWarpGroups)>> : std::true_type {};

  static constexpr int kNumEpilogueWarpGroups =
      HasNumEpiWGs<BaseEpilogue>::value ? 4 : 4;

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
  };

  // -----------------------------
  // Required static interface
  // -----------------------------
  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return BaseEpilogue::can_implement(problem_shape, args.base);
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return BaseEpilogue::get_workspace_size(problem_shape, args.base);
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const& problem_shape, Arguments const& args, int sm_count) {
    return BaseEpilogue::get_workspace_size(problem_shape, args.base, sm_count);
  }

  static size_t get_workspace_alignment() {
    return BaseEpilogue::get_workspace_alignment();
  }

  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& problem_shape,
                                       Arguments const& args,
                                       void* workspace) {
    Params p;
    p.base   = BaseEpilogue::to_underlying_arguments(problem_shape, args.base, workspace);
    p.signal = args.signal;
    return p;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(ProblemShape const& problem_shape,
                                              Arguments const& args,
                                              void* workspace,
                                              cudaStream_t stream,
                                              cutlass::CudaHostAdapter* cuda_adapter = nullptr) {
    return BaseEpilogue::initialize_workspace(problem_shape, args.base, workspace, stream, cuda_adapter);
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    BaseEpilogue::prefetch_tma_descriptors(params.base);
  }

  CUTLASS_DEVICE
  static int get_transaction_bytes(Params const& params) {
    return BaseEpilogue::get_transaction_bytes(params.base);
  }

  template<class TileShapeMNK>
  CUTLASS_HOST_DEVICE
  static constexpr int get_load_pipe_increment(TileShapeMNK tile_shape_mnk) {
    return BaseEpilogue::get_load_pipe_increment(tile_shape_mnk);
  }

  template<class TileShapeMNK>
  CUTLASS_HOST_DEVICE
  static constexpr int get_store_pipe_increment(TileShapeMNK tile_shape_mnk) {
    return BaseEpilogue::get_store_pipe_increment(tile_shape_mnk);
  }

  // -----------------------------
  // Stateful construction (REQUIRED)
  // -----------------------------
  CUTLASS_DEVICE
  ReorderSignalEpilogue(Params const& params, TensorStorage& tensor_storage)
    : params_(params)
    , base_(params.base, tensor_storage)
    , reordered_tile_(0)
    , M_(0)
    , N_(0) {}

  CUTLASS_DEVICE
  bool is_producer_load_needed() const {
    return base_.is_producer_load_needed();
  }

  // -----------------------------
  // Forwarders we don’t touch
  // -----------------------------
  template <class... Args>
  CUTLASS_DEVICE decltype(auto) load_init(Args&&... args) {
    return base_.load_init(std::forward<Args>(args)...);
  }

  template <class... Args>
  CUTLASS_DEVICE decltype(auto) store_init(Args&&... args) {
    return base_.store_init(std::forward<Args>(args)...);
  }

  template <class... Args>
  CUTLASS_DEVICE decltype(auto) load(Args&&... args) {
    return base_.load(std::forward<Args>(args)...);
  }

  template <class... Args>
  CUTLASS_DEVICE decltype(auto) load_tail(Args&&... args) {
    return base_.load_tail(std::forward<Args>(args)...);
  }

  // -----------------------------
  // store(): remap ONLY the tile coord (no reshape)
  // -----------------------------
  template <
    class EpiLoadPipe, class EpiLoadState,
    class EpiStorePipe, class EpiStoreState,
    class ProblemShape, class TileShape, class TileCoord,
    class AccumTensor, class TiledMma, class EpiSharedStorage
  >
  CUTLASS_DEVICE
  decltype(auto) store(
      EpiLoadPipe&&   epi_load_pipe,
      EpiLoadState&&  epi_load_state,
      EpiStorePipe&&  epi_store_pipe,
      EpiStoreState&& epi_store_state,
      ProblemShape const& problem_shape,
      TileShape   const& tile_shape,
      TileCoord   const& tile_coord,
      AccumTensor const& accum,
      TiledMma    const& tiled_mma,
      int thread_idx,
      EpiSharedStorage& shared_storage) {

    // cache M,N for store_tail bookkeeping
    M_ = int(cute::get<0>(problem_shape));
    N_ = int(cute::get<1>(problem_shape));

    int cta_m = int(cute::get<0>(tile_coord));
    int cta_n = int(cute::get<1>(tile_coord));

    int tile_cols = params_.signal.kMonitoredColumn;   // == N/TileN (bring-up)
    int logical_tile = cta_m * tile_cols + cta_n;

    int reordered_tile = params_.signal.ptr_Reorder_Array[logical_tile];
    reordered_tile_ = reordered_tile;

    int dst_m = reordered_tile / tile_cols;
    int dst_n = reordered_tile % tile_cols;

    auto mapped_tile_coord = cute::make_tuple(
      dst_m,
      dst_n,
      cute::get<2>(tile_coord),
      cute::get<3>(tile_coord)
    );

    return base_.store(
      std::forward<EpiLoadPipe>(epi_load_pipe),
      std::forward<EpiLoadState>(epi_load_state),
      std::forward<EpiStorePipe>(epi_store_pipe),
      std::forward<EpiStoreState>(epi_store_state),
      problem_shape,
      tile_shape,
      mapped_tile_coord,
      accum,
      tiled_mma,
      thread_idx,
      shared_storage
    );
  }

  // -----------------------------
  // store_tail(): forward + signal ONCE per CTA tile
  //
  // Strategy:
  //   - each epilogue warp-group does: tile_done[tile]++
  //   - the LAST one (old == kNumEpilogueWarpGroups-1) does:
  //         threadfence + MM[seg]++
  //
  // Layout of ptr_Monitored_Matrix (MM):
  //   MM[0 .. num_segments-1]          : segment counters
  //   MM[num_segments .. num_segments+num_tiles-1] : per-tile done counters
  //
  // Requirement: MM.numel() >= num_segments + num_tiles
  // -----------------------------
  template <class... Args>
  CUTLASS_DEVICE
  decltype(auto) store_tail(Args&&... args) {
    auto ret = base_.store_tail(std::forward<Args>(args)...);

    // One lane per participating warp performs the bookkeeping
    if (cute::elect_one_sync()) {

      int tile = reordered_tile_;

      // compute num_tiles (M/TileM)*(N/TileN)
      int tile_rows = M_ / params_.signal.ThreadblockM;
      int tile_cols = params_.signal.kMonitoredColumn;
      int num_tiles = tile_rows * tile_cols;

      // compute num_segments by summing CommSeg sizes until sum == num_tiles
      int num_segments = 0;
      int sum = 0;
      while (sum < num_tiles) {
        sum += params_.signal.kCommu_Seg_Array[num_segments];
        ++num_segments;
      }

      // locate this tile's segment
      int idx_bound = params_.signal.kCommu_Seg_Array[0];
      int seg = 0;
      while (idx_bound <= tile) {
        ++seg;
        idx_bound += params_.signal.kCommu_Seg_Array[seg];
      }

      // per-tile done counter base
      int* tile_done = params_.signal.ptr_Monitored_Matrix + num_segments;

      // last warp-group to arrive signals the segment counter once
      int old = atomicAdd(&tile_done[tile], 1);
      if (old == (kNumEpilogueWarpGroups - 1)) {

        __threadfence();
        atomicAdd(&params_.signal.ptr_Monitored_Matrix[seg], 1);

        if (params_.signal.if_monitor) {
          int global_order =
            atomicAdd(&params_.signal.ptr_Monitored_Matrix[tile_cols - 1], 1);

          cutlass::arch::global_store<int, sizeof(int)>(
            global_order,
            (void*)(params_.signal.ptr_Monitored_Matrix + tile_cols + tile),
            true
          );
        }
      }
    }

    return ret;
  }

private:
  Params params_;
  BaseEpilogue base_;
  int reordered_tile_;
  int M_;
  int N_;
};

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
