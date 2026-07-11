#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <type_traits>

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
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

#include "cute/tensor.hpp"
#include "epilogue/helper.h"

#ifndef OOVERLAP_DEVICE_INLINE
#define OOVERLAP_DEVICE_INLINE CUTLASS_DEVICE
#endif

#ifndef OOVERLAP_ENABLE_EPILOGUE_REORDER
#define OOVERLAP_ENABLE_EPILOGUE_REORDER 1
#endif

#ifndef OOVERLAP_ENABLE_EPILOGUE_SIGNAL
#define OOVERLAP_ENABLE_EPILOGUE_SIGNAL 1
#endif

#ifndef OOVERLAP_ENABLE_EPILOGUE_DEBUG
#define OOVERLAP_ENABLE_EPILOGUE_DEBUG 0
#endif

#ifndef OOVERLAP_ENABLE_EPILOGUE_MONITOR
#define OOVERLAP_ENABLE_EPILOGUE_MONITOR 1
#endif

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

struct SignalingEpilogueParams {
  int  *ptr_Monitored_Matrix;
  int  *ptr_Reorder_Array;
  int   kMonitoredColumn;
  int   kReorderedColumn;
  int  *kCommu_Seg_Array;
  bool  if_monitor;

  int   ThreadblockM;
  int   ThreadblockN;

  void *ptr_D;
  int   ld_D;

  int   kEpilogueArrivalsPerTile;
  int  *ptr_Debug_Arrivals;

  // Number of segment counters at the front of ptr_Monitored_Matrix.
  // tile_done starts at ptr_Monitored_Matrix + num_segments.
  int   num_segments;

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
    ld_D(0),
    kEpilogueArrivalsPerTile(0),
    ptr_Debug_Arrivals(nullptr),
    num_segments(0)
  {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class BaseEpilogue,
  class ThreadblockShape,
  bool DeepCoopSignal = false
>
struct ReorderSignalEpilogue {

  using ElementC         = typename BaseEpilogue::ElementC;
  using ElementD         = typename BaseEpilogue::ElementD;
  using StrideC          = typename BaseEpilogue::StrideC;
  using StrideD          = typename BaseEpilogue::StrideD;
  using ThreadEpilogueOp = typename BaseEpilogue::ThreadEpilogueOp;
  using DispatchPolicy   = typename BaseEpilogue::DispatchPolicy;

  using SharedStorage    = typename BaseEpilogue::SharedStorage;
  using TensorStorage    = typename BaseEpilogue::TensorStorage;
  using PipelineStorage  = typename BaseEpilogue::PipelineStorage;

  using LoadPipeline        = typename BaseEpilogue::LoadPipeline;
  using StorePipeline       = typename BaseEpilogue::StorePipeline;
  using LoadPipelineState   = typename BaseEpilogue::LoadPipelineState;
  using StorePipelineState  = typename BaseEpilogue::StorePipelineState;

  static constexpr bool RequiresTransactionBytes = BaseEpilogue::RequiresTransactionBytes;

  static constexpr int kNumEpilogueWarpGroups =
      EpiWGsFromBase<BaseEpilogue>::value ?
        EpiWGsFromBase<BaseEpilogue>::value :
      EpiWGsFromDispatchPolicy<BaseEpilogue>::value ?
        EpiWGsFromDispatchPolicy<BaseEpilogue>::value :
      EpiWGsFromSchedule<BaseEpilogue>::value ?
        EpiWGsFromSchedule<BaseEpilogue>::value :
        0;

  struct Arguments {
    typename BaseEpilogue::Arguments base;
    SignalingEpilogueParams signal;
  };

  struct Params {
    typename BaseEpilogue::Params base;
    SignalingEpilogueParams signal;

    // HERE forward CUTLASS TMA epilogue transaction bytes
    uint32_t tma_transaction_bytes;
  };


  // Return the epilogue/output problem shape that the base epilogue should use
  // when building descriptors.  For packed/reordered output this is different
  // from the logical GEMM MxN problem shape.
  template <class ProblemShape>
  CUTLASS_HOST_DEVICE
  static ProblemShape epilogue_problem_shape(
      ProblemShape problem_shape,
      SignalingEpilogueParams const& signal) {
#if OOVERLAP_ENABLE_EPILOGUE_REORDER
    int tile_m = signal.ThreadblockM;
    int tile_n = signal.ThreadblockN;

    int original_tile_cols = signal.kMonitoredColumn;
    int packed_tile_cols = signal.kReorderedColumn;

    if (packed_tile_cols <= 0) {
      packed_tile_cols = original_tile_cols;
    }

    if (tile_m <= 0 || tile_n <= 0 || original_tile_cols <= 0 || packed_tile_cols <= 0) {
      return problem_shape;
    }

    int M = int(cute::get<0>(problem_shape));

    int original_tile_rows = (M + tile_m - 1) / tile_m;
    int original_tile_num = original_tile_rows * original_tile_cols;

    int packed_tile_rows = (original_tile_num + packed_tile_cols - 1) / packed_tile_cols;

    cute::get<0>(problem_shape) = packed_tile_rows * tile_m;
    cute::get<1>(problem_shape) = packed_tile_cols * tile_n;
#endif

    return problem_shape;
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    auto epi_shape = epilogue_problem_shape(problem_shape, args.signal);
    return BaseEpilogue::can_implement(epi_shape, args.base);
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    auto epi_shape = epilogue_problem_shape(problem_shape, args.signal);
    return BaseEpilogue::get_workspace_size(epi_shape, args.base);
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const& problem_shape, Arguments const& args, int sm_count) {
    auto epi_shape = epilogue_problem_shape(problem_shape, args.signal);
    return BaseEpilogue::get_workspace_size(epi_shape, args.base, sm_count);
  }

  static size_t get_workspace_alignment() {
    return BaseEpilogue::get_workspace_alignment();
  }

  template <class ProblemShape>
  static Params to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      void* workspace) {
    Params p;
    auto epi_shape = epilogue_problem_shape(problem_shape, args.signal);
    p.base   = BaseEpilogue::to_underlying_arguments(epi_shape, args.base, workspace);
    p.signal = args.signal;

    if constexpr (BaseEpilogue::RequiresTransactionBytes) {
      p.tma_transaction_bytes = p.base.tma_transaction_bytes;
    }
    else {
      p.tma_transaction_bytes = 0;
    }

    return p;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
      ProblemShape const& problem_shape,
      Arguments const& args,
      void* workspace,
      cudaStream_t stream,
      cutlass::CudaHostAdapter* cuda_adapter = nullptr) {
    auto epi_shape = epilogue_problem_shape(problem_shape, args.signal);
    return BaseEpilogue::initialize_workspace(epi_shape, args.base, workspace, stream, cuda_adapter);
  }

  OOVERLAP_DEVICE_INLINE
  static void prefetch_tma_descriptors(Params const& params) {
    BaseEpilogue::prefetch_tma_descriptors(params.base);
  }

  OOVERLAP_DEVICE_INLINE
  static int get_transaction_bytes(Params const& params) {
    return params.tma_transaction_bytes;
    //return BaseEpilogue::get_transaction_bytes(params.base);
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

  OOVERLAP_DEVICE_INLINE
  ReorderSignalEpilogue(Params const& params, TensorStorage& tensor_storage)
    : params_(params)
    , base_(params.base, tensor_storage)
    , reordered_tile_(0)
    , M_(0)
    , N_(0) {}

  OOVERLAP_DEVICE_INLINE
  bool is_producer_load_needed() const {
    return base_.is_producer_load_needed();
  }

  template <class... Args>
  OOVERLAP_DEVICE_INLINE decltype(auto) load_init(Args&&... args) {
    return base_.load_init(static_cast<Args&&>(args)...);
  }

  template <class... Args>
  OOVERLAP_DEVICE_INLINE decltype(auto) store_init(Args&&... args) {
    return base_.store_init(static_cast<Args&&>(args)...);
  }

  template <class... Args>
  OOVERLAP_DEVICE_INLINE decltype(auto) load(Args&&... args) {
    return base_.load(static_cast<Args&&>(args)...);
  }

  template <class... Args>
  OOVERLAP_DEVICE_INLINE decltype(auto) load_tail(Args&&... args) {
    return base_.load_tail(static_cast<Args&&>(args)...);
  }

  // HERE extract cooperative subtile_idx if present
  OOVERLAP_DEVICE_INLINE
  static int get_subtile_idx() {
    return -1;
  }

  template <class T>
  OOVERLAP_DEVICE_INLINE
  static int get_subtile_idx(T const& x) {
    return int(x);
  }

  template <class T, class... Rest>
  OOVERLAP_DEVICE_INLINE
  static int get_subtile_idx(T const& x, Rest const&...) {
    return int(x);
  }


  //
  // NOTE:
  //   The normal WS/pingpong SM90 kernels call epilogue.store with:
  //
  //     (..., tiled_mma, thread_idx, shared_storage)
  //
  //   The cooperative SM90 kernel calls epilogue.store with one extra trailing
  //   scheduler/work-index argument:
  //
  //     (..., tiled_mma, thread_idx, shared_storage, some_int32)
  //
  //   Keep this store signature variadic at the end and forward those extra
  //   args to BaseEpilogue::store. Otherwise cooperative kernels fail with
  //   "no instance of ReorderSignalEpilogue::store matches the argument list".
  //
  template <
    class EpiLoadPipe, class EpiLoadState,
    class EpiStorePipe, class EpiStoreState,
    class ProblemShape, class TileShape, class TileCoord,
    class AccumTensor, class TiledMma, class EpiSharedStorage,
    class... ExtraArgs
  >
  OOVERLAP_DEVICE_INLINE
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
      EpiSharedStorage& shared_storage,
      ExtraArgs&&... extra_args) {

    M_ = int(cute::get<0>(problem_shape));
    N_ = int(cute::get<1>(problem_shape));

    int cta_m = int(cute::get<0>(tile_coord));
    int cta_n = int(cute::get<1>(tile_coord));

    int tile_m = params_.signal.ThreadblockM;
    int tile_n = params_.signal.ThreadblockN;

    int original_tile_cols = params_.signal.kMonitoredColumn;
    int packed_tile_cols   = params_.signal.kReorderedColumn;

    if (packed_tile_cols <= 0) {
      packed_tile_cols = original_tile_cols;
    }

    int logical_tile = cta_m * original_tile_cols + cta_n;

    int packed_tile = logical_tile;

#if OOVERLAP_ENABLE_EPILOGUE_REORDER
    if (params_.signal.ptr_Reorder_Array != nullptr) {
      packed_tile = params_.signal.ptr_Reorder_Array[logical_tile];
    }
#endif

    reordered_tile_ = packed_tile;

#if OOVERLAP_ENABLE_EPILOGUE_REORDER

    int original_tile_rows = (M_ + tile_m - 1) / tile_m;
    int original_tile_num  = original_tile_rows * original_tile_cols;

    int packed_tile_rows = (original_tile_num + packed_tile_cols - 1) / packed_tile_cols;

    int packed_M = packed_tile_rows * tile_m;
    int packed_N = packed_tile_cols * tile_n;

    int packed_cta_m = packed_tile / packed_tile_cols;
    int packed_cta_n = packed_tile - packed_cta_m * packed_tile_cols;

    auto packed_problem_shape = problem_shape;
    cute::get<0>(packed_problem_shape) = packed_M;
    cute::get<1>(packed_problem_shape) = packed_N;

    auto packed_tile_coord = tile_coord;
    cute::get<0>(packed_tile_coord) = packed_cta_m;
    cute::get<1>(packed_tile_coord) = packed_cta_n;

    if constexpr (DeepCoopSignal) {
      // HERE pass packed tile to CUTLASS TMA-subtile signal path
      typename BaseEpilogue::OoverlapCoopSignal sig;
      sig.MM = params_.signal.ptr_Monitored_Matrix;
      sig.cseg = params_.signal.kCommu_Seg_Array;
      sig.debug = params_.signal.ptr_Debug_Arrivals;
      sig.num_segments = params_.signal.num_segments;
      sig.tile = packed_tile;
      sig.expected = BaseEpilogue::get_store_pipe_increment(tile_shape);
      sig.if_monitor = params_.signal.if_monitor;

      int subtile_idx = get_subtile_idx(static_cast<ExtraArgs&&>(extra_args)...);

      return base_.store(
        static_cast<EpiLoadPipe&&>(epi_load_pipe),
        static_cast<EpiLoadState&&>(epi_load_state),
        static_cast<EpiStorePipe&&>(epi_store_pipe),
        static_cast<EpiStoreState&&>(epi_store_state),
        packed_problem_shape,
        tile_shape,
        packed_tile_coord,
        accum,
        tiled_mma,
        thread_idx,
        shared_storage,
        subtile_idx,
        sig
      );
    }
    else {
      return base_.store(
        static_cast<EpiLoadPipe&&>(epi_load_pipe),
        static_cast<EpiLoadState&&>(epi_load_state),
        static_cast<EpiStorePipe&&>(epi_store_pipe),
        static_cast<EpiStoreState&&>(epi_store_state),
        packed_problem_shape,
        tile_shape,
        packed_tile_coord,
        accum,
        tiled_mma,
        thread_idx,
        shared_storage,
        static_cast<ExtraArgs&&>(extra_args)...
      );
    }

#else

    if constexpr (DeepCoopSignal) {
      // HERE pass logical tile to CUTLASS TMA-subtile signal path
      typename BaseEpilogue::OoverlapCoopSignal sig;
      sig.MM = params_.signal.ptr_Monitored_Matrix;
      sig.cseg = params_.signal.kCommu_Seg_Array;
      sig.debug = params_.signal.ptr_Debug_Arrivals;
      sig.num_segments = params_.signal.num_segments;
      sig.tile = packed_tile;
      sig.expected = BaseEpilogue::get_store_pipe_increment(tile_shape);
      sig.if_monitor = params_.signal.if_monitor;

      int subtile_idx = get_subtile_idx(static_cast<ExtraArgs&&>(extra_args)...);

      return base_.store(
        static_cast<EpiLoadPipe&&>(epi_load_pipe),
        static_cast<EpiLoadState&&>(epi_load_state),
        static_cast<EpiStorePipe&&>(epi_store_pipe),
        static_cast<EpiStoreState&&>(epi_store_state),
        problem_shape,
        tile_shape,
        tile_coord,
        accum,
        tiled_mma,
        thread_idx,
        shared_storage,
        subtile_idx,
        sig
      );
    }
    else {
      return base_.store(
        static_cast<EpiLoadPipe&&>(epi_load_pipe),
        static_cast<EpiLoadState&&>(epi_load_state),
        static_cast<EpiStorePipe&&>(epi_store_pipe),
        static_cast<EpiStoreState&&>(epi_store_state),
        problem_shape,
        tile_shape,
        tile_coord,
        accum,
        tiled_mma,
        thread_idx,
        shared_storage,
        static_cast<ExtraArgs&&>(extra_args)...
      );
    }

#endif
  }

  template <class... Args>
  OOVERLAP_DEVICE_INLINE
  decltype(auto) store_tail(Args&&... args) {
    auto ret = base_.store_tail(static_cast<Args&&>(args)...);

#if OOVERLAP_ENABLE_EPILOGUE_SIGNAL
    // HERE old wrapper signal is disabled for cooperative deep path
    if constexpr (!DeepCoopSignal) {

      int linear_tid =
        int(threadIdx.x) +
        int(blockDim.x) * (int(threadIdx.y) + int(blockDim.y) * int(threadIdx.z));

      int lane_idx = linear_tid & 31;
      int warp_idx = linear_tid >> 5;

      constexpr int kWarpsPerWarpGroup = 4;
      int warp_idx_in_wg = warp_idx & (kWarpsPerWarpGroup - 1);

      bool one_thread_per_warpgroup = (lane_idx == 0) && (warp_idx_in_wg == 0);

      if (one_thread_per_warpgroup) {

        int tile = reordered_tile_;

#if OOVERLAP_ENABLE_EPILOGUE_DEBUG
        if (params_.signal.ptr_Debug_Arrivals) {
          atomicAdd(&params_.signal.ptr_Debug_Arrivals[tile], 1);
        }
#endif

        int* tile_done = params_.signal.ptr_Monitored_Matrix + params_.signal.num_segments;

        int expected_arrivals =
            (params_.signal.kEpilogueArrivalsPerTile > 0)
                ? params_.signal.kEpilogueArrivalsPerTile
                : kNumEpilogueWarpGroups;

        if (expected_arrivals <= 0) {
          expected_arrivals = 1;
        }

        int old = atomicAdd(&tile_done[tile], 1);

        if (old == (expected_arrivals - 1)) {
          __threadfence();

          int idx_bound = params_.signal.kCommu_Seg_Array[0];
          int seg = 0;

          while (idx_bound <= tile) {
            ++seg;
            idx_bound += params_.signal.kCommu_Seg_Array[seg];
          }

          atomicAdd(&params_.signal.ptr_Monitored_Matrix[seg], 1);

#if OOVERLAP_ENABLE_EPILOGUE_MONITOR
          if (params_.signal.if_monitor) {
            // MM layout used by the SM90 port:
            //   MM[0 : num_segments]
            //       per-segment ready counters, consumed by comm stream
            //   MM[num_segments : num_segments + tile_num]
            //       per-tile epilogue arrival counters, used internally here
            //   MM[num_segments + tile_num]
            //       global monitor/order counter, only used when if_monitor=true
            //   MM[num_segments + tile_num + 1 : num_segments + tile_num + 1 + tile_num]
            //       monitor output: monitor_order[tile] = tile completion order
            //
            // Do not store monitor data in tile_done. tile_done is live
            // synchronization state and must remain arrival counts.
            int tile_cols =
              (N_ + params_.signal.ThreadblockN - 1) /
              params_.signal.ThreadblockN;

            int tile_rows =
              (M_ + params_.signal.ThreadblockM - 1) /
              params_.signal.ThreadblockM;

            int tile_num = tile_rows * tile_cols;

            int* monitor_counter =
              params_.signal.ptr_Monitored_Matrix +
              params_.signal.num_segments + tile_num;

            int* monitor_order = monitor_counter + 1;

            int global_order =
              atomicAdd(monitor_counter, 1);

            cutlass::arch::global_store<int, sizeof(int)>(
              global_order,
              (void*)(monitor_order + tile),
              true
            );
          }
#endif
        }
      }
    }
#endif

    return ret;
  }

private:
  Params params_;
  BaseEpilogue base_;
  int reordered_tile_;
  int M_;
  int N_;
};

} // namespace cutlass
