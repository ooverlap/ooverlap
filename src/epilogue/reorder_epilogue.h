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
    // TODO: this might be wrong because it is per wrap.
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

}
