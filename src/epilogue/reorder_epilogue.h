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
#include "epilogue/helper.h"

#ifndef OOVERLAP_DEVICE_INLINE
#define OOVERLAP_DEVICE_INLINE CUTLASS_DEVICE
#endif

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Parameters needed for reorder + signaling inside epilogue.
struct SignalingEpilogueParams {
  int  *ptr_Monitored_Matrix;
  int  *ptr_Reorder_Array;      // logical tile idx -> reordered/packed tile idx
  int   kMonitoredColumn;       // original tile-cols = N / TileN
  int   kReorderedColumn;       // packed tile-cols = ReLDN
  int  *kCommu_Seg_Array;       // segment sizes in units of tiles; sum == total tiles
  bool  if_monitor;

  int   ThreadblockM;
  int   ThreadblockN;

  void *ptr_D;                  // base ptr of FINAL output buffer
  int   ld_D;                   // leading dim in elements of packed D:
                                //   kReorderedColumn * ThreadblockN

  // Optional override/debug:
  // If > 0, use this as the expected number of elected arrivals per tile in store_tail().
  // Otherwise fall back to the compile-time trait extracted from CUTLASS.
  int   kEpilogueArrivalsPerTile;

  // Optional debug buffer of length >= num_tiles.
  // If non-null, each elected arrival does:
  //   atomicAdd(ptr_Debug_Arrivals + reordered_tile, 1)
  int  *ptr_Debug_Arrivals;

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
    ptr_Debug_Arrivals(nullptr)
  {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Reorder + signal epilogue wrapper.
///
/// This wrapper matches the SM90 GemmUniversal warp-specialized epilogue interface.
/// We forward almost everything to BaseEpilogue, but:
///   1) remap CTA tile coords into the reordered packed output space
///   2) pass a packed problem shape to the base epilogue so predicates match packed D
///   3) emit one segment-ready signal when the last participating epilogue subgroup
///      for that tile reaches store_tail()
///
/// Packed D layout:
///
///   original logical GEMM output:
///     [M, N]
///
///   original logical tile grid:
///     tile_rows = ceil(M / ThreadblockM)
///     tile_cols = ceil(N / ThreadblockN)
///
///   RA:
///     RA[logical_tile] = packed_tile
///
///   packed tile grid:
///     packed_tile_cols = kReorderedColumn = ReLDN
///     packed_tile_rows = ceil(num_tiles / packed_tile_cols)
///
///   packed D shape:
///     [packed_tile_rows * ThreadblockM,
///      packed_tile_cols * ThreadblockN]
///
/// For fully contiguous per-tile communication, use:
///
///   ReLDN = 1
///
/// Then D is physically:
///
///   [num_tiles * ThreadblockM, ThreadblockN]
///
/// and tile p occupies:
///
///   D[p * ThreadblockM : (p+1) * ThreadblockM, 0 : ThreadblockN]
/////////////////////////////////////////////////////////////////////////////////////////////////

template <class BaseEpilogue, class ThreadblockShape>
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
  };

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

  OOVERLAP_DEVICE_INLINE
  static void prefetch_tma_descriptors(Params const& params) {
    BaseEpilogue::prefetch_tma_descriptors(params.base);
  }

  OOVERLAP_DEVICE_INLINE
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

  // --------------------------------------------------------------------------
  // store():
  //
  // This is the actual physical pack/reorder.
  //
  // Incoming tile_coord is the logical GEMM tile coordinate:
  //
  //   cta_m, cta_n
  //
  // Normal CUTLASS epilogue would store tile (cta_m, cta_n) to:
  //
  //   D[cta_m * TileM : ..., cta_n * TileN : ...]
  //
  // Here we remap:
  //
  //   logical_tile = cta_m * original_tile_cols + cta_n
  //   packed_tile  = RA[logical_tile]
  //
  // Then pass the base epilogue:
  //
  //   packed_cta_m = packed_tile / ReLDN
  //   packed_cta_n = packed_tile % ReLDN
  //
  // and a packed problem shape:
  //
  //   packed_M = ceil(num_tiles / ReLDN) * TileM
  //   packed_N = ReLDN * TileN
  //
  // The base epilogue then stores normally, but into packed coordinates.
  // --------------------------------------------------------------------------
  template <
    class EpiLoadPipe, class EpiLoadState,
    class EpiStorePipe, class EpiStoreState,
    class ProblemShape, class TileShape, class TileCoord,
    class AccumTensor, class TiledMma, class EpiSharedStorage
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
      EpiSharedStorage& shared_storage) {

/*    // Cache ORIGINAL logical M,N for store_tail bookkeeping.*/
    /*M_ = int(cute::get<0>(problem_shape));*/
    /*N_ = int(cute::get<1>(problem_shape));*/

    /*int cta_m = int(cute::get<0>(tile_coord));*/
    /*int cta_n = int(cute::get<1>(tile_coord));*/

    /*int tile_m = params_.signal.ThreadblockM;*/
    /*int tile_n = params_.signal.ThreadblockN;*/

    /*int original_tile_cols = params_.signal.kMonitoredColumn;*/
    /*int packed_tile_cols   = params_.signal.kReorderedColumn;*/

    /*if (packed_tile_cols <= 0) {*/
      /*packed_tile_cols = original_tile_cols;*/
    /*}*/

    /*int logical_tile = cta_m * original_tile_cols + cta_n;*/

    /*int packed_tile = logical_tile;*/
    /*if (params_.signal.ptr_Reorder_Array != nullptr) {*/
      /*packed_tile = params_.signal.ptr_Reorder_Array[logical_tile];*/
    /*}*/

    /*reordered_tile_ = packed_tile;*/

    /*int original_tile_rows = (M_ + tile_m - 1) / tile_m;*/
    /*int original_tile_num  = original_tile_rows * original_tile_cols;*/

    /*int packed_tile_rows = (original_tile_num + packed_tile_cols - 1) / packed_tile_cols;*/

    /*int packed_M = packed_tile_rows * tile_m;*/
    /*int packed_N = packed_tile_cols * tile_n;*/

    /*int packed_cta_m = packed_tile / packed_tile_cols;*/
    /*int packed_cta_n = packed_tile - packed_cta_m * packed_tile_cols;*/

    /*// Preserve the original type/rank of problem_shape and tile_coord.*/
    /*// This is less brittle than constructing a new CuTe coord and guessing rank.*/
    /*auto packed_problem_shape = problem_shape;*/
    /*cute::get<0>(packed_problem_shape) = packed_M;*/
    /*cute::get<1>(packed_problem_shape) = packed_N;*/

    /*auto packed_tile_coord = tile_coord;*/
    /*cute::get<0>(packed_tile_coord) = packed_cta_m;*/
    /*cute::get<1>(packed_tile_coord) = packed_cta_n;*/

    return base_.store(
      //std::forward<EpiLoadPipe>(epi_load_pipe),
      //std::forward<EpiLoadState>(epi_load_state),
      //std::forward<EpiStorePipe>(epi_store_pipe),
      //std::forward<EpiStoreState>(epi_store_state),
      static_cast<EpiLoadPipe&&>(epi_load_pipe),
      static_cast<EpiLoadState&&>(epi_load_state),
      static_cast<EpiStorePipe&&>(epi_store_pipe),
      static_cast<EpiStoreState&&>(epi_store_state),
      //packed_problem_shape,
      problem_shape,
      tile_shape,
      //packed_tile_coord,
      tile_coord,
      accum,
      tiled_mma,
      thread_idx,
      shared_storage
    );
  }

  // --------------------------------------------------------------------------
  // store_tail():
  //
  // MM layout expected here:
  //   MM[0 .. num_segments-1]                     : segment counters
  //   MM[num_segments .. num_segments+num_tiles-1]: per-tile done counters
  //
  // Optional:
  //   ptr_Debug_Arrivals[tile] counts actual elected arrivals for diagnosis.
  //
  // Important:
  //   tile_done[] and segment accounting are indexed by reordered/packed tile id,
  //   not logical tile id. This is what makes communication offsets simple:
  //
  //     offset = packed_tile_begin * TileM * TileN
  //
  // For correctness, the "expected arrivals per tile" must match the number of
  // times cute::elect_one_sync() fires for one CTA tile on this compiled kernel.
  // By default we use the CUTLASS trait above; if that mismatches reality, set
  // signal.kEpilogueArrivalsPerTile from the host after measuring.
  // --------------------------------------------------------------------------
  template <class... Args>
  OOVERLAP_DEVICE_INLINE
  decltype(auto) store_tail(Args&&... args) {
    auto ret = base_.store_tail(static_cast<Args&&>(args)...);

    /*int linear_tid =*/
      /*int(threadIdx.x) +*/
      /*int(blockDim.x) * (int(threadIdx.y) + int(blockDim.y) * int(threadIdx.z));*/

    /*int lane_idx = linear_tid & 31;*/
    /*int warp_idx = linear_tid >> 5;*/

    /*constexpr int kWarpsPerWarpGroup = 4;*/
    /*int warp_idx_in_wg = warp_idx & (kWarpsPerWarpGroup - 1);*/

    /*bool one_thread_per_warpgroup = (lane_idx == 0) && (warp_idx_in_wg == 0);*/

    /*if (one_thread_per_warpgroup) {*/

      /*int tile = reordered_tile_;*/

      /*// Total logical tile count.*/
      /*int tile_rows = (M_ + params_.signal.ThreadblockM - 1) / params_.signal.ThreadblockM;*/
      /*int tile_cols = (N_ + params_.signal.ThreadblockN - 1) / params_.signal.ThreadblockN;*/
      /*int num_tiles = tile_rows * tile_cols;*/

      /*// Number of segments.*/
      /*int num_segments = 0;*/
      /*int sum = 0;*/
      /*while (sum < num_tiles) {*/
        /*sum += params_.signal.kCommu_Seg_Array[num_segments];*/
        /*++num_segments;*/
      /*}*/

      /*// Which segment contains this reordered/packed tile?*/
      /*int idx_bound = params_.signal.kCommu_Seg_Array[0];*/
      /*int seg = 0;*/
      /*while (idx_bound <= tile) {*/
        /*++seg;*/
        /*idx_bound += params_.signal.kCommu_Seg_Array[seg];*/
      /*}*/

      /*// Optional debug: count actual bookkeeping arrivals per tile.*/
      /*if (params_.signal.ptr_Debug_Arrivals) {*/
        /*atomicAdd(&params_.signal.ptr_Debug_Arrivals[tile], 1);*/
      /*}*/

      /*// Count one arrival per warp-group, not per warp.*/
      /*int* tile_done = params_.signal.ptr_Monitored_Matrix + num_segments;*/

      /*int expected_arrivals =*/
          /*(params_.signal.kEpilogueArrivalsPerTile > 0)*/
              /*? params_.signal.kEpilogueArrivalsPerTile*/
              /*: kNumEpilogueWarpGroups;*/

      /*if (expected_arrivals <= 0) {*/
        /*expected_arrivals = 1;*/
      /*}*/

      /*int old = atomicAdd(&tile_done[tile], 1);*/

      /*// Last warp-group to finish this tile signals its segment exactly once.*/
      /*if (old == (expected_arrivals - 1)) {*/
        /*__threadfence();*/

        /*atomicAdd(&params_.signal.ptr_Monitored_Matrix[seg], 1);*/

        /*if (params_.signal.if_monitor) {*/
          /*int global_order =*/
            /*atomicAdd(&params_.signal.ptr_Monitored_Matrix[tile_cols - 1], 1);*/

          /*cutlass::arch::global_store<int, sizeof(int)>(*/
            /*global_order,*/
            /*(void*)(params_.signal.ptr_Monitored_Matrix + tile_cols + tile),*/
            /*true*/
          /*);*/
        /*}*/
      /*}*/
    /*}*/

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
