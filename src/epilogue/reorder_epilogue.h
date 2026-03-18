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
  int   ld_D;                   // leading dim (elements) of reshaped output:
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
///   2) emit one segment-ready signal when the last participating epilogue subgroup
///      for that tile reaches store_tail()
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

  // --------------------------------------------------------------------------
  // store():
  //   - logical tile id is flattened with kMonitoredColumn
  //   - reordered tile id is unflattened with kReorderedColumn
  //   - base epilogue receives a PACKED problem shape so its predicates match
  //     the reordered packed D layout
  // --------------------------------------------------------------------------
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

    // Cache ORIGINAL logical M,N for segment bookkeeping in store_tail()
    M_ = int(cute::get<0>(problem_shape));
    N_ = int(cute::get<1>(problem_shape));

    const int cta_m = int(cute::get<0>(tile_coord));
    const int cta_n = int(cute::get<1>(tile_coord));

    // Flatten logical tile in ORIGINAL tile grid
    const int logical_tile_cols = params_.signal.kMonitoredColumn;
    const int logical_tile = cta_m * logical_tile_cols + cta_n;

    // Reordered linear tile id
    const int reordered_tile = params_.signal.ptr_Reorder_Array[logical_tile];
    reordered_tile_ = reordered_tile;

    // Unflatten reordered tile id in PACKED tile grid
    const int packed_tile_cols = params_.signal.kReorderedColumn;
    const int dst_m = reordered_tile / packed_tile_cols;
    const int dst_n = reordered_tile % packed_tile_cols;

    auto mapped_tile_coord = cute::make_tuple(
      dst_m,
      dst_n,
      cute::get<2>(tile_coord),
      cute::get<3>(tile_coord)
    );

    // Base epilogue predicates should see the PACKED output extent
    const int64_t packed_cols_elems =
        int64_t(params_.signal.kReorderedColumn) * int64_t(params_.signal.ThreadblockN);
    const int64_t packed_rows =
        (int64_t(M_) * int64_t(N_)) / packed_cols_elems;

    auto packed_problem_shape = cute::make_shape(
      int(packed_rows),
      int(packed_cols_elems),
      int(cute::get<2>(problem_shape)),
      int(cute::get<3>(problem_shape))
    );

    return base_.store(
      std::forward<EpiLoadPipe>(epi_load_pipe),
      std::forward<EpiLoadState>(epi_load_state),
      std::forward<EpiStorePipe>(epi_store_pipe),
      std::forward<EpiStoreState>(epi_store_state),
      packed_problem_shape,
      tile_shape,
      mapped_tile_coord,
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
  // For correctness, the "expected arrivals per tile" must match the number of
  // times cute::elect_one_sync() fires for one CTA tile on this compiled kernel.
  // By default we use the CUTLASS trait above; if that mismatches reality, set
  // signal.kEpilogueArrivalsPerTile from the host after measuring.
  // --------------------------------------------------------------------------
  template <class... Args>
  CUTLASS_DEVICE
  decltype(auto) store_tail(Args&&... args) {
    auto ret = base_.store_tail(std::forward<Args>(args)...);

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

      // Total logical tile count
      int tile_rows = (M_ + params_.signal.ThreadblockM - 1) / params_.signal.ThreadblockM;
      int tile_cols = (N_ + params_.signal.ThreadblockN - 1) / params_.signal.ThreadblockN;
      int num_tiles = tile_rows * tile_cols;

      // Number of segments
      int num_segments = 0;
      int sum = 0;
      while (sum < num_tiles) {
        sum += params_.signal.kCommu_Seg_Array[num_segments];
        ++num_segments;
      }

      // Which segment contains this reordered tile?
      int idx_bound = params_.signal.kCommu_Seg_Array[0];
      int seg = 0;
      while (idx_bound <= tile) {
        ++seg;
        idx_bound += params_.signal.kCommu_Seg_Array[seg];
      }

      // Optional debug: count actual bookkeeping arrivals per tile
      if (params_.signal.ptr_Debug_Arrivals) {
        atomicAdd(&params_.signal.ptr_Debug_Arrivals[tile], 1);
      }

      // Count one arrival per warp-group, not per warp
      int* tile_done = params_.signal.ptr_Monitored_Matrix + num_segments;

      int expected_arrivals =
          (params_.signal.kEpilogueArrivalsPerTile > 0)
              ? params_.signal.kEpilogueArrivalsPerTile
              : kNumEpilogueWarpGroups;

      if (expected_arrivals <= 0) {
        expected_arrivals = 1;
      }

      int old = atomicAdd(&tile_done[tile], 1);

      // Last warp-group to finish this tile signals its segment exactly once
      if (old == (expected_arrivals - 1)) {
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

} // namespace cutlass
