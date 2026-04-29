/***************************************************************************************************
 * SM90 CUTLASS 3.x GEMM dispatch for packed/reordered output experiments.
 *
 * This file caches the CUTLASS GEMM object per template instantiation.
 *
 * First call for a unique key:
 *   pays CUTLASS setup.
 *
 * Later calls with same key:
 *   only run the initialized GEMM.
 **************************************************************************************************/

#include <ATen/core/interned_strings.h>
#include <cuda_fp16.h>

#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/fast_math.h"

#include "gemm_with_signal_sm90.h"
#include "gemm_signal_sm90_dispatch.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

struct GemmSignalCacheKey {
  bool valid;
  int device;
  int M;
  int N;
  int K;
  int ReLDN;
  int* CommThr;
  void* A;
  void* B;
  void* D;
  int* MM;
  int* RA;
  bool Monitor;
  int num_segments;

  GemmSignalCacheKey()
      : valid(false),
        device(-1),
        M(0),
        N(0),
        K(0),
        ReLDN(0),
        CommThr(nullptr),
        A(nullptr),
        B(nullptr),
        D(nullptr),
        MM(nullptr),
        RA(nullptr),
        Monitor(false),
        num_segments(0) {}

  bool same_as(GemmSignalCacheKey const& other) const {
    return valid &&
           other.valid &&
           device  == other.device &&
           M       == other.M &&
           N       == other.N &&
           K       == other.K &&
           ReLDN   == other.ReLDN &&
           CommThr == other.CommThr &&
           A       == other.A &&
           B       == other.B &&
           D       == other.D &&
           MM      == other.MM &&
           RA      == other.RA &&
           Monitor == other.Monitor &&
           num_segments == other.num_segments;
  }
};

} // namespace

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int TileM,
  int TileN,
  int TileK,
  typename ClusterShape,
  typename MainloopSchedule,
  typename EpilogueSchedule
>
void cutlass_gemm_signal_sm90(
  int M, int N, int K,
  int ReLDN,
  int num_segments,
  int* CommThr,
  half* A, half* B, half* D,
  int* MM, int* RA,
  bool Monitor,
  cudaStream_t stream = nullptr
) {
  using ElementA           = cutlass::half_t;
  using LayoutA            = cutlass::layout::RowMajor;
  using ElementB           = cutlass::half_t;
  using LayoutB            = cutlass::layout::ColumnMajor;
  using ElementC           = cutlass::half_t;
  using LayoutC            = cutlass::layout::RowMajor;
  using ElementAccumulator = float;

  using GemmSignal = cutlass::GemmSignalSm90<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    TileM,
    TileN,
    TileK,
    ClusterShape,
    MainloopSchedule,
    EpilogueSchedule
  >;

  static GemmSignal gemm_op;
  static GemmSignalCacheKey cached_key;

  int device_id = 0;
  cudaGetDevice(&device_id);

  GemmSignalCacheKey new_key;
  new_key.valid   = true;
  new_key.device  = device_id;
  new_key.M       = M;
  new_key.N       = N;
  new_key.K       = K;
  new_key.ReLDN   = ReLDN;
  new_key.CommThr = CommThr;
  new_key.A       = reinterpret_cast<void*>(A);
  new_key.B       = reinterpret_cast<void*>(B);
  new_key.D       = reinterpret_cast<void*>(D);
  new_key.MM      = MM;
  new_key.RA      = RA;
  new_key.Monitor = Monitor;
  new_key.num_segments = num_segments;

  cutlass::gemm::GemmCoord problem_size(M, N, K);

  int64_t ld_D_reshaped = int64_t(ReLDN) * int64_t(TileN);

  typename GemmSignal::Arguments arguments(
    problem_size,
    reinterpret_cast<cutlass::half_t*>(A),
    reinterpret_cast<cutlass::half_t*>(B),

    // C is unused because beta=0. Keep C and D same.
    reinterpret_cast<cutlass::half_t*>(D),
    reinterpret_cast<cutlass::half_t*>(D),

    int64_t(K),
    int64_t(K),
    ld_D_reshaped,
    ld_D_reshaped,

    ElementAccumulator(1.0f),
    ElementAccumulator(0.0f),

    MM,
    RA,
    N / TileN,
    ReLDN,
    CommThr,
    num_segments,
    Monitor
  );

  if (!cached_key.same_as(new_key)) {
    CUTLASS_CHECK_SM90(gemm_op.initialize(arguments, stream));
    cached_key = new_key;
  }

  CUTLASS_CHECK_SM90(gemm_op(stream));
}

// explicit instantiations
#include "inc/signal_instances_sm90.inc"

// function pointer table
#include "tiling/signal_tiling_sm90.cuh"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace ooverlap {

int gemm_signal_sm90_algo_count() {
  return signal_sm90_func_count;
}

bool gemm_signal_sm90_get_algo_meta(
    int algo,
    GemmSignalSm90AlgoMeta* out) {
  if (out == nullptr) {
    return false;
  }

  if (algo < 0 || algo >= signal_sm90_func_count) {
    return false;
  }

  auto const& src = signal_sm90_algo_meta[algo];

  out->tile_m = src.tile_m;
  out->tile_n = src.tile_n;
  out->tile_k = src.tile_k;
  out->cluster_m = src.cluster_m;
  out->cluster_n = src.cluster_n;
  out->cluster_k = src.cluster_k;
  out->mainloop = src.mainloop;
  out->epilogue = src.epilogue;

  return true;
}

bool gemm_signal_sm90_dispatch(
    int algo,
    int M, int N, int K,
    int ReLDN,
    int num_segments,
    int32_t* CommThr,
    void* A, void* B, void* D,
    int32_t* MM, int32_t* RA,
    bool Monitor,
    cudaStream_t stream) {

  if (algo < 0 || algo >= signal_sm90_func_count) {
    return false;
  }

  signal_sm90_func_table[algo](
      M, N, K,
      ReLDN,
      num_segments,
      reinterpret_cast<int*>(CommThr),
      reinterpret_cast<half*>(A),
      reinterpret_cast<half*>(B),
      reinterpret_cast<half*>(D),
      reinterpret_cast<int*>(MM),
      reinterpret_cast<int*>(RA),
      Monitor,
      stream);

  return true;
}

} // namespace ooverlap
