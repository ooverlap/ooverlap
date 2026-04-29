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
        Monitor(false) {}

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
           Monitor == other.Monitor;
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

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace ooverlap {

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
    
  using Cluster1x1x1 = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using Cluster1x2x1 = cute::Shape<cute::_1, cute::_2, cute::_1>;
  using Cluster2x1x1 = cute::Shape<cute::_2, cute::_1, cute::_1>;

  using WS = cutlass::gemm::KernelTmaWarpSpecialized;
  using EpiAuto = cutlass::epilogue::collective::EpilogueScheduleAuto;

  switch (algo) {
    case 0:
      cutlass_gemm_signal_sm90<
          128, 128, 32,
          Cluster1x1x1,
          WS,
          EpiAuto>(
          M, N, K, ReLDN, num_segments, reinterpret_cast<int*>(CommThr),
          reinterpret_cast<half*>(A), reinterpret_cast<half*>(B),
          reinterpret_cast<half*>(D), reinterpret_cast<int*>(MM),
          reinterpret_cast<int*>(RA), Monitor, stream);
      return true;

    case 1:
      cutlass_gemm_signal_sm90<
          128, 128, 64,
          Cluster1x1x1,
          WS,
          EpiAuto>(
          M, N, K, ReLDN, num_segments, reinterpret_cast<int*>(CommThr),
          reinterpret_cast<half*>(A), reinterpret_cast<half*>(B),
          reinterpret_cast<half*>(D), reinterpret_cast<int*>(MM),
          reinterpret_cast<int*>(RA), Monitor, stream);
      return true;

    case 2:
      cutlass_gemm_signal_sm90<
          128, 128, 128,
          Cluster1x1x1,
          WS,
          EpiAuto>(
          M, N, K, ReLDN, num_segments, reinterpret_cast<int*>(CommThr),
          reinterpret_cast<half*>(A), reinterpret_cast<half*>(B),
          reinterpret_cast<half*>(D), reinterpret_cast<int*>(MM),
          reinterpret_cast<int*>(RA), Monitor, stream);
      return true;

    case 3:
      cutlass_gemm_signal_sm90<
          128, 128, 64,
          Cluster1x2x1,
          WS,
          EpiAuto>(
          M, N, K, ReLDN, num_segments, reinterpret_cast<int*>(CommThr),
          reinterpret_cast<half*>(A), reinterpret_cast<half*>(B),
          reinterpret_cast<half*>(D), reinterpret_cast<int*>(MM),
          reinterpret_cast<int*>(RA), Monitor, stream);
      return true;

    case 4:
      cutlass_gemm_signal_sm90<
          128, 128, 64,
          Cluster2x1x1,
          WS,
          EpiAuto>(
          M, N, K, ReLDN, num_segments, reinterpret_cast<int*>(CommThr),
          reinterpret_cast<half*>(A), reinterpret_cast<half*>(B),
          reinterpret_cast<half*>(D), reinterpret_cast<int*>(MM),
          reinterpret_cast<int*>(RA), Monitor, stream);
      return true;

    default:
      return false;
  }
}

} // namespace ooverlap
