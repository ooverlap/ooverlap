#include "baseline_impl.h"
#include "nccl_utils.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <pybind11/stl.h>
#include <cuda_bf16.h>

#include "overlap/gemm_signal_sm90_dispatch.h"
#include "overlap/gemm_plain_sm90_dispatch.h"
#include "overlap/gemm_scatter_sm90_dispatch.h"
#include "overlap_impl.h"

#include "test/persistent_allreduce_2gpu_sm90.h"
#include "test/ipc_allreduce_2gpu_sm90.h"
#include "test/ipc_allreduce_benchmark_2gpu_sm90.h"
#include "test/tma_bandwidth_experiment_sm90.h"
#include "test/tma_allreduce_sweep_2gpu_sm90.h"
#include "test/public_allreduce_benchmark_2gpu_sm90.h"

namespace py = pybind11;

struct Sm90GemmAlgoInfo {
  int tile_m;
  int tile_n;
};

static Sm90GemmAlgoInfo get_sm90_gemm_algo_info(int64_t algo) {
  ooverlap::GemmSignalSm90AlgoMeta meta;

  bool ok = ooverlap::gemm_signal_sm90_get_algo_meta(
      static_cast<int>(algo),
      &meta);

  TORCH_CHECK(ok, "Unsupported algo=", algo,
              ". Generated SM90 algo count=",
              ooverlap::gemm_signal_sm90_algo_count());

  return {meta.tile_m, meta.tile_n};
}

// --------------------------------------------
// SM90 1-GPU signal GEMM wrapper (for testing)
// --------------------------------------------
static void gemm_signal_sm90(
    torch::Tensor A,
    torch::Tensor B_packed,
    torch::Tensor D,
    torch::Tensor MM,
    torch::Tensor RA,
    torch::Tensor CommThr,
    int64_t ReLDN,
    int64_t algo,
    bool monitor) {

  TORCH_CHECK(A.is_cuda() && B_packed.is_cuda() && D.is_cuda(), "A/B/D must be CUDA");
  TORCH_CHECK(MM.is_cuda() && RA.is_cuda() && CommThr.is_cuda(), "MM/RA/CommThr must be CUDA");

  TORCH_CHECK(A.scalar_type() == torch::kFloat16, "A must be float16");
  TORCH_CHECK(B_packed.scalar_type() == torch::kFloat16, "B_packed must be float16");
  TORCH_CHECK(D.scalar_type() == torch::kFloat16, "D must be float16");

  TORCH_CHECK(MM.scalar_type() == torch::kInt32, "MM must be int32");
  TORCH_CHECK(RA.scalar_type() == torch::kInt32, "RA must be int32");
  TORCH_CHECK(CommThr.scalar_type() == torch::kInt32, "CommThr must be int32");

  TORCH_CHECK(A.dim() == 2 && B_packed.dim() == 2 && D.dim() == 2, "A/B/D must be 2D");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B_packed.is_contiguous(), "B_packed must be contiguous");
  TORCH_CHECK(D.is_contiguous(), "D must be contiguous");

  const int64_t M = A.size(0);
  const int64_t K = A.size(1);
  const int64_t N = B_packed.size(0);
  TORCH_CHECK(B_packed.size(1) == K, "B_packed must be (N,K) where K matches A");
  TORCH_CHECK(ReLDN > 0, "ReLDN must be > 0");

  auto info = get_sm90_gemm_algo_info(algo);
  const int64_t TileM = info.tile_m;
  const int64_t TileN = info.tile_n;

  TORCH_CHECK(M % TileM == 0,
              "M must be multiple of TileM=", TileM,
              " for algo=", algo);
  
  TORCH_CHECK(N % TileN == 0,
              "N must be multiple of TileN=", TileN,
              " for algo=", algo);

  const int64_t tile_rows = M / TileM;
  const int64_t tile_cols = N / TileN;
  const int64_t num_tiles = tile_rows * tile_cols;

  TORCH_CHECK(RA.numel() == num_tiles, "RA must have length num_tiles=", num_tiles);
  TORCH_CHECK(CommThr.numel() >= 1, "CommThr must have >= 1 element");
  TORCH_CHECK(MM.numel() >= 1, "MM must have >= 1 element");

  const int64_t ldD = ReLDN * TileN;
  TORCH_CHECK(D.size(0) >= M, "D.size(0) must be >= M");
  TORCH_CHECK(D.size(1) >= ldD, "D.size(1) must be >= ReLDN*TileN (", ldD, ")");

  const int dev = A.get_device();
  cudaError_t err = cudaSetDevice(dev);
  TORCH_CHECK(err == cudaSuccess, "cudaSetDevice failed: ", cudaGetErrorString(err));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const int64_t num_segments = CommThr.numel();

  TORCH_CHECK(num_segments > 0, "CommThr must have at least one segment");
  TORCH_CHECK(MM.numel() >= num_segments + num_tiles,
            "MM must have at least num_segments + num_tiles elements");

  bool ok = ooverlap::gemm_signal_sm90_dispatch(
      static_cast<int>(algo),
      static_cast<int>(M),
      static_cast<int>(N),
      static_cast<int>(K),
      static_cast<int>(ReLDN),
      static_cast<int>(num_segments),
      reinterpret_cast<int32_t*>(CommThr.data_ptr<int32_t>()),
      static_cast<void*>(A.data_ptr<at::Half>()),
      static_cast<void*>(B_packed.data_ptr<at::Half>()),
      static_cast<void*>(D.data_ptr<at::Half>()),
      reinterpret_cast<int32_t*>(MM.data_ptr<int32_t>()),
      reinterpret_cast<int32_t*>(RA.data_ptr<int32_t>()),
      0,
      monitor,
      stream);

  TORCH_CHECK(ok, "Unsupported algo=", algo);
}

static void gemm_scatter_sm90(
    torch::Tensor A,
    torch::Tensor B_packed,
    torch::Tensor D,
    torch::Tensor MM,
    torch::Tensor RA,
    torch::Tensor RE,
    torch::Tensor CommThr,
    int64_t ReLDN,
    int64_t algo,
    bool monitor) {

  TORCH_CHECK(A.is_cuda() && B_packed.is_cuda() && D.is_cuda(), "A/B/D must be CUDA");
  TORCH_CHECK(MM.is_cuda() && RA.is_cuda() && RE.is_cuda() && CommThr.is_cuda(),
              "MM/RA/RE/CommThr must be CUDA");

  TORCH_CHECK(A.scalar_type() == torch::kFloat16, "A must be float16");
  TORCH_CHECK(B_packed.scalar_type() == torch::kFloat16, "B_packed must be float16");
  TORCH_CHECK(D.scalar_type() == torch::kFloat16, "D must be float16");

  TORCH_CHECK(MM.scalar_type() == torch::kInt32, "MM must be int32");
  TORCH_CHECK(RA.scalar_type() == torch::kInt32, "RA must be int32");
  TORCH_CHECK(RE.scalar_type() == torch::kInt32, "RE must be int32");
  TORCH_CHECK(CommThr.scalar_type() == torch::kInt32, "CommThr must be int32");

  TORCH_CHECK(A.dim() == 2 && B_packed.dim() == 2 && D.dim() == 2, "A/B/D must be 2D");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B_packed.is_contiguous(), "B_packed must be contiguous");
  TORCH_CHECK(D.is_contiguous(), "D must be contiguous");

  const int64_t M = A.size(0);
  const int64_t K = A.size(1);
  const int64_t N = B_packed.size(0);
  TORCH_CHECK(B_packed.size(1) == K, "B_packed must be (N,K) where K matches A");
  TORCH_CHECK(ReLDN > 0, "ReLDN must be > 0");

  auto info = get_sm90_gemm_algo_info(algo);
  const int64_t TileM = info.tile_m;
  const int64_t TileN = info.tile_n;

  //TORCH_CHECK(algo == 0, "Only algo=0 supported in bring-up");
  TORCH_CHECK(M % TileM == 0, "M must be multiple of 128 for algo=0");
  TORCH_CHECK(N % TileN == 0, "N must be multiple of 128 for algo=0");

  const int64_t tile_rows = M / TileM;
  const int64_t tile_cols = N / TileN;
  const int64_t num_tiles = tile_rows * tile_cols;

  TORCH_CHECK(RA.numel() == num_tiles, "RA must have length num_tiles=", num_tiles);
  TORCH_CHECK(RE.numel() >= 1, "RE must have >= 1 element");
  TORCH_CHECK(CommThr.numel() >= 1, "CommThr must have >= 1 element");
  TORCH_CHECK(MM.numel() >= 1, "MM must have >= 1 element");

  const int64_t ldD = ReLDN * TileN;
  TORCH_CHECK(D.size(0) >= M, "D.size(0) must be >= M");
  TORCH_CHECK(D.size(1) >= ldD, "D.size(1) must be >= ReLDN*TileN");

  const int dev = A.get_device();
  cudaError_t err = cudaSetDevice(dev);
  TORCH_CHECK(err == cudaSuccess, "cudaSetDevice failed: ", cudaGetErrorString(err));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  const int64_t num_segments = CommThr.numel();

  TORCH_CHECK(num_segments > 0, "CommThr must have at least one segment");
  TORCH_CHECK(MM.numel() >= num_segments + num_tiles,
            "MM must have at least num_segments + num_tiles elements");

  bool ok = ooverlap::gemm_scatter_sm90_dispatch(
    static_cast<int>(algo),
    static_cast<int>(M),
    static_cast<int>(N),
    static_cast<int>(K),
    static_cast<int>(ReLDN),
    static_cast<int>(num_segments),
    reinterpret_cast<int32_t*>(CommThr.data_ptr<int32_t>()),
    static_cast<void*>(A.data_ptr<at::Half>()),
    static_cast<void*>(B_packed.data_ptr<at::Half>()),
    static_cast<void*>(D.data_ptr<at::Half>()),
    reinterpret_cast<int32_t*>(MM.data_ptr<int32_t>()),
    reinterpret_cast<int32_t*>(RA.data_ptr<int32_t>()),
    reinterpret_cast<int32_t*>(RE.data_ptr<int32_t>()),
    monitor,
    stream); 

  TORCH_CHECK(ok, "Unsupported algo=", algo);
}

static void gemm_plain_sm90(
    torch::Tensor A,
    torch::Tensor B_col,
    torch::Tensor D_col,
    int64_t algo) {
  TORCH_CHECK(A.is_cuda() && B_col.is_cuda() && D_col.is_cuda(),
              "A/B_col/D_col must be CUDA");

  TORCH_CHECK(A.scalar_type() == torch::kFloat16, "A must be float16");
  TORCH_CHECK(B_col.scalar_type() == torch::kFloat16, "B_col must be float16");
  TORCH_CHECK(D_col.scalar_type() == torch::kFloat16, "D_col must be float16");

  TORCH_CHECK(A.dim() == 2 && B_col.dim() == 2 && D_col.dim() == 2,
              "A/B_col/D_col must be 2D");

  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B_col.is_contiguous(), "B_col must be contiguous");
  TORCH_CHECK(D_col.is_contiguous(), "D_col must be contiguous");

  const int64_t M = A.size(0);
  const int64_t K = A.size(1);
  const int64_t N = B_col.size(0);

  TORCH_CHECK(B_col.size(1) == K,
              "B_col must have physical shape (N, K)");

  TORCH_CHECK(D_col.size(0) == N && D_col.size(1) == M,
              "D_col must have physical shape (N, M), because logical D is column-major [M,N]");

  const int dev = A.get_device();
  cudaError_t err = cudaSetDevice(dev);
  TORCH_CHECK(err == cudaSuccess, "cudaSetDevice failed: ", cudaGetErrorString(err));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev).stream();

  bool ok = ooverlap::gemm_plain_sm90_dispatch(
      static_cast<int>(algo),
      static_cast<int>(M),
      static_cast<int>(N),
      static_cast<int>(K),
      static_cast<void*>(A.data_ptr<at::Half>()),
      static_cast<void*>(B_col.data_ptr<at::Half>()),
      static_cast<void*>(D_col.data_ptr<at::Half>()),
      stream);

  TORCH_CHECK(ok, "gemm_plain_sm90 failed for algo=", algo);
}

static const char* cublas_status_to_string(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    default: return "CUBLAS_STATUS_UNKNOWN";
  }
}

static void baseline_gemm_col(
    torch::Tensor A,
    torch::Tensor B_col,
    torch::Tensor D_col) {
  TORCH_CHECK(A.is_cuda() && B_col.is_cuda() && D_col.is_cuda(),
              "A/B_col/D_col must be CUDA");

  TORCH_CHECK(A.scalar_type() == torch::kFloat16, "A must be float16");
  TORCH_CHECK(B_col.scalar_type() == torch::kFloat16, "B_col must be float16");
  TORCH_CHECK(D_col.scalar_type() == torch::kFloat16, "D_col must be float16");

  TORCH_CHECK(A.dim() == 2 && B_col.dim() == 2 && D_col.dim() == 2,
              "A/B_col/D_col must be 2D");

  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B_col.is_contiguous(), "B_col must be contiguous");
  TORCH_CHECK(D_col.is_contiguous(), "D_col must be contiguous");

  const int64_t M64 = A.size(0);
  const int64_t K64 = A.size(1);
  const int64_t N64 = B_col.size(0);

  TORCH_CHECK(B_col.size(1) == K64,
              "B_col must have physical shape (N, K)");

  TORCH_CHECK(D_col.size(0) == N64 && D_col.size(1) == M64,
              "D_col must have physical shape (N, M)");

  TORCH_CHECK(M64 <= INT_MAX && N64 <= INT_MAX && K64 <= INT_MAX,
              "M/N/K exceed int range for cuBLAS");

  const int M = static_cast<int>(M64);
  const int N = static_cast<int>(N64);
  const int K = static_cast<int>(K64);

  const int dev = A.get_device();
  cudaError_t cuda_err = cudaSetDevice(dev);
  TORCH_CHECK(cuda_err == cudaSuccess,
              "cudaSetDevice failed: ", cudaGetErrorString(cuda_err));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev).stream();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

  cublasStatus_t st = cublasSetStream(handle, stream);
  TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS,
              "cublasSetStream failed: ", cublas_status_to_string(st));

  st = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS,
              "cublasSetMathMode failed: ", cublas_status_to_string(st));

  const half alpha = __float2half(1.0f);
  const half beta  = __float2half(0.0f);

  // We want logical:
  //
  //   D[M, N] = A[M, K] @ B_col.t()[K, N]
  //
  // but D_col is physical [N, M], i.e. logical column-major D[M, N].
  //
  // cuBLAS is column-major:
  //   A physical [M,K] row-major is column-major [K,M], use OP_T => [M,K]
  //   B_col physical [N,K] row-major is column-major [K,N], use OP_N => [K,N]
  //   D_col physical [N,M] row-major is column-major [M,N], ldc=M
  st = cublasGemmEx(
      handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      M, N, K,
      static_cast<const void*>(&alpha),
      static_cast<const void*>(A.data_ptr<at::Half>()),
      CUDA_R_16F,
      K,
      static_cast<const void*>(B_col.data_ptr<at::Half>()),
      CUDA_R_16F,
      K,
      static_cast<const void*>(&beta),
      static_cast<void*>(D_col.data_ptr<at::Half>()),
      CUDA_R_16F,
      M,
      CUBLAS_COMPUTE_16F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS,
              "cublasGemmEx baseline_gemm_col failed: ",
              cublas_status_to_string(st));
}

PYBIND11_MODULE(ooverlap_ext, m) {
  m.def("gemm_signal_sm90", &gemm_signal_sm90,
        "SM90 fused reorder+signal GEMM (bring-up: algo=0 only)");

  m.def("gemm_scatter_sm90", &gemm_scatter_sm90,
        "SM90 fused reorder+scatter GEMM (bring-up: algo=0 only)");

  m.def("generate_nccl_id", &generate_nccl_id,
        "Generate an NCCL unique ID as a Python list[int]");

  m.def("tma_persistent_two_gpu_allreduce_smoke_test",
        &ooverlap::tma_persistent_two_gpu_allreduce_smoke_test,
        py::arg("numel"),
        py::arg("dev0") = 0,
        py::arg("dev1") = 1,
        "Persistent 2-GPU all-reduce smoke test");

  m.def("benchmark_persistent_two_gpu_allreduce_sm90",
        &ooverlap::benchmark_persistent_two_gpu_allreduce_sm90,
        py::arg("numel"),
        py::arg("iters"),
        py::arg("warmup"),
        py::arg("dev0") = 0,
        py::arg("dev1") = 1,
        "Benchmark persistent 2-GPU all-reduce vs NCCL");

  m.def("tma_ipc_two_gpu_allreduce_rank_smoke_test",
        &ooverlap::tma_ipc_two_gpu_allreduce_rank_smoke_test,
        py::arg("numel"),
        py::arg("local_rank"),
        py::arg("dev0") = 0,
        py::arg("dev1") = 1,
        py::arg("broker_key"),
        py::arg("iters") = 1,
        "2-process CUDA IPC 2-GPU all-reduce smoke test; call once per local rank");

  m.def("benchmark_ipc_two_gpu_allreduce_rank_sm90",
        &ooverlap::benchmark_ipc_two_gpu_allreduce_rank_sm90,
        py::arg("numel"),
        py::arg("local_rank"),
        py::arg("dev0"),
        py::arg("dev1"),
        py::arg("broker_key"),
        py::arg("nccl_unique_id_bytes"),
        py::arg("iters"),
        py::arg("warmup"),
        py::arg("verify") = false,
        "Per-rank 2-process IPC benchmark: ooverlap IPC allreduce vs NCCL");

  m.def("benchmark_tma_bandwidth_experiment_sm90",
        &ooverlap::benchmark_tma_bandwidth_experiment_sm90,
        py::arg("min_bytes") = 512 * 1024,
        py::arg("max_bytes") = 1024LL * 1024LL * 1024LL,
        py::arg("iters") = 100,
        py::arg("warmup") = 20,
        py::arg("num_blocks") = 8,
        py::arg("dev0") = 0,
        py::arg("dev1") = 1,
        py::arg("include_mem_async") = false,
        py::arg("include_nccl") = false,
        "Experimental TMA/global-memory/NCCL bandwidth sweep.");

  m.def("benchmark_tma_two_gpu_allreduce_sweep_sm90",
        &ooverlap::benchmark_tma_two_gpu_allreduce_sweep_sm90,
        py::arg("numels"),
        py::arg("kernels"),
        py::arg("threads"),
        py::arg("max_ctas"),
        py::arg("window_chunks"),
        py::arg("chunk_bytes"),
        py::arg("stage_depths"),
        py::arg("iters"),
        py::arg("warmup"),
        py::arg("dev0") = 0,
        py::arg("dev1") = 1,
        "Sweep SM90 2-GPU TMA allreduce runtime launch configs and selected chunk/stage variants.");
  
  m.def("benchmark_public_allreduce_2gpu_sm90",
        &ooverlap::benchmark_public_allreduce_2gpu_sm90,
        py::arg("min_bytes"),
        py::arg("max_bytes"),
        py::arg("points"),
        py::arg("iters"),
        py::arg("warmup"),
        py::arg("tuning_mode"),
        py::arg("dev0") = 0,
        py::arg("dev1") = 1,
        "Benchmark public oo_allreduce_tuned API vs NCCL in one process.");
  
  m.def("gemm_plain_sm90", &gemm_plain_sm90,
      "Plain SM90 CUTLASS GEMM: A row-major, B column-major, D column-major");

  m.def("baseline_gemm_col", &baseline_gemm_col,
      "cuBLAS GEMM baseline: A row-major [M,K], B_col [N,K], D_col [N,M]");

  py::class_<BaselineImpl>(m, "BaselineImpl")
      .def(py::init<>())
      .def("nccl_init", &BaselineImpl::NcclInit)
      .def("cublas_init", &BaselineImpl::CublasInit)
      .def("gemm", &BaselineImpl::Gemm)
      .def("gemm_allreduce", &BaselineImpl::GemmAllReduce)
      .def("gemm_reducescatter", &BaselineImpl::GemmReduceScatter)
      .def("nccl_allreduce", &BaselineImpl::NcclAllReduce)
      .def("nccl_reducescatter", &BaselineImpl::NcclReduceScatter);

  py::class_<OverlapImpl>(m, "OverlapImpl")
      .def(py::init<>())
      .def("cutlass_init", &OverlapImpl::CutlassInit)
      .def("nccl_init", &OverlapImpl::NcclInit)
      .def("ooverlap_ipc_init", &OverlapImpl::OoverlapIpcInit)
      .def("ooverlap_release", &OverlapImpl::OoverlapRelease)
      .def("overlap_init", &OverlapImpl::OverlapInit)
      .def("gemm", &OverlapImpl::Gemm)
      .def("gemm_allreduce", &OverlapImpl::GemmAllReduce)
      .def("gemm_allreduce_overlap", &OverlapImpl::GemmAllReduceOverlap)
      .def("gemm_reducescatter_overlap", &OverlapImpl::GemmReduceScatterOverlap)
      .def("seg_allreduce", &OverlapImpl::SegAllReduce)
      .def("ooverlap_allreduce", &OverlapImpl::OoverlapAllReduce)
      .def("nccl_allreduce", &OverlapImpl::NcclAllReduce)
      .def("nccl_reducescatter", &OverlapImpl::NcclReduceScatter);
}

#ifndef OOVERLAP_ENABLE_TORCH_LIBRARY
#define OOVERLAP_ENABLE_TORCH_LIBRARY 0
#endif

#if OOVERLAP_ENABLE_TORCH_LIBRARY

#include <torch/script.h>

template<typename T>
void NcclInitWrapper(const c10::intrusive_ptr<T>& self,
                     const int64_t tp_rank,
                     const int64_t tp_size,
                     const std::vector<int64_t>& tp_id) {
  self->NcclInit(tp_rank, tp_size, tp_id);
}

template<typename T>
void CublasInitWrapper(const c10::intrusive_ptr<T>& self) {
  self->CublasInit();
}

template<typename T>
void GemmAllReduceWrapper(const c10::intrusive_ptr<T>& self,
                          at::Tensor A, at::Tensor B, at::Tensor C) {
  self->GemmAllReduce(A, B, C);
}

template<typename T>
void GemmReduceScatterWrapper(const c10::intrusive_ptr<T>& self,
                              at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D) {
  self->GemmReduceScatter(A, B, C, D);
}

template<typename T>
void GemmWrapper(const c10::intrusive_ptr<T>& self,
                 at::Tensor A, at::Tensor B, at::Tensor C) {
  self->Gemm(A, B, C);
}

template<typename T>
void NcclAllReduceWrapper(const c10::intrusive_ptr<T>& self, at::Tensor C) {
  self->NcclAllReduce(C);
}

template<typename T>
void NcclReduceScatterWrapper(const c10::intrusive_ptr<T>& self, at::Tensor C) {
  self->NcclReduceScatter(C);
}

TORCH_LIBRARY(ooverlap_class, m) {
  m.class_<BaselineImpl>("BaselineImpl")
    .def(torch::init())
    .def("nccl_init", &NcclInitWrapper<BaselineImpl>)
    .def("cublas_init", &CublasInitWrapper<BaselineImpl>)
    .def("gemm", &GemmWrapper<BaselineImpl>)
    .def("gemm_allreduce", &GemmAllReduceWrapper<BaselineImpl>)
    .def("gemm_reducescatter", &GemmReduceScatterWrapper<BaselineImpl>)
    .def("nccl_allreduce", &NcclAllReduceWrapper<BaselineImpl>)
    .def("nccl_reducescatter", &NcclReduceScatterWrapper<BaselineImpl>);
}

TORCH_LIBRARY(ooverlap_op, m) {
  m.def("generate_nccl_id", &generate_nccl_id);
}

#endif  // OOVERLAP_ENABLE_TORCH_LIBRARY
