#include "baseline_impl.h"
#include "nccl_utils.h"
#include "overlap_impl.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <pybind11/stl.h>

#include <climits>
#include <cstring>

#include "gemm/gemm_signal_sm90_dispatch.h"
#include "gemm/gemm_plain_sm90_dispatch.h"
#include "gemm/gemm_scatter_sm90_dispatch.h"

#include "test/persistent_allreduce_2gpu_sm90.h"
#include "test/tma_bandwidth_experiment_sm90.h"
#include "test/ipc_collective_sm90.h"
#include "test/tma_collective_sweep_2gpu.h"

namespace py = pybind11;

namespace {

void check_half_2d(torch::Tensor T, const char* name) {
  TORCH_CHECK(T.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(T.scalar_type() == torch::kFloat16, name, " must be float16");
  TORCH_CHECK(T.dim() == 2, name, " must be 2D");
  TORCH_CHECK(T.is_contiguous(), name, " must be contiguous");
}

void check_i32(torch::Tensor T, const char* name, bool cuda) {
  TORCH_CHECK(cuda ? T.is_cuda() : T.device().is_cpu(),
              name, cuda ? " must be CUDA" : " must be CPU");
  TORCH_CHECK(T.scalar_type() == torch::kInt32, name, " must be int32");
  TORCH_CHECK(T.is_contiguous(), name, " must be contiguous");
}

cudaStream_t current_stream_for(torch::Tensor T) {
  const int dev = T.get_device();
  cudaError_t err = cudaSetDevice(dev);
  TORCH_CHECK(err == cudaSuccess, "cudaSetDevice failed: ", cudaGetErrorString(err));
  return at::cuda::getCurrentCUDAStream(dev).stream();
}

struct TileInfo {
  int tile_m;
  int tile_n;
};

TileInfo signal_tile_info(int64_t algo) {
  ooverlap::GemmSignalSm90AlgoMeta meta{};
  TORCH_CHECK(
      ooverlap::gemm_signal_sm90_get_algo_meta(static_cast<int>(algo), &meta),
      "Unsupported signal GEMM algo=", algo,
      ". Count=", ooverlap::gemm_signal_sm90_algo_count());
  return {meta.tile_m, meta.tile_n};
}

TileInfo plain_tile_info(int64_t algo) {
  ooverlap::GemmPlainSm90AlgoMeta meta{};
  TORCH_CHECK(
      ooverlap::gemm_plain_sm90_get_algo_meta(static_cast<int>(algo), &meta),
      "Unsupported plain GEMM algo=", algo,
      ". Count=", ooverlap::gemm_plain_sm90_algo_count());
  return {meta.tile_m, meta.tile_n};
}

const char* cublas_status_to_string(cublasStatus_t status) {
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

void gemm_signal_sm90(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor D,
    torch::Tensor MM,
    torch::Tensor RA,
    torch::Tensor CommThr,
    int64_t ReLDN,
    int64_t algo,
    bool monitor) {

  check_half_2d(A, "A");
  check_half_2d(B, "B");
  check_half_2d(D, "D");
  check_i32(MM, "MM", true);
  check_i32(RA, "RA", true);
  check_i32(CommThr, "CommThr", true);

  const int M = static_cast<int>(A.size(0));
  const int K = static_cast<int>(A.size(1));
  const int N = static_cast<int>(B.size(0));
  const int segs = static_cast<int>(CommThr.numel());

  TORCH_CHECK(B.size(1) == K, "B must have shape [N, K]");
  TORCH_CHECK(ReLDN > 0 && segs > 0, "ReLDN and num_segments must be > 0");

  auto tile = signal_tile_info(algo);
  TORCH_CHECK(M % tile.tile_m == 0 && N % tile.tile_n == 0,
              "M/N must be compatible with selected algo");

  const int tile_num = (M / tile.tile_m) * (N / tile.tile_n);
  TORCH_CHECK(RA.numel() == tile_num, "RA.numel() must equal tile count");
  TORCH_CHECK(MM.numel() >= static_cast<int64_t>(segs) + tile_num, "MM is too small");

  bool ok = ooverlap::gemm_signal_sm90_dispatch(
      static_cast<int>(algo),
      M, N, K,
      static_cast<int>(ReLDN),
      segs,
      reinterpret_cast<int32_t*>(CommThr.data_ptr<int32_t>()),
      static_cast<void*>(A.data_ptr<at::Half>()),
      static_cast<void*>(B.data_ptr<at::Half>()),
      static_cast<void*>(D.data_ptr<at::Half>()),
      reinterpret_cast<int32_t*>(MM.data_ptr<int32_t>()),
      reinterpret_cast<int32_t*>(RA.data_ptr<int32_t>()),
      0,
      monitor,
      current_stream_for(A));

  TORCH_CHECK(ok, "gemm_signal_sm90_dispatch failed for algo=", algo);
}

void gemm_scatter_sm90(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor D,
    torch::Tensor MM,
    torch::Tensor RA,
    torch::Tensor RE,
    torch::Tensor CommThr,
    int64_t ReLDN,
    int64_t algo,
    bool monitor) {

  check_half_2d(A, "A");
  check_half_2d(B, "B");
  check_half_2d(D, "D");
  check_i32(MM, "MM", true);
  check_i32(RA, "RA", true);
  check_i32(RE, "RE", true);
  check_i32(CommThr, "CommThr", true);

  const int M = static_cast<int>(A.size(0));
  const int K = static_cast<int>(A.size(1));
  const int N = static_cast<int>(B.size(0));
  const int segs = static_cast<int>(CommThr.numel());

  TORCH_CHECK(B.size(1) == K, "B must have shape [N, K]");
  TORCH_CHECK(ReLDN > 0 && segs > 0, "ReLDN and num_segments must be > 0");

  auto tile = signal_tile_info(algo);
  TORCH_CHECK(M % tile.tile_m == 0 && N % tile.tile_n == 0,
              "M/N must be compatible with selected algo");

  const int tile_num = (M / tile.tile_m) * (N / tile.tile_n);
  TORCH_CHECK(RA.numel() == tile_num, "RA.numel() must equal tile count");
  TORCH_CHECK(MM.numel() >= static_cast<int64_t>(segs) + tile_num, "MM is too small");

  bool ok = ooverlap::gemm_scatter_sm90_dispatch(
      static_cast<int>(algo),
      M, N, K,
      static_cast<int>(ReLDN),
      segs,
      reinterpret_cast<int32_t*>(CommThr.data_ptr<int32_t>()),
      static_cast<void*>(A.data_ptr<at::Half>()),
      static_cast<void*>(B.data_ptr<at::Half>()),
      static_cast<void*>(D.data_ptr<at::Half>()),
      reinterpret_cast<int32_t*>(MM.data_ptr<int32_t>()),
      reinterpret_cast<int32_t*>(RA.data_ptr<int32_t>()),
      reinterpret_cast<int32_t*>(RE.data_ptr<int32_t>()),
      monitor,
      current_stream_for(A));

  TORCH_CHECK(ok, "gemm_scatter_sm90_dispatch failed for algo=", algo);
}

void gemm_plain_sm90(
    torch::Tensor A,
    torch::Tensor B_col,
    torch::Tensor D_col,
    int64_t algo) {

  check_half_2d(A, "A");
  check_half_2d(B_col, "B_col");
  check_half_2d(D_col, "D_col");

  const int M = static_cast<int>(A.size(0));
  const int K = static_cast<int>(A.size(1));
  const int N = static_cast<int>(B_col.size(0));

  TORCH_CHECK(B_col.size(1) == K, "B_col must have shape [N, K]");
  TORCH_CHECK(D_col.size(0) == N && D_col.size(1) == M,
              "D_col must have shape [N, M]");

  auto tile = plain_tile_info(algo);
  TORCH_CHECK(M % tile.tile_m == 0 && N % tile.tile_n == 0,
              "M/N must be compatible with selected algo");

  bool ok = ooverlap::gemm_plain_sm90_dispatch(
      static_cast<int>(algo),
      M, N, K,
      static_cast<void*>(A.data_ptr<at::Half>()),
      static_cast<void*>(B_col.data_ptr<at::Half>()),
      static_cast<void*>(D_col.data_ptr<at::Half>()),
      current_stream_for(A));

  TORCH_CHECK(ok, "gemm_plain_sm90_dispatch failed for algo=", algo);
}

void baseline_gemm_col(
    torch::Tensor A,
    torch::Tensor B_col,
    torch::Tensor D_col) {

  check_half_2d(A, "A");
  check_half_2d(B_col, "B_col");
  check_half_2d(D_col, "D_col");

  const int64_t M64 = A.size(0);
  const int64_t K64 = A.size(1);
  const int64_t N64 = B_col.size(0);

  TORCH_CHECK(B_col.size(1) == K64, "B_col must have shape [N, K]");
  TORCH_CHECK(D_col.size(0) == N64 && D_col.size(1) == M64,
              "D_col must have shape [N, M]");
  TORCH_CHECK(M64 <= INT_MAX && N64 <= INT_MAX && K64 <= INT_MAX,
              "M/N/K exceed int range");

  const int M = static_cast<int>(M64);
  const int N = static_cast<int>(N64);
  const int K = static_cast<int>(K64);

  cudaStream_t stream = current_stream_for(A);
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

  cublasStatus_t st = cublasSetStream(handle, stream);
  TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasSetStream failed: ",
              cublas_status_to_string(st));

  st = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasSetMathMode failed: ",
              cublas_status_to_string(st));

  const half alpha = __float2half(1.0f);
  const half beta = __float2half(0.0f);

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

  TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasGemmEx failed: ",
              cublas_status_to_string(st));
}

} // namespace

PYBIND11_MODULE(ooverlap_ext, m) {
  m.def("generate_nccl_id", &generate_nccl_id);

  m.def("gemm_signal_sm90", &gemm_signal_sm90);
  m.def("gemm_scatter_sm90", &gemm_scatter_sm90);
  m.def("gemm_plain_sm90", &gemm_plain_sm90);
  m.def("baseline_gemm_col", &baseline_gemm_col);

  m.def("tma_persistent_two_gpu_allreduce_smoke_test",
        &ooverlap::tma_persistent_two_gpu_allreduce_smoke_test,
        py::arg("numel"),
        py::arg("dev0") = 0,
        py::arg("dev1") = 1);

  m.def("benchmark_persistent_two_gpu_allreduce_sm90",
        &ooverlap::benchmark_persistent_two_gpu_allreduce_sm90,
        py::arg("numel"),
        py::arg("iters"),
        py::arg("warmup"),
        py::arg("dev0") = 0,
        py::arg("dev1") = 1);

  m.def("benchmark_persistent_two_gpu_collective_sm90",
        &ooverlap::benchmark_persistent_two_gpu_collective_sm90,
        py::arg("collective"),
        py::arg("numel"),
        py::arg("iters"),
        py::arg("warmup"),
        py::arg("dev0") = 0,
        py::arg("dev1") = 1);

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
        py::arg("include_nccl") = false);

  m.def("benchmark_tma_two_gpu_collective_sweep_json",
        &ooverlap::benchmark_tma_two_gpu_collective_sweep_json,
        py::arg("request_json"));

  m.def("smoke_ipc_collective_rank_sm90",
        &ooverlap::smoke_ipc_collective_rank_sm90,
        py::arg("collective"),
        py::arg("numel"),
        py::arg("local_rank"),
        py::arg("dev0"),
        py::arg("dev1"),
        py::arg("broker_key"),
        py::arg("nccl_unique_id_bytes"),
        py::arg("verify") = true);

  m.def("benchmark_ipc_collective_rank_sm90",
        &ooverlap::benchmark_ipc_collective_rank_sm90,
        py::arg("collective"),
        py::arg("sizes"),
        py::arg("local_rank"),
        py::arg("dev0"),
        py::arg("dev1"),
        py::arg("broker_key"),
        py::arg("nccl_unique_id_bytes"),
        py::arg("iters"),
        py::arg("warmup"),
        py::arg("verify"));

  py::class_<BaselineImpl>(m, "BaselineImpl")
      .def(py::init<>())
      .def("nccl_init", &BaselineImpl::NcclInit)
      .def("cublas_init", &BaselineImpl::CublasInit)
      .def("gemm", &BaselineImpl::Gemm)
      .def("gemm_allreduce", &BaselineImpl::GemmAllReduce)
      .def("gemm_reducescatter", &BaselineImpl::GemmReduceScatter)
      .def("gemm_all2all", &BaselineImpl::GemmAll2All)
      .def("gemm_plain", &BaselineImpl::GemmPlain)
      .def("gemm_plain_allreduce", &BaselineImpl::GemmPlainAllReduce)
      .def("gemm_plain_reducescatter", &BaselineImpl::GemmPlainReduceScatter)
      .def("gemm_plain_all2all", &BaselineImpl::GemmPlainAll2All)
      .def("nccl_allreduce", &BaselineImpl::NcclAllReduce)
      .def("nccl_reducescatter", &BaselineImpl::NcclReduceScatter)
      .def("nccl_all2all", &BaselineImpl::NcclAll2All);

  py::class_<OverlapImpl>(m, "OverlapImpl")
      .def(py::init<>())
      .def("cutlass_init", &OverlapImpl::CutlassInit)
      .def("nccl_init", &OverlapImpl::NcclInit)
      .def("ooverlap_ipc_init", &OverlapImpl::OoverlapIpcInit)
      .def("ooverlap_release", &OverlapImpl::OoverlapRelease)
      .def("overlap_init", &OverlapImpl::OverlapInit)
      .def("gemm_allreduce_overlap", &OverlapImpl::GemmAllReduceOverlap)
      .def("gemm_reducescatter_overlap", &OverlapImpl::GemmReduceScatterOverlap)
      .def("ooverlap_allreduce", &OverlapImpl::OoverlapAllReduce);
}
