#include "baseline_impl.h"
#include "nccl_utils.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <pybind11/stl.h>

#include "overlap/gemm_signal_sm90_dispatch.h"
#include "overlap/gemm_scatter_sm90_dispatch.h"
#include "overlap_impl.h"

#include "test/persistent_allreduce_2gpu_sm90.h"
#include "test/ipc_allreduce_2gpu_sm90.h"
#include "test/ipc_allreduce_benchmark_2gpu_sm90.h"
#include "test/tma_bandwidth_experiment_sm90.h"
#include "test/tma_allreduce_sweep_2gpu_sm90.h"

namespace py = pybind11;

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

  constexpr int64_t TileM = 128;
  constexpr int64_t TileN = 128;

  TORCH_CHECK(algo == 0, "Only algo=0 supported in bring-up");
  TORCH_CHECK(M % TileM == 0, "M must be multiple of 128 for algo=0");
  TORCH_CHECK(N % TileN == 0, "N must be multiple of 128 for algo=0");

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

  cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

  bool ok = ooverlap::gemm_signal_sm90_dispatch(
      static_cast<int>(algo),
      static_cast<int>(M),
      static_cast<int>(N),
      static_cast<int>(K),
      static_cast<int>(ReLDN),
      reinterpret_cast<int32_t*>(CommThr.data_ptr<int32_t>()),
      static_cast<void*>(A.data_ptr<at::Half>()),
      static_cast<void*>(B_packed.data_ptr<at::Half>()),
      static_cast<void*>(D.data_ptr<at::Half>()),
      reinterpret_cast<int32_t*>(MM.data_ptr<int32_t>()),
      reinterpret_cast<int32_t*>(RA.data_ptr<int32_t>()),
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

  constexpr int64_t TileM = 128;
  constexpr int64_t TileN = 128;

  TORCH_CHECK(algo == 0, "Only algo=0 supported in bring-up");
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

  cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

  bool ok = ooverlap::gemm_scatter_sm90_dispatch(
      static_cast<int>(algo),
      static_cast<int>(M),
      static_cast<int>(N),
      static_cast<int>(K),
      static_cast<int>(ReLDN),
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
  py::class_<OverlapImpl>(m, "OverlapImpl")
      .def(py::init<>())
      .def("cutlass_init", &OverlapImpl::CutlassInit)
      .def("nccl_init", &OverlapImpl::NcclInit)
      .def("overlap_init", &OverlapImpl::OverlapInit)
      .def("gemm_allreduce_overlap", &OverlapImpl::GemmAllReduceOverlap)
      .def("gemm_reducescatter_overlap", &OverlapImpl::GemmReduceScatterOverlap)
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
