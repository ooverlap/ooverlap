#include "rmsnorm.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "rmsnorm.cuh"

namespace ooverlap {

namespace {
inline int rms_threads_for_dim(int dim) {
  int threads = (dim + 15) / 16;   // 16 elems / thread
  if (threads < 32) threads = 32;  // reduction helper assumes at least one warp
  return threads;
}
} // namespace

void rmsnorm(at::Tensor X, at::Tensor O, at::Tensor RW) {
  TORCH_CHECK(X.is_cuda() && O.is_cuda() && RW.is_cuda(), "X/O/RW must be CUDA");
  TORCH_CHECK(X.scalar_type() == torch::kFloat16, "X must be float16");
  TORCH_CHECK(O.scalar_type() == torch::kFloat16, "O must be float16");
  TORCH_CHECK(RW.scalar_type() == torch::kFloat16, "RW must be float16");
  TORCH_CHECK(X.dim() == 2 && O.dim() == 2, "X and O must be 2D");
  TORCH_CHECK(X.size(0) == O.size(0) && X.size(1) == O.size(1), "O must match X shape");

  const int bs = static_cast<int>(X.size(0));
  const int dim = static_cast<int>(X.size(1));

  TORCH_CHECK(dim % 16 == 0, "rmsnorm kernel expects dim % 16 == 0");
  TORCH_CHECK(RW.numel() == dim, "RW must have shape [dim]");

  const int threads = rms_threads_for_dim(dim);
  TORCH_CHECK(threads <= 1024, "rmsnorm kernel needs <= 1024 threads, got ", threads);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  rmsnorm_kernel<<<dim3(bs), dim3(threads), 0, stream>>>(
      reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(RW.data_ptr<at::Half>()),
      reinterpret_cast<half*>(O.data_ptr<at::Half>()),
      bs,
      dim);
}

void reorder_rmsnorm(
    at::Tensor X,
    at::Tensor O,
    at::Tensor RW,
    int64_t BM,
    int64_t BN,
    int64_t rldn,
    at::Tensor RA) {

  TORCH_CHECK(X.is_cuda() && O.is_cuda() && RW.is_cuda() && RA.is_cuda(),
              "X/O/RW/RA must be CUDA");
  TORCH_CHECK(X.scalar_type() == torch::kFloat16, "X must be float16");
  TORCH_CHECK(O.scalar_type() == torch::kFloat16, "O must be float16");
  TORCH_CHECK(RW.scalar_type() == torch::kFloat16, "RW must be float16");
  TORCH_CHECK(RA.scalar_type() == torch::kInt32, "RA must be int32");
  TORCH_CHECK(X.dim() == 2 && O.dim() == 2, "X and O must be 2D");
  TORCH_CHECK(X.size(0) == O.size(0) && X.size(1) == O.size(1), "O must match X shape");

  const int bs = static_cast<int>(X.size(0));
  const int dim = static_cast<int>(X.size(1));

  TORCH_CHECK(BM > 0 && BN > 0 && rldn > 0, "BM/BN/rldn must be > 0");
  TORCH_CHECK(dim % 16 == 0, "reorder_rmsnorm kernel expects dim % 16 == 0");
  TORCH_CHECK(bs % BM == 0, "reorder_rmsnorm expects bs % BM == 0");
  TORCH_CHECK(dim % BN == 0, "reorder_rmsnorm expects dim % BN == 0");
  TORCH_CHECK(RW.numel() == dim, "RW must have shape [dim]");

  const int expected_ra = static_cast<int>((bs / BM) * (dim / BN));
  TORCH_CHECK(RA.numel() == expected_ra,
              "RA must have numel == (bs/BM)*(dim/BN) = ", expected_ra);

  const int threads = rms_threads_for_dim(dim);
  TORCH_CHECK(threads <= 1024, "reorder_rmsnorm kernel needs <= 1024 threads, got ", threads);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  reorder_rmsnorm_kernel<<<dim3(bs), dim3(threads), 0, stream>>>(
      reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(RW.data_ptr<at::Half>()),
      reinterpret_cast<half*>(O.data_ptr<at::Half>()),
      bs,
      dim,
      BM,
      BN,
      dim / BN,
      rldn,
      RA.data_ptr<int>());
}

} // namespace ooverlap
