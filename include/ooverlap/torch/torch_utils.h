#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

namespace ooverlap {
namespace torch_utils {

inline void refresh_gemm_stream(cudaStream_t& gemm_stream) {
    gemm_stream = at::cuda::getCurrentCUDAStream().stream();
}

inline void ensure_streams_ready(
    cudaStream_t& gemm_stream,
    cudaStream_t& comm_stream,
    bool& overlap_init_done) {
    refresh_gemm_stream(gemm_stream);

    if (comm_stream == nullptr) {
        cudaError_t err = cudaStreamCreateWithPriority(
            &comm_stream, cudaStreamNonBlocking, -5);
        TORCH_CHECK(err == cudaSuccess,
                    "cudaStreamCreateWithPriority failed: ",
                    cudaGetErrorString(err));
        overlap_init_done = true;
    }
}

inline size_t tensor_nbytes(const at::Tensor& t) {
    return static_cast<size_t>(t.numel()) *
           static_cast<size_t>(t.element_size());
}

inline void check_common_gemm_inputs(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A/B must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(B.scalar_type() == torch::kFloat16, "B must be float16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A/B must be 2D");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "A/B must be contiguous");
}

} // namespace torch_utils
} // namespace ooverlap
