#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "ooverlap/system/runtime_utils.cuh"

namespace ooverlap {
namespace testing {

__global__ void fill_pattern_kernel(
    half* ptr,
    int64_t n,
    float scale,
    float bias) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = bias + scale * static_cast<float>(idx % 1024);
    ptr[idx] = __float2half_rn(x);
}

inline void fill_pattern(
    half* ptr,
    int64_t n,
    float scale,
    float bias,
    cudaStream_t stream) {
    constexpr int kThreads = 256;
    int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    fill_pattern_kernel<<<blocks, kThreads, 0, stream>>>(ptr, n, scale, bias);
    ooverlap::system::runtime::check_cuda(cudaGetLastError(),
                                          "fill_pattern_kernel launch");
}

inline std::vector<float> host_reference_pattern(
    int64_t n,
    float scale,
    float bias) {
    std::vector<float> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        out[static_cast<size_t>(i)] = bias + scale * static_cast<float>(i % 1024);
    }
    return out;
}

inline std::vector<float> copy_half_device_to_host_float(
    const half* ptr,
    int64_t n,
    int dev) {
    std::vector<half> tmp(static_cast<size_t>(n));
    ooverlap::system::runtime::set_device(dev);
    ooverlap::system::runtime::check_cuda(
        cudaMemcpy(tmp.data(), ptr, static_cast<size_t>(n) * sizeof(half),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy(device->host)");

    std::vector<float> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        out[static_cast<size_t>(i)] = __half2float(tmp[static_cast<size_t>(i)]);
    }
    return out;
}

inline void expect_allclose(
    const std::vector<float>& got,
    const std::vector<float>& ref,
    const char* what,
    float atol = 1e-2f) {
    if (got.size() != ref.size()) {
        std::ostringstream oss;
        oss << what << " size mismatch: got=" << got.size()
            << " ref=" << ref.size();
        throw std::runtime_error(oss.str());
    }

    for (size_t i = 0; i < got.size(); ++i) {
        float diff = std::fabs(got[i] - ref[i]);
        if (diff > atol) {
            std::ostringstream oss;
            oss << what << " mismatch at i=" << i
                << " got=" << got[i]
                << " ref=" << ref[i]
                << " diff=" << diff;
            throw std::runtime_error(oss.str());
        }
    }
}

} // namespace testing
} // namespace ooverlap
