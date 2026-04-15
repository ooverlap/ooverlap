#include "test/tma_vmm_smoke_test.h"
#include "overlap/bulk_tma_copy_sm90.cuh"
#include "ooverlap/system/vmm.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace ooverlap {
namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << what << " failed: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

__global__ void fill_pattern_half_kernel(half* ptr, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Write deterministic raw 16-bit values.
    uint16_t v = static_cast<uint16_t>((idx * 17 + 3) & 0xffff);
    reinterpret_cast<uint16_t*>(ptr)[idx] = v;
}

inline void ensure_context_on_device(int dev) {
    check_cuda(cudaSetDevice(dev), "cudaSetDevice");
    // Forces runtime context creation
    check_cuda(cudaFree(nullptr), "cudaFree(nullptr)");
}

} // namespace

bool tma_vmm_smoke_test(
    int64_t num_elements,
    int src_device,
    int dst_device) {

    if (num_elements <= 0) {
        throw std::invalid_argument("tma_vmm_smoke_test: num_elements must be > 0");
    }
    if (src_device == dst_device) {
        throw std::invalid_argument("tma_vmm_smoke_test: src_device and dst_device must differ");
    }

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 2) {
        throw std::runtime_error("tma_vmm_smoke_test: need at least 2 CUDA devices");
    }
    if (src_device < 0 || src_device >= device_count ||
        dst_device < 0 || dst_device >= device_count) {
        throw std::invalid_argument("tma_vmm_smoke_test: invalid device id");
    }

    ensure_context_on_device(src_device);
    ensure_context_on_device(dst_device);

    const size_t bytes = static_cast<size_t>(num_elements) * sizeof(half);

    // Source buffer on GPU src_device
    check_cuda(cudaSetDevice(src_device), "cudaSetDevice(src)");
    half* src_ptr = nullptr;
    check_cuda(cudaMalloc(&src_ptr, bytes), "cudaMalloc(src)");

    cudaStream_t src_stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&src_stream, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags(src_stream)");

    const int threads = 256;
    const int blocks = static_cast<int>((num_elements + threads - 1) / threads);
    fill_pattern_half_kernel<<<blocks, threads, 0, src_stream>>>(src_ptr, num_elements);
    check_cuda(cudaGetLastError(), "fill_pattern_half_kernel launch");
    check_cuda(cudaStreamSynchronize(src_stream), "cudaStreamSynchronize(fill)");

    // Destination buffer owned by GPU dst_device, visible to src_device and dst_device
    void* remote_dst_ptr = nullptr;
    size_t mapped_size = 0;
    {
        std::vector<int> access_devices = {src_device, dst_device};
        ooverlap::system::vmm::vm_alloc_map_set_access(
            &remote_dst_ptr,
            &mapped_size,
            bytes,
            dst_device,
            access_devices);
    }

    // Optional zero init on destination owner GPU
    check_cuda(cudaSetDevice(dst_device), "cudaSetDevice(dst)");
    check_cuda(cudaMemset(remote_dst_ptr, 0, mapped_size), "cudaMemset(remote_dst_ptr)");

    // Launch the copy on GPU src_device. The destination pointer is remote GPU1 memory,
    // but VMM access has made it valid for GPU0 to write.
    check_cuda(cudaSetDevice(src_device), "cudaSetDevice(src)");
    check_cuda(
        enqueue_bulk_tma_copy_sm90(
            src_ptr,
            reinterpret_cast<half*>(remote_dst_ptr),
            static_cast<size_t>(num_elements),
            src_stream),
        "enqueue_bulk_tma_copy_sm90");
    check_cuda(cudaStreamSynchronize(src_stream), "cudaStreamSynchronize(copy)");

    // Read back source and destination to host and verify byte-for-byte.
    std::vector<uint16_t> src_host(num_elements);
    std::vector<uint16_t> dst_host(num_elements);

    check_cuda(cudaSetDevice(src_device), "cudaSetDevice(src)");
    check_cuda(
        cudaMemcpy(src_host.data(), src_ptr, bytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy(src->host)");

    check_cuda(cudaSetDevice(dst_device), "cudaSetDevice(dst)");
    check_cuda(
        cudaMemcpy(dst_host.data(), remote_dst_ptr, bytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy(remote_dst->host)");

    int64_t first_bad = -1;
    for (int64_t i = 0; i < num_elements; ++i) {
        if (src_host[i] != dst_host[i]) {
            first_bad = i;
            break;
        }
    }

    // Cleanup
    check_cuda(cudaSetDevice(src_device), "cudaSetDevice(src)");
    if (src_stream) cudaStreamDestroy(src_stream);
    if (src_ptr) cudaFree(src_ptr);

    ooverlap::system::vmm::vm_unmap(remote_dst_ptr, mapped_size);

    if (first_bad >= 0) {
        std::ostringstream oss;
        oss << "tma_vmm_smoke_test mismatch at idx=" << first_bad
            << " src=0x" << std::hex << src_host[first_bad]
            << " dst=0x" << std::hex << dst_host[first_bad];
        throw std::runtime_error(oss.str());
    }

    return true;
}

} // namespace ooverlap
