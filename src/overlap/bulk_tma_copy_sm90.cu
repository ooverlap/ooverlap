#include "overlap/bulk_tma_copy_sm90.cuh"
#include "ooverlap/sync/sync.cuh"
#include "ooverlap/tma/tma.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <algorithm>

namespace ooverlap {
namespace {

constexpr int kThreads = 128;
constexpr uint32_t kChunkBytes = 16 * 1024;

__global__ void bulk_tma_copy_kernel_sm90(
    const half* __restrict__ src,
    half* __restrict__ dst,
    size_t total_bytes) {

    extern __shared__ __align__(16) unsigned char smem[];

    const size_t chunk_base =
        static_cast<size_t>(blockIdx.x) * static_cast<size_t>(kChunkBytes);

    if (chunk_base >= total_bytes) {
        return;
    }

    const uint32_t this_bytes = static_cast<uint32_t>(
        std::min(static_cast<size_t>(kChunkBytes), total_bytes - chunk_base));

    const char* gsrc = reinterpret_cast<const char*>(src) + chunk_base;
    char* gdst = reinterpret_cast<char*>(dst) + chunk_base;

    // Stage from global into shared with a normal cooperative copy.
    for (uint32_t i = threadIdx.x; i < this_bytes; i += blockDim.x) {
        smem[i] = gsrc[i];
    }

    __syncthreads();

    // One thread issues the bulk TMA store from shared -> global.
    if (threadIdx.x == 0) {
        tma::store_async(
            reinterpret_cast<void*>(gdst),
            reinterpret_cast<void*>(smem),
            this_bytes);

        // Wait until the TMA store has finished reading shared memory.
        tma::store_async_read_wait<0>();
    }
}

} // namespace

cudaError_t enqueue_bulk_tma_copy_sm90(
    const half* src,
    half* dst,
    size_t num_elements,
    cudaStream_t stream) {

    if (src == nullptr || dst == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }
    if (num_elements == 0) {
        return cudaSuccess;
    }

    const size_t total_bytes = num_elements * sizeof(half);
    const int blocks = static_cast<int>(
        (total_bytes + static_cast<size_t>(kChunkBytes) - 1) / static_cast<size_t>(kChunkBytes));

    bulk_tma_copy_kernel_sm90<<<blocks, kThreads, kChunkBytes, stream>>>(
        src, dst, total_bytes);

    return cudaGetLastError();
}

} // namespace ooverlap
