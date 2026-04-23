#include "comm/tma_two_gpu_peer_allreduce_sm90.h"

#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/tma/tma.cuh"
#include "ooverlap/tma/tma_reduce.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace ooverlap {
namespace {

constexpr int kTwoGpuPeerThreads = 16;
constexpr size_t kTwoGpuPeerChunkBytes = 16 * 1024;
constexpr int kTwoGpuPeerMaxBlocks = 16;
constexpr int kTwoGpuPeerStageDepth = 1;
constexpr size_t kTwoGpuPeerStaticSharedBytes =
    static_cast<size_t>(kTwoGpuPeerStageDepth) * sizeof(sync::semaphore);
constexpr size_t kTwoGpuPeerProgressBytes = 2 * sizeof(int);

__host__ __device__ __forceinline__ size_t min_sz(size_t a, size_t b) {
    return (a < b) ? a : b;
}

__global__ void tma_two_gpu_copy_then_reduce_kernel_sm90(
    const half* local_in,
    half* local_out,
    const half* peer_out,
    int* local_progress,
    const int* peer_progress,
    size_t numel,
    int num_chunks,
    int num_blocks) {

    const int start_chunk = static_cast<int>(blockIdx.x);
    const int chunk_stride = static_cast<int>(gridDim.x);

    if (start_chunk >= num_chunks) {
        return;
    }

    extern __shared__ uint4 shared_storage_u4[];
    unsigned char* smem = reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore load_barrier;

    const unsigned char* local_in_bytes =
        reinterpret_cast<const unsigned char*>(local_in);
    unsigned char* local_out_bytes =
        reinterpret_cast<unsigned char*>(local_out);
    const unsigned char* peer_out_bytes =
        reinterpret_cast<const unsigned char*>(peer_out);

    const size_t total_bytes = numel * sizeof(half);

    // Phase 1: copy local input into our own peer-visible output buffer.
    for (int chunk = start_chunk; chunk < num_chunks; chunk += chunk_stride) {
        const size_t offset = static_cast<size_t>(chunk) * kTwoGpuPeerChunkBytes;
        const size_t bytes = min_sz(kTwoGpuPeerChunkBytes, total_bytes - offset);
        const size_t bulk_bytes = bytes & ~static_cast<size_t>(0xF);
        const size_t tail_bytes = bytes - bulk_bytes;

        if (threadIdx.x == 0) {
            sync::init_semaphore(load_barrier, 1);
            tma::expect_bytes(load_barrier, static_cast<uint32_t>(bytes));
            tma::load_async(
                smem,
                local_in_bytes + offset,
                static_cast<uint32_t>(bytes),
                load_barrier);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            sync::wait(load_barrier, 0);
        }
        __syncthreads();

        if (threadIdx.x == 0 && bulk_bytes > 0) {
            tma::store_async(
                local_out_bytes + offset,
                smem,
                static_cast<uint32_t>(bulk_bytes));
            tma::store_async_wait<0>();
        }
        __syncthreads();

        if (tail_bytes > 0) {
            for (size_t i = threadIdx.x; i < tail_bytes; i += blockDim.x) {
                local_out_bytes[offset + bulk_bytes + i] = smem[bulk_bytes + i];
            }
        }
        __syncthreads();
    }

    // One whole-buffer signal after all blocks finish the copy phase.
    if (threadIdx.x == 0) {
        const int finished = atomicAdd(local_progress + 1, 1) + 1;
        if (finished == num_blocks) {
            __threadfence_system();
            atomicExch(local_progress + 0, 1);
        }
    }
    __syncthreads();

    // Wait until the peer has finished copying its whole buffer.
    if (threadIdx.x == 0) {
        const volatile int* peer_ready =
            reinterpret_cast<const volatile int*>(peer_progress);
        while (peer_ready[0] == 0) {
#if defined(__CUDA_ARCH__)
            __nanosleep(64);
#endif
        }
    }
    __syncthreads();

    // Phase 2: reduce peer buffer into our local output buffer.
    for (int chunk = start_chunk; chunk < num_chunks; chunk += chunk_stride) {
        const size_t offset = static_cast<size_t>(chunk) * kTwoGpuPeerChunkBytes;
        const size_t bytes = min_sz(kTwoGpuPeerChunkBytes, total_bytes - offset);
        const size_t bulk_bytes = bytes & ~static_cast<size_t>(0xF);
        const size_t tail_bytes = bytes - bulk_bytes;

        if (threadIdx.x == 0) {
            sync::init_semaphore(load_barrier, 1);
            tma::expect_bytes(load_barrier, static_cast<uint32_t>(bytes));
            tma::load_async(
                smem,
                peer_out_bytes + offset,
                static_cast<uint32_t>(bytes),
                load_barrier);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            sync::wait(load_barrier, 0);
        }
        __syncthreads();

        if (threadIdx.x == 0 && bulk_bytes > 0) {
            tma::reduce_add_noftz_f16_async(
                local_out_bytes + offset,
                smem,
                static_cast<uint32_t>(bulk_bytes));
            tma::reduce_async_wait<0>();
        }
        __syncthreads();

        if (tail_bytes > 0) {
            const size_t bulk_elems = bulk_bytes / sizeof(half);
            const size_t tail_elems = tail_bytes / sizeof(half);
            const half* smem_half = reinterpret_cast<const half*>(smem);
            half* out_half = reinterpret_cast<half*>(local_out_bytes + offset);

            for (size_t i = threadIdx.x; i < tail_elems; i += blockDim.x) {
                const float oldv = __half2float(out_half[bulk_elems + i]);
                const float addv = __half2float(smem_half[bulk_elems + i]);
                out_half[bulk_elems + i] = __float2half_rn(oldv + addv);
            }
        }
        __syncthreads();
    }
}

} // namespace

int tma_two_gpu_peer_allreduce_compute_num_chunks(size_t numel) {
    const size_t total_bytes = numel * sizeof(half);
    return static_cast<int>(
        (total_bytes + kTwoGpuPeerChunkBytes - 1) / kTwoGpuPeerChunkBytes);
}

void tma_two_gpu_peer_allreduce_configure_kernel_once(int device) {
    struct CacheEntry {
        bool configured = false;
        size_t dynamic_smem_bytes = 0;
    };

    static std::mutex mutex;
    static std::unordered_map<int, CacheEntry> cache;

    const size_t dynamic_smem_bytes =
        static_cast<size_t>(kTwoGpuPeerStageDepth) * kTwoGpuPeerChunkBytes;
    const size_t total_smem_bytes =
        dynamic_smem_bytes + kTwoGpuPeerStaticSharedBytes;

    std::lock_guard<std::mutex> lock(mutex);

    auto it = cache.find(device);
    if (it != cache.end() &&
        it->second.configured &&
        it->second.dynamic_smem_bytes == dynamic_smem_bytes) {
        return;
    }

    system::runtime::set_device(device);

    cudaDeviceProp prop{};
    system::runtime::check_cuda(
        cudaGetDeviceProperties(&prop, device),
        "cudaGetDeviceProperties");

    if (total_smem_bytes > static_cast<size_t>(prop.sharedMemPerBlockOptin)) {
        throw std::runtime_error(
            "tma_two_gpu_peer_allreduce_configure_kernel_once: requested shared memory exceeds opt-in limit");
    }

    if (total_smem_bytes > static_cast<size_t>(prop.sharedMemPerBlock)) {
        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                tma_two_gpu_copy_then_reduce_kernel_sm90,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(dynamic_smem_bytes)),
            "cudaFuncSetAttribute(MaxDynamicSharedMemorySize)");

        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                tma_two_gpu_copy_then_reduce_kernel_sm90,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                100),
            "cudaFuncSetAttribute(PreferredSharedMemoryCarveout)");
    }

    cache[device] = {true, dynamic_smem_bytes};
}

void tma_two_gpu_peer_allreduce_state_init(
    TmaTwoGpuPeerAllreduceState* st,
    int dev0,
    int dev1,
    size_t numel) {
    if (st == nullptr) {
        throw std::invalid_argument("tma_two_gpu_peer_allreduce_state_init: state is null");
    }
    if (numel == 0) {
        throw std::invalid_argument("tma_two_gpu_peer_allreduce_state_init: numel must be > 0");
    }
    if (dev0 == dev1) {
        throw std::invalid_argument("tma_two_gpu_peer_allreduce_state_init: devices must differ");
    }

    tma_two_gpu_peer_allreduce_state_destroy(st);

    st->dev0 = dev0;
    st->dev1 = dev1;
    st->num_chunks = tma_two_gpu_peer_allreduce_compute_num_chunks(numel);

    tma_two_gpu_peer_allreduce_configure_kernel_once(dev0);
    tma_two_gpu_peer_allreduce_configure_kernel_once(dev1);

    const std::vector<int> access_devices = {dev0, dev1};
    st->progress0 = system::alloc_peer_visible_buffer(
        kTwoGpuPeerProgressBytes, dev0, access_devices);
    st->progress1 = system::alloc_peer_visible_buffer(
        kTwoGpuPeerProgressBytes, dev1, access_devices);

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(
        cudaMemset(st->progress0.ptr, 0, kTwoGpuPeerProgressBytes),
        "cudaMemset(progress0)");

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(
        cudaMemset(st->progress1.ptr, 0, kTwoGpuPeerProgressBytes),
        "cudaMemset(progress1)");
}

void tma_two_gpu_peer_allreduce_state_destroy(
    TmaTwoGpuPeerAllreduceState* st) {
    if (st == nullptr) {
        return;
    }

    system::free_peer_visible_buffer(st->progress0);
    system::free_peer_visible_buffer(st->progress1);

    st->dev0 = -1;
    st->dev1 = -1;
    st->num_chunks = 0;
}

void tma_two_gpu_peer_allreduce_outputs_init(
    TmaTwoGpuPeerAllreduceOutputs* outs,
    int dev0,
    int dev1,
    size_t numel) {
    if (outs == nullptr) {
        throw std::invalid_argument("tma_two_gpu_peer_allreduce_outputs_init: outputs is null");
    }
    if (numel == 0) {
        throw std::invalid_argument("tma_two_gpu_peer_allreduce_outputs_init: numel must be > 0");
    }

    tma_two_gpu_peer_allreduce_outputs_destroy(outs);

    const size_t bytes = numel * sizeof(half);
    const std::vector<int> access_devices = {dev0, dev1};

    outs->dev0 = dev0;
    outs->dev1 = dev1;
    outs->out0 = system::alloc_peer_visible_buffer(bytes, dev0, access_devices);
    outs->out1 = system::alloc_peer_visible_buffer(bytes, dev1, access_devices);
}

void tma_two_gpu_peer_allreduce_outputs_destroy(
    TmaTwoGpuPeerAllreduceOutputs* outs) {
    if (outs == nullptr) {
        return;
    }

    system::free_peer_visible_buffer(outs->out0);
    system::free_peer_visible_buffer(outs->out1);

    outs->dev0 = -1;
    outs->dev1 = -1;
}

cudaError_t prime_tma_two_gpu_peer_allreduce_outputs_sm90(
    TmaTwoGpuPeerAllreduceState* st,
    const half* rank0_in,
    const half* rank1_in,
    half* rank0_out_peer,
    half* rank1_out_peer,
    size_t numel,
    cudaStream_t stream0,
    cudaStream_t stream1) {
    (void)rank0_in;
    (void)rank1_in;
    (void)rank0_out_peer;
    (void)rank1_out_peer;
    (void)numel;
    (void)stream0;
    (void)stream1;

    if (st == nullptr) {
        return cudaErrorInvalidValue;
    }

    system::runtime::set_device(st->dev0);
    system::runtime::check_cuda(
        cudaMemset(st->progress0.ptr, 0, kTwoGpuPeerProgressBytes),
        "cudaMemset(progress0)");

    system::runtime::set_device(st->dev1);
    system::runtime::check_cuda(
        cudaMemset(st->progress1.ptr, 0, kTwoGpuPeerProgressBytes),
        "cudaMemset(progress1)");

    return cudaSuccess;
}

cudaError_t enqueue_tma_two_gpu_peer_allreduce_kernel_only_sm90(
    TmaTwoGpuPeerAllreduceState* st,
    const half* rank0_in,
    const half* rank1_in,
    half* rank0_out_peer,
    half* rank1_out_peer,
    size_t numel,
    cudaStream_t stream0,
    cudaStream_t stream1) {
    if (st == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (rank0_in == nullptr || rank1_in == nullptr ||
        rank0_out_peer == nullptr || rank1_out_peer == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }
    if (numel == 0) {
        return cudaErrorInvalidValue;
    }

    const int num_chunks = tma_two_gpu_peer_allreduce_compute_num_chunks(numel);
    if (num_chunks != st->num_chunks) {
        return cudaErrorInvalidValue;
    }

    const int num_blocks = std::min(kTwoGpuPeerMaxBlocks, num_chunks);
    const size_t smem_bytes =
        static_cast<size_t>(kTwoGpuPeerStageDepth) * kTwoGpuPeerChunkBytes;

    system::runtime::set_device(st->dev0);
    tma_two_gpu_copy_then_reduce_kernel_sm90<<<num_blocks, kTwoGpuPeerThreads, smem_bytes, stream0>>>(
        rank0_in,
        rank0_out_peer,
        rank1_out_peer,
        reinterpret_cast<int*>(st->progress0.ptr),
        reinterpret_cast<const int*>(st->progress1.ptr),
        numel,
        num_chunks,
        num_blocks);
    cudaError_t err0 = cudaGetLastError();
    if (err0 != cudaSuccess) {
        return err0;
    }

    system::runtime::set_device(st->dev1);
    tma_two_gpu_copy_then_reduce_kernel_sm90<<<num_blocks, kTwoGpuPeerThreads, smem_bytes, stream1>>>(
        rank1_in,
        rank1_out_peer,
        rank0_out_peer,
        reinterpret_cast<int*>(st->progress1.ptr),
        reinterpret_cast<const int*>(st->progress0.ptr),
        numel,
        num_chunks,
        num_blocks);
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        return err1;
    }

    return cudaSuccess;
}

cudaError_t enqueue_tma_two_gpu_peer_allreduce_sm90(
    TmaTwoGpuPeerAllreduceState* st,
    const half* rank0_in,
    const half* rank1_in,
    half* rank0_out_peer,
    half* rank1_out_peer,
    size_t numel,
    cudaStream_t stream0,
    cudaStream_t stream1) {
    cudaError_t err = prime_tma_two_gpu_peer_allreduce_outputs_sm90(
        st,
        rank0_in,
        rank1_in,
        rank0_out_peer,
        rank1_out_peer,
        numel,
        stream0,
        stream1);
    if (err != cudaSuccess) {
        return err;
    }

    return enqueue_tma_two_gpu_peer_allreduce_kernel_only_sm90(
        st,
        rank0_in,
        rank1_in,
        rank0_out_peer,
        rank1_out_peer,
        numel,
        stream0,
        stream1);
}

} // namespace ooverlap
