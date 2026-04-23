#include "comm/tma_two_gpu_peer_allreduce_sm90.h"

#include "ooverlap/tma/tma.cuh"
#include "ooverlap/tma/tma_reduce.cuh"
#include "ooverlap/system/runtime_utils.cuh"

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
constexpr int kTwoGpuPeerStageDepth = 8;
constexpr size_t kTwoGpuPeerStaticSharedBytes =
    static_cast<size_t>(kTwoGpuPeerStageDepth) * sizeof(sync::semaphore);

__host__ __device__ __forceinline__ size_t min_sz(size_t a, size_t b) {
    return (a < b) ? a : b;
}

__global__ void tma_two_gpu_peer_reduce_kernel_sm90(
    const half* local_in,
    half* peer_out,
    size_t numel,
    int num_chunks) {

    const int start_chunk = static_cast<int>(blockIdx.x);
    const int chunk_stride = static_cast<int>(gridDim.x);

    if (start_chunk >= num_chunks) {
        return;
    }

    extern __shared__ uint4 shared_storage_u4[];
    unsigned char* shared_raw = reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore load_barriers[kTwoGpuPeerStageDepth];

    auto stage_ptr = [&](int stage) -> unsigned char* {
        return shared_raw + static_cast<size_t>(stage) * kTwoGpuPeerChunkBytes;
    };

    const unsigned char* local_bytes = reinterpret_cast<const unsigned char*>(local_in);
    unsigned char* peer_bytes = reinterpret_cast<unsigned char*>(peer_out);
    const size_t total_bytes = numel * sizeof(half);

    int local_iter = 0;
    int cur_chunk = start_chunk;
    int next_chunk_to_load = start_chunk + chunk_stride;

    {
        const size_t cur_offset = static_cast<size_t>(cur_chunk) * kTwoGpuPeerChunkBytes;
        const size_t cur_bytes = min_sz(kTwoGpuPeerChunkBytes, total_bytes - cur_offset);

        if (threadIdx.x == 0) {
            sync::init_semaphore(load_barriers[0], 1);
            tma::expect_bytes(load_barriers[0], static_cast<uint32_t>(cur_bytes));
            tma::load_async(
                stage_ptr(0),
                local_bytes + cur_offset,
                static_cast<uint32_t>(cur_bytes),
                load_barriers[0]);
        }
        __syncthreads();
    }

    while (cur_chunk < num_chunks) {
        const int cur_stage = local_iter % kTwoGpuPeerStageDepth;

        if (threadIdx.x == 0) {
            sync::wait(load_barriers[cur_stage], 0);
        }
        __syncthreads();

        unsigned char* cur_smem = stage_ptr(cur_stage);

        const size_t cur_offset = static_cast<size_t>(cur_chunk) * kTwoGpuPeerChunkBytes;
        const size_t cur_bytes = min_sz(kTwoGpuPeerChunkBytes, total_bytes - cur_offset);

        const size_t bulk_bytes = cur_bytes & ~static_cast<size_t>(0xF);
        const size_t tail_bytes = cur_bytes - bulk_bytes;

        if (threadIdx.x == 0 && bulk_bytes > 0) {
            tma::reduce_add_noftz_f16_async(
                peer_bytes + cur_offset,
                cur_smem,
                static_cast<uint32_t>(bulk_bytes));
        }

        if (next_chunk_to_load < num_chunks) {
            const int next_stage = (local_iter + 1) % kTwoGpuPeerStageDepth;

            if (threadIdx.x == 0) {
                if ((local_iter + 1) >= kTwoGpuPeerStageDepth) {
                    tma::reduce_async_read_wait<kTwoGpuPeerStageDepth - 1>();
                }

                const size_t next_offset =
                    static_cast<size_t>(next_chunk_to_load) * kTwoGpuPeerChunkBytes;
                const size_t next_bytes =
                    min_sz(kTwoGpuPeerChunkBytes, total_bytes - next_offset);

                sync::init_semaphore(load_barriers[next_stage], 1);
                tma::expect_bytes(load_barriers[next_stage], static_cast<uint32_t>(next_bytes));
                tma::load_async(
                    stage_ptr(next_stage),
                    local_bytes + next_offset,
                    static_cast<uint32_t>(next_bytes),
                    load_barriers[next_stage]);
            }
        }

        __syncthreads();

        if (tail_bytes > 0) {
            const size_t bulk_elems = bulk_bytes / sizeof(half);
            const size_t tail_elems = tail_bytes / sizeof(half);
            const half* cur_half = reinterpret_cast<const half*>(cur_smem);

            for (size_t i = threadIdx.x; i < tail_elems; i += blockDim.x) {
                const size_t idx = (cur_offset / sizeof(half)) + bulk_elems + i;
                const float oldv = __half2float(peer_out[idx]);
                const float addv = __half2float(cur_half[bulk_elems + i]);
                peer_out[idx] = __float2half_rn(oldv + addv);
            }
        }

        __syncthreads();

        cur_chunk = next_chunk_to_load;
        next_chunk_to_load += chunk_stride;
        ++local_iter;
    }
}

void create_event_on_device(
    int device,
    cudaEvent_t* ev,
    const char* what) {
    system::runtime::set_device(device);
    system::runtime::check_cuda(
        cudaEventCreateWithFlags(ev, cudaEventDisableTiming),
        what);
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
                tma_two_gpu_peer_reduce_kernel_sm90,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(dynamic_smem_bytes)),
            "cudaFuncSetAttribute(MaxDynamicSharedMemorySize)");

        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                tma_two_gpu_peer_reduce_kernel_sm90,
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

    create_event_on_device(dev0, &st->init_done0, "cudaEventCreateWithFlags(init_done0)");
    create_event_on_device(dev1, &st->init_done1, "cudaEventCreateWithFlags(init_done1)");
    create_event_on_device(dev0, &st->reduce_done0, "cudaEventCreateWithFlags(reduce_done0)");
    create_event_on_device(dev1, &st->reduce_done1, "cudaEventCreateWithFlags(reduce_done1)");
}

void tma_two_gpu_peer_allreduce_state_destroy(
    TmaTwoGpuPeerAllreduceState* st) {
    if (st == nullptr) {
        return;
    }

    if (st->init_done0 != nullptr) {
        system::runtime::set_device(st->dev0);
        cudaEventDestroy(st->init_done0);
        st->init_done0 = nullptr;
    }
    if (st->init_done1 != nullptr) {
        system::runtime::set_device(st->dev1);
        cudaEventDestroy(st->init_done1);
        st->init_done1 = nullptr;
    }
    if (st->reduce_done0 != nullptr) {
        system::runtime::set_device(st->dev0);
        cudaEventDestroy(st->reduce_done0);
        st->reduce_done0 = nullptr;
    }
    if (st->reduce_done1 != nullptr) {
        system::runtime::set_device(st->dev1);
        cudaEventDestroy(st->reduce_done1);
        st->reduce_done1 = nullptr;
    }

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
    std::vector<int> access_devices = {dev0, dev1};

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

    const size_t bytes = numel * sizeof(half);

    system::runtime::set_device(st->dev0);
    system::runtime::check_cuda(
        cudaMemcpyAsync(
            rank0_out_peer,
            rank0_in,
            bytes,
            cudaMemcpyDeviceToDevice,
            stream0),
        "cudaMemcpyAsync(rank0_in -> rank0_out_peer)");
    system::runtime::check_cuda(
        cudaEventRecord(st->init_done0, stream0),
        "cudaEventRecord(init_done0)");

    system::runtime::set_device(st->dev1);
    system::runtime::check_cuda(
        cudaMemcpyAsync(
            rank1_out_peer,
            rank1_in,
            bytes,
            cudaMemcpyDeviceToDevice,
            stream1),
        "cudaMemcpyAsync(rank1_in -> rank1_out_peer)");
    system::runtime::check_cuda(
        cudaEventRecord(st->init_done1, stream1),
        "cudaEventRecord(init_done1)");

    system::runtime::set_device(st->dev0);
    system::runtime::check_cuda(
        cudaStreamWaitEvent(stream0, st->init_done1, 0),
        "cudaStreamWaitEvent(stream0, init_done1)");

    system::runtime::set_device(st->dev1);
    system::runtime::check_cuda(
        cudaStreamWaitEvent(stream1, st->init_done0, 0),
        "cudaStreamWaitEvent(stream1, init_done0)");

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
    tma_two_gpu_peer_reduce_kernel_sm90<<<num_blocks, kTwoGpuPeerThreads, smem_bytes, stream0>>>(
        rank0_in,
        rank1_out_peer,
        numel,
        num_chunks);
    cudaError_t err0 = cudaGetLastError();
    if (err0 != cudaSuccess) {
        return err0;
    }
    system::runtime::check_cuda(
        cudaEventRecord(st->reduce_done0, stream0),
        "cudaEventRecord(reduce_done0)");

    system::runtime::set_device(st->dev1);
    tma_two_gpu_peer_reduce_kernel_sm90<<<num_blocks, kTwoGpuPeerThreads, smem_bytes, stream1>>>(
        rank1_in,
        rank0_out_peer,
        numel,
        num_chunks);
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        return err1;
    }
    system::runtime::check_cuda(
        cudaEventRecord(st->reduce_done1, stream1),
        "cudaEventRecord(reduce_done1)");

    system::runtime::set_device(st->dev0);
    system::runtime::check_cuda(
        cudaStreamWaitEvent(stream0, st->reduce_done1, 0),
        "cudaStreamWaitEvent(stream0, reduce_done1)");

    system::runtime::set_device(st->dev1);
    system::runtime::check_cuda(
        cudaStreamWaitEvent(stream1, st->reduce_done0, 0),
        "cudaStreamWaitEvent(stream1, reduce_done0)");

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
