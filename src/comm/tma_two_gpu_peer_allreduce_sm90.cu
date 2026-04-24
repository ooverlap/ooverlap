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

// -----------------------------------------------------------------------------
// Tunables
// -----------------------------------------------------------------------------

constexpr int kTwoGpuPeerThreads = 16;
constexpr int kTwoGpuPeerMaxWindows = 4;
constexpr size_t kTwoGpuPeerChunkBytes = 16 * 1024;

// phase 1: owner rank reduces its local window into peer buffer
constexpr int kTwoGpuPeerReduceStageDepth = 8;
constexpr int kTwoGpuPeerReduceStageGap = kTwoGpuPeerReduceStageDepth / 2;

// phase 2: non-owner rank copies finalized local window back to peer buffer
constexpr int kTwoGpuPeerCopyStageDepth = 8;
constexpr int kTwoGpuPeerCopyStageGap = kTwoGpuPeerCopyStageDepth / 2;

static_assert(kTwoGpuPeerReduceStageDepth % 2 == 0,
              "kTwoGpuPeerReduceStageDepth must be even");
static_assert(kTwoGpuPeerCopyStageDepth % 2 == 0,
              "kTwoGpuPeerCopyStageDepth must be even");
static_assert(kTwoGpuPeerReduceStageGap >= 1,
              "kTwoGpuPeerReduceStageGap must be >= 1");
static_assert(kTwoGpuPeerCopyStageGap >= 1,
              "kTwoGpuPeerCopyStageGap must be >= 1");

constexpr size_t kTwoGpuPeerReduceSharedBytes =
    static_cast<size_t>(kTwoGpuPeerReduceStageDepth) * kTwoGpuPeerChunkBytes;

constexpr size_t kTwoGpuPeerCopySharedBytes =
    static_cast<size_t>(kTwoGpuPeerCopyStageDepth) * kTwoGpuPeerChunkBytes;

constexpr size_t kTwoGpuPeerDynamicSharedBytes =
    (kTwoGpuPeerReduceSharedBytes > kTwoGpuPeerCopySharedBytes)
        ? kTwoGpuPeerReduceSharedBytes
        : kTwoGpuPeerCopySharedBytes;

constexpr int kTwoGpuPeerBarrierCount =
    (kTwoGpuPeerReduceStageDepth > kTwoGpuPeerCopyStageDepth)
        ? kTwoGpuPeerReduceStageDepth
        : kTwoGpuPeerCopyStageDepth;

constexpr size_t kTwoGpuPeerStaticSharedBytes =
    static_cast<size_t>(kTwoGpuPeerBarrierCount) * sizeof(sync::semaphore);

// progress0 / progress1 layout:
//   [0 .. kMaxWindows-1]               reduce-done flags
//   [kMaxWindows .. 2*kMaxWindows-1]   copy-done flags
constexpr size_t kTwoGpuPeerProgressBytes =
    static_cast<size_t>(2 * kTwoGpuPeerMaxWindows) * sizeof(int);

__host__ __device__ __forceinline__ size_t min_sz(size_t a, size_t b) {
    return (a < b) ? a : b;
}

__host__ __device__ __forceinline__ int ceil_div_int64_to_int(
    size_t num,
    size_t den) {
    return static_cast<int>((num + den - 1) / den);
}

__host__ __device__ __forceinline__ int min_int(int a, int b) {
    return (a < b) ? a : b;
}

__host__ __device__ __forceinline__ int window_num_chunks(
    int num_chunks) {
    return min_int(num_chunks, kTwoGpuPeerMaxWindows);
}

__host__ __device__ __forceinline__ int window_chunk_start(
    int window_idx,
    int num_chunks,
    int num_windows) {
    const int base = num_chunks / num_windows;
    const int rem = num_chunks % num_windows;
    return window_idx * base + ((window_idx < rem) ? window_idx : rem);
}

__host__ __device__ __forceinline__ int window_chunk_count(
    int window_idx,
    int num_chunks,
    int num_windows) {
    const int base = num_chunks / num_windows;
    const int rem = num_chunks % num_windows;
    return base + ((window_idx < rem) ? 1 : 0);
}

__global__ void tma_two_gpu_reduce_or_copy_windows_kernel_sm90(
    const half* local_in,
    half* local_buf,
    half* peer_buf,
    int* local_progress,
    const int* peer_progress,
    size_t numel,
    int num_chunks,
    int num_windows,
    int rank) {

    const int window_idx = static_cast<int>(blockIdx.x);
    if (window_idx >= num_windows) {
        return;
    }

    const int win_start_chunk =
        window_chunk_start(window_idx, num_chunks, num_windows);
    const int win_chunk_count =
        window_chunk_count(window_idx, num_chunks, num_windows);

    if (win_chunk_count <= 0) {
        return;
    }

    const int owner_rank = window_idx & 1;
    const bool i_own_this_window = (rank == owner_rank);

    const size_t total_bytes = numel * sizeof(half);

    extern __shared__ uint4 shared_storage_u4[];
    unsigned char* shared_raw = reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore barriers[kTwoGpuPeerBarrierCount];

    auto reduce_stage_ptr = [&](int stage) -> unsigned char* {
        return shared_raw + static_cast<size_t>(stage) * kTwoGpuPeerChunkBytes;
    };

    auto copy_stage_ptr = [&](int stage) -> unsigned char* {
        return shared_raw + static_cast<size_t>(stage) * kTwoGpuPeerChunkBytes;
    };

    const unsigned char* local_in_bytes =
        reinterpret_cast<const unsigned char*>(local_in);
    const unsigned char* local_buf_bytes =
        reinterpret_cast<const unsigned char*>(local_buf);
    unsigned char* peer_buf_bytes =
        reinterpret_cast<unsigned char*>(peer_buf);

    int* local_reduce_done = local_progress;
    int* local_copy_done = local_progress + kTwoGpuPeerMaxWindows;

    const volatile int* peer_reduce_done =
        reinterpret_cast<const volatile int*>(peer_progress);
    const volatile int* peer_copy_done =
        reinterpret_cast<const volatile int*>(peer_progress + kTwoGpuPeerMaxWindows);

    // -------------------------------------------------------------------------
    // Phase 1: owner rank reduces local input window into peer buffer
    // -------------------------------------------------------------------------

    if (i_own_this_window) {
        for (int warm = 0; warm < kTwoGpuPeerReduceStageGap; ++warm) {
            if (warm >= win_chunk_count) {
                break;
            }

            const int chunk = win_start_chunk + warm;
            const int slot = warm;

            const size_t offset =
                static_cast<size_t>(chunk) * kTwoGpuPeerChunkBytes;
            const size_t bytes =
                min_sz(kTwoGpuPeerChunkBytes, total_bytes - offset);

            if (threadIdx.x == 0) {
                sync::init_semaphore(barriers[slot], 1);
                tma::expect_bytes(barriers[slot], static_cast<uint32_t>(bytes));
                tma::load_async(
                    reduce_stage_ptr(slot),
                    local_in_bytes + offset,
                    static_cast<uint32_t>(bytes),
                    barriers[slot]);
            }
            __syncthreads();
        }

        for (int iter = 0; iter < win_chunk_count; ++iter) {
            const int chunk = win_start_chunk + iter;
            const int cur_slot = iter % kTwoGpuPeerReduceStageDepth;

            const size_t offset =
                static_cast<size_t>(chunk) * kTwoGpuPeerChunkBytes;
            const size_t bytes =
                min_sz(kTwoGpuPeerChunkBytes, total_bytes - offset);
            const size_t bulk_bytes = bytes & ~static_cast<size_t>(0xF);
            const size_t tail_bytes = bytes - bulk_bytes;

            if (threadIdx.x == 0) {
                sync::wait(barriers[cur_slot], 0);
            }
            __syncthreads();

            const int future_iter = iter + kTwoGpuPeerReduceStageGap;
            if (future_iter < win_chunk_count) {
                const int future_chunk = win_start_chunk + future_iter;
                const int future_slot =
                    future_iter % kTwoGpuPeerReduceStageDepth;

                const size_t future_offset =
                    static_cast<size_t>(future_chunk) * kTwoGpuPeerChunkBytes;
                const size_t future_bytes =
                    min_sz(kTwoGpuPeerChunkBytes, total_bytes - future_offset);

                if (threadIdx.x == 0) {
                    if (iter >= kTwoGpuPeerReduceStageGap) {
                        tma::reduce_async_read_wait<kTwoGpuPeerReduceStageGap - 1>();
                    }

                    sync::init_semaphore(barriers[future_slot], 1);
                    tma::expect_bytes(
                        barriers[future_slot],
                        static_cast<uint32_t>(future_bytes));
                    tma::load_async(
                        reduce_stage_ptr(future_slot),
                        local_in_bytes + future_offset,
                        static_cast<uint32_t>(future_bytes),
                        barriers[future_slot]);
                }
            }

            __syncthreads();

            if (threadIdx.x == 0 && bulk_bytes > 0) {
                tma::reduce_add_noftz_f16_async(
                    peer_buf_bytes + offset,
                    reduce_stage_ptr(cur_slot),
                    static_cast<uint32_t>(bulk_bytes));
            }

            if (tail_bytes > 0) {
                const size_t bulk_elems = bulk_bytes / sizeof(half);
                const size_t tail_elems = tail_bytes / sizeof(half);
                const half* smem_half =
                    reinterpret_cast<const half*>(reduce_stage_ptr(cur_slot));
                half* peer_half =
                    reinterpret_cast<half*>(peer_buf_bytes + offset);

                for (size_t i = threadIdx.x; i < tail_elems; i += blockDim.x) {
                    const float oldv = __half2float(peer_half[bulk_elems + i]);
                    const float addv = __half2float(smem_half[bulk_elems + i]);
                    peer_half[bulk_elems + i] = __float2half_rn(oldv + addv);
                }
            }

            __syncthreads();
        }

        if (threadIdx.x == 0) {
            tma::reduce_async_wait<0>();
            __threadfence_system();
            atomicExch(local_reduce_done + window_idx, 1);
        }
        __syncthreads();
        return;
    }

    // -------------------------------------------------------------------------
    // Phase 2: non-owner waits for owner to finish, then copies finalized local
    // window back into peer buffer
    // -------------------------------------------------------------------------

    if (threadIdx.x == 0) {
        while (peer_reduce_done[window_idx] == 0) {
#if defined(__CUDA_ARCH__)
            __nanosleep(64);
#endif
        }
    }
    __syncthreads();

    for (int warm = 0; warm < kTwoGpuPeerCopyStageGap; ++warm) {
        if (warm >= win_chunk_count) {
            break;
        }

        const int chunk = win_start_chunk + warm;
        const int slot = warm;

        const size_t offset =
            static_cast<size_t>(chunk) * kTwoGpuPeerChunkBytes;
        const size_t bytes =
            min_sz(kTwoGpuPeerChunkBytes, total_bytes - offset);

        if (threadIdx.x == 0) {
            sync::init_semaphore(barriers[slot], 1);
            tma::expect_bytes(barriers[slot], static_cast<uint32_t>(bytes));
            tma::load_async(
                copy_stage_ptr(slot),
                local_buf_bytes + offset,
                static_cast<uint32_t>(bytes),
                barriers[slot]);
        }
        __syncthreads();
    }

    for (int iter = 0; iter < win_chunk_count; ++iter) {
        const int chunk = win_start_chunk + iter;
        const int cur_slot = iter % kTwoGpuPeerCopyStageDepth;

        const size_t offset =
            static_cast<size_t>(chunk) * kTwoGpuPeerChunkBytes;
        const size_t bytes =
            min_sz(kTwoGpuPeerChunkBytes, total_bytes - offset);
        const size_t bulk_bytes = bytes & ~static_cast<size_t>(0xF);
        const size_t tail_bytes = bytes - bulk_bytes;

        if (threadIdx.x == 0) {
            sync::wait(barriers[cur_slot], 0);
        }
        __syncthreads();

        const int future_iter = iter + kTwoGpuPeerCopyStageGap;
        if (future_iter < win_chunk_count) {
            const int future_chunk = win_start_chunk + future_iter;
            const int future_slot = future_iter % kTwoGpuPeerCopyStageDepth;

            const size_t future_offset =
                static_cast<size_t>(future_chunk) * kTwoGpuPeerChunkBytes;
            const size_t future_bytes =
                min_sz(kTwoGpuPeerChunkBytes, total_bytes - future_offset);

            if (threadIdx.x == 0) {
                if (iter >= kTwoGpuPeerCopyStageGap) {
                    tma::store_async_read_wait<kTwoGpuPeerCopyStageGap - 1>();
                }

                sync::init_semaphore(barriers[future_slot], 1);
                tma::expect_bytes(
                    barriers[future_slot],
                    static_cast<uint32_t>(future_bytes));
                tma::load_async(
                    copy_stage_ptr(future_slot),
                    local_buf_bytes + future_offset,
                    static_cast<uint32_t>(future_bytes),
                    barriers[future_slot]);
            }
        }

        __syncthreads();

        if (threadIdx.x == 0 && bulk_bytes > 0) {
            tma::store_async(
                peer_buf_bytes + offset,
                copy_stage_ptr(cur_slot),
                static_cast<uint32_t>(bulk_bytes));
        }

        if (tail_bytes > 0) {
            for (size_t i = threadIdx.x; i < tail_bytes; i += blockDim.x) {
                peer_buf_bytes[offset + bulk_bytes + i] =
                    copy_stage_ptr(cur_slot)[bulk_bytes + i];
            }
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        tma::store_async_wait<0>();
        __threadfence_system();
        atomicExch(local_copy_done + window_idx, 1);
    }
    __syncthreads();

    (void)peer_copy_done;
}

} // namespace

int tma_two_gpu_peer_allreduce_compute_num_chunks(size_t numel) {
    const size_t total_bytes = numel * sizeof(half);
    return ceil_div_int64_to_int(total_bytes, kTwoGpuPeerChunkBytes);
}

void tma_two_gpu_peer_allreduce_configure_kernel_once(int device) {
    struct CacheEntry {
        bool configured = false;
        size_t dynamic_smem_bytes = 0;
    };

    static std::mutex mutex;
    static std::unordered_map<int, CacheEntry> cache;

    const size_t dynamic_smem_bytes = kTwoGpuPeerDynamicSharedBytes;
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
                tma_two_gpu_reduce_or_copy_windows_kernel_sm90,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(dynamic_smem_bytes)),
            "cudaFuncSetAttribute(MaxDynamicSharedMemorySize)");

        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                tma_two_gpu_reduce_or_copy_windows_kernel_sm90,
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

    if (st == nullptr) {
        return cudaErrorInvalidValue;
    }

    system::runtime::set_device(st->dev0);
    system::runtime::check_cuda(
        cudaMemsetAsync(st->progress0.ptr, 0, kTwoGpuPeerProgressBytes, stream0),
        "cudaMemsetAsync(progress0)");

    system::runtime::set_device(st->dev1);
    system::runtime::check_cuda(
        cudaMemsetAsync(st->progress1.ptr, 0, kTwoGpuPeerProgressBytes, stream1),
        "cudaMemsetAsync(progress1)");

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

    const int num_windows = window_num_chunks(num_chunks);
    const int num_blocks = num_windows;
    const size_t smem_bytes = kTwoGpuPeerDynamicSharedBytes;

    system::runtime::set_device(st->dev0);
    tma_two_gpu_reduce_or_copy_windows_kernel_sm90<<<num_blocks, kTwoGpuPeerThreads, smem_bytes, stream0>>>(
        rank0_in,
        rank0_out_peer,
        rank1_out_peer,
        reinterpret_cast<int*>(st->progress0.ptr),
        reinterpret_cast<const int*>(st->progress1.ptr),
        numel,
        num_chunks,
        num_windows,
        0);
    cudaError_t err0 = cudaGetLastError();
    if (err0 != cudaSuccess) {
        return err0;
    }

    system::runtime::set_device(st->dev1);
    tma_two_gpu_reduce_or_copy_windows_kernel_sm90<<<num_blocks, kTwoGpuPeerThreads, smem_bytes, stream1>>>(
        rank1_in,
        rank1_out_peer,
        rank0_out_peer,
        reinterpret_cast<int*>(st->progress1.ptr),
        reinterpret_cast<const int*>(st->progress0.ptr),
        numel,
        num_chunks,
        num_windows,
        1);
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
