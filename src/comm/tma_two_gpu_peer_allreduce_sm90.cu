#include "comm/tma_two_gpu_peer_allreduce_sm90.h"

#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/tma/tma.cuh"
#include "ooverlap/tma/tma_reduce.cuh"

#include "comm/params.h"
#include "comm/utils.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace ooverlap {
namespace {

__device__ __forceinline__ unsigned char* stage_ptr(
    unsigned char* shared_raw,
    int stage) {
    return shared_raw +
           static_cast<size_t>(stage) * TMA_TWO_GPU_PEER_CHUNK_BYTES;
}

__device__ void reduce_window_to_peer_sm90(
    const unsigned char* local_in_bytes,
    unsigned char* peer_buf_bytes,
    comm::utils::Window window,
    size_t total_bytes,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    // Warm the load pipeline.
    for (int warm = 0; warm < TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP; ++warm) {
        if (warm >= window.chunk_count) {
            break;
        }

        const int chunk = window.start_chunk + warm;
        const int slot = warm;

        const size_t offset =
            static_cast<size_t>(chunk) * TMA_TWO_GPU_PEER_CHUNK_BYTES;
        const size_t bytes =
            comm::utils::min_sz(TMA_TWO_GPU_PEER_CHUNK_BYTES,
                                total_bytes - offset);

        if (threadIdx.x == 0) {
            sync::init_semaphore(barriers[slot], 1);
            tma::expect_bytes(barriers[slot], static_cast<uint32_t>(bytes));
            tma::load_async(
                stage_ptr(shared_raw, slot),
                local_in_bytes + offset,
                static_cast<uint32_t>(bytes),
                barriers[slot]);
        }
        __syncthreads();
    }

    for (int iter = 0; iter < window.chunk_count; ++iter) {
        const int chunk = window.start_chunk + iter;
        const int cur_slot = iter % TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH;

        const size_t offset =
            static_cast<size_t>(chunk) * TMA_TWO_GPU_PEER_CHUNK_BYTES;
        const size_t bytes =
            comm::utils::min_sz(TMA_TWO_GPU_PEER_CHUNK_BYTES,
                                total_bytes - offset);
        const size_t bulk_bytes = bytes & ~static_cast<size_t>(0xF);
        const size_t tail_bytes = bytes - bulk_bytes;

        if (threadIdx.x == 0) {
            sync::wait(barriers[cur_slot], 0);
        }
        __syncthreads();

        const int future_iter = iter + TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP;
        if (future_iter < window.chunk_count) {
            const int future_chunk = window.start_chunk + future_iter;
            const int future_slot =
                future_iter % TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH;

            const size_t future_offset =
                static_cast<size_t>(future_chunk) * TMA_TWO_GPU_PEER_CHUNK_BYTES;
            const size_t future_bytes =
                comm::utils::min_sz(TMA_TWO_GPU_PEER_CHUNK_BYTES,
                                    total_bytes - future_offset);

            if (threadIdx.x == 0) {
                if (iter >= TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP) {
                    tma::reduce_async_read_wait<
                        TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP - 1>();
                }

                sync::init_semaphore(barriers[future_slot], 1);
                tma::expect_bytes(
                    barriers[future_slot],
                    static_cast<uint32_t>(future_bytes));
                tma::load_async(
                    stage_ptr(shared_raw, future_slot),
                    local_in_bytes + future_offset,
                    static_cast<uint32_t>(future_bytes),
                    barriers[future_slot]);
            }
        }

        __syncthreads();

        if (threadIdx.x == 0 && bulk_bytes > 0) {
            tma::reduce_add_noftz_f16_async(
                peer_buf_bytes + offset,
                stage_ptr(shared_raw, cur_slot),
                static_cast<uint32_t>(bulk_bytes));
        }

        if (tail_bytes > 0) {
            const size_t bulk_elems = bulk_bytes / sizeof(half);
            const size_t tail_elems = tail_bytes / sizeof(half);
            const half* smem_half =
                reinterpret_cast<const half*>(stage_ptr(shared_raw, cur_slot));
            half* peer_half = reinterpret_cast<half*>(peer_buf_bytes + offset);

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
    }
    __syncthreads();
}

__device__ void copy_window_sm90(
    const unsigned char* src_bytes,
    unsigned char* dst_bytes,
    comm::utils::Window window,
    size_t total_bytes,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    // Warm the load pipeline from source buffer to shared memory.
    for (int warm = 0; warm < TMA_TWO_GPU_PEER_COPY_STAGE_GAP; ++warm) {
        if (warm >= window.chunk_count) {
            break;
        }

        const int chunk = window.start_chunk + warm;
        const int slot = warm;

        const size_t offset =
            static_cast<size_t>(chunk) * TMA_TWO_GPU_PEER_CHUNK_BYTES;
        const size_t bytes =
            comm::utils::min_sz(TMA_TWO_GPU_PEER_CHUNK_BYTES,
                                total_bytes - offset);

        if (threadIdx.x == 0) {
            sync::init_semaphore(barriers[slot], 1);
            tma::expect_bytes(barriers[slot], static_cast<uint32_t>(bytes));
            tma::load_async(
                stage_ptr(shared_raw, slot),
                src_bytes + offset,
                static_cast<uint32_t>(bytes),
                barriers[slot]);
        }
        __syncthreads();
    }

    for (int iter = 0; iter < window.chunk_count; ++iter) {
        const int chunk = window.start_chunk + iter;
        const int cur_slot = iter % TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH;

        const size_t offset =
            static_cast<size_t>(chunk) * TMA_TWO_GPU_PEER_CHUNK_BYTES;
        const size_t bytes =
            comm::utils::min_sz(TMA_TWO_GPU_PEER_CHUNK_BYTES,
                                total_bytes - offset);
        const size_t bulk_bytes = bytes & ~static_cast<size_t>(0xF);
        const size_t tail_bytes = bytes - bulk_bytes;

        if (threadIdx.x == 0) {
            sync::wait(barriers[cur_slot], 0);
        }
        __syncthreads();

        const int future_iter = iter + TMA_TWO_GPU_PEER_COPY_STAGE_GAP;
        if (future_iter < window.chunk_count) {
            const int future_chunk = window.start_chunk + future_iter;
            const int future_slot =
                future_iter % TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH;

            const size_t future_offset =
                static_cast<size_t>(future_chunk) * TMA_TWO_GPU_PEER_CHUNK_BYTES;
            const size_t future_bytes =
                comm::utils::min_sz(TMA_TWO_GPU_PEER_CHUNK_BYTES,
                                    total_bytes - future_offset);

            if (threadIdx.x == 0) {
                if (iter >= TMA_TWO_GPU_PEER_COPY_STAGE_GAP) {
                    tma::store_async_read_wait<
                        TMA_TWO_GPU_PEER_COPY_STAGE_GAP - 1>();
                }

                sync::init_semaphore(barriers[future_slot], 1);
                tma::expect_bytes(
                    barriers[future_slot],
                    static_cast<uint32_t>(future_bytes));
                tma::load_async(
                    stage_ptr(shared_raw, future_slot),
                    src_bytes + future_offset,
                    static_cast<uint32_t>(future_bytes),
                    barriers[future_slot]);
            }
        }

        __syncthreads();

        if (threadIdx.x == 0 && bulk_bytes > 0) {
            tma::store_async(
                dst_bytes + offset,
                stage_ptr(shared_raw, cur_slot),
                static_cast<uint32_t>(bulk_bytes));
        }

        if (tail_bytes > 0) {
            unsigned char* smem = stage_ptr(shared_raw, cur_slot);
            for (size_t i = threadIdx.x; i < tail_bytes; i += blockDim.x) {
                dst_bytes[offset + bulk_bytes + i] =
                    smem[bulk_bytes + i];
            }
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        tma::store_async_wait<0>();
        __threadfence_system();
    }
    __syncthreads();
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
    (void)local_progress;
    (void)peer_progress;

    // Each rank only launches CTAs for windows it owns:
    //   rank 0 -> windows 0, 2, 4, ...
    //   rank 1 -> windows 1, 3, 5, ...
    const int window_idx = 2 * static_cast<int>(blockIdx.x) + rank;
    if (window_idx >= num_windows) {
        return;
    }

    const comm::utils::Window window =
        comm::utils::make_window(window_idx, num_chunks, num_windows);

    if (window.chunk_count <= 0) {
        return;
    }

    const size_t total_bytes = numel * sizeof(half);

    extern __shared__ uint4 shared_storage_u4[];
    unsigned char* shared_raw =
        reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore barriers[TMA_TWO_GPU_PEER_BARRIER_COUNT];

    const unsigned char* local_in_bytes =
        reinterpret_cast<const unsigned char*>(local_in);
    unsigned char* local_buf_bytes =
        reinterpret_cast<unsigned char*>(local_buf);
    unsigned char* peer_buf_bytes =
        reinterpret_cast<unsigned char*>(peer_buf);

    reduce_window_to_peer_sm90(
        local_in_bytes,
        peer_buf_bytes,
        window,
        total_bytes,
        shared_raw,
        barriers);

    copy_window_sm90(
        peer_buf_bytes,
        local_buf_bytes,
        window,
        total_bytes,
        shared_raw,
        barriers);
}

} // namespace

int tma_two_gpu_peer_allreduce_compute_num_chunks(size_t numel) {
    const size_t total_bytes = numel * sizeof(half);
    return comm::utils::ceil_div_int64_to_int(
        total_bytes,
        TMA_TWO_GPU_PEER_CHUNK_BYTES);
}

void tma_two_gpu_peer_allreduce_configure_kernel_once(int device) {
    struct CacheEntry {
        bool configured = false;
        size_t dynamic_smem_bytes = 0;
    };

    static std::mutex mutex;
    static std::unordered_map<int, CacheEntry> cache;

    const size_t dynamic_smem_bytes = TMA_TWO_GPU_PEER_DYNAMIC_SHARED_BYTES;
    const size_t total_smem_bytes =
        dynamic_smem_bytes + TMA_TWO_GPU_PEER_STATIC_SHARED_BYTES;

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
        TMA_TWO_GPU_PEER_PROGRESS_BYTES,
        dev0,
        access_devices);
    st->progress1 = system::alloc_peer_visible_buffer(
        TMA_TWO_GPU_PEER_PROGRESS_BYTES,
        dev1,
        access_devices);

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(
        cudaMemset(st->progress0.ptr, 0, TMA_TWO_GPU_PEER_PROGRESS_BYTES),
        "cudaMemset(progress0)");

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(
        cudaMemset(st->progress1.ptr, 0, TMA_TWO_GPU_PEER_PROGRESS_BYTES),
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
        cudaMemsetAsync(
            st->progress0.ptr,
            0,
            TMA_TWO_GPU_PEER_PROGRESS_BYTES,
            stream0),
        "cudaMemsetAsync(progress0)");

    system::runtime::set_device(st->dev1);
    system::runtime::check_cuda(
        cudaMemsetAsync(
            st->progress1.ptr,
            0,
            TMA_TWO_GPU_PEER_PROGRESS_BYTES,
            stream1),
        "cudaMemsetAsync(progress1)");

    return cudaSuccess;
}

cudaError_t enqueue_tma_two_gpu_peer_allreduce_rank_sm90(
    const half* local_in,
    half* local_buf,
    half* peer_buf,
    size_t numel,
    int rank,
    int dev0,
    int dev1,
    cudaStream_t stream) {
    if (local_in == nullptr || local_buf == nullptr || peer_buf == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }
    if (numel == 0) {
        return cudaErrorInvalidValue;
    }
    if (rank != 0 && rank != 1) {
        return cudaErrorInvalidValue;
    }
    if (dev0 == dev1) {
        return cudaErrorInvalidValue;
    }

    const int device = (rank == 0) ? dev0 : dev1;
    const int num_chunks = tma_two_gpu_peer_allreduce_compute_num_chunks(numel);
    const int num_windows = comm::utils::window_num_chunks(num_chunks);
    const int num_blocks =
        (rank == 0) ? ((num_windows + 1) / 2) : (num_windows / 2);

    if (num_blocks <= 0) {
        return cudaSuccess;
    }

    tma_two_gpu_peer_allreduce_configure_kernel_once(device);

    const size_t smem_bytes = TMA_TWO_GPU_PEER_DYNAMIC_SHARED_BYTES;

    system::runtime::set_device(device);
    tma_two_gpu_reduce_or_copy_windows_kernel_sm90<<<
        num_blocks,
        TMA_TWO_GPU_PEER_THREADS,
        smem_bytes,
        stream>>>(
            local_in,
            local_buf,
            peer_buf,
            nullptr,
            nullptr,
            numel,
            num_chunks,
            num_windows,
            rank);

    return cudaGetLastError();
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

    const int num_windows = comm::utils::window_num_chunks(num_chunks);
    const int num_blocks_rank0 = (num_windows + 1) / 2;
    const int num_blocks_rank1 = num_windows / 2;
    const size_t smem_bytes = TMA_TWO_GPU_PEER_DYNAMIC_SHARED_BYTES;

    system::runtime::set_device(st->dev0);
    tma_two_gpu_reduce_or_copy_windows_kernel_sm90<<<
        num_blocks_rank0,
        TMA_TWO_GPU_PEER_THREADS,
        smem_bytes,
        stream0>>>(
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

    if (num_blocks_rank1 > 0) {
        system::runtime::set_device(st->dev1);
        tma_two_gpu_reduce_or_copy_windows_kernel_sm90<<<
            num_blocks_rank1,
            TMA_TWO_GPU_PEER_THREADS,
            smem_bytes,
            stream1>>>(
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
