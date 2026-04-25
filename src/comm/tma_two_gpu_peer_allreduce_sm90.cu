#include "comm/tma_two_gpu_peer_allreduce_sm90.h"

#include "ooverlap/system/runtime_utils.cuh"

#include "comm/exec/pipeline_stage.h"
#include "comm/exec/pipeline_tma_load.h"
#include "comm/exec/pipeline_tma_copy.h"
#include "comm/exec/pipeline_tma_reduce.h"

#include "comm/params.h"
#include "comm/utils.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace ooverlap {
namespace {

__host__ __device__ __forceinline__ size_t dtype_size_bytes(
    oo_dtype_t dtype) {
    switch (dtype) {
        case OO_DTYPE_FLOAT16:
            return sizeof(half);
        default:
            return 0;
    }
}

__device__ __forceinline__ unsigned char* stage_ptr(
    unsigned char* shared_raw,
    int stage) {
    return shared_raw +
           static_cast<size_t>(stage) * TMA_TWO_GPU_PEER_CHUNK_BYTES;
}

__device__ __forceinline__ void wait_for_collective_ready_sm90(
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch) {
    if (local_ready_signal == nullptr ||
        peer_ready_signal == nullptr ||
        collective_epoch <= 0) {
        return;
    }

    if (threadIdx.x == 0) {
        atomicMax(local_ready_signal, collective_epoch);
        __threadfence_system();

        const volatile int* peer_ready =
            reinterpret_cast<const volatile int*>(peer_ready_signal);

        while (peer_ready[0] < collective_epoch) {
#if defined(__CUDA_ARCH__)
            __nanosleep(64);
#endif
        }
    }

    __syncthreads();
}

template <int StageDepth, int FillDepth, typename Apply>
__device__ void run_tma_window_pipeline_sm90(
    const unsigned char* src_bytes,
    unsigned char* dst_bytes,
    comm::utils::Window window,
    size_t total_bytes,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    comm::PipelineTMALoad load{};
    Apply apply{};

    for (int warm = 0; warm < FillDepth; ++warm) {
        if (warm >= window.chunk_count) {
            break;
        }

        const int chunk = window.start_chunk + warm;
        const int slot = warm;

        const size_t offset =
            static_cast<size_t>(chunk) * TMA_TWO_GPU_PEER_CHUNK_BYTES;
        const size_t bytes =
            comm::utils::min_sz(
                TMA_TWO_GPU_PEER_CHUNK_BYTES,
                total_bytes - offset);

        comm::PipelineStage stage = comm::make_pipeline_stage(
            comm::make_pipeline_chunk(
                src_bytes + offset,
                dst_bytes + offset,
                bytes),
            stage_ptr(shared_raw, slot),
            &barriers[slot]);

        if (threadIdx.x == 0) {
            load.issue(&stage);
        }

        __syncthreads();
    }

    for (int iter = 0; iter < window.chunk_count; ++iter) {
        const int chunk = window.start_chunk + iter;
        const int cur_slot = iter % StageDepth;

        const size_t offset =
            static_cast<size_t>(chunk) * TMA_TWO_GPU_PEER_CHUNK_BYTES;
        const size_t bytes =
            comm::utils::min_sz(
                TMA_TWO_GPU_PEER_CHUNK_BYTES,
                total_bytes - offset);

        comm::PipelineStage cur_stage = comm::make_pipeline_stage(
            comm::make_pipeline_chunk(
                src_bytes + offset,
                dst_bytes + offset,
                bytes),
            stage_ptr(shared_raw, cur_slot),
            &barriers[cur_slot]);

        if (threadIdx.x == 0) {
            load.wait_ready(&cur_stage);
        }

        __syncthreads();

        const int future_iter = iter + FillDepth;
        if (future_iter < window.chunk_count) {
            const int future_chunk = window.start_chunk + future_iter;
            const int future_slot = future_iter % StageDepth;

            const size_t future_offset =
                static_cast<size_t>(future_chunk) *
                TMA_TWO_GPU_PEER_CHUNK_BYTES;
            const size_t future_bytes =
                comm::utils::min_sz(
                    TMA_TWO_GPU_PEER_CHUNK_BYTES,
                    total_bytes - future_offset);

            comm::PipelineStage future_stage = comm::make_pipeline_stage(
                comm::make_pipeline_chunk(
                    src_bytes + future_offset,
                    dst_bytes + future_offset,
                    future_bytes),
                stage_ptr(shared_raw, future_slot),
                &barriers[future_slot]);

            if (threadIdx.x == 0) {
                if (iter >= FillDepth) {
                    apply.wait_before_stage_reuse();
                }

                load.issue(&future_stage);
            }
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            apply.issue_bulk(&cur_stage);
        }

        apply.finish_tail(&cur_stage);

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        apply.wait_complete();
        __threadfence_system();
    }

    __syncthreads();
}

template <typename ReduceApply>
__device__ void reduce_window_to_peer_sm90(
    const unsigned char* local_in_bytes,
    unsigned char* peer_buf_bytes,
    comm::utils::Window window,
    size_t total_bytes,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    run_tma_window_pipeline_sm90<
        TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH,
        TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP,
        ReduceApply>(
            local_in_bytes,
            peer_buf_bytes,
            window,
            total_bytes,
            shared_raw,
            barriers);
}

__device__ void copy_window_sm90(
    const unsigned char* src_bytes,
    unsigned char* dst_bytes,
    comm::utils::Window window,
    size_t total_bytes,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    using CopyApply = comm::PipelineTMACopy<
        TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH,
        TMA_TWO_GPU_PEER_COPY_STAGE_GAP>;

    run_tma_window_pipeline_sm90<
        TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH,
        TMA_TWO_GPU_PEER_COPY_STAGE_GAP,
        CopyApply>(
            src_bytes,
            dst_bytes,
            window,
            total_bytes,
            shared_raw,
            barriers);
}

template <typename ReduceApply, int ElemBytes>
__global__ void tma_two_gpu_allreduce_rank_kernel_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    int rank,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch) {
    wait_for_collective_ready_sm90(
        local_ready_signal,
        peer_ready_signal,
        collective_epoch);

    const size_t total_bytes = count * static_cast<size_t>(ElemBytes);
    const int num_chunks = comm::utils::ceil_div_int64_to_int(
        total_bytes,
        TMA_TWO_GPU_PEER_CHUNK_BYTES);
    const int num_windows = comm::utils::window_num_chunks(num_chunks);

    const int window_idx = 2 * static_cast<int>(blockIdx.x) + rank;
    if (window_idx >= num_windows) {
        return;
    }

    const comm::utils::Window window =
        comm::utils::make_window(window_idx, num_chunks, num_windows);

    if (window.chunk_count <= 0) {
        return;
    }

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

    reduce_window_to_peer_sm90<ReduceApply>(
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

template <typename ReduceApply, int ElemBytes>
void configure_kernel_once_for(int device) {
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
                tma_two_gpu_allreduce_rank_kernel_sm90<
                    ReduceApply,
                    ElemBytes>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(dynamic_smem_bytes)),
            "cudaFuncSetAttribute(MaxDynamicSharedMemorySize)");

        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                tma_two_gpu_allreduce_rank_kernel_sm90<
                    ReduceApply,
                    ElemBytes>,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                100),
            "cudaFuncSetAttribute(PreferredSharedMemoryCarveout)");
    }

    cache[device] = {true, dynamic_smem_bytes};
}

template <typename ReduceApply, int ElemBytes>
cudaError_t launch_rank_kernel_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    int rank,
    int dev0,
    int dev1,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch) {
    const int device = (rank == 0) ? dev0 : dev1;

    const size_t total_bytes = count * static_cast<size_t>(ElemBytes);
    const int num_chunks = comm::utils::ceil_div_int64_to_int(
        total_bytes,
        TMA_TWO_GPU_PEER_CHUNK_BYTES);
    const int num_windows = comm::utils::window_num_chunks(num_chunks);
    const int owned_blocks =
        (rank == 0) ? ((num_windows + 1) / 2) : (num_windows / 2);

    const bool needs_rendezvous =
        local_ready_signal != nullptr &&
        peer_ready_signal != nullptr &&
        collective_epoch > 0;

    const int num_blocks =
        needs_rendezvous ? std::max(1, owned_blocks) : owned_blocks;

    if (num_blocks <= 0) {
        return cudaSuccess;
    }

    configure_kernel_once_for<ReduceApply, ElemBytes>(device);

    system::runtime::set_device(device);

    tma_two_gpu_allreduce_rank_kernel_sm90<
        ReduceApply,
        ElemBytes><<<
            num_blocks,
            TMA_TWO_GPU_PEER_THREADS,
            TMA_TWO_GPU_PEER_DYNAMIC_SHARED_BYTES,
            stream>>>(
                local_in,
                local_buf,
                peer_buf,
                count,
                rank,
                local_ready_signal,
                peer_ready_signal,
                collective_epoch);

    return cudaGetLastError();
}

template <typename ReduceApply, int ElemBytes>
void configure_dispatch_for(int device) {
    configure_kernel_once_for<ReduceApply, ElemBytes>(device);
}

cudaError_t dispatch_rank_kernel_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int rank,
    int dev0,
    int dev1,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch) {
    if (dtype == OO_DTYPE_FLOAT16 && op == OO_REDUCE_SUM) {
        return launch_rank_kernel_sm90<
            comm::PipelineTMAReduce<
                TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH,
                TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP,
                comm::PipelineReduceAddNoFtzF16>,
            static_cast<int>(sizeof(half))>(
                local_in,
                local_buf,
                peer_buf,
                count,
                rank,
                dev0,
                dev1,
                stream,
                local_ready_signal,
                peer_ready_signal,
                collective_epoch);
    }

    return cudaErrorInvalidValue;
}

void configure_dispatch_sm90(
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int device) {
    if (dtype == OO_DTYPE_FLOAT16 && op == OO_REDUCE_SUM) {
        configure_dispatch_for<
            comm::PipelineTMAReduce<
                TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH,
                TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP,
                comm::PipelineReduceAddNoFtzF16>,
            static_cast<int>(sizeof(half))>(device);
        return;
    }

    throw std::invalid_argument(
        "tma_two_gpu_peer_allreduce_configure_kernel_once: unsupported dtype/op");
}

} // namespace

void tma_two_gpu_peer_allreduce_configure_kernel_once(
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int device) {
    configure_dispatch_sm90(dtype, op, device);
}

cudaError_t enqueue_tma_two_gpu_peer_allreduce_rank_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int rank,
    int dev0,
    int dev1,
    cudaStream_t stream,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch) {
    if (local_in == nullptr || local_buf == nullptr || peer_buf == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }
    if (count == 0) {
        return cudaErrorInvalidValue;
    }
    if (rank != 0 && rank != 1) {
        return cudaErrorInvalidValue;
    }
    if (dev0 == dev1) {
        return cudaErrorInvalidValue;
    }
    if (dtype_size_bytes(dtype) == 0) {
        return cudaErrorInvalidValue;
    }

    return dispatch_rank_kernel_sm90(
        local_in,
        local_buf,
        peer_buf,
        count,
        dtype,
        op,
        rank,
        dev0,
        dev1,
        stream,
        local_ready_signal,
        peer_ready_signal,
        collective_epoch);
}

} // namespace ooverlap
