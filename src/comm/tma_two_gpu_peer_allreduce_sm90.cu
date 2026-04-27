#include "comm/tma_two_gpu_peer_allreduce_sm90.h"

#include "ooverlap/system/runtime_utils.cuh"

#include "comm/params.h"
#include "comm/pipeline_tma_reduce.h"
#include "comm/utils.h"
#include "comm/window_pipeline_sm90.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
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
        case OO_DTYPE_BFLOAT16:
            return sizeof(__nv_bfloat16);
        case OO_DTYPE_FLOAT32:
            return sizeof(float);
        default:
            return 0;
    }
}

__host__ __device__ __forceinline__ size_t window_begin_byte_sm90(
    comm::utils::Window window) {
    return static_cast<size_t>(window.start_chunk) *
           TMA_TWO_GPU_PEER_CHUNK_BYTES;
}

__host__ __device__ __forceinline__ size_t window_end_byte_sm90(
    comm::utils::Window window,
    size_t total_bytes) {
    const size_t end =
        static_cast<size_t>(window.start_chunk + window.chunk_count) *
        TMA_TWO_GPU_PEER_CHUNK_BYTES;

    return comm::utils::min_sz(end, total_bytes);
}

__host__ __device__ __forceinline__ size_t window_size_bytes_sm90(
    comm::utils::Window window,
    size_t total_bytes) {
    const size_t begin = window_begin_byte_sm90(window);
    const size_t end = window_end_byte_sm90(window, total_bytes);

    return (begin < end) ? (end - begin) : 0;
}

template <typename ReduceApply>
__device__ void reduce_window_to_peer_sm90(
    const unsigned char* local_in_bytes,
    unsigned char* peer_buf_bytes,
    comm::utils::Window window,
    size_t total_bytes,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    const size_t begin = window_begin_byte_sm90(window);
    const size_t bytes = window_size_bytes_sm90(window, total_bytes);

    comm::window_pipeline::run_window<
        TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH,
        TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP,
        TMA_TWO_GPU_PEER_CHUNK_BYTES,
        ReduceApply>(
            local_in_bytes + begin,
            peer_buf_bytes + begin,
            bytes,
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
    const size_t begin = window_begin_byte_sm90(window);
    const size_t bytes = window_size_bytes_sm90(window, total_bytes);

    comm::window_pipeline::copy_window_tma<
        TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH,
        TMA_TWO_GPU_PEER_COPY_STAGE_GAP,
        TMA_TWO_GPU_PEER_CHUNK_BYTES>(
            src_bytes + begin,
            dst_bytes + begin,
            bytes,
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
    comm::window_pipeline::wait_for_collective_ready(
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

template <typename ReduceOp, int ElemBytes>
cudaError_t launch_reduce_op_sm90(
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
    using ReduceApply = comm::PipelineTMAReduce<
        TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH,
        TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP,
        ReduceOp>;

    return launch_rank_kernel_sm90<ReduceApply, ElemBytes>(
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

template <typename ReduceOp, int ElemBytes>
void configure_reduce_op_sm90(int device) {
    using ReduceApply = comm::PipelineTMAReduce<
        TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH,
        TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP,
        ReduceOp>;

    configure_dispatch_for<ReduceApply, ElemBytes>(device);
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
    if (dtype == OO_DTYPE_FLOAT16) {
        if (op == OO_REDUCE_ADD) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceAddNoFtzF16,
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

        if (op == OO_REDUCE_MIN) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceMinF16,
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

        if (op == OO_REDUCE_MAX) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceMaxF16,
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
    }

    if (dtype == OO_DTYPE_BFLOAT16) {
        if (op == OO_REDUCE_ADD) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceAddBF16,
                static_cast<int>(sizeof(__nv_bfloat16))>(
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

        if (op == OO_REDUCE_MIN) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceMinBF16,
                static_cast<int>(sizeof(__nv_bfloat16))>(
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

        if (op == OO_REDUCE_MAX) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceMaxBF16,
                static_cast<int>(sizeof(__nv_bfloat16))>(
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
    }

    if (dtype == OO_DTYPE_FLOAT32) {
        if (op == OO_REDUCE_ADD) {
            return launch_reduce_op_sm90<
                comm::PipelineReduceAddF32,
                static_cast<int>(sizeof(float))>(
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
    }

    return cudaErrorInvalidValue;
}

void configure_dispatch_sm90(
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    int device) {
    if (dtype == OO_DTYPE_FLOAT16) {
        if (op == OO_REDUCE_ADD) {
            configure_reduce_op_sm90<
                comm::PipelineReduceAddNoFtzF16,
                static_cast<int>(sizeof(half))>(device);
            return;
        }

        if (op == OO_REDUCE_MIN) {
            configure_reduce_op_sm90<
                comm::PipelineReduceMinF16,
                static_cast<int>(sizeof(half))>(device);
            return;
        }

        if (op == OO_REDUCE_MAX) {
            configure_reduce_op_sm90<
                comm::PipelineReduceMaxF16,
                static_cast<int>(sizeof(half))>(device);
            return;
        }
    }

    if (dtype == OO_DTYPE_BFLOAT16) {
        if (op == OO_REDUCE_ADD) {
            configure_reduce_op_sm90<
                comm::PipelineReduceAddBF16,
                static_cast<int>(sizeof(__nv_bfloat16))>(device);
            return;
        }

        if (op == OO_REDUCE_MIN) {
            configure_reduce_op_sm90<
                comm::PipelineReduceMinBF16,
                static_cast<int>(sizeof(__nv_bfloat16))>(device);
            return;
        }

        if (op == OO_REDUCE_MAX) {
            configure_reduce_op_sm90<
                comm::PipelineReduceMaxBF16,
                static_cast<int>(sizeof(__nv_bfloat16))>(device);
            return;
        }
    }

    if (dtype == OO_DTYPE_FLOAT32) {
        if (op == OO_REDUCE_ADD) {
            configure_reduce_op_sm90<
                comm::PipelineReduceAddF32,
                static_cast<int>(sizeof(float))>(device);
            return;
        }
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
