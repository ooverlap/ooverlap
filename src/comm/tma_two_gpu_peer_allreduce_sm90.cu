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

template <typename ReduceApply, int ElemBytes>
__global__ void tma_two_gpu_allreduce_rank_kernel_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    int rank,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    int ctas_per_rank) {
    comm::window_pipeline::wait_for_collective_ready(
        local_ready_signal,
        peer_ready_signal,
        collective_epoch);

    const int cta_idx = static_cast<int>(blockIdx.x);

    if (ctas_per_rank <= 0 || cta_idx >= ctas_per_rank) {
        return;
    }

    const size_t total_bytes = count * static_cast<size_t>(ElemBytes);

    const int num_chunks =
        comm::utils::ceil_div_int64_to_int(
            total_bytes,
            TMA_TWO_GPU_PEER_CHUNK_BYTES);

    const int num_windows =
        comm::utils::window_count_for_chunks(
            num_chunks,
            TMA_TWO_GPU_PEER_WINDOW_CHUNKS);

    const comm::utils::WindowRange rank_range =
        comm::utils::rank_window_range(num_windows, rank);

    const comm::utils::WindowRange cta_range =
        comm::utils::cta_window_range(
            cta_idx,
            ctas_per_rank,
            rank_range);

    if (cta_range.begin >= cta_range.end) {
        return;
    }

    extern __shared__ uint4 shared_storage_u4[];

    unsigned char* shared_raw =
        reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore barriers[TMA_TWO_GPU_PEER_BARRIER_COUNT];

    comm::window_pipeline::run_window_range<
        TMA_TWO_GPU_PEER_REDUCE_STAGE_DEPTH,
        TMA_TWO_GPU_PEER_REDUCE_STAGE_GAP,
        TMA_TWO_GPU_PEER_CHUNK_BYTES,
        TMA_TWO_GPU_PEER_WINDOW_CHUNKS,
        ReduceApply>(
            local_in,
            peer_buf,
            total_bytes,
            cta_range.begin,
            cta_range.end,
            shared_raw,
            barriers);

    comm::window_pipeline::copy_window_range_tma<
        TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH,
        TMA_TWO_GPU_PEER_COPY_STAGE_GAP,
        TMA_TWO_GPU_PEER_CHUNK_BYTES,
        TMA_TWO_GPU_PEER_WINDOW_CHUNKS>(
            peer_buf,
            local_buf,
            total_bytes,
            cta_range.begin,
            cta_range.end,
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

    const int num_chunks =
        comm::utils::ceil_div_int64_to_int(
            total_bytes,
            TMA_TWO_GPU_PEER_CHUNK_BYTES);

    const int num_windows =
        comm::utils::window_count_for_chunks(
            num_chunks,
            TMA_TWO_GPU_PEER_WINDOW_CHUNKS);

    const int owned_windows =
        comm::utils::rank_window_count(num_windows, rank);

    const int ctas_per_rank =
        comm::utils::cta_count_for_windows(
            owned_windows,
            TMA_TWO_GPU_PEER_MAX_CTAS);

    const bool needs_rendezvous =
        local_ready_signal != nullptr &&
        peer_ready_signal != nullptr &&
        collective_epoch > 0;

    const int num_blocks =
        needs_rendezvous ? std::max(1, ctas_per_rank) : ctas_per_rank;

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
                collective_epoch,
                ctas_per_rank);

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
