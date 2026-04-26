#include "comm/tma_two_gpu_peer_allreduce_pivot_sm90.h"

#include "ooverlap/system/runtime_utils.cuh"

#include "comm/pipeline_stage.h"
#include "comm/pipeline_tma_load.h"
#include "comm/pipeline_tma_copy.h"
#include "comm/pipeline_tma_reduce.h"

#include "comm/params.h"
#include "comm/utils.h"

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

constexpr int kPivotProducerWarpThreads = 32;

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

__device__ __forceinline__ int compute_pivot_chunk_count(
    int window_chunk_count,
    int pivot_numerator,
    int pivot_denominator) {
    if (window_chunk_count <= 0 ||
        pivot_numerator <= 0 ||
        pivot_denominator <= 0) {
        return 0;
    }

    int pivot =
        static_cast<int>(
            (static_cast<long long>(window_chunk_count) *
             static_cast<long long>(pivot_numerator)) /
            static_cast<long long>(pivot_denominator));

    if (pivot < 0) {
        pivot = 0;
    }

    if (pivot > window_chunk_count) {
        pivot = window_chunk_count;
    }

    return pivot;
}

__device__ __forceinline__ comm::utils::Window make_sub_window_sm90(
    comm::utils::Window window,
    int local_start,
    int local_count) {
    comm::utils::Window out{};
    out.start_chunk = window.start_chunk + local_start;
    out.chunk_count = local_count;
    return out;
}

__device__ __forceinline__ uint4 load_copy_u128_no_allocate(
    const uint4* __restrict__ ptr) {
    uint4 value;

    asm volatile(
        "{\n"
        "  .reg .u64 addr;\n"
        "  cvta.to.global.u64 addr, %4;\n"
        "  ld.global.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [addr];\n"
        "}\n"
        : "=r"(value.x),
          "=r"(value.y),
          "=r"(value.z),
          "=r"(value.w)
        : "l"(ptr));

    return value;
}

__device__ __forceinline__ void store_copy_u128_no_allocate(
    uint4* __restrict__ ptr,
    uint4 value) {
    asm volatile(
        "{\n"
        "  .reg .u64 addr;\n"
        "  cvta.to.global.u64 addr, %0;\n"
        "  st.global.L1::no_allocate.v4.u32 [addr], {%1, %2, %3, %4};\n"
        "}\n"
        :
        : "l"(ptr),
          "r"(value.x),
          "r"(value.y),
          "r"(value.z),
          "r"(value.w)
        : "memory");
}

template <typename ReduceOp>
__device__ __forceinline__ void finish_reduce_tail_warp_sm90(
    const comm::PipelineStage* stage,
    int lane) {
    using scalar_t = typename ReduceOp::scalar_t;

    const size_t bulk_bytes = comm::pipeline_stage_bulk_bytes(stage);
    const size_t tail_bytes = comm::pipeline_stage_tail_bytes(stage);

    if (tail_bytes == 0) {
        return;
    }

    const size_t bulk_elems = bulk_bytes / sizeof(scalar_t);
    const size_t tail_elems = tail_bytes / sizeof(scalar_t);

    scalar_t* dst =
        reinterpret_cast<scalar_t*>(stage->chunk.dst);
    const scalar_t* src =
        reinterpret_cast<const scalar_t*>(stage->smem);

    for (size_t i = static_cast<size_t>(lane);
         i < tail_elems;
         i += static_cast<size_t>(kPivotProducerWarpThreads)) {
        const size_t idx = bulk_elems + i;
        dst[idx] = ReduceOp::apply_tail(dst[idx], src[idx]);
    }
}

__device__ __forceinline__ void finish_copy_tail_warp_sm90(
    const comm::PipelineStage* stage,
    int lane) {
    const size_t bulk_bytes = comm::pipeline_stage_bulk_bytes(stage);
    const size_t tail_bytes = comm::pipeline_stage_tail_bytes(stage);

    if (tail_bytes == 0) {
        return;
    }

    unsigned char* dst = stage->chunk.dst + bulk_bytes;
    const unsigned char* src = stage->smem + bulk_bytes;

    for (size_t i = static_cast<size_t>(lane);
         i < tail_bytes;
         i += static_cast<size_t>(kPivotProducerWarpThreads)) {
        dst[i] = src[i];
    }
}

__device__ __forceinline__ void fast_copy_chunk_u128_no_allocate_sm90(
    const unsigned char* __restrict__ src_bytes,
    unsigned char* __restrict__ dst_bytes,
    size_t bytes,
    int copy_thread,
    int copy_threads) {
    const size_t num_u128 = bytes / sizeof(uint4);

    const uint4* __restrict__ src_u128 =
        reinterpret_cast<const uint4*>(src_bytes);
    uint4* __restrict__ dst_u128 =
        reinterpret_cast<uint4*>(dst_bytes);

    for (size_t i = static_cast<size_t>(copy_thread);
         i < num_u128;
         i += static_cast<size_t>(copy_threads)) {
        const uint4 value = load_copy_u128_no_allocate(src_u128 + i);
        store_copy_u128_no_allocate(dst_u128 + i, value);
    }

    const size_t tail_begin = num_u128 * sizeof(uint4);

    for (size_t i = tail_begin + static_cast<size_t>(copy_thread);
         i < bytes;
         i += static_cast<size_t>(copy_threads)) {
        dst_bytes[i] = src_bytes[i];
    }
}

template <typename ReduceOp>
__device__ void reduce_window_and_signal_pivot_sm90(
    const unsigned char* src_bytes,
    unsigned char* dst_bytes,
    comm::utils::Window window,
    size_t total_bytes,
    int pivot_count,
    volatile int* reduced_pivot_count,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    const unsigned int mask = 0xffffffffu;
    const int lane = static_cast<int>(threadIdx.x) & 31;

    comm::PipelineTMALoad load{};

    for (int iter = 0; iter < window.chunk_count; ++iter) {
        const int chunk = window.start_chunk + iter;

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
            stage_ptr(shared_raw, 0),
            &barriers[0]);

        if (lane == 0) {
            load.issue(&stage);
            load.wait_ready(&stage);

            const size_t bulk_bytes =
                comm::pipeline_stage_bulk_bytes(&stage);

            if (bulk_bytes != 0) {
                ReduceOp::issue_bulk(
                    stage.chunk.dst,
                    stage.smem,
                    static_cast<uint32_t>(bulk_bytes));
            }
        }

        __syncwarp(mask);

        finish_reduce_tail_warp_sm90<ReduceOp>(&stage, lane);

        __syncwarp(mask);

        if (lane == 0) {
            const size_t bulk_bytes =
                comm::pipeline_stage_bulk_bytes(&stage);

            if (bulk_bytes != 0) {
                tma::reduce_async_wait<0>();
            }

            __threadfence();

            if (iter < pivot_count) {
                reduced_pivot_count[0] = iter + 1;
            }
        }

        __syncwarp(mask);
    }

    if (lane == 0) {
        __threadfence();

        if (reduced_pivot_count[0] < pivot_count) {
            reduced_pivot_count[0] = pivot_count;
        }
    }

    __syncwarp(mask);
}

template <int StageDepth, int FillDepth>
__device__ void copy_window_tma_warp_sm90(
    const unsigned char* src_bytes,
    unsigned char* dst_bytes,
    comm::utils::Window window,
    size_t total_bytes,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    static_assert(StageDepth > 0, "StageDepth must be > 0");
    static_assert(FillDepth > 0, "FillDepth must be > 0");
    static_assert(FillDepth <= StageDepth, "FillDepth must be <= StageDepth");

    if (window.chunk_count <= 0) {
        return;
    }

    const unsigned int mask = 0xffffffffu;
    const int lane = static_cast<int>(threadIdx.x) & 31;

    comm::PipelineTMALoad load{};

    using CopyApply =
        comm::PipelineTMACopy<StageDepth, FillDepth>;

    CopyApply apply{};

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

        if (lane == 0) {
            load.issue(&stage);
        }

        __syncwarp(mask);
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

        if (lane == 0) {
            load.wait_ready(&cur_stage);
        }

        __syncwarp(mask);

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

            if (lane == 0) {
                if (iter >= FillDepth) {
                    apply.wait_before_stage_reuse();
                }

                load.issue(&future_stage);
            }
        }

        __syncwarp(mask);

        if (lane == 0) {
            apply.issue_bulk(&cur_stage);
        }

        finish_copy_tail_warp_sm90(&cur_stage, lane);

        __syncwarp(mask);
    }

    if (lane == 0) {
        apply.wait_complete();
        __threadfence_system();
    }

    __syncwarp(mask);
}

template <typename ReduceOp, int ElemBytes>
__global__ void tma_two_gpu_allreduce_rank_pivot_kernel_sm90(
    const void* local_in,
    void* local_buf,
    void* peer_buf,
    size_t count,
    int rank,
    int* local_ready_signal,
    const int* peer_ready_signal,
    int collective_epoch,
    int pivot_numerator,
    int pivot_denominator) {
    wait_for_collective_ready_sm90(
        local_ready_signal,
        peer_ready_signal,
        collective_epoch);

    const size_t total_bytes = count * static_cast<size_t>(ElemBytes);

    const int num_chunks =
        comm::utils::ceil_div_int64_to_int(
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
    __shared__ volatile int reduced_pivot_count;

    if (threadIdx.x == 0) {
        reduced_pivot_count = 0;
    }

    __syncthreads();

    const unsigned char* local_in_bytes =
        reinterpret_cast<const unsigned char*>(local_in);

    unsigned char* local_buf_bytes =
        reinterpret_cast<unsigned char*>(local_buf);

    unsigned char* peer_buf_bytes =
        reinterpret_cast<unsigned char*>(peer_buf);

    (void)local_in_bytes;

    const int pivot_count =
        compute_pivot_chunk_count(
            window.chunk_count,
            pivot_numerator,
            pivot_denominator);

    if (threadIdx.x < kPivotProducerWarpThreads) {
        /*
         * Same direction as the current fast version:
         *
         *     reduce peer_buf -> local_buf
         *
         * For chunks [0, pivot_count), the copy warps copy local_buf -> peer_buf
         * as soon as the producer warp marks each chunk ready.
         *
         * For chunks [pivot_count, window.chunk_count), producer warp performs
         * TMA copy after the whole reduce window finishes.
         */
        reduce_window_and_signal_pivot_sm90<ReduceOp>(
            peer_buf_bytes,
            local_buf_bytes,
            window,
            total_bytes,
            pivot_count,
            &reduced_pivot_count,
            shared_raw,
            barriers);

        const int rest_count = window.chunk_count - pivot_count;

        if (rest_count > 0) {
            const comm::utils::Window rest_window =
                make_sub_window_sm90(
                    window,
                    pivot_count,
                    rest_count);

            copy_window_tma_warp_sm90<
                TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH,
                TMA_TWO_GPU_PEER_COPY_STAGE_GAP>(
                    local_buf_bytes,
                    peer_buf_bytes,
                    rest_window,
                    total_bytes,
                    shared_raw,
                    barriers);
        }

        return;
    }

    const int copy_thread =
        static_cast<int>(threadIdx.x) - kPivotProducerWarpThreads;

    const int copy_threads =
        static_cast<int>(blockDim.x) - kPivotProducerWarpThreads;

    if (copy_threads <= 0 || pivot_count <= 0) {
        return;
    }

    for (int iter = 0; iter < pivot_count; ++iter) {
        while (reduced_pivot_count <= iter) {
#if defined(__CUDA_ARCH__)
            __nanosleep(64);
#endif
        }

        const int chunk = window.start_chunk + iter;

        const size_t offset =
            static_cast<size_t>(chunk) * TMA_TWO_GPU_PEER_CHUNK_BYTES;

        const size_t bytes =
            comm::utils::min_sz(
                TMA_TWO_GPU_PEER_CHUNK_BYTES,
                total_bytes - offset);

        fast_copy_chunk_u128_no_allocate_sm90(
            local_buf_bytes + offset,
            peer_buf_bytes + offset,
            bytes,
            copy_thread,
            copy_threads);
    }

    __threadfence_system();
}

template <typename ReduceOp, int ElemBytes>
void configure_pivot_kernel_once_for(int device) {
    struct CacheEntry {
        bool configured = false;
        size_t dynamic_smem_bytes = 0;
    };

    static std::mutex mutex;
    static std::unordered_map<int, CacheEntry> cache;

    const size_t dynamic_smem_bytes =
        TMA_TWO_GPU_PEER_DYNAMIC_SHARED_BYTES;

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

    if (total_smem_bytes >
        static_cast<size_t>(prop.sharedMemPerBlockOptin)) {
        throw std::runtime_error(
            "tma_two_gpu_peer_allreduce_pivot_configure_kernel_once: requested shared memory exceeds opt-in limit");
    }

    if (total_smem_bytes >
        static_cast<size_t>(prop.sharedMemPerBlock)) {
        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                tma_two_gpu_allreduce_rank_pivot_kernel_sm90<
                    ReduceOp,
                    ElemBytes>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(dynamic_smem_bytes)),
            "cudaFuncSetAttribute(MaxDynamicSharedMemorySize pivot)");

        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                tma_two_gpu_allreduce_rank_pivot_kernel_sm90<
                    ReduceOp,
                    ElemBytes>,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                100),
            "cudaFuncSetAttribute(PreferredSharedMemoryCarveout pivot)");
    }

    cache[device] = {true, dynamic_smem_bytes};
}

template <typename ReduceOp, int ElemBytes>
cudaError_t launch_rank_pivot_kernel_sm90(
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
    int collective_epoch,
    int pivot_numerator,
    int pivot_denominator) {
    const int device = (rank == 0) ? dev0 : dev1;

    const size_t total_bytes = count * static_cast<size_t>(ElemBytes);

    const int num_chunks =
        comm::utils::ceil_div_int64_to_int(
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

    configure_pivot_kernel_once_for<ReduceOp, ElemBytes>(device);

    system::runtime::set_device(device);

    tma_two_gpu_allreduce_rank_pivot_kernel_sm90<
        ReduceOp,
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
                pivot_numerator,
                pivot_denominator);

    return cudaGetLastError();
}

template <typename ReduceOp, int ElemBytes>
cudaError_t launch_reduce_op_pivot_sm90(
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
    int collective_epoch,
    int pivot_numerator,
    int pivot_denominator) {
    return launch_rank_pivot_kernel_sm90<ReduceOp, ElemBytes>(
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
        collective_epoch,
        pivot_numerator,
        pivot_denominator);
}

cudaError_t dispatch_rank_pivot_kernel_sm90(
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
    int collective_epoch,
    int pivot_numerator,
    int pivot_denominator) {
    if (dtype == OO_DTYPE_FLOAT16) {
        if (op == OO_REDUCE_ADD) {
            return launch_reduce_op_pivot_sm90<
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
                    collective_epoch,
                    pivot_numerator,
                    pivot_denominator);
        }

        if (op == OO_REDUCE_MIN) {
            return launch_reduce_op_pivot_sm90<
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
                    collective_epoch,
                    pivot_numerator,
                    pivot_denominator);
        }

        if (op == OO_REDUCE_MAX) {
            return launch_reduce_op_pivot_sm90<
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
                    collective_epoch,
                    pivot_numerator,
                    pivot_denominator);
        }
    }

    if (dtype == OO_DTYPE_BFLOAT16) {
        if (op == OO_REDUCE_ADD) {
            return launch_reduce_op_pivot_sm90<
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
                    collective_epoch,
                    pivot_numerator,
                    pivot_denominator);
        }

        if (op == OO_REDUCE_MIN) {
            return launch_reduce_op_pivot_sm90<
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
                    collective_epoch,
                    pivot_numerator,
                    pivot_denominator);
        }

        if (op == OO_REDUCE_MAX) {
            return launch_reduce_op_pivot_sm90<
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
                    collective_epoch,
                    pivot_numerator,
                    pivot_denominator);
        }
    }

    if (dtype == OO_DTYPE_FLOAT32) {
        if (op == OO_REDUCE_ADD) {
            return launch_reduce_op_pivot_sm90<
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
                    collective_epoch,
                    pivot_numerator,
                    pivot_denominator);
        }
    }

    return cudaErrorInvalidValue;
}

} // namespace

cudaError_t enqueue_tma_two_gpu_peer_allreduce_rank_pivot_sm90(
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
    int collective_epoch,
    int pivot_numerator,
    int pivot_denominator) {
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

    if (pivot_denominator <= 0) {
        return cudaErrorInvalidValue;
    }

    return dispatch_rank_pivot_kernel_sm90(
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
        collective_epoch,
        pivot_numerator,
        pivot_denominator);
}

} // namespace ooverlap
