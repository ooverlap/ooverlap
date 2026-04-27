#include "test/tma_bandwidth_experiment_sm90.h"

#include "comm/fast_gmem_copy.cuh"
#include "comm/params.h"
#include "comm/pipeline_stage.h"
#include "comm/pipeline_tma_copy.h"
#include "comm/pipeline_tma_load.h"
#include "comm/pipeline_tma_reduce.h"

#include "ooverlap/system/peer_buffer.cuh"
#include "ooverlap/system/runtime_utils.cuh"

#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#define OOVERLAP_EXPERIMENT_NCCL_CHECK(cmd)                                      \
    do {                                                                         \
        ncclResult_t result__ = (cmd);                                           \
        if (result__ != ncclSuccess) {                                           \
            throw std::runtime_error(                                            \
                std::string("NCCL error: ") + ncclGetErrorString(result__));     \
        }                                                                        \
    } while (0)

namespace ooverlap {
namespace {

constexpr int kExperimentThreads = TMA_TWO_GPU_PEER_DEFAULT_THREADS;
constexpr int kExperimentGmemThreads = 1024;

constexpr int kExperimentChunkBytes = TMA_TWO_GPU_PEER_CHUNK_BYTES;
constexpr int kExperimentStageDepth = TMA_TWO_GPU_PEER_COPY_STAGE_DEPTH;
constexpr int kExperimentFillDepth = TMA_TWO_GPU_PEER_COPY_STAGE_GAP;
constexpr int kExperimentBarrierCount = TMA_TWO_GPU_PEER_BARRIER_COUNT;

constexpr size_t kExperimentTmaSmemBytes =
    static_cast<size_t>(kExperimentStageDepth) * kExperimentChunkBytes;

constexpr size_t kExperimentMemAsyncSmemBytes = kExperimentChunkBytes;

static_assert(kExperimentStageDepth > 0, "stage depth must be positive");
static_assert(kExperimentFillDepth > 0, "fill depth must be positive");
static_assert(kExperimentFillDepth <= kExperimentStageDepth,
              "fill depth must be <= stage depth");

struct ChunkRange {
    int start_chunk = 0;
    int chunk_count = 0;
};

enum class ExperimentMethod {
    kTmaCopy = 0,
    kTmaReduce = 1,
    kMemAsyncCopy = 2,
    kGmemCopyU32 = 3,
    kNcclSendRecv = 4,
    kGmemCopyU64 = 5,
    kGmemCopyU128 = 6,
    kGmemAddF16U128 = 7,
};

enum ExperimentScenarioId {
    kScenarioLocalToPeer = 0,
    kScenarioPeerToLocal = 1,
    kScenarioSameDev = 2,
};

__host__ __device__ __forceinline__ int ceil_div_int64_to_int(
    size_t x,
    size_t y) {
    return static_cast<int>((x + y - 1) / y);
}

__host__ __device__ __forceinline__ size_t min_sz(
    size_t a,
    size_t b) {
    return (a < b) ? a : b;
}

__host__ __device__ __forceinline__ ChunkRange make_block_chunk_range(
    int block_idx,
    int num_blocks,
    int num_chunks) {
    const int chunks_per_block =
        (num_chunks + num_blocks - 1) / num_blocks;

    ChunkRange range{};
    range.start_chunk = block_idx * chunks_per_block;
    range.chunk_count = chunks_per_block;

    if (range.start_chunk >= num_chunks) {
        range.chunk_count = 0;
        return range;
    }

    const int remaining = num_chunks - range.start_chunk;
    if (range.chunk_count > remaining) {
        range.chunk_count = remaining;
    }

    return range;
}

__device__ __forceinline__ unsigned char* stage_ptr(
    unsigned char* shared_raw,
    int stage) {
    return shared_raw + static_cast<size_t>(stage) * kExperimentChunkBytes;
}

template <int StageDepth, int FillDepth, typename Apply>
__device__ void run_window_pipeline(
    const unsigned char* src_bytes,
    unsigned char* dst_bytes,
    ChunkRange range,
    size_t total_bytes,
    unsigned char* shared_raw,
    sync::semaphore* barriers) {
    comm::PipelineTMALoad load{};
    Apply apply{};

    for (int warm = 0; warm < FillDepth; ++warm) {
        if (warm >= range.chunk_count) {
            break;
        }

        const int chunk = range.start_chunk + warm;
        const int slot = warm;

        const size_t offset =
            static_cast<size_t>(chunk) * kExperimentChunkBytes;
        const size_t bytes =
            min_sz(kExperimentChunkBytes, total_bytes - offset);

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

    for (int iter = 0; iter < range.chunk_count; ++iter) {
        const int chunk = range.start_chunk + iter;
        const int cur_slot = iter % StageDepth;

        const size_t offset =
            static_cast<size_t>(chunk) * kExperimentChunkBytes;
        const size_t bytes =
            min_sz(kExperimentChunkBytes, total_bytes - offset);

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
        if (future_iter < range.chunk_count) {
            const int future_chunk = range.start_chunk + future_iter;
            const int future_slot = future_iter % StageDepth;

            const size_t future_offset =
                static_cast<size_t>(future_chunk) * kExperimentChunkBytes;
            const size_t future_bytes =
                min_sz(kExperimentChunkBytes, total_bytes - future_offset);

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

__global__ void tma_pipeline_copy_kernel(
    const void* src,
    void* dst,
    size_t total_bytes,
    int num_chunks) {
    const ChunkRange range =
        make_block_chunk_range(
            static_cast<int>(blockIdx.x),
            static_cast<int>(gridDim.x),
            num_chunks);

    if (range.chunk_count <= 0) {
        return;
    }

    extern __shared__ uint4 shared_storage_u4[];
    unsigned char* shared_raw =
        reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore barriers[kExperimentBarrierCount];

    using CopyApply =
        comm::PipelineTMACopy<kExperimentStageDepth, kExperimentFillDepth>;

    run_window_pipeline<
        kExperimentStageDepth,
        kExperimentFillDepth,
        CopyApply>(
            reinterpret_cast<const unsigned char*>(src),
            reinterpret_cast<unsigned char*>(dst),
            range,
            total_bytes,
            shared_raw,
            barriers);
}

__global__ void tma_pipeline_reduce_add_f16_kernel(
    const void* src,
    void* dst,
    size_t total_bytes,
    int num_chunks) {
    const ChunkRange range =
        make_block_chunk_range(
            static_cast<int>(blockIdx.x),
            static_cast<int>(gridDim.x),
            num_chunks);

    if (range.chunk_count <= 0) {
        return;
    }

    extern __shared__ uint4 shared_storage_u4[];
    unsigned char* shared_raw =
        reinterpret_cast<unsigned char*>(shared_storage_u4);

    __shared__ sync::semaphore barriers[kExperimentBarrierCount];

    using ReduceApply =
        comm::PipelineTMAReduce<
            kExperimentStageDepth,
            kExperimentFillDepth,
            comm::PipelineReduceAddNoFtzF16>;

    run_window_pipeline<
        kExperimentStageDepth,
        kExperimentFillDepth,
        ReduceApply>(
            reinterpret_cast<const unsigned char*>(src),
            reinterpret_cast<unsigned char*>(dst),
            range,
            total_bytes,
            shared_raw,
            barriers);
}

__global__ void mem_async_copy_kernel(
    const void* src,
    void* dst,
    size_t total_bytes,
    int num_chunks) {
    namespace cg = cooperative_groups;

    const ChunkRange range =
        make_block_chunk_range(
            static_cast<int>(blockIdx.x),
            static_cast<int>(gridDim.x),
            num_chunks);

    if (range.chunk_count <= 0) {
        return;
    }

    extern __shared__ unsigned char smem[];

    auto block = cg::this_thread_block();

    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope_block,
        1> pipeline_state;

    auto pipe = cuda::make_pipeline(block, &pipeline_state);

    const unsigned char* src_bytes =
        reinterpret_cast<const unsigned char*>(src);
    unsigned char* dst_bytes =
        reinterpret_cast<unsigned char*>(dst);

    for (int iter = 0; iter < range.chunk_count; ++iter) {
        const int chunk = range.start_chunk + iter;
        const size_t offset =
            static_cast<size_t>(chunk) * kExperimentChunkBytes;
        const size_t bytes =
            min_sz(kExperimentChunkBytes, total_bytes - offset);

        pipe.producer_acquire();
        cuda::memcpy_async(
            block,
            smem,
            src_bytes + offset,
            bytes,
            pipe);
        pipe.producer_commit();

        pipe.consumer_wait();

        for (size_t i = threadIdx.x; i < bytes; i += blockDim.x) {
            dst_bytes[offset + i] = smem[i];
        }

        pipe.consumer_release();
        block.sync();
    }

    __threadfence_system();
}

void configure_one_kernel(
    const void* kernel,
    size_t dynamic_smem_bytes,
    int device,
    const char* name) {
    system::runtime::set_device(device);

    cudaDeviceProp prop{};
    system::runtime::check_cuda(
        cudaGetDeviceProperties(&prop, device),
        "cudaGetDeviceProperties");

    const size_t static_smem_bytes =
        static_cast<size_t>(kExperimentBarrierCount) *
        sizeof(sync::semaphore);

    const size_t total_smem_bytes =
        dynamic_smem_bytes + static_smem_bytes;

    if (total_smem_bytes > static_cast<size_t>(prop.sharedMemPerBlockOptin)) {
        throw std::runtime_error(
            std::string(name) +
            ": requested shared memory exceeds opt-in limit");
    }

    if (dynamic_smem_bytes > static_cast<size_t>(prop.sharedMemPerBlock)) {
        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(dynamic_smem_bytes)),
            "cudaFuncSetAttribute(MaxDynamicSharedMemorySize)");
    }

    system::runtime::check_cuda(
        cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            100),
        "cudaFuncSetAttribute(PreferredSharedMemoryCarveout)");
}

void configure_experiment_kernels_once(int device) {
    static bool configured[16] = {};

    if (device >= 0 && device < 16 && configured[device]) {
        return;
    }

    configure_one_kernel(
        reinterpret_cast<const void*>(tma_pipeline_copy_kernel),
        kExperimentTmaSmemBytes,
        device,
        "tma_pipeline_copy_kernel");

    configure_one_kernel(
        reinterpret_cast<const void*>(tma_pipeline_reduce_add_f16_kernel),
        kExperimentTmaSmemBytes,
        device,
        "tma_pipeline_reduce_add_f16_kernel");

    configure_one_kernel(
        reinterpret_cast<const void*>(mem_async_copy_kernel),
        kExperimentMemAsyncSmemBytes,
        device,
        "mem_async_copy_kernel");

    configure_one_kernel(
        reinterpret_cast<const void*>(
            comm::fast_copy::gmem_copy_coalesced_kernel<uint32_t>),
        0,
        device,
        "gmem_copy_coalesced_kernel<uint32_t>");

    configure_one_kernel(
        reinterpret_cast<const void*>(
            comm::fast_copy::gmem_copy_coalesced_kernel<uint2>),
        0,
        device,
        "gmem_copy_coalesced_kernel<uint2>");

    configure_one_kernel(
        reinterpret_cast<const void*>(
            comm::fast_copy::gmem_copy_coalesced_kernel<uint4>),
        0,
        device,
        "gmem_copy_coalesced_kernel<uint4>");

    configure_one_kernel(
        reinterpret_cast<const void*>(
            comm::fast_copy::gmem_add_f16_u128_kernel<4>),
        0,
        device,
        "gmem_add_f16_u128_kernel<4>");

    if (device >= 0 && device < 16) {
        configured[device] = true;
    }
}

cudaError_t launch_method(
    ExperimentMethod method,
    const void* src,
    void* dst,
    size_t bytes,
    int num_blocks,
    cudaStream_t stream) {
    const int num_chunks =
        ceil_div_int64_to_int(bytes, kExperimentChunkBytes);

    switch (method) {
        case ExperimentMethod::kTmaCopy:
            tma_pipeline_copy_kernel<<<
                num_blocks,
                kExperimentThreads,
                kExperimentTmaSmemBytes,
                stream>>>(
                    src,
                    dst,
                    bytes,
                    num_chunks);
            return cudaGetLastError();

        case ExperimentMethod::kTmaReduce:
            tma_pipeline_reduce_add_f16_kernel<<<
                num_blocks,
                kExperimentThreads,
                kExperimentTmaSmemBytes,
                stream>>>(
                    src,
                    dst,
                    bytes,
                    num_chunks);
            return cudaGetLastError();

        case ExperimentMethod::kMemAsyncCopy:
            mem_async_copy_kernel<<<
                num_blocks,
                kExperimentThreads,
                kExperimentMemAsyncSmemBytes,
                stream>>>(
                    src,
                    dst,
                    bytes,
                    num_chunks);
            return cudaGetLastError();

        case ExperimentMethod::kGmemCopyU32:
            comm::fast_copy::gmem_copy_coalesced_kernel<uint32_t><<<
                num_blocks,
                kExperimentGmemThreads,
                0,
                stream>>>(
                    src,
                    dst,
                    bytes);
            return cudaGetLastError();

        case ExperimentMethod::kGmemCopyU64:
            comm::fast_copy::gmem_copy_coalesced_kernel<uint2><<<
                num_blocks,
                kExperimentGmemThreads,
                0,
                stream>>>(
                    src,
                    dst,
                    bytes);
            return cudaGetLastError();

        case ExperimentMethod::kGmemCopyU128:
            comm::fast_copy::gmem_copy_coalesced_kernel<uint4><<<
                num_blocks,
                kExperimentGmemThreads,
                0,
                stream>>>(
                    src,
                    dst,
                    bytes);
            return cudaGetLastError();

        case ExperimentMethod::kGmemAddF16U128:
            comm::fast_copy::gmem_add_f16_u128_kernel<4><<<
                num_blocks,
                kExperimentGmemThreads,
                0,
                stream>>>(
                    src,
                    dst,
                    bytes);
            return cudaGetLastError();

        default:
            return cudaErrorInvalidValue;
    }
}

double benchmark_one_method_ms(
    ExperimentMethod method,
    const void* src,
    void* dst,
    size_t bytes,
    int num_blocks,
    int kernel_device,
    cudaStream_t stream,
    int iters,
    int warmup) {
    system::runtime::set_device(kernel_device);

    for (int i = 0; i < warmup; ++i) {
        system::runtime::check_cuda(
            launch_method(method, src, dst, bytes, num_blocks, stream),
            "launch warmup");
    }

    system::runtime::check_cuda(
        cudaStreamSynchronize(stream),
        "cudaStreamSynchronize(warmup)");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    system::runtime::check_cuda(
        cudaEventCreate(&start),
        "cudaEventCreate(start)");
    system::runtime::check_cuda(
        cudaEventCreate(&stop),
        "cudaEventCreate(stop)");

    system::runtime::check_cuda(
        cudaEventRecord(start, stream),
        "cudaEventRecord(start)");

    for (int i = 0; i < iters; ++i) {
        system::runtime::check_cuda(
            launch_method(method, src, dst, bytes, num_blocks, stream),
            "launch timed");
    }

    system::runtime::check_cuda(
        cudaEventRecord(stop, stream),
        "cudaEventRecord(stop)");

    system::runtime::check_cuda(
        cudaEventSynchronize(stop),
        "cudaEventSynchronize(stop)");

    float total_ms = 0.0f;
    system::runtime::check_cuda(
        cudaEventElapsedTime(&total_ms, start, stop),
        "cudaEventElapsedTime");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return static_cast<double>(total_ms) / static_cast<double>(iters);
}

double benchmark_nccl_sendrecv_ms(
    const void* src,
    void* dst,
    size_t bytes,
    int src_device,
    int dst_device,
    int src_rank,
    int dst_rank,
    ncclComm_t src_comm,
    ncclComm_t dst_comm,
    int iters,
    int warmup) {
    cudaStream_t src_stream =
        system::runtime::create_stream_on_device(src_device);
    cudaStream_t dst_stream =
        system::runtime::create_stream_on_device(dst_device);

    cudaEvent_t src_start = nullptr;
    cudaEvent_t src_stop = nullptr;
    cudaEvent_t dst_start = nullptr;
    cudaEvent_t dst_stop = nullptr;

    try {
        auto launch_once = [&]() {
            OOVERLAP_EXPERIMENT_NCCL_CHECK(ncclGroupStart());

            system::runtime::set_device(src_device);
            OOVERLAP_EXPERIMENT_NCCL_CHECK(
                ncclSend(
                    src,
                    bytes,
                    ncclUint8,
                    dst_rank,
                    src_comm,
                    src_stream));

            system::runtime::set_device(dst_device);
            OOVERLAP_EXPERIMENT_NCCL_CHECK(
                ncclRecv(
                    dst,
                    bytes,
                    ncclUint8,
                    src_rank,
                    dst_comm,
                    dst_stream));

            OOVERLAP_EXPERIMENT_NCCL_CHECK(ncclGroupEnd());
        };

        for (int i = 0; i < warmup; ++i) {
            launch_once();
        }

        system::runtime::sync_stream_on_device(
            src_device,
            src_stream,
            "sync NCCL send warmup");
        system::runtime::sync_stream_on_device(
            dst_device,
            dst_stream,
            "sync NCCL recv warmup");

        system::runtime::set_device(src_device);
        system::runtime::check_cuda(
            cudaEventCreate(&src_start),
            "cudaEventCreate(src_start)");
        system::runtime::check_cuda(
            cudaEventCreate(&src_stop),
            "cudaEventCreate(src_stop)");
        system::runtime::check_cuda(
            cudaEventRecord(src_start, src_stream),
            "cudaEventRecord(src_start)");

        system::runtime::set_device(dst_device);
        system::runtime::check_cuda(
            cudaEventCreate(&dst_start),
            "cudaEventCreate(dst_start)");
        system::runtime::check_cuda(
            cudaEventCreate(&dst_stop),
            "cudaEventCreate(dst_stop)");
        system::runtime::check_cuda(
            cudaEventRecord(dst_start, dst_stream),
            "cudaEventRecord(dst_start)");

        for (int i = 0; i < iters; ++i) {
            launch_once();
        }

        system::runtime::set_device(src_device);
        system::runtime::check_cuda(
            cudaEventRecord(src_stop, src_stream),
            "cudaEventRecord(src_stop)");

        system::runtime::set_device(dst_device);
        system::runtime::check_cuda(
            cudaEventRecord(dst_stop, dst_stream),
            "cudaEventRecord(dst_stop)");

        system::runtime::set_device(src_device);
        system::runtime::check_cuda(
            cudaEventSynchronize(src_stop),
            "cudaEventSynchronize(src_stop)");

        system::runtime::set_device(dst_device);
        system::runtime::check_cuda(
            cudaEventSynchronize(dst_stop),
            "cudaEventSynchronize(dst_stop)");

        float src_ms = 0.0f;
        float dst_ms = 0.0f;

        system::runtime::set_device(src_device);
        system::runtime::check_cuda(
            cudaEventElapsedTime(&src_ms, src_start, src_stop),
            "cudaEventElapsedTime(src)");

        system::runtime::set_device(dst_device);
        system::runtime::check_cuda(
            cudaEventElapsedTime(&dst_ms, dst_start, dst_stop),
            "cudaEventElapsedTime(dst)");

        system::runtime::set_device(src_device);
        cudaEventDestroy(src_start);
        cudaEventDestroy(src_stop);

        system::runtime::set_device(dst_device);
        cudaEventDestroy(dst_start);
        cudaEventDestroy(dst_stop);

        system::runtime::destroy_stream_on_device(src_device, src_stream);
        system::runtime::destroy_stream_on_device(dst_device, dst_stream);

        return static_cast<double>(std::max(src_ms, dst_ms)) /
               static_cast<double>(iters);
    } catch (...) {
        if (src_start != nullptr) {
            system::runtime::set_device(src_device);
            cudaEventDestroy(src_start);
        }
        if (src_stop != nullptr) {
            system::runtime::set_device(src_device);
            cudaEventDestroy(src_stop);
        }
        if (dst_start != nullptr) {
            system::runtime::set_device(dst_device);
            cudaEventDestroy(dst_start);
        }
        if (dst_stop != nullptr) {
            system::runtime::set_device(dst_device);
            cudaEventDestroy(dst_stop);
        }

        system::runtime::destroy_stream_on_device(src_device, src_stream);
        system::runtime::destroy_stream_on_device(dst_device, dst_stream);

        throw;
    }
}

void memset_buffer(
    int device,
    void* ptr,
    int value,
    size_t bytes,
    cudaStream_t stream) {
    system::runtime::set_device(device);
    system::runtime::check_cuda(
        cudaMemsetAsync(ptr, value, bytes, stream),
        "cudaMemsetAsync");
}

void add_result(
    std::vector<std::map<std::string, double>>& results,
    int scenario_id,
    ExperimentMethod method,
    int src_device,
    int dst_device,
    int kernel_device,
    size_t bytes,
    int num_blocks,
    double latency_ms) {
    const double seconds = latency_ms * 1.0e-3;
    const double gbps =
        seconds > 0.0
            ? static_cast<double>(bytes) / seconds / 1.0e9
            : 0.0;

    std::map<std::string, double> row;
    row["scenario"] = static_cast<double>(scenario_id);
    row["method"] = static_cast<double>(static_cast<int>(method));
    row["src_device"] = static_cast<double>(src_device);
    row["dst_device"] = static_cast<double>(dst_device);
    row["kernel_device"] = static_cast<double>(kernel_device);
    row["bytes"] = static_cast<double>(bytes);
    row["num_blocks"] = static_cast<double>(num_blocks);
    row["latency_ms"] = latency_ms;
    row["gbps"] = gbps;

    results.push_back(row);
}

void run_case_for_size(
    std::vector<std::map<std::string, double>>& results,
    int scenario_id,
    int src_device,
    int dst_device,
    int kernel_device,
    size_t bytes,
    int num_blocks,
    int iters,
    int warmup,
    bool include_mem_async,
    bool include_nccl,
    ncclComm_t* comms) {
    std::vector<int> access_devices;
    auto add_access_device = [&](int device) {
        for (int d : access_devices) {
            if (d == device) {
                return;
            }
        }
        access_devices.push_back(device);
    };

    add_access_device(src_device);
    add_access_device(dst_device);
    add_access_device(kernel_device);

    system::mapped_peer_buffer src_buf =
        system::alloc_peer_visible_buffer(bytes, src_device, access_devices);
    system::mapped_peer_buffer dst_buf =
        system::alloc_peer_visible_buffer(bytes, dst_device, access_devices);

    cudaStream_t stream = nullptr;

    try {
        configure_experiment_kernels_once(kernel_device);

        stream = system::runtime::create_stream_on_device(kernel_device);

        memset_buffer(dst_device, dst_buf.ptr, 0, bytes, stream);

        cudaStream_t src_stream =
            system::runtime::create_stream_on_device(src_device);
        memset_buffer(src_device, src_buf.ptr, 1, bytes, src_stream);
        system::runtime::sync_stream_on_device(
            src_device,
            src_stream,
            "sync source memset");
        system::runtime::destroy_stream_on_device(src_device, src_stream);

        system::runtime::sync_stream_on_device(
            dst_device,
            stream,
            "sync destination memset");

        std::vector<ExperimentMethod> methods = {
            ExperimentMethod::kTmaCopy,
            ExperimentMethod::kTmaReduce,
            ExperimentMethod::kGmemAddF16U128,
            ExperimentMethod::kGmemCopyU32,
            ExperimentMethod::kGmemCopyU64,
            ExperimentMethod::kGmemCopyU128,
        };

        if (include_mem_async) {
            methods.push_back(ExperimentMethod::kMemAsyncCopy);
        }

        for (ExperimentMethod method : methods) {
            const double latency_ms =
                benchmark_one_method_ms(
                    method,
                    src_buf.ptr,
                    dst_buf.ptr,
                    bytes,
                    num_blocks,
                    kernel_device,
                    stream,
                    iters,
                    warmup);

            add_result(
                results,
                scenario_id,
                method,
                src_device,
                dst_device,
                kernel_device,
                bytes,
                num_blocks,
                latency_ms);
        }

        if (include_nccl && comms != nullptr && src_device != dst_device) {
            const double latency_ms =
                benchmark_nccl_sendrecv_ms(
                    src_buf.ptr,
                    dst_buf.ptr,
                    bytes,
                    src_device,
                    dst_device,
                    0,
                    1,
                    comms[0],
                    comms[1],
                    iters,
                    warmup);

            add_result(
                results,
                scenario_id,
                ExperimentMethod::kNcclSendRecv,
                src_device,
                dst_device,
                -1,
                bytes,
                num_blocks,
                latency_ms);
        }

        system::runtime::destroy_stream_on_device(kernel_device, stream);
        system::free_peer_visible_buffer(src_buf);
        system::free_peer_visible_buffer(dst_buf);
    } catch (...) {
        if (stream != nullptr) {
            system::runtime::destroy_stream_on_device(kernel_device, stream);
        }

        system::free_peer_visible_buffer(src_buf);
        system::free_peer_visible_buffer(dst_buf);
        throw;
    }
}

void run_all_cases_for_size(
    std::vector<std::map<std::string, double>>& results,
    size_t bytes,
    int num_blocks,
    int iters,
    int warmup,
    int dev0,
    int dev1,
    bool include_mem_async,
    bool include_nccl,
    ncclComm_t* comms) {
    if (dev0 != dev1) {
        run_case_for_size(
            results,
            kScenarioLocalToPeer,
            dev0,
            dev1,
            dev0,
            bytes,
            num_blocks,
            iters,
            warmup,
            include_mem_async,
            include_nccl,
            comms);

        run_case_for_size(
            results,
            kScenarioPeerToLocal,
            dev0,
            dev1,
            dev1,
            bytes,
            num_blocks,
            iters,
            warmup,
            include_mem_async,
            include_nccl,
            comms);
    }

    run_case_for_size(
        results,
        kScenarioSameDev,
        dev0,
        dev0,
        dev0,
        bytes,
        num_blocks,
        iters,
        warmup,
        include_mem_async,
        false,
        nullptr);
}

} // namespace

std::vector<std::map<std::string, double>>
benchmark_tma_bandwidth_experiment_sm90(
    int64_t min_bytes,
    int64_t max_bytes,
    int iters,
    int warmup,
    int num_blocks,
    int dev0,
    int dev1,
    bool include_mem_async,
    bool include_nccl) {
    if (min_bytes <= 0 || max_bytes <= 0 || min_bytes > max_bytes) {
        throw std::invalid_argument(
            "benchmark_tma_bandwidth_experiment_sm90: invalid byte range");
    }
    if (iters <= 0 || warmup < 0) {
        throw std::invalid_argument(
            "benchmark_tma_bandwidth_experiment_sm90: invalid iteration counts");
    }
    if (num_blocks <= 0) {
        throw std::invalid_argument(
            "benchmark_tma_bandwidth_experiment_sm90: num_blocks must be > 0");
    }
    if (dev0 < 0 || dev1 < 0) {
        throw std::invalid_argument(
            "benchmark_tma_bandwidth_experiment_sm90: invalid devices");
    }

    system::runtime::ensure_context_on_device(dev0);
    system::runtime::ensure_context_on_device(dev1);

    std::vector<std::map<std::string, double>> results;

    ncclComm_t comms[2] = {nullptr, nullptr};
    bool nccl_initialized = false;

    try {
        if (include_nccl && dev0 != dev1) {
            int devices[2] = {dev0, dev1};
            OOVERLAP_EXPERIMENT_NCCL_CHECK(
                ncclCommInitAll(comms, 2, devices));
            nccl_initialized = true;
        }

        for (int64_t bytes_i = min_bytes;
             bytes_i <= max_bytes;
             bytes_i *= 2) {
            const size_t bytes = static_cast<size_t>(bytes_i);

            run_all_cases_for_size(
                results,
                bytes,
                num_blocks,
                iters,
                warmup,
                dev0,
                dev1,
                include_mem_async,
                include_nccl,
                nccl_initialized ? comms : nullptr);
        }

        if (nccl_initialized) {
            ncclCommDestroy(comms[0]);
            ncclCommDestroy(comms[1]);
        }

        return results;
    } catch (...) {
        if (nccl_initialized) {
            ncclCommDestroy(comms[0]);
            ncclCommDestroy(comms[1]);
        }
        throw;
    }
}

} // namespace ooverlap
