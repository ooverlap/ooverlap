#include "test/persistent_allreduce_4gpu_sm90.h"

#include "test/tma_benchmark_sm90.h"
#include "ooverlap/tma/tma.cuh"
#include "ooverlap/tma/tma_reduce.cuh"
#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/system/peer_buffer.cuh"
#include "ooverlap/testing/test_utils.cuh"
#include "comm/communicator.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ooverlap {

static constexpr int kPersistentThreads = 16;
static constexpr size_t kPersistentChunkBytes = 16 * 1024;
static constexpr int kPersistentMaxBlocks = 16;
static constexpr int kPersistentStageDepth = 3;
static constexpr size_t kPersistentStaticSharedBytes =
    static_cast<size_t>(kPersistentStageDepth) * sizeof(sync::semaphore);

namespace {

struct PersistentPairState {
    int num_chunks = 0;
    cudaEvent_t init_done0 = nullptr;
    cudaEvent_t init_done1 = nullptr;
    cudaEvent_t reduce_done0 = nullptr;
    cudaEvent_t reduce_done1 = nullptr;
};

struct PersistentPairOutputs {
    system::mapped_peer_buffer out0; // owned by dev0
    system::mapped_peer_buffer out1; // owned by dev1
};

struct PersistentPairContext {
    int dev0 = -1;
    int dev1 = -1;
    comm::Communicator comm{};
    PersistentPairState st{};
    PersistentPairOutputs outs{};
};

struct FourGpuScratch {
    std::array<half*, 4> temp{nullptr, nullptr, nullptr, nullptr};
    std::array<cudaEvent_t, 4> temp_ready{nullptr, nullptr, nullptr, nullptr};
    std::array<cudaEvent_t, 4> temp_consumed{nullptr, nullptr, nullptr, nullptr};
};

__host__ __device__ __forceinline__ size_t min_sz(size_t a, size_t b) {
    return (a < b) ? a : b;
}

__global__ void persistent_pair_reduce_to_peer_kernel_sm90(
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

    __shared__ sync::semaphore load_barriers[kPersistentStageDepth];

    auto stage_ptr = [&](int stage) -> unsigned char* {
        return shared_raw + static_cast<size_t>(stage) * kPersistentChunkBytes;
    };

    const unsigned char* local_bytes = reinterpret_cast<const unsigned char*>(local_in);
    unsigned char* peer_bytes = reinterpret_cast<unsigned char*>(peer_out);
    const size_t total_bytes = numel * sizeof(half);

    int local_iter = 0;
    int cur_chunk = start_chunk;
    int next_chunk_to_load = start_chunk + chunk_stride;

    {
        const size_t cur_offset = static_cast<size_t>(cur_chunk) * kPersistentChunkBytes;
        const size_t cur_bytes = min_sz(kPersistentChunkBytes, total_bytes - cur_offset);

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
        const int cur_stage = local_iter % kPersistentStageDepth;

        if (threadIdx.x == 0) {
            sync::wait(load_barriers[cur_stage], 0);
        }
        __syncthreads();

        unsigned char* cur_smem = stage_ptr(cur_stage);

        const size_t cur_offset = static_cast<size_t>(cur_chunk) * kPersistentChunkBytes;
        const size_t cur_bytes = min_sz(kPersistentChunkBytes, total_bytes - cur_offset);

        const size_t bulk_bytes = cur_bytes & ~static_cast<size_t>(0xF);
        const size_t tail_bytes = cur_bytes - bulk_bytes;

        if (threadIdx.x == 0 && bulk_bytes > 0) {
            tma::reduce_add_noftz_f16_async(
                peer_bytes + cur_offset,
                cur_smem,
                static_cast<uint32_t>(bulk_bytes));
        }

        if (next_chunk_to_load < num_chunks) {
            const int next_stage = (local_iter + 1) % kPersistentStageDepth;

            if (threadIdx.x == 0) {
                if ((local_iter + 1) >= kPersistentStageDepth) {
                    tma::reduce_async_read_wait<kPersistentStageDepth - 1>();
                }

                const size_t next_offset =
                    static_cast<size_t>(next_chunk_to_load) * kPersistentChunkBytes;
                const size_t next_bytes =
                    min_sz(kPersistentChunkBytes, total_bytes - next_offset);

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
                float oldv = __half2float(peer_out[idx]);
                float addv = __half2float(cur_half[bulk_elems + i]);
                peer_out[idx] = __float2half_rn(oldv + addv);
            }
        }

        __syncthreads();

        cur_chunk = next_chunk_to_load;
        next_chunk_to_load += chunk_stride;
        ++local_iter;
    }
}

inline void validate_four_devices(int dev0, int dev1, int dev2, int dev3) {
    int ndev = 0;
    system::runtime::check_cuda(cudaGetDeviceCount(&ndev), "cudaGetDeviceCount");
    if (ndev < 4) {
        throw std::runtime_error("Need at least 4 CUDA devices");
    }

    const std::array<int, 4> devs = {dev0, dev1, dev2, dev3};
    for (int d : devs) {
        if (d < 0 || d >= ndev) {
            throw std::invalid_argument("Invalid CUDA device id");
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            if (devs[i] == devs[j]) {
                throw std::invalid_argument("Devices must be distinct");
            }
        }
    }
}

inline void configure_persistent_kernel_smem_once(int device, size_t dynamic_smem_bytes) {
    struct KernelConfigCacheEntry {
        bool configured = false;
        size_t dynamic_smem_bytes = 0;
    };

    static std::mutex mutex;
    static std::unordered_map<int, KernelConfigCacheEntry> cache;

    std::lock_guard<std::mutex> lock(mutex);

    auto it = cache.find(device);
    if (it != cache.end() &&
        it->second.configured &&
        it->second.dynamic_smem_bytes == dynamic_smem_bytes) {
        return;
    }

    system::runtime::set_device(device);

    cudaDeviceProp prop{};
    system::runtime::check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    const size_t total_smem_bytes = dynamic_smem_bytes + kPersistentStaticSharedBytes;

    if (total_smem_bytes > static_cast<size_t>(prop.sharedMemPerBlockOptin)) {
        throw std::runtime_error(
            "persistent kernel requested total shared memory exceeds device opt-in limit");
    }

    if (total_smem_bytes > static_cast<size_t>(prop.sharedMemPerBlock)) {
        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                persistent_pair_reduce_to_peer_kernel_sm90,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(dynamic_smem_bytes)),
            "cudaFuncSetAttribute(MaxDynamicSharedMemorySize)");

        system::runtime::check_cuda(
            cudaFuncSetAttribute(
                persistent_pair_reduce_to_peer_kernel_sm90,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                100),
            "cudaFuncSetAttribute(PreferredSharedMemoryCarveout)");
    }

    cache[device] = {true, dynamic_smem_bytes};
}

inline int compute_num_chunks(size_t numel) {
    const size_t total_bytes = numel * sizeof(half);
    return static_cast<int>((total_bytes + kPersistentChunkBytes - 1) / kPersistentChunkBytes);
}

inline void create_event_on_device(int device, cudaEvent_t* ev, const char* what) {
    system::runtime::set_device(device);
    system::runtime::check_cuda(
        cudaEventCreateWithFlags(ev, cudaEventDisableTiming),
        what);
}

inline void alloc_pair_state(
    PersistentPairState* st,
    int dev0,
    int dev1,
    int num_chunks) {

    if (st == nullptr) {
        throw std::invalid_argument("alloc_pair_state: state is null");
    }
    if (num_chunks <= 0) {
        throw std::invalid_argument("alloc_pair_state: num_chunks must be > 0");
    }

    st->num_chunks = num_chunks;

    create_event_on_device(dev0, &st->init_done0, "cudaEventCreate(init_done0)");
    create_event_on_device(dev1, &st->init_done1, "cudaEventCreate(init_done1)");
    create_event_on_device(dev0, &st->reduce_done0, "cudaEventCreate(reduce_done0)");
    create_event_on_device(dev1, &st->reduce_done1, "cudaEventCreate(reduce_done1)");
}

inline void free_pair_state(PersistentPairState* st, int dev0, int dev1) {
    if (st == nullptr) return;

    if (st->init_done0) {
        system::runtime::set_device(dev0);
        cudaEventDestroy(st->init_done0);
        st->init_done0 = nullptr;
    }
    if (st->init_done1) {
        system::runtime::set_device(dev1);
        cudaEventDestroy(st->init_done1);
        st->init_done1 = nullptr;
    }
    if (st->reduce_done0) {
        system::runtime::set_device(dev0);
        cudaEventDestroy(st->reduce_done0);
        st->reduce_done0 = nullptr;
    }
    if (st->reduce_done1) {
        system::runtime::set_device(dev1);
        cudaEventDestroy(st->reduce_done1);
        st->reduce_done1 = nullptr;
    }

    st->num_chunks = 0;
}

inline void alloc_pair_outputs(
    PersistentPairOutputs* outs,
    int dev0,
    int dev1,
    size_t bytes) {

    if (outs == nullptr) {
        throw std::invalid_argument("alloc_pair_outputs: outs is null");
    }

    std::vector<int> access_devices = {dev0, dev1};
    outs->out0 = system::alloc_peer_visible_buffer(bytes, dev0, access_devices);
    outs->out1 = system::alloc_peer_visible_buffer(bytes, dev1, access_devices);
}

inline void free_pair_outputs(PersistentPairOutputs* outs) {
    if (outs == nullptr) return;
    system::free_peer_visible_buffer(outs->out0);
    system::free_peer_visible_buffer(outs->out1);
}

inline half* pair_output_ptr(PersistentPairContext* ctx, int local_rank) {
    return reinterpret_cast<half*>(local_rank == 0 ? ctx->outs.out0.ptr : ctx->outs.out1.ptr);
}

inline void init_pair_context(
    PersistentPairContext* ctx,
    int dev0,
    int dev1,
    size_t numel) {

    if (ctx == nullptr) {
        throw std::invalid_argument("init_pair_context: ctx is null");
    }

    ctx->dev0 = dev0;
    ctx->dev1 = dev1;

    communicator_init(&ctx->comm, {dev0, dev1}, numel, 1);

    const size_t smem_bytes = static_cast<size_t>(kPersistentStageDepth) * kPersistentChunkBytes;
    configure_persistent_kernel_smem_once(dev0, smem_bytes);
    configure_persistent_kernel_smem_once(dev1, smem_bytes);

    alloc_pair_state(&ctx->st, dev0, dev1, compute_num_chunks(numel));
    alloc_pair_outputs(&ctx->outs, dev0, dev1, numel * sizeof(half));
}

inline void destroy_pair_context(PersistentPairContext* ctx) {
    if (ctx == nullptr) return;
    free_pair_outputs(&ctx->outs);
    free_pair_state(&ctx->st, ctx->dev0, ctx->dev1);
    communicator_destroy(&ctx->comm);
    ctx->dev0 = -1;
    ctx->dev1 = -1;
}

inline void alloc_four_gpu_scratch(
    FourGpuScratch* scratch,
    int dev0,
    int dev1,
    int dev2,
    int dev3,
    size_t bytes,
    PersistentPairContext* pair02,
    PersistentPairContext* pair13) {

    if (scratch == nullptr) {
        throw std::invalid_argument("alloc_four_gpu_scratch: scratch is null");
    }

    const std::array<int, 4> devs = {dev0, dev1, dev2, dev3};
    for (int i = 0; i < 4; ++i) {
        system::runtime::set_device(devs[i]);
        system::runtime::check_cuda(cudaMalloc(&scratch->temp[i], bytes), "cudaMalloc(scratch temp)");
        create_event_on_device(devs[i], &scratch->temp_ready[i], "cudaEventCreate(temp_ready)");
        create_event_on_device(devs[i], &scratch->temp_consumed[i], "cudaEventCreate(temp_consumed)");
    }

    // Make first iteration copies immediately legal.
    system::runtime::set_device(dev0);
    system::runtime::check_cuda(
        cudaEventRecord(scratch->temp_consumed[0], pair02->comm.streams[0]),
        "cudaEventRecord(initial temp_consumed0)");
    system::runtime::set_device(dev1);
    system::runtime::check_cuda(
        cudaEventRecord(scratch->temp_consumed[1], pair13->comm.streams[0]),
        "cudaEventRecord(initial temp_consumed1)");
    system::runtime::set_device(dev2);
    system::runtime::check_cuda(
        cudaEventRecord(scratch->temp_consumed[2], pair02->comm.streams[1]),
        "cudaEventRecord(initial temp_consumed2)");
    system::runtime::set_device(dev3);
    system::runtime::check_cuda(
        cudaEventRecord(scratch->temp_consumed[3], pair13->comm.streams[1]),
        "cudaEventRecord(initial temp_consumed3)");
}

inline void free_four_gpu_scratch(
    FourGpuScratch* scratch,
    int dev0,
    int dev1,
    int dev2,
    int dev3) {

    if (scratch == nullptr) return;
    const std::array<int, 4> devs = {dev0, dev1, dev2, dev3};
    for (int i = 0; i < 4; ++i) {
        system::runtime::set_device(devs[i]);
        if (scratch->temp[i]) {
            cudaFree(scratch->temp[i]);
            scratch->temp[i] = nullptr;
        }
        if (scratch->temp_ready[i]) {
            cudaEventDestroy(scratch->temp_ready[i]);
            scratch->temp_ready[i] = nullptr;
        }
        if (scratch->temp_consumed[i]) {
            cudaEventDestroy(scratch->temp_consumed[i]);
            scratch->temp_consumed[i] = nullptr;
        }
    }
}

inline cudaError_t enqueue_persistent_pair_with_state(
    PersistentPairContext* ctx,
    half* in0,
    half* in1,
    half* out0_peer,
    half* out1_peer,
    size_t numel) {

    if (ctx == nullptr) return cudaErrorInvalidValue;
    if (in0 == nullptr || in1 == nullptr || out0_peer == nullptr || out1_peer == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }
    if (ctx->comm.world_size != 2) return cudaErrorInvalidValue;
    if (numel == 0 || numel > ctx->comm.max_full_numel) return cudaErrorInvalidValue;

    const size_t bytes = numel * sizeof(half);
    const int num_chunks = compute_num_chunks(numel);
    if (num_chunks != ctx->st.num_chunks) return cudaErrorInvalidValue;

    const int num_blocks = std::min(kPersistentMaxBlocks, num_chunks);
    const size_t smem_bytes = static_cast<size_t>(kPersistentStageDepth) * kPersistentChunkBytes;

    system::runtime::set_device(ctx->dev0);
    system::runtime::check_cuda(
        cudaMemcpyAsync(
            out0_peer,
            in0,
            bytes,
            cudaMemcpyDeviceToDevice,
            ctx->comm.streams[0]),
        "cudaMemcpyAsync(in0 -> out0_peer)");
    system::runtime::check_cuda(
        cudaEventRecord(ctx->st.init_done0, ctx->comm.streams[0]),
        "cudaEventRecord(init_done0)");

    system::runtime::set_device(ctx->dev1);
    system::runtime::check_cuda(
        cudaMemcpyAsync(
            out1_peer,
            in1,
            bytes,
            cudaMemcpyDeviceToDevice,
            ctx->comm.streams[1]),
        "cudaMemcpyAsync(in1 -> out1_peer)");
    system::runtime::check_cuda(
        cudaEventRecord(ctx->st.init_done1, ctx->comm.streams[1]),
        "cudaEventRecord(init_done1)");

    system::runtime::set_device(ctx->dev0);
    system::runtime::check_cuda(
        cudaStreamWaitEvent(ctx->comm.streams[0], ctx->st.init_done1, 0),
        "cudaStreamWaitEvent(stream0, init_done1)");

    system::runtime::set_device(ctx->dev1);
    system::runtime::check_cuda(
        cudaStreamWaitEvent(ctx->comm.streams[1], ctx->st.init_done0, 0),
        "cudaStreamWaitEvent(stream1, init_done0)");

    system::runtime::set_device(ctx->dev0);
    persistent_pair_reduce_to_peer_kernel_sm90<<<
        num_blocks,
        kPersistentThreads,
        smem_bytes,
        ctx->comm.streams[0]>>>(
        in0,
        out1_peer,
        numel,
        num_chunks);
    cudaError_t err0 = cudaGetLastError();
    if (err0 != cudaSuccess) return err0;
    system::runtime::check_cuda(
        cudaEventRecord(ctx->st.reduce_done0, ctx->comm.streams[0]),
        "cudaEventRecord(reduce_done0)");

    system::runtime::set_device(ctx->dev1);
    persistent_pair_reduce_to_peer_kernel_sm90<<<
        num_blocks,
        kPersistentThreads,
        smem_bytes,
        ctx->comm.streams[1]>>>(
        in1,
        out0_peer,
        numel,
        num_chunks);
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) return err1;
    system::runtime::check_cuda(
        cudaEventRecord(ctx->st.reduce_done1, ctx->comm.streams[1]),
        "cudaEventRecord(reduce_done1)");

    system::runtime::set_device(ctx->dev0);
    system::runtime::check_cuda(
        cudaStreamWaitEvent(ctx->comm.streams[0], ctx->st.reduce_done1, 0),
        "cudaStreamWaitEvent(stream0, reduce_done1)");

    system::runtime::set_device(ctx->dev1);
    system::runtime::check_cuda(
        cudaStreamWaitEvent(ctx->comm.streams[1], ctx->st.reduce_done0, 0),
        "cudaStreamWaitEvent(stream1, reduce_done0)");

    return cudaSuccess;
}

inline void sync_one_stream(int dev, cudaStream_t stream, const char* what) {
    system::runtime::sync_stream_on_device(dev, stream, what);
}

inline void sync_all_final_streams(
    PersistentPairContext* pair02,
    PersistentPairContext* pair13,
    const char* what) {

    sync_one_stream(pair02->dev0, pair02->comm.streams[0], what);
    sync_one_stream(pair13->dev0, pair13->comm.streams[0], what);
    sync_one_stream(pair02->dev1, pair02->comm.streams[1], what);
    sync_one_stream(pair13->dev1, pair13->comm.streams[1], what);
}

inline void enqueue_four_gpu_persistent_once(
    PersistentPairContext* pair01,
    PersistentPairContext* pair23,
    PersistentPairContext* pair02,
    PersistentPairContext* pair13,
    FourGpuScratch* scratch,
    const std::array<half*, 4>& inputs,
    size_t numel) {

    // Phase 1: (0,1) and (2,3)
    system::runtime::check_cuda(
        enqueue_persistent_pair_with_state(
            pair01,
            inputs[0],
            inputs[1],
            pair_output_ptr(pair01, 0),
            pair_output_ptr(pair01, 1),
            numel),
        "enqueue_persistent_pair_with_state(pair01)");

    system::runtime::check_cuda(
        enqueue_persistent_pair_with_state(
            pair23,
            inputs[2],
            inputs[3],
            pair_output_ptr(pair23, 0),
            pair_output_ptr(pair23, 1),
            numel),
        "enqueue_persistent_pair_with_state(pair23)");

    // Copy phase-1 outputs into regular device temp buffers.
    // Guard against overwriting temps before previous phase-2 consumed them.
    system::runtime::set_device(pair01->dev0);
    system::runtime::check_cuda(
        cudaStreamWaitEvent(pair01->comm.streams[0], scratch->temp_consumed[0], 0),
        "cudaStreamWaitEvent(pair01 dev0, temp_consumed0)");
    system::runtime::check_cuda(
        cudaMemcpyAsync(
            scratch->temp[0],
            pair_output_ptr(pair01, 0),
            numel * sizeof(half),
            cudaMemcpyDeviceToDevice,
            pair01->comm.streams[0]),
        "cudaMemcpyAsync(temp0 <- pair01.out0)");
    system::runtime::check_cuda(
        cudaEventRecord(scratch->temp_ready[0], pair01->comm.streams[0]),
        "cudaEventRecord(temp_ready0)");

    system::runtime::set_device(pair01->dev1);
    system::runtime::check_cuda(
        cudaStreamWaitEvent(pair01->comm.streams[1], scratch->temp_consumed[1], 0),
        "cudaStreamWaitEvent(pair01 dev1, temp_consumed1)");
    system::runtime::check_cuda(
        cudaMemcpyAsync(
            scratch->temp[1],
            pair_output_ptr(pair01, 1),
            numel * sizeof(half),
            cudaMemcpyDeviceToDevice,
            pair01->comm.streams[1]),
        "cudaMemcpyAsync(temp1 <- pair01.out1)");
    system::runtime::check_cuda(
        cudaEventRecord(scratch->temp_ready[1], pair01->comm.streams[1]),
        "cudaEventRecord(temp_ready1)");

    system::runtime::set_device(pair23->dev0);
    system::runtime::check_cuda(
        cudaStreamWaitEvent(pair23->comm.streams[0], scratch->temp_consumed[2], 0),
        "cudaStreamWaitEvent(pair23 dev2, temp_consumed2)");
    system::runtime::check_cuda(
        cudaMemcpyAsync(
            scratch->temp[2],
            pair_output_ptr(pair23, 0),
            numel * sizeof(half),
            cudaMemcpyDeviceToDevice,
            pair23->comm.streams[0]),
        "cudaMemcpyAsync(temp2 <- pair23.out0)");
    system::runtime::check_cuda(
        cudaEventRecord(scratch->temp_ready[2], pair23->comm.streams[0]),
        "cudaEventRecord(temp_ready2)");

    system::runtime::set_device(pair23->dev1);
    system::runtime::check_cuda(
        cudaStreamWaitEvent(pair23->comm.streams[1], scratch->temp_consumed[3], 0),
        "cudaStreamWaitEvent(pair23 dev3, temp_consumed3)");
    system::runtime::check_cuda(
        cudaMemcpyAsync(
            scratch->temp[3],
            pair_output_ptr(pair23, 1),
            numel * sizeof(half),
            cudaMemcpyDeviceToDevice,
            pair23->comm.streams[1]),
        "cudaMemcpyAsync(temp3 <- pair23.out1)");
    system::runtime::check_cuda(
        cudaEventRecord(scratch->temp_ready[3], pair23->comm.streams[1]),
        "cudaEventRecord(temp_ready3)");

    // Phase 2 waits on temp readiness.
    system::runtime::set_device(pair02->dev0);
    system::runtime::check_cuda(
        cudaStreamWaitEvent(pair02->comm.streams[0], scratch->temp_ready[0], 0),
        "cudaStreamWaitEvent(pair02 dev0, temp_ready0)");
    system::runtime::set_device(pair13->dev0);
    system::runtime::check_cuda(
        cudaStreamWaitEvent(pair13->comm.streams[0], scratch->temp_ready[1], 0),
        "cudaStreamWaitEvent(pair13 dev1, temp_ready1)");
    system::runtime::set_device(pair02->dev1);
    system::runtime::check_cuda(
        cudaStreamWaitEvent(pair02->comm.streams[1], scratch->temp_ready[2], 0),
        "cudaStreamWaitEvent(pair02 dev2, temp_ready2)");
    system::runtime::set_device(pair13->dev1);
    system::runtime::check_cuda(
        cudaStreamWaitEvent(pair13->comm.streams[1], scratch->temp_ready[3], 0),
        "cudaStreamWaitEvent(pair13 dev3, temp_ready3)");

    // Phase 2: (0,2) and (1,3)
    system::runtime::check_cuda(
        enqueue_persistent_pair_with_state(
            pair02,
            scratch->temp[0],
            scratch->temp[2],
            pair_output_ptr(pair02, 0),
            pair_output_ptr(pair02, 1),
            numel),
        "enqueue_persistent_pair_with_state(pair02)");

    system::runtime::check_cuda(
        enqueue_persistent_pair_with_state(
            pair13,
            scratch->temp[1],
            scratch->temp[3],
            pair_output_ptr(pair13, 0),
            pair_output_ptr(pair13, 1),
            numel),
        "enqueue_persistent_pair_with_state(pair13)");

    // Mark temps safe to overwrite on the next iteration.
    system::runtime::set_device(pair02->dev0);
    system::runtime::check_cuda(
        cudaEventRecord(scratch->temp_consumed[0], pair02->comm.streams[0]),
        "cudaEventRecord(temp_consumed0)");
    system::runtime::set_device(pair13->dev0);
    system::runtime::check_cuda(
        cudaEventRecord(scratch->temp_consumed[1], pair13->comm.streams[0]),
        "cudaEventRecord(temp_consumed1)");
    system::runtime::set_device(pair02->dev1);
    system::runtime::check_cuda(
        cudaEventRecord(scratch->temp_consumed[2], pair02->comm.streams[1]),
        "cudaEventRecord(temp_consumed2)");
    system::runtime::set_device(pair13->dev1);
    system::runtime::check_cuda(
        cudaEventRecord(scratch->temp_consumed[3], pair13->comm.streams[1]),
        "cudaEventRecord(temp_consumed3)");
}

inline std::vector<float> reference_four_gpu_sum_fp16(int64_t numel) {
    auto ref0 = testing::host_reference_pattern_fp16(numel, 0.25f, 1.0f);
    auto ref1 = testing::host_reference_pattern_fp16(numel, 0.50f, 2.0f);
    auto ref2 = testing::host_reference_pattern_fp16(numel, 0.75f, 3.0f);
    auto ref3 = testing::host_reference_pattern_fp16(numel, 1.00f, 4.0f);

    std::vector<float> ref(static_cast<size_t>(numel));
    for (int64_t i = 0; i < numel; ++i) {
        const size_t idx = static_cast<size_t>(i);
        float s01 = testing::round_to_half(ref0[idx] + ref1[idx]);
        float s23 = testing::round_to_half(ref2[idx] + ref3[idx]);
        ref[idx] = testing::round_to_half(s01 + s23);
    }
    return ref;
}

inline double elapsed_ms_four_gpu_persistent_e2e(
    const std::array<int, 4>& start_devices,
    const std::array<cudaStream_t, 4>& start_streams,
    const std::array<int, 4>& stop_devices,
    const std::array<cudaStream_t, 4>& stop_streams,
    int iters,
    const std::function<void(int)>& launch_once) {

    std::array<cudaEvent_t, 4> starts{nullptr, nullptr, nullptr, nullptr};
    std::array<cudaEvent_t, 4> stops{nullptr, nullptr, nullptr, nullptr};

    for (int i = 0; i < 4; ++i) {
        system::runtime::set_device(start_devices[i]);
        system::runtime::check_cuda(cudaEventCreate(&starts[i]), "cudaEventCreate(start)");
        system::runtime::check_cuda(cudaEventRecord(starts[i], start_streams[i]), "cudaEventRecord(start)");
    }

    for (int i = 0; i < iters; ++i) {
        launch_once(i);
    }

    for (int i = 0; i < 4; ++i) {
        system::runtime::set_device(stop_devices[i]);
        system::runtime::check_cuda(cudaEventCreate(&stops[i]), "cudaEventCreate(stop)");
        system::runtime::check_cuda(cudaEventRecord(stops[i], stop_streams[i]), "cudaEventRecord(stop)");
    }

    double max_ms = 0.0;
    for (int i = 0; i < 4; ++i) {
        system::runtime::set_device(stop_devices[i]);
        system::runtime::check_cuda(cudaEventSynchronize(stops[i]), "cudaEventSynchronize(stop)");
        float ms = 0.0f;
        system::runtime::check_cuda(cudaEventElapsedTime(&ms, starts[i], stops[i]), "cudaEventElapsedTime");
        max_ms = std::max(max_ms, static_cast<double>(ms));
        cudaEventDestroy(starts[i]);
        cudaEventDestroy(stops[i]);
    }

    return max_ms;
}

inline half* final_output_ptr_for_rank(
    PersistentPairContext* pair02,
    PersistentPairContext* pair13,
    int rank) {

    switch (rank) {
        case 0: return pair_output_ptr(pair02, 0);
        case 1: return pair_output_ptr(pair13, 0);
        case 2: return pair_output_ptr(pair02, 1);
        case 3: return pair_output_ptr(pair13, 1);
        default: return nullptr;
    }
}

} // namespace

bool tma_persistent_four_gpu_allreduce_smoke_test(
    int64_t numel,
    int dev0,
    int dev1,
    int dev2,
    int dev3) {

    if (numel <= 0) {
        throw std::invalid_argument("tma_persistent_four_gpu_allreduce_smoke_test: numel must be > 0");
    }
    validate_four_devices(dev0, dev1, dev2, dev3);

    const size_t numel_sz = static_cast<size_t>(numel);
    const size_t bytes = numel_sz * sizeof(half);

    PersistentPairContext pair01{};
    PersistentPairContext pair23{};
    PersistentPairContext pair02{};
    PersistentPairContext pair13{};
    FourGpuScratch scratch{};

    init_pair_context(&pair01, dev0, dev1, numel_sz);
    init_pair_context(&pair23, dev2, dev3, numel_sz);
    init_pair_context(&pair02, dev0, dev2, numel_sz);
    init_pair_context(&pair13, dev1, dev3, numel_sz);
    alloc_four_gpu_scratch(&scratch, dev0, dev1, dev2, dev3, bytes, &pair02, &pair13);

    std::array<half*, 4> inputs{nullptr, nullptr, nullptr, nullptr};

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(cudaMalloc(&inputs[0], bytes), "cudaMalloc(input0)");
    testing::fill_pattern(inputs[0], numel, 0.25f, 1.0f, pair01.comm.streams[0]);

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(cudaMalloc(&inputs[1], bytes), "cudaMalloc(input1)");
    testing::fill_pattern(inputs[1], numel, 0.50f, 2.0f, pair01.comm.streams[1]);

    system::runtime::set_device(dev2);
    system::runtime::check_cuda(cudaMalloc(&inputs[2], bytes), "cudaMalloc(input2)");
    testing::fill_pattern(inputs[2], numel, 0.75f, 3.0f, pair23.comm.streams[0]);

    system::runtime::set_device(dev3);
    system::runtime::check_cuda(cudaMalloc(&inputs[3], bytes), "cudaMalloc(input3)");
    testing::fill_pattern(inputs[3], numel, 1.00f, 4.0f, pair23.comm.streams[1]);

    system::runtime::sync_stream_on_device(dev0, pair01.comm.streams[0], "sync fill input0");
    system::runtime::sync_stream_on_device(dev1, pair01.comm.streams[1], "sync fill input1");
    system::runtime::sync_stream_on_device(dev2, pair23.comm.streams[0], "sync fill input2");
    system::runtime::sync_stream_on_device(dev3, pair23.comm.streams[1], "sync fill input3");

    enqueue_four_gpu_persistent_once(&pair01, &pair23, &pair02, &pair13, &scratch, inputs, numel_sz);
    sync_all_final_streams(&pair02, &pair13, "sync four-gpu persistent final");

    auto ref = reference_four_gpu_sum_fp16(numel);

    auto got0 = testing::copy_half_device_to_host_float(final_output_ptr_for_rank(&pair02, &pair13, 0), numel, dev0);
    auto got1 = testing::copy_half_device_to_host_float(final_output_ptr_for_rank(&pair02, &pair13, 1), numel, dev1);
    auto got2 = testing::copy_half_device_to_host_float(final_output_ptr_for_rank(&pair02, &pair13, 2), numel, dev2);
    auto got3 = testing::copy_half_device_to_host_float(final_output_ptr_for_rank(&pair02, &pair13, 3), numel, dev3);

    testing::expect_allclose(got0, ref, "persistent four-gpu allreduce rank0");
    testing::expect_allclose(got1, ref, "persistent four-gpu allreduce rank1");
    testing::expect_allclose(got2, ref, "persistent four-gpu allreduce rank2");
    testing::expect_allclose(got3, ref, "persistent four-gpu allreduce rank3");

    for (int i = 0; i < 4; ++i) {
        if (inputs[i] != nullptr) {
            const int dev = (i == 0 ? dev0 : i == 1 ? dev1 : i == 2 ? dev2 : dev3);
            system::runtime::set_device(dev);
            system::runtime::check_cuda(cudaFree(inputs[i]), "cudaFree(input)");
        }
    }

    free_four_gpu_scratch(&scratch, dev0, dev1, dev2, dev3);
    destroy_pair_context(&pair01);
    destroy_pair_context(&pair23);
    destroy_pair_context(&pair02);
    destroy_pair_context(&pair13);
    return true;
}

std::map<std::string, double> benchmark_persistent_four_gpu_allreduce_sm90(
    int64_t numel,
    int iters,
    int warmup,
    int dev0,
    int dev1,
    int dev2,
    int dev3) {

    if (numel <= 0 || iters <= 0 || warmup < 0) {
        throw std::invalid_argument("benchmark_persistent_four_gpu_allreduce_sm90: invalid args");
    }
    validate_four_devices(dev0, dev1, dev2, dev3);

    const size_t numel_sz = static_cast<size_t>(numel);
    const size_t bytes = numel_sz * sizeof(half);

    PersistentPairContext pair01{};
    PersistentPairContext pair23{};
    PersistentPairContext pair02{};
    PersistentPairContext pair13{};
    FourGpuScratch scratch{};

    init_pair_context(&pair01, dev0, dev1, numel_sz);
    init_pair_context(&pair23, dev2, dev3, numel_sz);
    init_pair_context(&pair02, dev0, dev2, numel_sz);
    init_pair_context(&pair13, dev1, dev3, numel_sz);
    alloc_four_gpu_scratch(&scratch, dev0, dev1, dev2, dev3, bytes, &pair02, &pair13);

    std::array<half*, 4> inputs{nullptr, nullptr, nullptr, nullptr};

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(cudaMalloc(&inputs[0], bytes), "cudaMalloc(input0)");
    testing::fill_pattern(inputs[0], numel, 0.25f, 1.0f, pair01.comm.streams[0]);

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(cudaMalloc(&inputs[1], bytes), "cudaMalloc(input1)");
    testing::fill_pattern(inputs[1], numel, 0.50f, 2.0f, pair01.comm.streams[1]);

    system::runtime::set_device(dev2);
    system::runtime::check_cuda(cudaMalloc(&inputs[2], bytes), "cudaMalloc(input2)");
    testing::fill_pattern(inputs[2], numel, 0.75f, 3.0f, pair23.comm.streams[0]);

    system::runtime::set_device(dev3);
    system::runtime::check_cuda(cudaMalloc(&inputs[3], bytes), "cudaMalloc(input3)");
    testing::fill_pattern(inputs[3], numel, 1.00f, 4.0f, pair23.comm.streams[1]);

    system::runtime::sync_stream_on_device(dev0, pair01.comm.streams[0], "sync fill input0");
    system::runtime::sync_stream_on_device(dev1, pair01.comm.streams[1], "sync fill input1");
    system::runtime::sync_stream_on_device(dev2, pair23.comm.streams[0], "sync fill input2");
    system::runtime::sync_stream_on_device(dev3, pair23.comm.streams[1], "sync fill input3");

    for (int i = 0; i < warmup; ++i) {
        enqueue_four_gpu_persistent_once(&pair01, &pair23, &pair02, &pair13, &scratch, inputs, numel_sz);
        sync_all_final_streams(&pair02, &pair13, "sync four-gpu persistent warmup");
    }

    const std::array<int, 4> start_devices = {dev0, dev1, dev2, dev3};
    const std::array<cudaStream_t, 4> start_streams = {
        pair01.comm.streams[0],
        pair01.comm.streams[1],
        pair23.comm.streams[0],
        pair23.comm.streams[1],
    };
    const std::array<int, 4> stop_devices = {dev0, dev1, dev2, dev3};
    const std::array<cudaStream_t, 4> stop_streams = {
        pair02.comm.streams[0],
        pair13.comm.streams[0],
        pair02.comm.streams[1],
        pair13.comm.streams[1],
    };

    const double persistent_total_ms = elapsed_ms_four_gpu_persistent_e2e(
        start_devices,
        start_streams,
        stop_devices,
        stop_streams,
        iters,
        [&](int) {
            enqueue_four_gpu_persistent_once(&pair01, &pair23, &pair02, &pair13, &scratch, inputs, numel_sz);
        });

    const std::vector<int64_t> devices64 = {dev0, dev1, dev2, dev3};
    auto basic_metrics =
        benchmark_basic_ngpu_collective_sm90("allreduce", numel, devices64, iters, warmup);

    const double avg_persistent_ms = persistent_total_ms / static_cast<double>(iters);
    const double avg_basic_ms = basic_metrics.at("avg_ms_tma");
    const double avg_nccl_ms = basic_metrics.at("avg_ms_nccl");

    for (int i = 0; i < 4; ++i) {
        if (inputs[i] != nullptr) {
            const int dev = (i == 0 ? dev0 : i == 1 ? dev1 : i == 2 ? dev2 : dev3);
            system::runtime::set_device(dev);
            system::runtime::check_cuda(cudaFree(inputs[i]), "cudaFree(input)");
        }
    }

    free_four_gpu_scratch(&scratch, dev0, dev1, dev2, dev3);
    destroy_pair_context(&pair01);
    destroy_pair_context(&pair23);
    destroy_pair_context(&pair02);
    destroy_pair_context(&pair13);

    return {
        {"world_size", 4.0},
        {"numel", static_cast<double>(numel)},
        {"avg_ms_persistent", avg_persistent_ms},
        {"avg_ms_basic", avg_basic_ms},
        {"avg_ms_nccl", avg_nccl_ms},
        {"speedup_basic_over_persistent", avg_basic_ms / avg_persistent_ms},
        {"speedup_nccl_over_persistent", avg_nccl_ms / avg_persistent_ms},
    };
}

} // namespace ooverlap
