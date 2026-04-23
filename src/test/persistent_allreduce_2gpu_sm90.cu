#include "test/persistent_allreduce_2gpu_sm90.h"

#include "comm/collective/operation.h"
#include "comm/endpoint_persistent_kernel.h"
#include "comm/endpoint_runtime.h"
#include "comm/group.h"
#include "comm/transport/buffer.h"
#include "ooverlap/system/runtime_utils.cuh"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define OOVERLAP_PERSIST_NCCL_CHECK(cmd)                                                     \
    do {                                                                                     \
        ncclResult_t result__ = (cmd);                                                       \
        if (result__ != ncclSuccess) {                                                       \
            throw std::runtime_error(                                                        \
                std::string("NCCL error: ") + ncclGetErrorString(result__));                 \
        }                                                                                    \
    } while (0)

namespace ooverlap {
namespace {

void bench_trace(const char* msg) {
    return;
    std::printf("[bench] %s\n", msg);
    std::fflush(stdout);
}

void bench_trace_iter(const char* phase, int iter, const char* msg) {
    return;
    std::printf("[bench] phase=%s iter=%d %s\n", phase, iter, msg);
    std::fflush(stdout);
}

__global__ void fp16_add_inplace_kernel(
    half* dst,
    const half* src,
    int64_t n) {
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < n;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const float a = __half2float(dst[idx]);
        const float b = __half2float(src[idx]);
        dst[idx] = __float2half_rn(a + b);
    }
}

cudaError_t enqueue_fp16_add_inplace(
    half* dst,
    const half* src,
    size_t numel,
    cudaStream_t stream) {
    if (dst == nullptr || src == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }
    if (numel == 0) {
        return cudaSuccess;
    }

    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((numel + kThreads - 1) / kThreads);
    fp16_add_inplace_kernel<<<blocks, kThreads, 0, stream>>>(
        dst, src, static_cast<int64_t>(numel));
    return cudaGetLastError();
}

std::vector<half> make_host_pattern(
    int64_t numel,
    int rank) {
    const float base = 0.125f * static_cast<float>(rank + 1);
    const float step = 0.010f + 0.002f * static_cast<float>(rank);

    std::vector<half> out(static_cast<size_t>(numel));
    for (int64_t i = 0; i < numel; ++i) {
        const float x = base + step * static_cast<float>(i % 97);
        out[static_cast<size_t>(i)] = __float2half_rn(x);
    }
    return out;
}

std::vector<half> reference_two_gpu_sum(
    int64_t numel) {
    auto a = make_host_pattern(numel, 0);
    auto b = make_host_pattern(numel, 1);

    std::vector<half> out(static_cast<size_t>(numel));
    for (int64_t i = 0; i < numel; ++i) {
        const float acc =
            __half2float(a[static_cast<size_t>(i)]) +
            __half2float(b[static_cast<size_t>(i)]);
        out[static_cast<size_t>(i)] = __float2half_rn(acc);
    }
    return out;
}

std::vector<half> copy_half_device_to_host(
    const half* src,
    int64_t numel,
    int device) {
    std::vector<half> host(static_cast<size_t>(numel));
    system::runtime::set_device(device);
    system::runtime::check_cuda(
        cudaMemcpy(
            host.data(),
            src,
            static_cast<size_t>(numel) * sizeof(half),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(device -> host)");
    return host;
}

void expect_half_vectors_close(
    const std::vector<half>& got,
    const std::vector<half>& ref,
    const char* what) {
    if (got.size() != ref.size()) {
        throw std::runtime_error(std::string(what) + ": size mismatch");
    }

    for (size_t i = 0; i < got.size(); ++i) {
        const float g = __half2float(got[i]);
        const float r = __half2float(ref[i]);
        const float err = std::fabs(g - r);
        if (err > 1.0e-3f) {
            throw std::runtime_error(
                std::string(what) +
                ": mismatch at idx=" + std::to_string(i) +
                " got=" + std::to_string(g) +
                " ref=" + std::to_string(r));
        }
    }
}

void upload_host_half_vector(
    half* dst,
    const std::vector<half>& src,
    int device,
    cudaStream_t stream,
    const char* what) {
    system::runtime::set_device(device);
    system::runtime::check_cuda(
        cudaMemcpyAsync(
            dst,
            src.data(),
            src.size() * sizeof(half),
            cudaMemcpyHostToDevice,
            stream),
        what);
}

void sync_two_streams(
    int dev0,
    cudaStream_t stream0,
    int dev1,
    cudaStream_t stream1,
    const char* what) {
    system::runtime::sync_stream_on_device(dev0, stream0, what);
    system::runtime::sync_stream_on_device(dev1, stream1, what);
}

template <typename Fn>
double measure_host_ms(Fn&& fn) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

bool all_u32_equal_to_one(
    const uint32_t* ptr,
    size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (ptr[i] != 1u) {
            return false;
        }
    }
    return true;
}

int prev_rank_of(
    int rank,
    int world_size) {
    return (rank - 1 + world_size) % world_size;
}

void check_driver(
    CUresult result,
    const char* what) {
    if (result == CUDA_SUCCESS) {
        return;
    }

    const char* name = nullptr;
    const char* desc = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &desc);

    throw std::runtime_error(
        std::string(what) +
        ": " +
        (name != nullptr ? name : "CUDA_DRIVER_ERROR") +
        (desc != nullptr ? std::string(" (") + desc + ")" : std::string()));
}

struct DeviceMailbox {
    comm::transport::CommBuffer buf{};
    std::vector<uint32_t> host_cache{};
    size_t count = 0;
    int owner_rank = -1;

    void init(
        const std::vector<int>& devices,
        int owner_rank_,
        const std::vector<int>& access_ranks,
        size_t n) {
        destroy(devices);

        owner_rank = owner_rank_;
        count = n;
        host_cache.assign(n, 0u);
        buf = comm::transport::alloc_peer_visible_buffer_for_rank_with_access_ranks(
            devices,
            owner_rank,
            access_ranks,
            n * sizeof(uint32_t));
    }

    uint32_t* device_ptr_for_rank(size_t rank) const {
        return reinterpret_cast<uint32_t*>(buf.device_ptr_for_rank(rank));
    }

    uint32_t* owner_device_ptr() const {
        return reinterpret_cast<uint32_t*>(
            buf.device_ptr_for_rank(static_cast<size_t>(owner_rank)));
    }

    void copy_owner_to_host(
        const std::vector<int>& devices) {
        system::runtime::set_device(devices[static_cast<size_t>(owner_rank)]);
        system::runtime::check_cuda(
            cudaMemcpy(
                host_cache.data(),
                owner_device_ptr(),
                count * sizeof(uint32_t),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(DeviceMailbox owner -> host)");
    }

    void destroy(const std::vector<int>& devices) {
        if (owner_rank >= 0 && buf.bytes != 0) {
            comm::transport::free_comm_buffer(devices, buf);
        }
        buf = comm::transport::CommBuffer{};
        host_cache.clear();
        count = 0;
        owner_rank = -1;
    }
};

struct PersistentTimingBreakdown {
    double device_done_ms = 0.0;
    double host_wait_done_ms = 0.0;
    double stop_join_ms = 0.0;
    double total_ms = 0.0;
};

struct PersistentTwoGpuState {
    std::vector<int> devices{};

    comm::Group group{};
    std::vector<comm::EndpointRuntime> runtimes;
    std::vector<comm::EndpointPersistentControl> controls;

    std::vector<comm::transport::CommBuffer> accums;
    std::vector<comm::transport::CommBuffer> ready_queues;
    std::vector<DeviceMailbox> done_flags;

    std::vector<comm::collective::ChunkStateTable> chunk_states;
    std::vector<comm::collective::OperationDesc> ops;
    std::vector<comm::collective::DeviceOperationDesc> op_devs;

    std::vector<cudaStream_t> prep_streams;
    std::vector<cudaStream_t> timing_streams;
    std::vector<cudaEvent_t> timing_start_events;
    std::vector<cudaEvent_t> timing_stop_events;

    std::vector<uint32_t*> completion_counts;
    std::vector<uint32_t*> completion_flags;
    std::vector<uint32_t> host_completion_flags;

    size_t bytes = 0;
    uint32_t num_chunks = 0;
    bool initialized = false;
    bool kernels_running = false;
};

struct BasicCudaMemcpyState {
    int dev0 = -1;
    int dev1 = -1;
    cudaStream_t stream0 = nullptr;
    cudaStream_t stream1 = nullptr;
    half* work0 = nullptr;
    half* work1 = nullptr;
    half* inbox0 = nullptr;
    half* inbox1 = nullptr;
    size_t bytes = 0;
    bool initialized = false;
};

struct NcclAllreduceState {
    int dev0 = -1;
    int dev1 = -1;
    cudaStream_t stream0 = nullptr;
    cudaStream_t stream1 = nullptr;
    ncclComm_t comm0 = nullptr;
    ncclComm_t comm1 = nullptr;
    half* work0 = nullptr;
    half* work1 = nullptr;
    size_t bytes = 0;
    bool initialized = false;
};

void destroy_basic_cuda_memcpy_state(
    BasicCudaMemcpyState* st) {
    if (st == nullptr) {
        return;
    }

    try {
        if (st->work0 != nullptr) {
            system::runtime::set_device(st->dev0);
            cudaFree(st->work0);
        }
    } catch (...) {}
    try {
        if (st->inbox0 != nullptr) {
            system::runtime::set_device(st->dev0);
            cudaFree(st->inbox0);
        }
    } catch (...) {}
    try {
        if (st->work1 != nullptr) {
            system::runtime::set_device(st->dev1);
            cudaFree(st->work1);
        }
    } catch (...) {}
    try {
        if (st->inbox1 != nullptr) {
            system::runtime::set_device(st->dev1);
            cudaFree(st->inbox1);
        }
    } catch (...) {}

    try {
        if (st->stream0 != nullptr) {
            system::runtime::destroy_stream_on_device(st->dev0, st->stream0);
        }
    } catch (...) {}
    try {
        if (st->stream1 != nullptr) {
            system::runtime::destroy_stream_on_device(st->dev1, st->stream1);
        }
    } catch (...) {}

    st->dev0 = -1;
    st->dev1 = -1;
    st->stream0 = nullptr;
    st->stream1 = nullptr;
    st->work0 = nullptr;
    st->work1 = nullptr;
    st->inbox0 = nullptr;
    st->inbox1 = nullptr;
    st->bytes = 0;
    st->initialized = false;
}

void destroy_nccl_allreduce_state(
    NcclAllreduceState* st) {
    if (st == nullptr) {
        return;
    }

    try {
        if (st->comm0 != nullptr) {
            system::runtime::set_device(st->dev0);
            ncclCommDestroy(st->comm0);
        }
    } catch (...) {}
    try {
        if (st->comm1 != nullptr) {
            system::runtime::set_device(st->dev1);
            ncclCommDestroy(st->comm1);
        }
    } catch (...) {}

    try {
        if (st->work0 != nullptr) {
            system::runtime::set_device(st->dev0);
            cudaFree(st->work0);
        }
    } catch (...) {}
    try {
        if (st->work1 != nullptr) {
            system::runtime::set_device(st->dev1);
            cudaFree(st->work1);
        }
    } catch (...) {}

    try {
        if (st->stream0 != nullptr) {
            system::runtime::destroy_stream_on_device(st->dev0, st->stream0);
        }
    } catch (...) {}
    try {
        if (st->stream1 != nullptr) {
            system::runtime::destroy_stream_on_device(st->dev1, st->stream1);
        }
    } catch (...) {}

    st->dev0 = -1;
    st->dev1 = -1;
    st->stream0 = nullptr;
    st->stream1 = nullptr;
    st->comm0 = nullptr;
    st->comm1 = nullptr;
    st->work0 = nullptr;
    st->work1 = nullptr;
    st->bytes = 0;
    st->initialized = false;
}

void operation_desc_write_local_state(
    int device,
    const comm::collective::OperationDesc* desc,
    bool mark_done_one,
    bool seed_step0_queue) {
    if (device < 0) {
        throw std::invalid_argument("operation_desc_write_local_state: invalid device");
    }
    if (!comm::collective::operation_desc_is_valid(desc)) {
        throw std::invalid_argument("operation_desc_write_local_state: invalid desc");
    }

    const size_t queue_words = 2u + 2u * static_cast<size_t>(desc->num_chunks);
    std::vector<uint32_t> queue(queue_words, 0u);
    std::vector<uint32_t> done(desc->num_chunks, mark_done_one ? 1u : 0u);

    uint32_t* head_ptr = queue.data();
    uint32_t* tail_ptr = queue.data() + 1;
    auto* items = reinterpret_cast<comm::collective::ReadyItem*>(queue.data() + 2);

    *head_ptr = 0u;
    *tail_ptr = 0u;

    if (seed_step0_queue) {
        uint32_t tail = 0u;
        for (uint32_t idx = 0; idx < desc->num_chunks; ++idx) {
            if (comm::collective::operation_desc_actor_rank_for_step(desc, idx, 0u) ==
                desc->rank) {
                items[tail].chunk_idx = idx;
                items[tail].step = 0u;
                ++tail;
            }
        }
        *tail_ptr = tail;
    }

    system::runtime::set_device(device);
    system::runtime::check_cuda(
        cudaMemcpy(
            reinterpret_cast<void*>(desc->ready_queue_ptr),
            queue.data(),
            queue.size() * sizeof(uint32_t),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(operation ready_queue state)");

    system::runtime::check_cuda(
        cudaMemcpy(
            reinterpret_cast<void*>(desc->done_ptr),
            done.data(),
            static_cast<size_t>(desc->num_chunks) * sizeof(uint32_t),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(operation done state)");

    system::runtime::check_cuda(
        cudaMemset(
            reinterpret_cast<void*>(desc->chunk_states_ptr),
            0,
            static_cast<size_t>(desc->num_chunks) *
                sizeof(comm::collective::ChunkState)),
        "cudaMemset(operation chunk_states state)");
}

void operation_desc_seed_step0_only(
    int device,
    const comm::collective::OperationDesc* desc) {
    if (device < 0) {
        throw std::invalid_argument("operation_desc_seed_step0_only: invalid device");
    }
    if (!comm::collective::operation_desc_is_valid(desc)) {
        throw std::invalid_argument("operation_desc_seed_step0_only: invalid desc");
    }

    const size_t queue_words = 2u + 2u * static_cast<size_t>(desc->num_chunks);
    std::vector<uint32_t> queue(queue_words, 0u);

    uint32_t* head_ptr = queue.data();
    uint32_t* tail_ptr = queue.data() + 1;
    auto* items = reinterpret_cast<comm::collective::ReadyItem*>(queue.data() + 2);

    *head_ptr = 0u;
    *tail_ptr = 0u;

    uint32_t tail = 0u;
    for (uint32_t idx = 0; idx < desc->num_chunks; ++idx) {
        if (comm::collective::operation_desc_actor_rank_for_step(desc, idx, 0u) ==
            desc->rank) {
            items[tail].chunk_idx = idx;
            items[tail].step = 0u;
            ++tail;
        }
    }
    *tail_ptr = tail;

    system::runtime::set_device(device);
    system::runtime::check_cuda(
        cudaMemcpy(
            reinterpret_cast<void*>(desc->ready_queue_ptr),
            queue.data(),
            queue.size() * sizeof(uint32_t),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(operation ready_queue seed)");
}

void dump_persistent_debug_state(
    PersistentTwoGpuState* st,
    const char* tag) {
    if (st == nullptr) {
        return;
    }

    std::printf("[debug] persistent dump: %s\n", tag);

    for (size_t r = 0; r < st->devices.size(); ++r) {
        auto& done = st->done_flags[r];
        done.copy_owner_to_host(st->devices);

        uint32_t queue_meta[2] = {0, 0};
        system::runtime::set_device(st->devices[r]);
        system::runtime::check_cuda(
            cudaMemcpy(
                queue_meta,
                st->ready_queues[r].device_ptr_for_rank(r),
                2 * sizeof(uint32_t),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(queue meta -> host)");

        std::printf(
            "[debug] rank=%zu device=%d queue_head=%u queue_tail=%u\n",
            r,
            st->devices[r],
            queue_meta[0],
            queue_meta[1]);

        const size_t show = (st->num_chunks < 4u) ? st->num_chunks : 4u;
        for (size_t i = 0; i < show; ++i) {
            std::printf(
                "  done chunk=%zu value=%u\n",
                i,
                done.host_cache[i]);
        }

        std::vector<comm::collective::ChunkState> host_states(show);
        system::runtime::set_device(st->devices[r]);
        system::runtime::check_cuda(
            cudaMemcpy(
                host_states.data(),
                st->chunk_states[r].records,
                show * sizeof(comm::collective::ChunkState),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(chunk_states -> host)");

        for (size_t i = 0; i < show; ++i) {
            const auto& cs = host_states[i];
            std::printf(
                "  state chunk=%zu flags=%u started=%u completed=%u bytes=%u offset=%zu\n",
                i,
                cs.flags,
                cs.last_step_started,
                cs.last_step_completed,
                cs.bytes,
                cs.offset_bytes);
        }
    }
    std::fflush(stdout);
}

static void dump_full_done_progress(
    PersistentTwoGpuState* st,
    const char* tag) {
    std::printf("[debug] full done progress: %s\n", tag);

    for (size_t r = 0; r < st->done_flags.size(); ++r) {
        auto& done = st->done_flags[r];
        done.copy_owner_to_host(st->devices);

        size_t ones = 0;
        size_t first_bad = done.count;
        std::vector<size_t> bads;
        bads.reserve(8);

        for (size_t i = 0; i < done.count; ++i) {
            if (done.host_cache[i] == 1u) {
                ++ones;
            } else {
                if (first_bad == done.count) {
                    first_bad = i;
                }
                if (bads.size() < 8) {
                    bads.push_back(i);
                }
            }
        }

        uint32_t meta[2] = {0, 0};
        system::runtime::set_device(st->devices[r]);
        system::runtime::check_cuda(
            cudaMemcpy(
                meta,
                st->ready_queues[r].device_ptr_for_rank(r),
                2 * sizeof(uint32_t),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(queue meta -> host)");

        std::printf(
            "[debug] rank=%zu ones=%zu/%zu head=%u tail=%u",
            r, ones, done.count, meta[0], meta[1]);

        if (first_bad == done.count) {
            std::printf(" first_bad=none");
        } else {
            std::printf(" first_bad=%zu bads=", first_bad);
            for (size_t k = 0; k < bads.size(); ++k) {
                std::printf("%s%zu", (k == 0 ? "" : ","), bads[k]);
            }
        }
        std::printf("\n");
    }
    std::fflush(stdout);
}

struct WaitPollStats {
    uint64_t polls = 0;
    double memcpy_ms = 0.0;
};

bool wait_until_all_ranks_done_no_sleep(
    PersistentTwoGpuState* st,
    int timeout_ms,
    WaitPollStats* stats) {
    const auto start = std::chrono::steady_clock::now();

    while (true) {
        bool all_done = true;

        const auto copy_begin = std::chrono::steady_clock::now();
        for (size_t r = 0; r < st->devices.size(); ++r) {
            system::runtime::set_device(st->devices[r]);
            system::runtime::check_cuda(
                cudaMemcpy(
                    &st->host_completion_flags[r],
                    st->completion_flags[r],
                    sizeof(uint32_t),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(operation completion flag -> host)");

            if (st->host_completion_flags[r] != 1u) {
                all_done = false;
            }
        }
        const auto copy_end = std::chrono::steady_clock::now();

        if (stats != nullptr) {
            ++stats->polls;
            stats->memcpy_ms +=
                std::chrono::duration<double, std::milli>(copy_end - copy_begin).count();
        }

        if (all_done) {
            return true;
        }

        const auto now = std::chrono::steady_clock::now();
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed_ms > timeout_ms) {
            return false;
        }

        std::this_thread::yield();
    }
}

bool wait_until_all_timing_events_complete(
    PersistentTwoGpuState* st,
    int timeout_ms) {
    const auto start = std::chrono::steady_clock::now();
    uint64_t spins = 0;

    while (true) {
        bool all_done = true;

        for (size_t r = 0; r < st->timing_stop_events.size(); ++r) {
            system::runtime::set_device(st->devices[r]);
            const cudaError_t q = cudaEventQuery(st->timing_stop_events[r]);
            if (q == cudaSuccess) {
                continue;
            }
            if (q != cudaErrorNotReady) {
                system::runtime::check_cuda(q, "cudaEventQuery(persistent stop event)");
            }
            all_done = false;
            break;
        }

        if (all_done) {
            return true;
        }

        ++spins;
        if ((spins & ((1ull << 20) - 1ull)) == 0ull) {
            std::printf("[bench] wait timing events: still waiting spins=%llu\n",
                        static_cast<unsigned long long>(spins));
            std::fflush(stdout);
            dump_persistent_debug_state(st, "wait_until_all_timing_events_complete");
        }

        const auto now = std::chrono::steady_clock::now();
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed_ms > timeout_ms) {
            return false;
        }

        std::this_thread::yield();
    }
}

void destroy_persistent_two_gpu_state(
    PersistentTwoGpuState* st) {
    if (st == nullptr) {
        return;
    }

    if (st->kernels_running) {
        for (size_t r = 0; r < st->controls.size(); ++r) {
            try {
                comm::endpoint_persistent_control_request_stop(&st->controls[r]);
            } catch (...) {
            }
        }
        for (size_t r = 0; r < st->runtimes.size(); ++r) {
            try {
                system::runtime::check_cuda(
                    cudaStreamSynchronize(st->runtimes[r].endpoint.stream),
                    "cudaStreamSynchronize(destroy persistent stream)");
            } catch (...) {
            }
        }
        st->kernels_running = false;
    }

    for (size_t r = 0; r < st->timing_start_events.size(); ++r) {
        try {
            if (st->devices.size() == st->timing_start_events.size()) {
                system::runtime::set_device(st->devices[r]);
            }
            if (st->timing_start_events[r] != nullptr) {
                cudaEventDestroy(st->timing_start_events[r]);
            }
        } catch (...) {
        }
    }
    for (size_t r = 0; r < st->timing_stop_events.size(); ++r) {
        try {
            if (st->devices.size() == st->timing_stop_events.size()) {
                system::runtime::set_device(st->devices[r]);
            }
            if (st->timing_stop_events[r] != nullptr) {
                cudaEventDestroy(st->timing_stop_events[r]);
            }
        } catch (...) {
        }
    }
    for (size_t r = 0; r < st->prep_streams.size(); ++r) {
    try {
        if (st->prep_streams[r] != nullptr) {
            system::runtime::destroy_stream_on_device(
                st->devices[r],
                st->prep_streams[r]);
        }
    } catch (...) {
    }
}
st->prep_streams.clear();
    for (size_t r = 0; r < st->timing_streams.size(); ++r) {
        try {
            if (st->timing_streams[r] != nullptr) {
                system::runtime::destroy_stream_on_device(
                    st->devices[r],
                    st->timing_streams[r]);
            }
        } catch (...) {
        }
    }

    for (auto& op_dev : st->op_devs) {
        try {
            comm::collective::device_operation_desc_destroy(&op_dev);
        } catch (...) {
        }
    }
    for (auto& table : st->chunk_states) {
        try {
            comm::collective::chunk_state_table_destroy(&table);
        } catch (...) {
        }
    }
    for (auto& box : st->done_flags) {
        try {
            box.destroy(st->devices);
        } catch (...) {
        }
    }
    for (auto& q : st->ready_queues) {
        try {
            comm::transport::free_comm_buffer(st->devices, q);
        } catch (...) {
        }
    }
    for (auto& accum : st->accums) {
        try {
            comm::transport::free_comm_buffer(st->devices, accum);
        } catch (...) {
        }
    }
    for (auto& ctl : st->controls) {
        try {
            comm::endpoint_persistent_control_destroy(&ctl);
        } catch (...) {
        }
    }
    for (auto& rt : st->runtimes) {
        try {
            comm::endpoint_runtime_destroy(&rt);
        } catch (...) {
        }
    }
    for (size_t r = 0; r < st->completion_counts.size(); ++r) {
        try {
            if (st->completion_counts[r] != nullptr) {
                system::runtime::set_device(st->devices[r]);
                cudaFree(st->completion_counts[r]);
            }
        } catch (...) {}
    }
    for (size_t r = 0; r < st->completion_flags.size(); ++r) {
        try {
            if (st->completion_flags[r] != nullptr) {
                system::runtime::set_device(st->devices[r]);
                cudaFree(st->completion_flags[r]);
            }
        } catch (...) {}
    }

    try {
        comm::group_destroy(&st->group);
    } catch (...) {
    }

    st->devices.clear();
    st->runtimes.clear();
    st->controls.clear();
    st->accums.clear();
    st->ready_queues.clear();
    st->done_flags.clear();
    st->chunk_states.clear();
    st->ops.clear();
    st->op_devs.clear();
    st->timing_streams.clear();
    st->timing_start_events.clear();
    st->timing_stop_events.clear();
    st->completion_counts.clear();
    st->completion_flags.clear();
    st->host_completion_flags.clear();
    st->bytes = 0;
    st->num_chunks = 0;
    st->initialized = false;
    st->kernels_running = false;
}

void init_persistent_two_gpu_state(
    PersistentTwoGpuState* st,
    int dev0,
    int dev1,
    size_t numel) {
    if (st == nullptr) {
        throw std::invalid_argument("init_persistent_two_gpu_state: st is null");
    }

    destroy_persistent_two_gpu_state(st);

    st->devices = {dev0, dev1};
    st->bytes = numel * sizeof(half);
    st->num_chunks = comm::collective::operation_desc_compute_num_chunks(
        st->bytes,
        comm::kEndpointPersistentChunkBytes);

    st->runtimes.resize(2);
    st->controls.resize(2);
    st->accums.resize(2);
    st->ready_queues.resize(2);
    st->done_flags.resize(2);
    st->chunk_states.resize(2);
    st->ops.resize(2);
    st->op_devs.resize(2);
    st->timing_streams.resize(2, nullptr);
    st->timing_start_events.resize(2, nullptr);
    st->timing_stop_events.resize(2, nullptr);
    st->prep_streams.resize(2, nullptr);
    st->completion_counts.resize(2, nullptr);
    st->completion_flags.resize(2, nullptr);
    st->host_completion_flags.assign(2, 0u);

    comm::group_init(&st->group, st->devices, comm::kEndpointPersistentChunkBytes);

    for (int r = 0; r < 2; ++r) {
        comm::endpoint_runtime_init(&st->runtimes[static_cast<size_t>(r)], &st->group, r);
    }

    const size_t queue_bytes =
        2 * sizeof(uint32_t) +
        static_cast<size_t>(st->num_chunks) * sizeof(comm::collective::ReadyItem);

    for (int r = 0; r < 2; ++r) {
        const int prev_rank = prev_rank_of(r, 2);

        st->accums[static_cast<size_t>(r)] =
            comm::transport::alloc_peer_visible_buffer_for_rank_with_access_ranks(
                st->group.devices,
                r,
                {r, prev_rank},
                st->bytes);

        st->ready_queues[static_cast<size_t>(r)] =
            comm::transport::alloc_peer_visible_buffer_for_rank_with_access_ranks(
                st->group.devices,
                r,
                {r, prev_rank},
                queue_bytes);

        st->done_flags[static_cast<size_t>(r)].init(
            st->group.devices,
            r,
            {r, prev_rank},
            st->num_chunks);

        comm::collective::chunk_state_table_init(
            &st->chunk_states[static_cast<size_t>(r)],
            st->group.devices[static_cast<size_t>(r)],
            st->num_chunks);
    }

    for (int r = 0; r < 2; ++r) {
        const int next_rank = (r + 1) % 2;

        comm::endpoint_runtime_build_ring_allreduce_operation(
            &st->runtimes[static_cast<size_t>(r)],
            &st->ops[static_cast<size_t>(r)],
            st->accums[static_cast<size_t>(r)].device_ptr_for_rank(static_cast<size_t>(r)),
            st->accums[static_cast<size_t>(next_rank)].device_ptr_for_rank(static_cast<size_t>(r)),
            st->bytes,
            comm::kEndpointPersistentChunkBytes,
            st->ready_queues[static_cast<size_t>(r)].device_ptr_for_rank(static_cast<size_t>(r)),
            st->ready_queues[static_cast<size_t>(next_rank)].device_ptr_for_rank(static_cast<size_t>(r)),
            st->done_flags[static_cast<size_t>(r)].device_ptr_for_rank(static_cast<size_t>(r)),
            st->done_flags[static_cast<size_t>(next_rank)].device_ptr_for_rank(static_cast<size_t>(r)),
            st->chunk_states[static_cast<size_t>(r)].records,
            1);

        system::runtime::set_device(st->group.devices[static_cast<size_t>(r)]);

        system::runtime::check_cuda(
            cudaMalloc(&st->completion_counts[static_cast<size_t>(r)], sizeof(uint32_t)),
            "cudaMalloc(completion count)");
        
        system::runtime::check_cuda(
            cudaMalloc(&st->completion_flags[static_cast<size_t>(r)], sizeof(uint32_t)),
            "cudaMalloc(completion flag)");
        
        st->ops[static_cast<size_t>(r)].completion_count_ptr =
            reinterpret_cast<uint64_t>(st->completion_counts[static_cast<size_t>(r)]);
        st->ops[static_cast<size_t>(r)].completion_flag_ptr =
            reinterpret_cast<uint64_t>(st->completion_flags[static_cast<size_t>(r)]);
        st->ops[static_cast<size_t>(r)].completion_target = comm::collective::operation_desc_local_completion_target(&st->ops[r]);

        comm::collective::device_operation_desc_create(
            &st->op_devs[static_cast<size_t>(r)],
            st->group.devices[static_cast<size_t>(r)],
            &st->ops[static_cast<size_t>(r)]);

        comm::endpoint_persistent_control_init(
            &st->controls[static_cast<size_t>(r)],
            st->group.devices[static_cast<size_t>(r)]);

        st->prep_streams[static_cast<size_t>(r)] =
            system::runtime::create_stream_on_device(
                st->group.devices[static_cast<size_t>(r)]);

        st->timing_streams[static_cast<size_t>(r)] =
            system::runtime::create_stream_on_device(st->group.devices[static_cast<size_t>(r)]);

        system::runtime::set_device(st->group.devices[static_cast<size_t>(r)]);
        system::runtime::check_cuda(
            cudaEventCreate(&st->timing_start_events[static_cast<size_t>(r)]),
            "cudaEventCreate(persistent timing start)");
        system::runtime::check_cuda(
            cudaEventCreate(&st->timing_stop_events[static_cast<size_t>(r)]),
            "cudaEventCreate(persistent timing stop)");

        operation_desc_write_local_state(
            st->group.devices[static_cast<size_t>(r)],
            &st->ops[static_cast<size_t>(r)],
            true,
            false);
    }

    st->initialized = true;
    st->kernels_running = false;
}

void launch_persistent_two_gpu_run(
    PersistentTwoGpuState* st) {
    if (st == nullptr || !st->initialized) {
        throw std::invalid_argument("launch_persistent_two_gpu_run: state is not initialized");
    }
    if (st->kernels_running) {
        return;
    }

    for (int r = 0; r < 2; ++r) {
        system::runtime::check_cuda(
            comm::launch_endpoint_persistent_kernel_sm90(
                comm::endpoint_runtime_device_handle(&st->runtimes[static_cast<size_t>(r)]),
                st->op_devs[static_cast<size_t>(r)].ptr,
                &st->controls[static_cast<size_t>(r)],
                st->runtimes[static_cast<size_t>(r)].endpoint.stream),
            "launch_endpoint_persistent_kernel_sm90");
    }

    st->kernels_running = true;
}

void arm_persistent_timing_for_current_run(
    PersistentTwoGpuState* st) {
    if (st == nullptr || !st->initialized) {
        throw std::invalid_argument("arm_persistent_timing_for_current_run: state is not initialized");
    }
    if (st->num_chunks == 0) {
        throw std::invalid_argument("arm_persistent_timing_for_current_run: num_chunks is zero");
    }

    for (int r = 0; r < 2; ++r) {
        system::runtime::set_device(st->devices[static_cast<size_t>(r)]);

        system::runtime::check_cuda(
            cudaEventRecord(
                st->timing_start_events[static_cast<size_t>(r)],
                st->timing_streams[static_cast<size_t>(r)]),
            "cudaEventRecord(persistent timing start)");

        auto* last_done_ptr =
            st->done_flags[static_cast<size_t>(r)].owner_device_ptr() +
            static_cast<ptrdiff_t>(st->num_chunks - 1u);

        check_driver(
            cuStreamWaitValue32(
                reinterpret_cast<CUstream>(st->timing_streams[static_cast<size_t>(r)]),
                static_cast<CUdeviceptr>(
                    reinterpret_cast<uintptr_t>(last_done_ptr)),
                1u,
                CU_STREAM_WAIT_VALUE_EQ),
            "cuStreamWaitValue32(last done == 1)");

        system::runtime::check_cuda(
            cudaEventRecord(
                st->timing_stop_events[static_cast<size_t>(r)],
                st->timing_streams[static_cast<size_t>(r)]),
            "cudaEventRecord(persistent timing stop)");
    }
}

void prepare_persistent_two_gpu_run(
    PersistentTwoGpuState* st,
    half* rank0_src,
    half* rank1_src) {
    if (st == nullptr || !st->initialized) {
        throw std::invalid_argument("prepare_persistent_two_gpu_run: state is not initialized");
    }

    half* inputs[2] = {rank0_src, rank1_src};

    for (int r = 0; r < 2; ++r) {
        comm::endpoint_persistent_control_reset(&st->controls[static_cast<size_t>(r)]);
    }

    for (int r = 0; r < 2; ++r) {
        system::runtime::set_device(st->group.devices[static_cast<size_t>(r)]);
        system::runtime::check_cuda(
            cudaMemcpyAsync(
                st->accums[static_cast<size_t>(r)].device_ptr_for_rank(static_cast<size_t>(r)),
                inputs[static_cast<size_t>(r)],
                st->bytes,
                cudaMemcpyDeviceToDevice,
                st->runtimes[static_cast<size_t>(r)].endpoint.stream),
            "cudaMemcpyAsync(src -> accum)");
    }

    sync_two_streams(
        st->group.devices[0], st->runtimes[0].endpoint.stream,
        st->group.devices[1], st->runtimes[1].endpoint.stream,
        "sync prepare_persistent");

    for (int r = 0; r < 2; ++r) {
        comm::collective::operation_desc_reset_local_state(
            st->group.devices[static_cast<size_t>(r)],
            &st->ops[static_cast<size_t>(r)]);
    }
}

PersistentTimingBreakdown measure_persistent_host_breakdown_ms(
    PersistentTwoGpuState* st,
    int timeout_ms) {
    if (st == nullptr || !st->initialized) {
        throw std::invalid_argument("measure_persistent_host_breakdown_ms: state is not initialized");
    }

    WaitPollStats stats{};
    const auto t0 = std::chrono::steady_clock::now();

    launch_persistent_two_gpu_run(st);

    const bool all_done =
        wait_until_all_ranks_done_no_sleep(st, timeout_ms, &stats);

    const auto t1 = std::chrono::steady_clock::now();

    printf("[wait-prof] polls=%llu memcpy_ms=%.6f avg_copy_us=%.3f\n",
           (unsigned long long)stats.polls,
           stats.memcpy_ms,
           stats.polls ? (1000.0 * stats.memcpy_ms / (double)stats.polls) : 0.0);
    fflush(stdout);

    if (!all_done) {
        for (int r = 0; r < 2; ++r) {
            comm::endpoint_persistent_control_request_stop(
                &st->controls[static_cast<size_t>(r)]);
        }

        for (int r = 0; r < 2; ++r) {
            system::runtime::check_cuda(
                cudaStreamSynchronize(
                    st->runtimes[static_cast<size_t>(r)].endpoint.stream),
                "cudaStreamSynchronize(persistent stream timeout flush)");
        }

        st->kernels_running = false;
        //dump_persistent_debug_state(st, "timeout in measure_persistent_host_breakdown_ms");
        //dump_full_done_progress(st, "timeout");
        throw std::runtime_error(
            "measure_persistent_host_breakdown_ms: timeout waiting for done flags");
    }

    // Cleanup is intentionally OUTSIDE the timed window.
    for (int r = 0; r < 2; ++r) {
        comm::endpoint_persistent_control_request_stop(
            &st->controls[static_cast<size_t>(r)]);
    }

    const auto stop_begin = std::chrono::steady_clock::now();

    for (int r = 0; r < 2; ++r) {
        system::runtime::check_cuda(
            cudaStreamSynchronize(
                st->runtimes[static_cast<size_t>(r)].endpoint.stream),
            "cudaStreamSynchronize(persistent stream after timing)");
    }

    st->kernels_running = false;

    const auto stop_end = std::chrono::steady_clock::now();

    PersistentTimingBreakdown out{};
    out.device_done_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    out.host_wait_done_ms = out.device_done_ms;
    out.stop_join_ms =
        std::chrono::duration<double, std::milli>(stop_end - stop_begin).count();

    // "total" now means launch -> all_done only.
    out.total_ms = out.device_done_ms;
    return out;
}

void verify_persistent_result(
    PersistentTwoGpuState* st,
    int64_t numel) {
    auto ref = reference_two_gpu_sum(numel);

    auto got0 = copy_half_device_to_host(
        reinterpret_cast<const half*>(st->accums[0].device_ptr_for_rank(0)),
        numel,
        st->devices[0]);

    auto got1 = copy_half_device_to_host(
        reinterpret_cast<const half*>(st->accums[1].device_ptr_for_rank(1)),
        numel,
        st->devices[1]);

    expect_half_vectors_close(got0, ref, "persistent verify rank0");
    expect_half_vectors_close(got1, ref, "persistent verify rank1");
}

PersistentTimingBreakdown run_one_fresh_persistent_iteration(
    int dev0,
    int dev1,
    size_t numel,
    half* rank0_src,
    half* rank1_src,
    int timeout_ms,
    bool verify_result) {
    PersistentTwoGpuState st{};

    try {
        init_persistent_two_gpu_state(&st, dev0, dev1, numel);
        prepare_persistent_two_gpu_run(&st, rank0_src, rank1_src);

        const auto timing =
            measure_persistent_host_breakdown_ms(&st, timeout_ms);

        if (false) {
            verify_persistent_result(&st, static_cast<int64_t>(numel));
        }

        destroy_persistent_two_gpu_state(&st);
        return timing;
    } catch (...) {
        destroy_persistent_two_gpu_state(&st);
        throw;
    }
}

void init_basic_cuda_memcpy_state(
    BasicCudaMemcpyState* st,
    int dev0,
    int dev1,
    size_t numel) {
    if (st == nullptr) {
        throw std::invalid_argument("init_basic_cuda_memcpy_state: st is null");
    }

    destroy_basic_cuda_memcpy_state(st);

    st->dev0 = dev0;
    st->dev1 = dev1;
    st->bytes = numel * sizeof(half);

    system::runtime::ensure_context_on_device(dev0);
    system::runtime::ensure_context_on_device(dev1);

    st->stream0 = system::runtime::create_stream_on_device(dev0);
    st->stream1 = system::runtime::create_stream_on_device(dev1);

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(cudaMalloc(&st->work0, st->bytes), "cudaMalloc(work0)");
    system::runtime::check_cuda(cudaMalloc(&st->inbox0, st->bytes), "cudaMalloc(inbox0)");

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(cudaMalloc(&st->work1, st->bytes), "cudaMalloc(work1)");
    system::runtime::check_cuda(cudaMalloc(&st->inbox1, st->bytes), "cudaMalloc(inbox1)");

    st->initialized = true;
}

void prepare_basic_cuda_memcpy_run(
    BasicCudaMemcpyState* st,
    half* rank0_src,
    half* rank1_src) {
    if (st == nullptr || !st->initialized) {
        throw std::invalid_argument("prepare_basic_cuda_memcpy_run: state is not initialized");
    }

    system::runtime::set_device(st->dev0);
    system::runtime::check_cuda(
        cudaMemcpyAsync(
            st->work0,
            rank0_src,
            st->bytes,
            cudaMemcpyDeviceToDevice,
            st->stream0),
        "cudaMemcpyAsync(src0 -> work0)");
    system::runtime::check_cuda(
        cudaMemsetAsync(
            st->inbox0,
            0,
            st->bytes,
            st->stream0),
        "cudaMemsetAsync(inbox0)");

    system::runtime::set_device(st->dev1);
    system::runtime::check_cuda(
        cudaMemcpyAsync(
            st->work1,
            rank1_src,
            st->bytes,
            cudaMemcpyDeviceToDevice,
            st->stream1),
        "cudaMemcpyAsync(src1 -> work1)");
    system::runtime::check_cuda(
        cudaMemsetAsync(
            st->inbox1,
            0,
            st->bytes,
            st->stream1),
        "cudaMemsetAsync(inbox1)");

    sync_two_streams(
        st->dev0, st->stream0,
        st->dev1, st->stream1,
        "sync prepare_basic_cuda_memcpy_run");
}

void run_basic_cuda_memcpy_allreduce(
    BasicCudaMemcpyState* st,
    size_t numel) {
    if (st == nullptr || !st->initialized) {
        throw std::invalid_argument("run_basic_cuda_memcpy_allreduce: state is not initialized");
    }

    system::runtime::set_device(st->dev0);
    system::runtime::check_cuda(
        cudaMemcpyPeerAsync(
            st->inbox1,
            st->dev1,
            st->work0,
            st->dev0,
            st->bytes,
            st->stream0),
        "cudaMemcpyPeerAsync(work0 -> inbox1)");

    system::runtime::set_device(st->dev1);
    system::runtime::check_cuda(
        cudaMemcpyPeerAsync(
            st->inbox0,
            st->dev0,
            st->work1,
            st->dev1,
            st->bytes,
            st->stream1),
        "cudaMemcpyPeerAsync(work1 -> inbox0)");

    sync_two_streams(
        st->dev0, st->stream0,
        st->dev1, st->stream1,
        "sync memcpy phase");

    system::runtime::set_device(st->dev0);
    system::runtime::check_cuda(
        enqueue_fp16_add_inplace(st->work0, st->inbox0, numel, st->stream0),
        "enqueue_fp16_add_inplace rank0");

    system::runtime::set_device(st->dev1);
    system::runtime::check_cuda(
        enqueue_fp16_add_inplace(st->work1, st->inbox1, numel, st->stream1),
        "enqueue_fp16_add_inplace rank1");

    sync_two_streams(
        st->dev0, st->stream0,
        st->dev1, st->stream1,
        "sync add phase");
}

void verify_basic_cuda_memcpy_result(
    BasicCudaMemcpyState* st,
    int64_t numel) {
    auto ref = reference_two_gpu_sum(numel);

    auto got0 = copy_half_device_to_host(st->work0, numel, st->dev0);
    auto got1 = copy_half_device_to_host(st->work1, numel, st->dev1);

    expect_half_vectors_close(got0, ref, "basic memcpy verify rank0");
    expect_half_vectors_close(got1, ref, "basic memcpy verify rank1");
}


void prepare_nccl_allreduce_run(
    NcclAllreduceState* st,
    half* rank0_src,
    half* rank1_src) {
    if (st == nullptr || !st->initialized) {
        throw std::invalid_argument("prepare_nccl_allreduce_run: state is not initialized");
    }

    system::runtime::set_device(st->dev0);
    system::runtime::check_cuda(
        cudaMemcpyAsync(
            st->work0,
            rank0_src,
            st->bytes,
            cudaMemcpyDeviceToDevice,
            st->stream0),
        "cudaMemcpyAsync(src0 -> nccl work0)");

    system::runtime::set_device(st->dev1);
    system::runtime::check_cuda(
        cudaMemcpyAsync(
            st->work1,
            rank1_src,
            st->bytes,
            cudaMemcpyDeviceToDevice,
            st->stream1),
        "cudaMemcpyAsync(src1 -> nccl work1)");
}

void init_nccl_allreduce_state(
    NcclAllreduceState* st,
    int dev0,
    int dev1,
    size_t numel) {
    if (st == nullptr) {
        throw std::invalid_argument("init_nccl_allreduce_state: st is null");
    }

    destroy_nccl_allreduce_state(st);

    st->dev0 = dev0;
    st->dev1 = dev1;
    st->bytes = numel * sizeof(half);

    system::runtime::ensure_context_on_device(dev0);
    system::runtime::ensure_context_on_device(dev1);

    st->stream0 = system::runtime::create_stream_on_device(dev0);
    st->stream1 = system::runtime::create_stream_on_device(dev1);

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(cudaMalloc(&st->work0, st->bytes), "cudaMalloc(nccl work0)");

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(cudaMalloc(&st->work1, st->bytes), "cudaMalloc(nccl work1)");

    ncclUniqueId id{};
    OOVERLAP_PERSIST_NCCL_CHECK(ncclGetUniqueId(&id));

    OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupStart());

    system::runtime::set_device(dev0);
    OOVERLAP_PERSIST_NCCL_CHECK(ncclCommInitRank(&st->comm0, 2, id, 0));

    system::runtime::set_device(dev1);
    OOVERLAP_PERSIST_NCCL_CHECK(ncclCommInitRank(&st->comm1, 2, id, 1));

    OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupEnd());

    st->initialized = true;
}

void run_nccl_allreduce(
    NcclAllreduceState* st,
    size_t numel) {
    if (st == nullptr || !st->initialized) {
        throw std::invalid_argument("run_nccl_allreduce: state is not initialized");
    }

    OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupStart());

    system::runtime::set_device(st->dev0);
    OOVERLAP_PERSIST_NCCL_CHECK(
        ncclAllReduce(
            st->work0,
            st->work0,
            numel,
            ncclHalf,
            ncclSum,
            st->comm0,
            st->stream0));

    system::runtime::set_device(st->dev1);
    OOVERLAP_PERSIST_NCCL_CHECK(
        ncclAllReduce(
            st->work1,
            st->work1,
            numel,
            ncclHalf,
            ncclSum,
            st->comm1,
            st->stream1));

    OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupEnd());

    sync_two_streams(
        st->dev0, st->stream0,
        st->dev1, st->stream1,
        "sync nccl allreduce");
}

void verify_nccl_result(
    NcclAllreduceState* st,
    int64_t numel) {
    auto ref = reference_two_gpu_sum(numel);

    auto got0 = copy_half_device_to_host(st->work0, numel, st->dev0);
    auto got1 = copy_half_device_to_host(st->work1, numel, st->dev1);

    expect_half_vectors_close(got0, ref, "nccl verify rank0");
    expect_half_vectors_close(got1, ref, "nccl verify rank1");
}

} // namespace

std::map<std::string, double> benchmark_persistent_two_gpu_allreduce_sm90(
    int64_t numel,
    int iters,
    int warmup,
    int dev0,
    int dev1) {
    if (numel <= 0 || iters <= 0 || warmup < 0) {
        throw std::invalid_argument("benchmark_persistent_two_gpu_allreduce_sm90: invalid args");
    }
    if (dev0 == dev1) {
        throw std::invalid_argument("benchmark_persistent_two_gpu_allreduce_sm90: dev0 and dev1 must differ");
    }

    const size_t bytes = static_cast<size_t>(numel) * sizeof(half);
    const auto host0 = make_host_pattern(numel, 0);
    const auto host1 = make_host_pattern(numel, 1);

    half* rank0_src = nullptr;
    half* rank1_src = nullptr;

    cudaStream_t upload0 = nullptr;
    cudaStream_t upload1 = nullptr;

    BasicCudaMemcpyState basic{};
    NcclAllreduceState nccl{};

    try {
        system::runtime::ensure_context_on_device(dev0);
        system::runtime::ensure_context_on_device(dev1);

        upload0 = system::runtime::create_stream_on_device(dev0);
        upload1 = system::runtime::create_stream_on_device(dev1);

        system::runtime::set_device(dev0);
        system::runtime::check_cuda(cudaMalloc(&rank0_src, bytes), "cudaMalloc(rank0_src)");

        system::runtime::set_device(dev1);
        system::runtime::check_cuda(cudaMalloc(&rank1_src, bytes), "cudaMalloc(rank1_src)");

        upload_host_half_vector(
            rank0_src,
            host0,
            dev0,
            upload0,
            "cudaMemcpyAsync(host0 -> rank0_src)");
        upload_host_half_vector(
            rank1_src,
            host1,
            dev1,
            upload1,
            "cudaMemcpyAsync(host1 -> rank1_src)");

        sync_two_streams(
            dev0, upload0,
            dev1, upload1,
            "sync source upload");

        init_basic_cuda_memcpy_state(&basic, dev0, dev1, static_cast<size_t>(numel));
        init_nccl_allreduce_state(&nccl, dev0, dev1, static_cast<size_t>(numel));

        // Persistent warmup: fresh state every iteration
        double persistent_device_done_ms = 0.0;
        double persistent_host_wait_done_ms = 0.0;
        double persistent_stop_join_ms = 0.0;
        double persistent_total_ms = 0.0;

        for (int i = 0; i < warmup; ++i) {
            (void)run_one_fresh_persistent_iteration(
                dev0,
                dev1,
                static_cast<size_t>(numel),
                rank0_src,
                rank1_src,
                5000,
                false);
        }

        for (int i = 0; i < iters; ++i) {
            const auto timing =
                run_one_fresh_persistent_iteration(
                    dev0,
                    dev1,
                    static_cast<size_t>(numel),
                    rank0_src,
                    rank1_src,
                    5000,
                    false);

            persistent_device_done_ms += timing.device_done_ms;
            persistent_host_wait_done_ms += timing.host_wait_done_ms;
            persistent_stop_join_ms += timing.stop_join_ms;
            persistent_total_ms += timing.total_ms;
        }

        // One extra untimed correctness check with a fresh run.
        (void)run_one_fresh_persistent_iteration(
            dev0,
            dev1,
            static_cast<size_t>(numel),
            rank0_src,
            rank1_src,
            5000,
            true);

        // Basic memcpy baseline
        for (int i = 0; i < warmup; ++i) {
            prepare_basic_cuda_memcpy_run(&basic, rank0_src, rank1_src);
            run_basic_cuda_memcpy_allreduce(&basic, static_cast<size_t>(numel));
        }

        double basic_ms = 0.0;
        for (int i = 0; i < iters; ++i) {
            prepare_basic_cuda_memcpy_run(&basic, rank0_src, rank1_src);
            basic_ms += measure_host_ms([&]() {
                run_basic_cuda_memcpy_allreduce(&basic, static_cast<size_t>(numel));
            });
        }

        verify_basic_cuda_memcpy_result(&basic, numel);

        // NCCL baseline
        for (int i = 0; i < warmup; ++i) {
            prepare_nccl_allreduce_run(&nccl, rank0_src, rank1_src);
            run_nccl_allreduce(&nccl, static_cast<size_t>(numel));
        }

        double nccl_ms = 0.0;
        for (int i = 0; i < iters; ++i) {
            prepare_nccl_allreduce_run(&nccl, rank0_src, rank1_src);
            nccl_ms += measure_host_ms([&]() {
                run_nccl_allreduce(&nccl, static_cast<size_t>(numel));
            });
        }

        verify_nccl_result(&nccl, numel);

        destroy_nccl_allreduce_state(&nccl);
        destroy_basic_cuda_memcpy_state(&basic);

        if (upload0 != nullptr) {
            system::runtime::destroy_stream_on_device(dev0, upload0);
            upload0 = nullptr;
        }
        if (upload1 != nullptr) {
            system::runtime::destroy_stream_on_device(dev1, upload1);
            upload1 = nullptr;
        }

        if (rank0_src != nullptr) {
            system::runtime::set_device(dev0);
            cudaFree(rank0_src);
            rank0_src = nullptr;
        }
        if (rank1_src != nullptr) {
            system::runtime::set_device(dev1);
            cudaFree(rank1_src);
            rank1_src = nullptr;
        }

        const double avg_ms_persistent_launch_to_done =
            persistent_device_done_ms / static_cast<double>(iters);
        const double avg_ms_persistent_host_wait_done =
            persistent_host_wait_done_ms / static_cast<double>(iters);
        const double avg_ms_persistent_stop_join =
            persistent_stop_join_ms / static_cast<double>(iters);
        const double avg_ms_persistent_total =
            persistent_total_ms / static_cast<double>(iters);

        const double avg_ms_cuda_memcpy =
            basic_ms / static_cast<double>(iters);
        const double avg_ms_nccl =
            nccl_ms / static_cast<double>(iters);

        return {
            {"avg_ms_cuda_memcpy", avg_ms_cuda_memcpy},
            {"avg_ms_nccl", avg_ms_nccl},
            {"avg_ms_persistent_host_wait_done", avg_ms_persistent_host_wait_done},
            {"avg_ms_persistent_launch_to_done", avg_ms_persistent_launch_to_done},
            {"avg_ms_persistent_stop_join", avg_ms_persistent_stop_join},
            {"avg_ms_persistent_total", avg_ms_persistent_total},
            {"cuda_memcpy_over_persistent_total",
             avg_ms_persistent_total > 0.0
                 ? avg_ms_cuda_memcpy / avg_ms_persistent_total
                 : 0.0},
            {"nccl_over_persistent_total",
             avg_ms_persistent_total > 0.0
                 ? avg_ms_nccl / avg_ms_persistent_total
                 : 0.0},
            {"numel", static_cast<double>(numel)},
        };
    } catch (...) {
        destroy_nccl_allreduce_state(&nccl);
        destroy_basic_cuda_memcpy_state(&basic);

        try {
            if (upload0 != nullptr) {
                system::runtime::destroy_stream_on_device(dev0, upload0);
            }
        } catch (...) {
        }
        try {
            if (upload1 != nullptr) {
                system::runtime::destroy_stream_on_device(dev1, upload1);
            }
        } catch (...) {
        }

        try {
            if (rank0_src != nullptr) {
                system::runtime::set_device(dev0);
                cudaFree(rank0_src);
            }
        } catch (...) {
        }
        try {
            if (rank1_src != nullptr) {
                system::runtime::set_device(dev1);
                cudaFree(rank1_src);
            }
        } catch (...) {
        }

        throw;
    }
}

} // namespace ooverlap
