#include "test/persistent_allreduce_2gpu_sm90.h"

#include "comm/collective/operation.h"
#include "comm/endpoint_persistent_kernel.h"
#include "comm/endpoint_runtime.h"
#include "comm/group.h"
#include "comm/transport/buffer.h"
#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define OOVERLAP_PERSIST_NCCL_CHECK(cmd)                                                     \
    do {                                                                                     \
        ncclResult_t result__ = (cmd);                                                       \
        if (result__ != ncclSuccess) {                                                       \
            throw std::runtime_error(std::string("NCCL error: ") + ncclGetErrorString(result__)); \
        }                                                                                    \
    } while (0)

namespace ooverlap {
namespace {

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

__global__ void wait_done_flags_kernel(
    const uint32_t* done,
    size_t count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    volatile const uint32_t* vdone =
        reinterpret_cast<volatile const uint32_t*>(done);

    while (true) {
        bool all_done = true;
        for (size_t i = 0; i < count; ++i) {
            if (vdone[i] != 1u) {
                all_done = false;
                break;
            }
        }

        if (all_done) {
            return;
        }

#if defined(__CUDA_ARCH__)
        __nanosleep(256);
#endif
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

struct HostMappedMailbox {
    uint32_t* host_ptr = nullptr;
    std::vector<uint32_t*> device_ptrs;
    size_t count = 0;

    void init(size_t n, const std::vector<int>& devices) {
        if (n == 0) {
            throw std::invalid_argument("HostMappedMailbox::init: n must be > 0");
        }
        destroy();

        count = n;

        cudaError_t err = cudaHostAlloc(
            reinterpret_cast<void**>(&host_ptr),
            count * sizeof(uint32_t),
            cudaHostAllocMapped | cudaHostAllocPortable);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaHostAlloc(mapped mailbox) failed: ") +
                cudaGetErrorString(err));
        }

        device_ptrs.resize(devices.size(), nullptr);

        for (size_t i = 0; i < devices.size(); ++i) {
            system::runtime::set_device(devices[i]);
            err = cudaHostGetDevicePointer(
                reinterpret_cast<void**>(&device_ptrs[i]),
                host_ptr,
                0);
            if (err != cudaSuccess) {
                cudaFreeHost(host_ptr);
                host_ptr = nullptr;
                device_ptrs.clear();
                count = 0;
                throw std::runtime_error(
                    std::string("cudaHostGetDevicePointer(mapped mailbox) failed: ") +
                    cudaGetErrorString(err));
            }
        }
    }

    void reset(uint32_t value) {
        if (host_ptr == nullptr) {
            throw std::invalid_argument("HostMappedMailbox::reset: not initialized");
        }
        for (size_t i = 0; i < count; ++i) {
            host_ptr[i] = value;
        }
    }

    uint32_t* device_ptr_for_rank(size_t rank) const {
        return device_ptrs.at(rank);
    }

    void destroy() {
        if (host_ptr != nullptr) {
            cudaFreeHost(host_ptr);
        }
        host_ptr = nullptr;
        device_ptrs.clear();
        count = 0;
    }
};

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

bool wait_until_all_ranks_done(
    const std::vector<HostMappedMailbox>& done_flags,
    int timeout_ms) {
    const auto start = std::chrono::steady_clock::now();

    while (true) {
        bool all_done = true;
        for (const auto& done : done_flags) {
            if (!all_u32_equal_to_one(done.host_ptr, done.count)) {
                all_done = false;
                break;
            }
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

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

int prev_rank_of(
    int rank,
    int world_size) {
    return (rank - 1 + world_size) % world_size;
}

struct PersistentTwoGpuState {
    std::vector<int> devices{};

    comm::Group group{};
    std::vector<comm::EndpointRuntime> runtimes;
    std::vector<comm::EndpointPersistentControl> controls;

    std::vector<comm::transport::CommBuffer> accums;
    std::vector<HostMappedMailbox> inbound_steps;
    std::vector<HostMappedMailbox> done_flags;

    std::vector<comm::collective::ChunkStateTable> chunk_states;
    std::vector<comm::collective::OperationDesc> ops;
    std::vector<comm::collective::DeviceOperationDesc> op_devs;

    size_t bytes = 0;
    uint32_t num_chunks = 0;
    bool initialized = false;
};

void destroy_persistent_two_gpu_state(
    PersistentTwoGpuState* st) {
    if (st == nullptr) {
        return;
    }

    for (auto& op_dev : st->op_devs) {
        try { comm::collective::device_operation_desc_destroy(&op_dev); } catch (...) {}
    }
    for (auto& table : st->chunk_states) {
        try { comm::collective::chunk_state_table_destroy(&table); } catch (...) {}
    }
    for (auto& box : st->done_flags) {
        try { box.destroy(); } catch (...) {}
    }
    for (auto& box : st->inbound_steps) {
        try { box.destroy(); } catch (...) {}
    }
    for (auto& accum : st->accums) {
        try { comm::transport::free_comm_buffer(st->devices, accum); } catch (...) {}
    }
    for (auto& ctl : st->controls) {
        try { comm::endpoint_persistent_control_destroy(&ctl); } catch (...) {}
    }
    for (auto& rt : st->runtimes) {
        try { comm::endpoint_runtime_destroy(&rt); } catch (...) {}
    }
    try { comm::group_destroy(&st->group); } catch (...) {}

    st->devices.clear();
    st->runtimes.clear();
    st->controls.clear();
    st->accums.clear();
    st->inbound_steps.clear();
    st->done_flags.clear();
    st->chunk_states.clear();
    st->ops.clear();
    st->op_devs.clear();
    st->bytes = 0;
    st->num_chunks = 0;
    st->initialized = false;
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
    st->inbound_steps.resize(2);
    st->done_flags.resize(2);
    st->chunk_states.resize(2);
    st->ops.resize(2);
    st->op_devs.resize(2);

    comm::group_init(&st->group, st->devices, comm::kEndpointPersistentChunkBytes);

    for (int r = 0; r < 2; ++r) {
        comm::endpoint_runtime_init(&st->runtimes[static_cast<size_t>(r)], &st->group, r);
    }

    for (int r = 0; r < 2; ++r) {
        const int prev_rank = prev_rank_of(r, 2);

        st->accums[static_cast<size_t>(r)] =
            comm::transport::alloc_peer_visible_buffer_for_rank_with_access_ranks(
                st->group.devices,
                r,
                {prev_rank},
                st->bytes);

        st->inbound_steps[static_cast<size_t>(r)].init(st->num_chunks, st->devices);
        st->done_flags[static_cast<size_t>(r)].init(st->num_chunks, st->devices);

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
            st->inbound_steps[static_cast<size_t>(r)].device_ptr_for_rank(static_cast<size_t>(r)),
            st->inbound_steps[static_cast<size_t>(next_rank)].device_ptr_for_rank(static_cast<size_t>(r)),
            st->done_flags[static_cast<size_t>(r)].device_ptr_for_rank(static_cast<size_t>(r)),
            st->done_flags[static_cast<size_t>(next_rank)].device_ptr_for_rank(static_cast<size_t>(r)),
            st->chunk_states[static_cast<size_t>(r)].records,
            1);

        comm::collective::device_operation_desc_create(
            &st->op_devs[static_cast<size_t>(r)],
            st->group.devices[static_cast<size_t>(r)],
            &st->ops[static_cast<size_t>(r)]);

        comm::endpoint_persistent_control_init(
            &st->controls[static_cast<size_t>(r)],
            st->group.devices[static_cast<size_t>(r)]);
    }

    st->initialized = true;
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
        st->inbound_steps[static_cast<size_t>(r)].reset(
            comm::collective::kOperationInboundStepInvalid);
        st->done_flags[static_cast<size_t>(r)].reset(0u);
        comm::collective::chunk_state_table_reset(&st->chunk_states[static_cast<size_t>(r)]);
    }

    for (int r = 0; r < 2; ++r) {
        for (uint32_t idx = 0; idx < st->num_chunks; ++idx) {
            if (comm::collective::operation_desc_actor_rank_for_step(
                    &st->ops[static_cast<size_t>(r)],
                    idx,
                    0u) == r) {
                st->inbound_steps[static_cast<size_t>(r)].host_ptr[idx] = 0u;
            }
        }
    }
}

void launch_persistent_two_gpu_run(
    PersistentTwoGpuState* st) {
    if (st == nullptr || !st->initialized) {
        throw std::invalid_argument("launch_persistent_two_gpu_run: state is not initialized");
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
}

void wait_and_stop_persistent_two_gpu_run(
    PersistentTwoGpuState* st,
    int timeout_ms) {
    if (st == nullptr || !st->initialized) {
        throw std::invalid_argument("wait_and_stop_persistent_two_gpu_run: state is not initialized");
    }

    const bool all_done = wait_until_all_ranks_done(st->done_flags, timeout_ms);
    if (!all_done) {
        throw std::runtime_error("persistent allreduce timeout waiting for done flags");
    }

    for (int r = 0; r < 2; ++r) {
        comm::endpoint_persistent_control_request_stop(&st->controls[static_cast<size_t>(r)]);
    }

    for (int r = 0; r < 2; ++r) {
        system::runtime::check_cuda(
            cudaStreamSynchronize(st->runtimes[static_cast<size_t>(r)].endpoint.stream),
            "cudaStreamSynchronize(persistent stream)");
    }
}

double measure_persistent_device_ms(
    PersistentTwoGpuState* st,
    cudaStream_t watch_stream0,
    cudaStream_t watch_stream1,
    cudaEvent_t start0,
    cudaEvent_t stop0,
    cudaEvent_t start1,
    cudaEvent_t stop1,
    int timeout_ms) {
    if (st == nullptr || !st->initialized) {
        throw std::invalid_argument("measure_persistent_device_ms: state is not initialized");
    }

    launch_persistent_two_gpu_run(st);

    system::runtime::set_device(st->devices[0]);
    system::runtime::check_cuda(
        cudaEventRecord(start0, watch_stream0),
        "cudaEventRecord(start0 persistent)");
    wait_done_flags_kernel<<<1, 1, 0, watch_stream0>>>(
        st->done_flags[0].device_ptr_for_rank(0),
        st->num_chunks);
    system::runtime::check_cuda(
        cudaGetLastError(),
        "launch wait_done_flags_kernel rank0");
    system::runtime::check_cuda(
        cudaEventRecord(stop0, watch_stream0),
        "cudaEventRecord(stop0 persistent)");

    system::runtime::set_device(st->devices[1]);
    system::runtime::check_cuda(
        cudaEventRecord(start1, watch_stream1),
        "cudaEventRecord(start1 persistent)");
    wait_done_flags_kernel<<<1, 1, 0, watch_stream1>>>(
        st->done_flags[1].device_ptr_for_rank(1),
        st->num_chunks);
    system::runtime::check_cuda(
        cudaGetLastError(),
        "launch wait_done_flags_kernel rank1");
    system::runtime::check_cuda(
        cudaEventRecord(stop1, watch_stream1),
        "cudaEventRecord(stop1 persistent)");

    system::runtime::set_device(st->devices[0]);
    cudaError_t err0 = cudaEventSynchronize(stop0);

    system::runtime::set_device(st->devices[1]);
    cudaError_t err1 = cudaEventSynchronize(stop1);

    if (err0 == cudaErrorNotReady || err1 == cudaErrorNotReady) {
        throw std::runtime_error("measure_persistent_device_ms: unexpected cudaErrorNotReady");
    }
    if (err0 != cudaSuccess) {
        throw std::runtime_error(
            std::string("measure_persistent_device_ms: rank0 event sync failed: ") +
            cudaGetErrorString(err0));
    }
    if (err1 != cudaSuccess) {
        throw std::runtime_error(
            std::string("measure_persistent_device_ms: rank1 event sync failed: ") +
            cudaGetErrorString(err1));
    }

    float ms0 = 0.0f;
    float ms1 = 0.0f;

    system::runtime::set_device(st->devices[0]);
    system::runtime::check_cuda(
        cudaEventElapsedTime(&ms0, start0, stop0),
        "cudaEventElapsedTime persistent rank0");

    system::runtime::set_device(st->devices[1]);
    system::runtime::check_cuda(
        cudaEventElapsedTime(&ms1, start1, stop1),
        "cudaEventElapsedTime persistent rank1");

    // Shutdown is intentionally NOT part of the measured region.
    for (int r = 0; r < 2; ++r) {
        comm::endpoint_persistent_control_request_stop(&st->controls[static_cast<size_t>(r)]);
    }

    for (int r = 0; r < 2; ++r) {
        system::runtime::check_cuda(
            cudaStreamSynchronize(st->runtimes[static_cast<size_t>(r)].endpoint.stream),
            "cudaStreamSynchronize(persistent stream after timing)");
    }

    return static_cast<double>((ms0 > ms1) ? ms0 : ms1);
}

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

void verify_nccl_result(
    half* out0,
    half* out1,
    int dev0,
    int dev1,
    int64_t numel) {
    auto ref = reference_two_gpu_sum(numel);

    auto got0 = copy_half_device_to_host(out0, numel, dev0);
    auto got1 = copy_half_device_to_host(out1, numel, dev1);

    expect_half_vectors_close(got0, ref, "nccl verify rank0");
    expect_half_vectors_close(got1, ref, "nccl verify rank1");
}

inline void verify_persistent_result(
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

    cudaStream_t watch_stream0 = nullptr;
    cudaStream_t watch_stream1 = nullptr;
    cudaEvent_t persistent_start0 = nullptr;
    cudaEvent_t persistent_stop0 = nullptr;
    cudaEvent_t persistent_start1 = nullptr;
    cudaEvent_t persistent_stop1 = nullptr;

    const size_t bytes = static_cast<size_t>(numel) * sizeof(half);
    const auto host0 = make_host_pattern(numel, 0);
    const auto host1 = make_host_pattern(numel, 1);

    half* rank0_src = nullptr;
    half* rank1_src = nullptr;
    half* rank0_nccl = nullptr;
    half* rank1_nccl = nullptr;

    cudaStream_t nccl_stream0 = nullptr;
    cudaStream_t nccl_stream1 = nullptr;
    ncclComm_t nccl_comms[2] = {nullptr, nullptr};

    PersistentTwoGpuState persistent{};
    BasicCudaMemcpyState basic{};

    try {
        init_persistent_two_gpu_state(&persistent, dev0, dev1, static_cast<size_t>(numel));
        init_basic_cuda_memcpy_state(&basic, dev0, dev1, static_cast<size_t>(numel));

        system::runtime::ensure_context_on_device(dev0);
        system::runtime::ensure_context_on_device(dev1);

        nccl_stream0 = system::runtime::create_stream_on_device(dev0);
        nccl_stream1 = system::runtime::create_stream_on_device(dev1);

        watch_stream0 = system::runtime::create_stream_on_device(dev0);
        watch_stream1 = system::runtime::create_stream_on_device(dev1);

        system::runtime::set_device(dev0);
        system::runtime::check_cuda(
            cudaEventCreate(&persistent_start0),
            "cudaEventCreate(persistent_start0)");
        system::runtime::check_cuda(
            cudaEventCreate(&persistent_stop0),
            "cudaEventCreate(persistent_stop0)");

        system::runtime::set_device(dev1);
        system::runtime::check_cuda(
            cudaEventCreate(&persistent_start1),
            "cudaEventCreate(persistent_start1)");
        system::runtime::check_cuda(
            cudaEventCreate(&persistent_stop1),
            "cudaEventCreate(persistent_stop1)");

        system::runtime::set_device(dev0);
        system::runtime::check_cuda(cudaMalloc(&rank0_src, bytes), "cudaMalloc(rank0_src)");
        system::runtime::check_cuda(cudaMalloc(&rank0_nccl, bytes), "cudaMalloc(rank0_nccl)");
        upload_host_half_vector(rank0_src, host0, dev0, nccl_stream0, "upload host0");

        system::runtime::set_device(dev1);
        system::runtime::check_cuda(cudaMalloc(&rank1_src, bytes), "cudaMalloc(rank1_src)");
        system::runtime::check_cuda(cudaMalloc(&rank1_nccl, bytes), "cudaMalloc(rank1_nccl)");
        upload_host_half_vector(rank1_src, host1, dev1, nccl_stream1, "upload host1");

        sync_two_streams(dev0, nccl_stream0, dev1, nccl_stream1, "sync upload inputs");

        int nccl_devices[2] = {dev0, dev1};
        OOVERLAP_PERSIST_NCCL_CHECK(ncclCommInitAll(nccl_comms, 2, nccl_devices));

        for (int i = 0; i < warmup; ++i) {
            prepare_basic_cuda_memcpy_run(&basic, rank0_src, rank1_src);
            run_basic_cuda_memcpy_allreduce(&basic, static_cast<size_t>(numel));
        }

        for (int i = 0; i < warmup; ++i) {
            system::runtime::set_device(dev0);
            system::runtime::check_cuda(
                cudaMemcpyAsync(rank0_nccl, rank0_src, bytes, cudaMemcpyDeviceToDevice, nccl_stream0),
                "cudaMemcpyAsync(rank0 src -> nccl)");
            system::runtime::set_device(dev1);
            system::runtime::check_cuda(
                cudaMemcpyAsync(rank1_nccl, rank1_src, bytes, cudaMemcpyDeviceToDevice, nccl_stream1),
                "cudaMemcpyAsync(rank1 src -> nccl)");

            sync_two_streams(dev0, nccl_stream0, dev1, nccl_stream1, "sync nccl warmup prepare");

            OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupStart());
            OOVERLAP_PERSIST_NCCL_CHECK(
                ncclAllReduce(
                    rank0_nccl,
                    rank0_nccl,
                    static_cast<size_t>(numel),
                    ncclFloat16,
                    ncclSum,
                    nccl_comms[0],
                    nccl_stream0));
            OOVERLAP_PERSIST_NCCL_CHECK(
                ncclAllReduce(
                    rank1_nccl,
                    rank1_nccl,
                    static_cast<size_t>(numel),
                    ncclFloat16,
                    ncclSum,
                    nccl_comms[1],
                    nccl_stream1));
            OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupEnd());

            sync_two_streams(dev0, nccl_stream0, dev1, nccl_stream1, "sync nccl warmup");
        }

        for (int i = 0; i < warmup; ++i) {
            prepare_persistent_two_gpu_run(&persistent, rank0_src, rank1_src);
            launch_persistent_two_gpu_run(&persistent);
            wait_and_stop_persistent_two_gpu_run(&persistent, 30000);
        }

        double basic_total_ms = 0.0;
        for (int i = 0; i < iters; ++i) {
            prepare_basic_cuda_memcpy_run(&basic, rank0_src, rank1_src);
            basic_total_ms += measure_host_ms([&] {
                run_basic_cuda_memcpy_allreduce(&basic, static_cast<size_t>(numel));
            });
        }

        double nccl_total_ms = 0.0;
        for (int i = 0; i < iters; ++i) {
            system::runtime::set_device(dev0);
            system::runtime::check_cuda(
                cudaMemcpyAsync(rank0_nccl, rank0_src, bytes, cudaMemcpyDeviceToDevice, nccl_stream0),
                "cudaMemcpyAsync(rank0 src -> nccl measured)");
            system::runtime::set_device(dev1);
            system::runtime::check_cuda(
                cudaMemcpyAsync(rank1_nccl, rank1_src, bytes, cudaMemcpyDeviceToDevice, nccl_stream1),
                "cudaMemcpyAsync(rank1 src -> nccl measured)");

            sync_two_streams(dev0, nccl_stream0, dev1, nccl_stream1, "sync nccl measured prepare");

            nccl_total_ms += measure_host_ms([&] {
                OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupStart());
                OOVERLAP_PERSIST_NCCL_CHECK(
                    ncclAllReduce(
                        rank0_nccl,
                        rank0_nccl,
                        static_cast<size_t>(numel),
                        ncclFloat16,
                        ncclSum,
                        nccl_comms[0],
                        nccl_stream0));
                OOVERLAP_PERSIST_NCCL_CHECK(
                    ncclAllReduce(
                        rank1_nccl,
                        rank1_nccl,
                        static_cast<size_t>(numel),
                        ncclFloat16,
                        ncclSum,
                        nccl_comms[1],
                        nccl_stream1));
                OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupEnd());

                sync_two_streams(dev0, nccl_stream0, dev1, nccl_stream1, "sync nccl measured");
            });
        }

        double persistent_total_ms = 0.0;
        for (int i = 0; i < iters; ++i) {
            prepare_persistent_two_gpu_run(&persistent, rank0_src, rank1_src);
            persistent_total_ms += measure_persistent_device_ms(
                &persistent,
                watch_stream0,
                watch_stream1,
                persistent_start0,
                persistent_stop0,
                persistent_start1,
                persistent_stop1,
                30000);
        }

        prepare_basic_cuda_memcpy_run(&basic, rank0_src, rank1_src);
        run_basic_cuda_memcpy_allreduce(&basic, static_cast<size_t>(numel));
        verify_basic_cuda_memcpy_result(&basic, numel);

        system::runtime::set_device(dev0);
        system::runtime::check_cuda(
            cudaMemcpyAsync(rank0_nccl, rank0_src, bytes, cudaMemcpyDeviceToDevice, nccl_stream0),
            "cudaMemcpyAsync(rank0 src -> nccl verify)");
        system::runtime::set_device(dev1);
        system::runtime::check_cuda(
            cudaMemcpyAsync(rank1_nccl, rank1_src, bytes, cudaMemcpyDeviceToDevice, nccl_stream1),
            "cudaMemcpyAsync(rank1 src -> nccl verify)");

        sync_two_streams(dev0, nccl_stream0, dev1, nccl_stream1, "sync nccl verify prepare");

        OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupStart());
        OOVERLAP_PERSIST_NCCL_CHECK(
            ncclAllReduce(
                rank0_nccl,
                rank0_nccl,
                static_cast<size_t>(numel),
                ncclFloat16,
                ncclSum,
                nccl_comms[0],
                nccl_stream0));
        OOVERLAP_PERSIST_NCCL_CHECK(
            ncclAllReduce(
                rank1_nccl,
                rank1_nccl,
                static_cast<size_t>(numel),
                ncclFloat16,
                ncclSum,
                nccl_comms[1],
                nccl_stream1));
        OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupEnd());

        sync_two_streams(dev0, nccl_stream0, dev1, nccl_stream1, "sync nccl verify");
        verify_nccl_result(rank0_nccl, rank1_nccl, dev0, dev1, numel);

        prepare_persistent_two_gpu_run(&persistent, rank0_src, rank1_src);
        launch_persistent_two_gpu_run(&persistent);
        wait_and_stop_persistent_two_gpu_run(&persistent, 30000);
        verify_persistent_result(&persistent, numel);

        const double avg_basic_ms = basic_total_ms / static_cast<double>(iters);
        const double avg_nccl_ms = nccl_total_ms / static_cast<double>(iters);
        const double avg_persistent_ms = persistent_total_ms / static_cast<double>(iters);

        OOVERLAP_PERSIST_NCCL_CHECK(ncclCommDestroy(nccl_comms[0]));
        OOVERLAP_PERSIST_NCCL_CHECK(ncclCommDestroy(nccl_comms[1]));
        nccl_comms[0] = nullptr;
        nccl_comms[1] = nullptr;

        system::runtime::destroy_stream_on_device(dev0, nccl_stream0);
        system::runtime::destroy_stream_on_device(dev1, nccl_stream1);
        nccl_stream0 = nullptr;
        nccl_stream1 = nullptr;

        system::runtime::set_device(dev0);
        system::runtime::check_cuda(
            cudaEventDestroy(persistent_start0),
            "cudaEventDestroy(persistent_start0)");
        system::runtime::check_cuda(
            cudaEventDestroy(persistent_stop0),
            "cudaEventDestroy(persistent_stop0)");
        persistent_start0 = nullptr;
        persistent_stop0 = nullptr;

        system::runtime::set_device(dev1);
        system::runtime::check_cuda(
            cudaEventDestroy(persistent_start1),
            "cudaEventDestroy(persistent_start1)");
        system::runtime::check_cuda(
            cudaEventDestroy(persistent_stop1),
            "cudaEventDestroy(persistent_stop1)");
        persistent_start1 = nullptr;
        persistent_stop1 = nullptr;

        system::runtime::destroy_stream_on_device(dev0, watch_stream0);
        system::runtime::destroy_stream_on_device(dev1, watch_stream1);
        watch_stream0 = nullptr;
        watch_stream1 = nullptr;

        system::runtime::set_device(dev0);
        system::runtime::check_cuda(cudaFree(rank0_src), "cudaFree(rank0_src)");
        system::runtime::check_cuda(cudaFree(rank0_nccl), "cudaFree(rank0_nccl)");
        system::runtime::set_device(dev1);
        system::runtime::check_cuda(cudaFree(rank1_src), "cudaFree(rank1_src)");
        system::runtime::check_cuda(cudaFree(rank1_nccl), "cudaFree(rank1_nccl)");
        rank0_src = nullptr;
        rank1_src = nullptr;
        rank0_nccl = nullptr;
        rank1_nccl = nullptr;

        destroy_basic_cuda_memcpy_state(&basic);
        destroy_persistent_two_gpu_state(&persistent);

        return {
            {"numel", static_cast<double>(numel)},
            {"avg_ms_persistent", avg_persistent_ms},
            {"avg_ms_cuda_memcpy", avg_basic_ms},
            {"avg_ms_nccl", avg_nccl_ms},
            {"cuda_memcpy_over_persistent", avg_basic_ms / avg_persistent_ms},
            {"nccl_over_persistent", avg_nccl_ms / avg_persistent_ms}
        };
    } catch (...) {
        if (nccl_comms[0] != nullptr) {
            try { ncclCommDestroy(nccl_comms[0]); } catch (...) {}
        }
        if (nccl_comms[1] != nullptr) {
            try { ncclCommDestroy(nccl_comms[1]); } catch (...) {}
        }

        if (nccl_stream0 != nullptr) {
            try { system::runtime::destroy_stream_on_device(dev0, nccl_stream0); } catch (...) {}
        }
        if (nccl_stream1 != nullptr) {
            try { system::runtime::destroy_stream_on_device(dev1, nccl_stream1); } catch (...) {}
        }

        if (rank0_src != nullptr) {
            try {
                system::runtime::set_device(dev0);
                cudaFree(rank0_src);
            } catch (...) {}
        }
        if (rank0_nccl != nullptr) {
            try {
                system::runtime::set_device(dev0);
                cudaFree(rank0_nccl);
            } catch (...) {}
        }
        if (rank1_src != nullptr) {
            try {
                system::runtime::set_device(dev1);
                cudaFree(rank1_src);
            } catch (...) {}
        }
        if (rank1_nccl != nullptr) {
            try {
                system::runtime::set_device(dev1);
                cudaFree(rank1_nccl);
            } catch (...) {}
        }

        destroy_basic_cuda_memcpy_state(&basic);
        destroy_persistent_two_gpu_state(&persistent);
        throw;
    }
}

} // namespace ooverlap
