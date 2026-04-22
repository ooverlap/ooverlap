#include "test/persistent_allreduce_2gpu_sm90.h"

#include "test/tma_collective_sm90.h"

#include "comm/collective/operation.h"
#include "comm/endpoint_persistent_kernel.h"
#include "comm/endpoint_runtime.h"
#include "comm/group.h"
#include "comm/transport/buffer.h"
#include "ooverlap/system/peer_buffer.cuh"
#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/testing/test_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <chrono>
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

struct BasicTwoGpuState {
    int dev0 = -1;
    int dev1 = -1;
    cudaStream_t stream0 = nullptr;
    cudaStream_t stream1 = nullptr;

    system::mapped_peer_buffer inbox0{};
    system::mapped_peer_buffer inbox1{};

    half* work0 = nullptr;
    half* work1 = nullptr;

    size_t bytes = 0;
    bool initialized = false;
};

inline std::vector<int> normalize_two_devices(
    int dev0,
    int dev1) {
    if (dev0 < 0 || dev1 < 0) {
        throw std::invalid_argument("device ids must be >= 0");
    }
    if (dev0 == dev1) {
        throw std::invalid_argument("dev0 and dev1 must differ");
    }
    return {dev0, dev1};
}

inline int prev_rank_of(
    int rank,
    int world_size) {
    return (rank - 1 + world_size) % world_size;
}

inline void sync_two_streams(
    int dev0,
    cudaStream_t stream0,
    int dev1,
    cudaStream_t stream1,
    const char* what) {
    system::runtime::sync_stream_on_device(dev0, stream0, what);
    system::runtime::sync_stream_on_device(dev1, stream1, what);
}

inline std::vector<float> reference_two_gpu_sum_fp16(
    int64_t numel) {
    auto ref0 = testing::host_reference_pattern_fp16(numel, 0.25f, 1.0f);
    auto ref1 = testing::host_reference_pattern_fp16(numel, 0.50f, 2.0f);

    std::vector<float> ref(static_cast<size_t>(numel));
    for (int64_t i = 0; i < numel; ++i) {
        float acc = ref0[static_cast<size_t>(i)];
        acc = testing::round_to_half(acc + ref1[static_cast<size_t>(i)]);
        ref[static_cast<size_t>(i)] = acc;
    }
    return ref;
}

inline bool all_u32_equal_to_one(
    const uint32_t* ptr,
    size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (ptr[i] != 1u) {
            return false;
        }
    }
    return true;
}

inline bool wait_until_all_ranks_done(
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

inline void destroy_persistent_two_gpu_state(
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

inline void init_persistent_two_gpu_state(
    PersistentTwoGpuState* st,
    int dev0,
    int dev1,
    size_t numel) {
    if (st == nullptr) {
        throw std::invalid_argument("init_persistent_two_gpu_state: st is null");
    }

    destroy_persistent_two_gpu_state(st);

    st->devices = normalize_two_devices(dev0, dev1);
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

inline void prepare_persistent_two_gpu_run(
    PersistentTwoGpuState* st,
    half* rank0_in,
    half* rank1_in) {
    if (st == nullptr || !st->initialized) {
        throw std::invalid_argument("prepare_persistent_two_gpu_run: state is not initialized");
    }
    if (rank0_in == nullptr || rank1_in == nullptr) {
        throw std::invalid_argument("prepare_persistent_two_gpu_run: input pointer is null");
    }

    half* inputs[2] = {rank0_in, rank1_in};

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
            "cudaMemcpyAsync(input -> accum)");
    }

    sync_two_streams(
        st->group.devices[0], st->runtimes[0].endpoint.stream,
        st->group.devices[1], st->runtimes[1].endpoint.stream,
        "sync prepare_persistent_two_gpu_run copies");

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

inline void launch_persistent_two_gpu_run(
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

inline void wait_and_stop_persistent_two_gpu_run(
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

inline void copy_persistent_outputs_to_device(
    PersistentTwoGpuState* st,
    half* rank0_out,
    half* rank1_out) {
    if (st == nullptr || !st->initialized) {
        throw std::invalid_argument("copy_persistent_outputs_to_device: state is not initialized");
    }
    if (rank0_out == nullptr || rank1_out == nullptr) {
        throw std::invalid_argument("copy_persistent_outputs_to_device: output pointer is null");
    }

    half* outputs[2] = {rank0_out, rank1_out};

    for (int r = 0; r < 2; ++r) {
        system::runtime::set_device(st->group.devices[static_cast<size_t>(r)]);
        system::runtime::check_cuda(
            cudaMemcpyAsync(
                outputs[static_cast<size_t>(r)],
                st->accums[static_cast<size_t>(r)].device_ptr_for_rank(static_cast<size_t>(r)),
                st->bytes,
                cudaMemcpyDeviceToDevice,
                st->runtimes[static_cast<size_t>(r)].endpoint.stream),
            "cudaMemcpyAsync(accum -> output)");
    }

    sync_two_streams(
        st->group.devices[0], st->runtimes[0].endpoint.stream,
        st->group.devices[1], st->runtimes[1].endpoint.stream,
        "sync copy_persistent_outputs_to_device");
}

inline void verify_persistent_result(
    PersistentTwoGpuState* st,
    int64_t numel) {
    auto ref = reference_two_gpu_sum_fp16(numel);

    auto got0 = testing::copy_half_device_to_host_float(
        reinterpret_cast<half*>(st->accums[0].device_ptr_for_rank(0)),
        numel,
        st->devices[0]);

    auto got1 = testing::copy_half_device_to_host_float(
        reinterpret_cast<half*>(st->accums[1].device_ptr_for_rank(1)),
        numel,
        st->devices[1]);

    testing::expect_allclose(got0, ref, "persistent verify rank0");
    testing::expect_allclose(got1, ref, "persistent verify rank1");
}

inline void destroy_basic_two_gpu_state(
    BasicTwoGpuState* st) {
    if (st == nullptr) {
        return;
    }

    try {
        if (st->work0 != nullptr) {
            system::runtime::set_device(st->dev0);
            cudaFree(st->work0);
        }
    } catch (...) {
    }
    try {
        if (st->work1 != nullptr) {
            system::runtime::set_device(st->dev1);
            cudaFree(st->work1);
        }
    } catch (...) {
    }
    try {
        if (st->stream0 != nullptr) {
            system::runtime::destroy_stream_on_device(st->dev0, st->stream0);
        }
    } catch (...) {
    }
    try {
        if (st->stream1 != nullptr) {
            system::runtime::destroy_stream_on_device(st->dev1, st->stream1);
        }
    } catch (...) {
    }
    try { system::free_peer_visible_buffer(st->inbox0); } catch (...) {}
    try { system::free_peer_visible_buffer(st->inbox1); } catch (...) {}

    st->dev0 = -1;
    st->dev1 = -1;
    st->stream0 = nullptr;
    st->stream1 = nullptr;
    st->work0 = nullptr;
    st->work1 = nullptr;
    st->bytes = 0;
    st->initialized = false;
}

inline void init_basic_two_gpu_state(
    BasicTwoGpuState* st,
    int dev0,
    int dev1,
    size_t numel) {
    if (st == nullptr) {
        throw std::invalid_argument("init_basic_two_gpu_state: st is null");
    }

    destroy_basic_two_gpu_state(st);

    st->dev0 = dev0;
    st->dev1 = dev1;
    st->bytes = numel * sizeof(half);

    system::runtime::ensure_context_on_device(dev0);
    system::runtime::ensure_context_on_device(dev1);

    st->stream0 = system::runtime::create_stream_on_device(dev0);
    st->stream1 = system::runtime::create_stream_on_device(dev1);

    std::vector<int> access_devices = {dev0, dev1};
    st->inbox0 = system::alloc_peer_visible_buffer(st->bytes, dev0, access_devices);
    st->inbox1 = system::alloc_peer_visible_buffer(st->bytes, dev1, access_devices);

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(cudaMalloc(&st->work0, st->bytes), "cudaMalloc(basic work0)");
    system::runtime::check_cuda(cudaMemset(st->inbox0.ptr, 0, st->inbox0.mapped_size), "cudaMemset(inbox0)");

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(cudaMalloc(&st->work1, st->bytes), "cudaMalloc(basic work1)");
    system::runtime::check_cuda(cudaMemset(st->inbox1.ptr, 0, st->inbox1.mapped_size), "cudaMemset(inbox1)");

    st->initialized = true;
}

inline void prepare_basic_two_gpu_run(
    BasicTwoGpuState* st,
    half* rank0_src,
    half* rank1_src) {
    if (st == nullptr || !st->initialized) {
        throw std::invalid_argument("prepare_basic_two_gpu_run: state is not initialized");
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
            st->inbox0.ptr,
            0,
            st->inbox0.mapped_size,
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
            st->inbox1.ptr,
            0,
            st->inbox1.mapped_size,
            st->stream1),
        "cudaMemsetAsync(inbox1)");

    sync_two_streams(
        st->dev0, st->stream0,
        st->dev1, st->stream1,
        "sync prepare_basic_two_gpu_run");
}

inline void run_basic_two_gpu_allreduce(
    BasicTwoGpuState* st,
    size_t numel) {
    if (st == nullptr || !st->initialized) {
        throw std::invalid_argument("run_basic_two_gpu_allreduce: state is not initialized");
    }

    system::runtime::check_cuda(
        enqueue_two_gpu_all_reduce_tma_sm90(
            st->work0,
            st->work1,
            reinterpret_cast<half*>(st->inbox0.ptr),
            reinterpret_cast<half*>(st->inbox1.ptr),
            numel,
            st->dev0,
            st->dev1,
            st->stream0,
            st->stream1),
        "enqueue_two_gpu_all_reduce_tma_sm90");
}

inline void verify_basic_result(
    BasicTwoGpuState* st,
    int64_t numel) {
    auto ref = reference_two_gpu_sum_fp16(numel);

    auto got0 = testing::copy_half_device_to_host_float(st->work0, numel, st->dev0);
    auto got1 = testing::copy_half_device_to_host_float(st->work1, numel, st->dev1);

    testing::expect_allclose(got0, ref, "basic verify rank0");
    testing::expect_allclose(got1, ref, "basic verify rank1");
}

template <typename Fn>
double measure_host_ms(Fn&& fn) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

} // namespace

cudaError_t enqueue_persistent_two_gpu_allreduce_sm90(
    int dev0,
    int dev1,
    half* rank0_in,
    half* rank1_in,
    half* rank0_out,
    half* rank1_out,
    size_t numel,
    int timeout_ms) {
    if (dev0 < 0 || dev1 < 0 || dev0 == dev1) {
        return cudaErrorInvalidValue;
    }
    if (rank0_in == nullptr || rank1_in == nullptr ||
        rank0_out == nullptr || rank1_out == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }
    if (numel == 0) {
        return cudaErrorInvalidValue;
    }

    try {
        PersistentTwoGpuState st{};
        init_persistent_two_gpu_state(&st, dev0, dev1, numel);
        prepare_persistent_two_gpu_run(&st, rank0_in, rank1_in);
        launch_persistent_two_gpu_run(&st);
        wait_and_stop_persistent_two_gpu_run(&st, timeout_ms);
        copy_persistent_outputs_to_device(&st, rank0_out, rank1_out);
        destroy_persistent_two_gpu_state(&st);
        return cudaSuccess;
    } catch (...) {
        return cudaErrorUnknown;
    }
}

bool tma_persistent_two_gpu_allreduce_smoke_test(
    int64_t numel,
    int dev0,
    int dev1) {
    if (numel <= 0) {
        throw std::invalid_argument("tma_persistent_two_gpu_allreduce_smoke_test: numel must be > 0");
    }

    const size_t bytes = static_cast<size_t>(numel) * sizeof(half);

    half* rank0_in = nullptr;
    half* rank1_in = nullptr;

    PersistentTwoGpuState st{};

    try {
        init_persistent_two_gpu_state(&st, dev0, dev1, static_cast<size_t>(numel));

        system::runtime::set_device(dev0);
        system::runtime::check_cuda(cudaMalloc(&rank0_in, bytes), "cudaMalloc(rank0_in)");
        testing::fill_pattern(rank0_in, numel, 0.25f, 1.0f, st.runtimes[0].endpoint.stream);

        system::runtime::set_device(dev1);
        system::runtime::check_cuda(cudaMalloc(&rank1_in, bytes), "cudaMalloc(rank1_in)");
        testing::fill_pattern(rank1_in, numel, 0.50f, 2.0f, st.runtimes[1].endpoint.stream);

        sync_two_streams(
            dev0, st.runtimes[0].endpoint.stream,
            dev1, st.runtimes[1].endpoint.stream,
            "sync fill persistent smoke");

        prepare_persistent_two_gpu_run(&st, rank0_in, rank1_in);
        launch_persistent_two_gpu_run(&st);
        wait_and_stop_persistent_two_gpu_run(&st, 30000);
        verify_persistent_result(&st, numel);

        system::runtime::set_device(dev0);
        system::runtime::check_cuda(cudaFree(rank0_in), "cudaFree(rank0_in)");
        system::runtime::set_device(dev1);
        system::runtime::check_cuda(cudaFree(rank1_in), "cudaFree(rank1_in)");

        destroy_persistent_two_gpu_state(&st);
        return true;
    } catch (...) {
        if (rank0_in != nullptr) {
            try {
                system::runtime::set_device(dev0);
                cudaFree(rank0_in);
            } catch (...) {}
        }
        if (rank1_in != nullptr) {
            try {
                system::runtime::set_device(dev1);
                cudaFree(rank1_in);
            } catch (...) {}
        }
        destroy_persistent_two_gpu_state(&st);
        throw;
    }
}

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

    PersistentTwoGpuState persistent{};
    BasicTwoGpuState basic{};

    half* rank0_src = nullptr;
    half* rank1_src = nullptr;
    half* rank0_nccl_out = nullptr;
    half* rank1_nccl_out = nullptr;

    cudaStream_t nccl_stream0 = nullptr;
    cudaStream_t nccl_stream1 = nullptr;
    ncclComm_t nccl_comms[2] = {nullptr, nullptr};

    try {
        init_persistent_two_gpu_state(&persistent, dev0, dev1, static_cast<size_t>(numel));
        init_basic_two_gpu_state(&basic, dev0, dev1, static_cast<size_t>(numel));

        system::runtime::ensure_context_on_device(dev0);
        system::runtime::ensure_context_on_device(dev1);
        nccl_stream0 = system::runtime::create_stream_on_device(dev0);
        nccl_stream1 = system::runtime::create_stream_on_device(dev1);

        system::runtime::set_device(dev0);
        system::runtime::check_cuda(cudaMalloc(&rank0_src, bytes), "cudaMalloc(rank0_src)");
        system::runtime::check_cuda(cudaMalloc(&rank0_nccl_out, bytes), "cudaMalloc(rank0_nccl_out)");
        testing::fill_pattern(rank0_src, numel, 0.25f, 1.0f, nccl_stream0);

        system::runtime::set_device(dev1);
        system::runtime::check_cuda(cudaMalloc(&rank1_src, bytes), "cudaMalloc(rank1_src)");
        system::runtime::check_cuda(cudaMalloc(&rank1_nccl_out, bytes), "cudaMalloc(rank1_nccl_out)");
        testing::fill_pattern(rank1_src, numel, 0.50f, 2.0f, nccl_stream1);

        sync_two_streams(dev0, nccl_stream0, dev1, nccl_stream1, "sync fill benchmark inputs");

        int nccl_devices[2] = {dev0, dev1};
        OOVERLAP_PERSIST_NCCL_CHECK(ncclCommInitAll(nccl_comms, 2, nccl_devices));

        for (int i = 0; i < warmup; ++i) {
            prepare_basic_two_gpu_run(&basic, rank0_src, rank1_src);
            run_basic_two_gpu_allreduce(&basic, static_cast<size_t>(numel));
        }

        for (int i = 0; i < warmup; ++i) {
            system::runtime::set_device(dev0);
            system::runtime::check_cuda(
                cudaMemcpyAsync(rank0_nccl_out, rank0_src, bytes, cudaMemcpyDeviceToDevice, nccl_stream0),
                "cudaMemcpyAsync(rank0 src -> nccl out)");
            system::runtime::set_device(dev1);
            system::runtime::check_cuda(
                cudaMemcpyAsync(rank1_nccl_out, rank1_src, bytes, cudaMemcpyDeviceToDevice, nccl_stream1),
                "cudaMemcpyAsync(rank1 src -> nccl out)");

            sync_two_streams(dev0, nccl_stream0, dev1, nccl_stream1, "sync nccl warmup prepare");

            OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupStart());
            OOVERLAP_PERSIST_NCCL_CHECK(
                ncclAllReduce(
                    rank0_nccl_out,
                    rank0_nccl_out,
                    static_cast<size_t>(numel),
                    ncclFloat16,
                    ncclSum,
                    nccl_comms[0],
                    nccl_stream0));
            OOVERLAP_PERSIST_NCCL_CHECK(
                ncclAllReduce(
                    rank1_nccl_out,
                    rank1_nccl_out,
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
            prepare_basic_two_gpu_run(&basic, rank0_src, rank1_src);
            basic_total_ms += measure_host_ms([&] {
                run_basic_two_gpu_allreduce(&basic, static_cast<size_t>(numel));
            });
        }

        double nccl_total_ms = 0.0;
        for (int i = 0; i < iters; ++i) {
            system::runtime::set_device(dev0);
            system::runtime::check_cuda(
                cudaMemcpyAsync(rank0_nccl_out, rank0_src, bytes, cudaMemcpyDeviceToDevice, nccl_stream0),
                "cudaMemcpyAsync(rank0 src -> nccl out)");
            system::runtime::set_device(dev1);
            system::runtime::check_cuda(
                cudaMemcpyAsync(rank1_nccl_out, rank1_src, bytes, cudaMemcpyDeviceToDevice, nccl_stream1),
                "cudaMemcpyAsync(rank1 src -> nccl out)");

            sync_two_streams(dev0, nccl_stream0, dev1, nccl_stream1, "sync nccl measured prepare");

            nccl_total_ms += measure_host_ms([&] {
                OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupStart());
                OOVERLAP_PERSIST_NCCL_CHECK(
                    ncclAllReduce(
                        rank0_nccl_out,
                        rank0_nccl_out,
                        static_cast<size_t>(numel),
                        ncclFloat16,
                        ncclSum,
                        nccl_comms[0],
                        nccl_stream0));
                OOVERLAP_PERSIST_NCCL_CHECK(
                    ncclAllReduce(
                        rank1_nccl_out,
                        rank1_nccl_out,
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
            persistent_total_ms += measure_host_ms([&] {
                launch_persistent_two_gpu_run(&persistent);
                wait_and_stop_persistent_two_gpu_run(&persistent, 30000);
            });
        }

        prepare_basic_two_gpu_run(&basic, rank0_src, rank1_src);
        run_basic_two_gpu_allreduce(&basic, static_cast<size_t>(numel));
        verify_basic_result(&basic, numel);

        system::runtime::set_device(dev0);
        system::runtime::check_cuda(
            cudaMemcpyAsync(rank0_nccl_out, rank0_src, bytes, cudaMemcpyDeviceToDevice, nccl_stream0),
            "cudaMemcpyAsync(rank0 src -> nccl out verify)");
        system::runtime::set_device(dev1);
        system::runtime::check_cuda(
            cudaMemcpyAsync(rank1_nccl_out, rank1_src, bytes, cudaMemcpyDeviceToDevice, nccl_stream1),
            "cudaMemcpyAsync(rank1 src -> nccl out verify)");

        sync_two_streams(dev0, nccl_stream0, dev1, nccl_stream1, "sync nccl verify prepare");

        OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupStart());
        OOVERLAP_PERSIST_NCCL_CHECK(
            ncclAllReduce(
                rank0_nccl_out,
                rank0_nccl_out,
                static_cast<size_t>(numel),
                ncclFloat16,
                ncclSum,
                nccl_comms[0],
                nccl_stream0));
        OOVERLAP_PERSIST_NCCL_CHECK(
            ncclAllReduce(
                rank1_nccl_out,
                rank1_nccl_out,
                static_cast<size_t>(numel),
                ncclFloat16,
                ncclSum,
                nccl_comms[1],
                nccl_stream1));
        OOVERLAP_PERSIST_NCCL_CHECK(ncclGroupEnd());

        sync_two_streams(dev0, nccl_stream0, dev1, nccl_stream1, "sync nccl verify");

        {
            auto ref = reference_two_gpu_sum_fp16(numel);
            auto got0 = testing::copy_half_device_to_host_float(rank0_nccl_out, numel, dev0);
            auto got1 = testing::copy_half_device_to_host_float(rank1_nccl_out, numel, dev1);
            testing::expect_allclose(got0, ref, "nccl verify rank0");
            testing::expect_allclose(got1, ref, "nccl verify rank1");
        }

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
        system::runtime::check_cuda(cudaFree(rank0_src), "cudaFree(rank0_src)");
        system::runtime::check_cuda(cudaFree(rank0_nccl_out), "cudaFree(rank0_nccl_out)");
        system::runtime::set_device(dev1);
        system::runtime::check_cuda(cudaFree(rank1_src), "cudaFree(rank1_src)");
        system::runtime::check_cuda(cudaFree(rank1_nccl_out), "cudaFree(rank1_nccl_out)");
        rank0_src = nullptr;
        rank1_src = nullptr;
        rank0_nccl_out = nullptr;
        rank1_nccl_out = nullptr;

        destroy_basic_two_gpu_state(&basic);
        destroy_persistent_two_gpu_state(&persistent);

        return {
            {"numel", static_cast<double>(numel)},
            {"avg_ms_persistent", avg_persistent_ms},
            {"avg_ms_basic", avg_basic_ms},
            {"avg_ms_nccl", avg_nccl_ms},
            {"speedup_basic_over_persistent", avg_basic_ms / avg_persistent_ms},
            {"speedup_nccl_over_persistent", avg_nccl_ms / avg_persistent_ms}
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
        if (rank0_nccl_out != nullptr) {
            try {
                system::runtime::set_device(dev0);
                cudaFree(rank0_nccl_out);
            } catch (...) {}
        }
        if (rank1_src != nullptr) {
            try {
                system::runtime::set_device(dev1);
                cudaFree(rank1_src);
            } catch (...) {}
        }
        if (rank1_nccl_out != nullptr) {
            try {
                system::runtime::set_device(dev1);
                cudaFree(rank1_nccl_out);
            } catch (...) {}
        }

        destroy_basic_two_gpu_state(&basic);
        destroy_persistent_two_gpu_state(&persistent);
        throw;
    }
}

} // namespace ooverlap
