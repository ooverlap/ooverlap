#include "test/endpoint_persistent_smoke_test.h"

#include "comm/collective/operation.h"
#include "comm/endpoint_persistent_kernel.h"
#include "comm/endpoint_runtime.h"
#include "comm/group.h"
#include "comm/transport/buffer.h"
#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ooverlap {
namespace {

std::vector<half> make_host_pattern(
    int64_t numel,
    float base,
    float step) {
    std::vector<half> out(static_cast<size_t>(numel));
    for (int64_t i = 0; i < numel; ++i) {
        const float x = base + step * static_cast<float>(i % 97);
        out[static_cast<size_t>(i)] = __float2half_rn(x);
    }
    return out;
}

std::vector<half> add_host_vectors(
    const std::vector<half>& a,
    const std::vector<half>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("add_host_vectors: size mismatch");
    }

    std::vector<half> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        const float av = __half2float(a[i]);
        const float bv = __half2float(b[i]);
        out[i] = __float2half_rn(av + bv);
    }
    return out;
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

void copy_u32_buffer_from_owner(
    int owner_device,
    const void* ptr,
    std::vector<uint32_t>* out) {
    if (out == nullptr) {
        throw std::invalid_argument("copy_u32_buffer_from_owner: out is null");
    }

    system::runtime::set_device(owner_device);
    system::runtime::check_cuda(
        cudaMemcpy(
            out->data(),
            ptr,
            out->size() * sizeof(uint32_t),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(u32 buffer -> host)");
}

} // namespace

bool endpoint_persistent_smoke_test(
    int64_t numel,
    int dev0,
    int dev1) {
    if (numel <= 0) {
        throw std::invalid_argument("endpoint_persistent_smoke_test: numel must be > 0");
    }
    if (dev0 == dev1) {
        throw std::invalid_argument("endpoint_persistent_smoke_test: dev0 and dev1 must differ");
    }

    comm::Group group{};
    comm::EndpointRuntime runtime0{};
    comm::EndpointRuntime runtime1{};
    comm::EndpointPersistentControl control0{};
    comm::EndpointPersistentControl control1{};

    comm::transport::CommBuffer accum0{};
    comm::transport::CommBuffer accum1{};
    comm::transport::CommBuffer chunk_steps{};
    comm::transport::CommBuffer chunk_done{};

    comm::collective::ChunkStateTable chunk_states0{};
    comm::collective::ChunkStateTable chunk_states1{};

    comm::collective::DeviceOperationDesc op0_dev{};
    comm::collective::DeviceOperationDesc op1_dev{};

    bool control0_initialized = false;
    bool control1_initialized = false;
    bool kernel0_launched = false;
    bool kernel1_launched = false;

    try {
        const size_t bytes = static_cast<size_t>(numel) * sizeof(half);
        const size_t chunk_bytes = comm::kEndpointPersistentChunkBytes;
        const uint32_t num_chunks =
            comm::collective::operation_desc_compute_num_chunks(bytes, chunk_bytes);

        const auto host_src0 = make_host_pattern(numel, 0.125f, 0.010f);
        const auto host_src1 = make_host_pattern(numel, 1.250f, 0.020f);
        const auto host_ref = add_host_vectors(host_src0, host_src1);

        std::printf("[smoke] before group_init\n"); std::fflush(stdout);
        comm::group_init(&group, {dev0, dev1}, comm::kEndpointPersistentChunkBytes);
        std::printf("[smoke] after group_init\n"); std::fflush(stdout);

        comm::endpoint_runtime_init(&runtime0, &group, 0);
        comm::endpoint_runtime_init(&runtime1, &group, 1);
        std::printf("[smoke] after endpoint_runtime_init\n"); std::fflush(stdout);

        accum0 = comm::transport::alloc_peer_visible_buffer_for_rank(
            group.devices, 0, bytes);
        accum1 = comm::transport::alloc_peer_visible_buffer_for_rank(
            group.devices, 1, bytes);

        chunk_steps = comm::transport::alloc_peer_visible_buffer_for_rank(
            group.devices,
            0,
            static_cast<size_t>(num_chunks) * sizeof(uint32_t));
        chunk_done = comm::transport::alloc_peer_visible_buffer_for_rank(
            group.devices,
            0,
            static_cast<size_t>(num_chunks) * sizeof(uint32_t));

        system::runtime::set_device(dev0);
        system::runtime::check_cuda(
            cudaMemcpy(accum0.ptr, host_src0.data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy(host_src0 -> accum0)");

        system::runtime::set_device(dev1);
        system::runtime::check_cuda(
            cudaMemcpy(accum1.ptr, host_src1.data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy(host_src1 -> accum1)");

        comm::collective::chunk_state_table_init(&chunk_states0, dev0, num_chunks);
        comm::collective::chunk_state_table_init(&chunk_states1, dev1, num_chunks);

        comm::collective::OperationDesc op0{};
        comm::collective::OperationDesc op1{};

        comm::endpoint_runtime_build_ring_allreduce_operation(
            &runtime0,
            &op0,
            accum0.ptr,
            accum1.ptr,
            bytes,
            chunk_bytes,
            chunk_steps.ptr,
            chunk_done.ptr,
            chunk_states0.records,
            1);

        comm::endpoint_runtime_build_ring_allreduce_operation(
            &runtime1,
            &op1,
            accum1.ptr,
            accum0.ptr,
            bytes,
            chunk_bytes,
            chunk_steps.ptr,
            chunk_done.ptr,
            chunk_states1.records,
            1);

        // One shared reset for progress arrays, then per-rank local chunk-state reset.
        comm::collective::operation_desc_reset_shared_progress(dev0, &op0);
        comm::collective::operation_desc_reset_local_chunk_state(dev0, &op0);
        comm::collective::operation_desc_reset_local_chunk_state(dev1, &op1);

        comm::collective::device_operation_desc_create(&op0_dev, dev0, &op0);
        comm::collective::device_operation_desc_create(&op1_dev, dev1, &op1);

        comm::endpoint_runtime_configure_submission(
            &runtime0,
            accum0.ptr,
            accum0.ptr,
            bytes,
            comm::exec::ChunkOpKind::kReduceAddNoFtzF16,
            1,
            1,
            0);

        comm::endpoint_runtime_configure_submission(
            &runtime1,
            accum1.ptr,
            accum1.ptr,
            bytes,
            comm::exec::ChunkOpKind::kReduceAddNoFtzF16,
            1,
            0,
            0);

        std::printf("[smoke] after operation setup\n"); std::fflush(stdout);

        comm::endpoint_persistent_control_init(&control0, dev0);
        comm::endpoint_persistent_control_init(&control1, dev1);
        control0_initialized = true;
        control1_initialized = true;

        std::printf("[smoke] before persistent launch\n"); std::fflush(stdout);
        system::runtime::check_cuda(
            comm::launch_endpoint_persistent_kernel_sm90(
                comm::endpoint_runtime_device_handle(&runtime0),
                op0_dev.ptr,
                &control0,
                runtime0.endpoint.stream),
            "launch_endpoint_persistent_kernel_sm90(rank0)");
        kernel0_launched = true;

        system::runtime::check_cuda(
            comm::launch_endpoint_persistent_kernel_sm90(
                comm::endpoint_runtime_device_handle(&runtime1),
                op1_dev.ptr,
                &control1,
                runtime1.endpoint.stream),
            "launch_endpoint_persistent_kernel_sm90(rank1)");
        kernel1_launched = true;
        std::printf("[smoke] after persistent launch\n"); std::fflush(stdout);

        std::printf("[smoke] letting persistent kernels run\n"); std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        std::printf("[smoke] requesting stop\n"); std::fflush(stdout);
        comm::endpoint_persistent_control_request_stop(&control0);
        comm::endpoint_persistent_control_request_stop(&control1);

        system::runtime::check_cuda(
            cudaStreamSynchronize(runtime0.endpoint.stream),
            "cudaStreamSynchronize(rank0 persistent stream)");
        system::runtime::check_cuda(
            cudaStreamSynchronize(runtime1.endpoint.stream),
            "cudaStreamSynchronize(rank1 persistent stream)");
        std::printf("[smoke] persistent streams joined\n"); std::fflush(stdout);

        std::vector<uint32_t> host_steps(static_cast<size_t>(num_chunks), 0u);
        std::vector<uint32_t> host_done(static_cast<size_t>(num_chunks), 0u);
        copy_u32_buffer_from_owner(dev0, chunk_steps.ptr, &host_steps);
        copy_u32_buffer_from_owner(dev0, chunk_done.ptr, &host_done);

        const uint32_t expected_final_step =
            comm::collective::operation_desc_total_ring_steps(&op0);

        for (uint32_t i = 0; i < num_chunks; ++i) {
            if (host_steps[static_cast<size_t>(i)] != expected_final_step) {
                throw std::runtime_error(
                    "endpoint_persistent_smoke_test: chunk step did not reach final value");
            }
            if (host_done[static_cast<size_t>(i)] != 1u) {
                throw std::runtime_error(
                    "endpoint_persistent_smoke_test: chunk done flag was not set");
            }
        }

        std::vector<half> host_out0(static_cast<size_t>(numel));
        std::vector<half> host_out1(static_cast<size_t>(numel));

        system::runtime::set_device(dev0);
        system::runtime::check_cuda(
            cudaMemcpy(host_out0.data(), accum0.ptr, bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy(accum0 -> host_out0)");

        system::runtime::set_device(dev1);
        system::runtime::check_cuda(
            cudaMemcpy(host_out1.data(), accum1.ptr, bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy(accum1 -> host_out1)");

        expect_half_vectors_close(host_out0, host_ref, "endpoint_persistent_smoke_test(rank0)");
        expect_half_vectors_close(host_out1, host_ref, "endpoint_persistent_smoke_test(rank1)");

        comm::collective::device_operation_desc_destroy(&op1_dev);
        comm::collective::device_operation_desc_destroy(&op0_dev);

        comm::collective::chunk_state_table_destroy(&chunk_states1);
        comm::collective::chunk_state_table_destroy(&chunk_states0);

        comm::transport::free_comm_buffer(group.devices, chunk_done);
        comm::transport::free_comm_buffer(group.devices, chunk_steps);
        comm::transport::free_comm_buffer(group.devices, accum1);
        comm::transport::free_comm_buffer(group.devices, accum0);

        if (control1_initialized) {
            comm::endpoint_persistent_control_destroy(&control1);
            control1_initialized = false;
        }
        if (control0_initialized) {
            comm::endpoint_persistent_control_destroy(&control0);
            control0_initialized = false;
        }

        comm::endpoint_runtime_destroy(&runtime1);
        comm::endpoint_runtime_destroy(&runtime0);
        comm::group_destroy(&group);
        return true;
    } catch (...) {
        try {
            if (kernel1_launched && control1_initialized) {
                comm::endpoint_persistent_control_request_stop(&control1);
                cudaStreamSynchronize(runtime1.endpoint.stream);
            }
        } catch (...) {
        }

        try {
            if (kernel0_launched && control0_initialized) {
                comm::endpoint_persistent_control_request_stop(&control0);
                cudaStreamSynchronize(runtime0.endpoint.stream);
            }
        } catch (...) {
        }

        comm::collective::device_operation_desc_destroy(&op1_dev);
        comm::collective::device_operation_desc_destroy(&op0_dev);

        comm::collective::chunk_state_table_destroy(&chunk_states1);
        comm::collective::chunk_state_table_destroy(&chunk_states0);

        comm::transport::free_comm_buffer(group.devices, chunk_done);
        comm::transport::free_comm_buffer(group.devices, chunk_steps);
        comm::transport::free_comm_buffer(group.devices, accum1);
        comm::transport::free_comm_buffer(group.devices, accum0);

        if (control1_initialized) {
            comm::endpoint_persistent_control_destroy(&control1);
        }
        if (control0_initialized) {
            comm::endpoint_persistent_control_destroy(&control0);
        }

        comm::endpoint_runtime_destroy(&runtime1);
        comm::endpoint_runtime_destroy(&runtime0);
        comm::group_destroy(&group);
        throw;
    }
}

} // namespace ooverlap
