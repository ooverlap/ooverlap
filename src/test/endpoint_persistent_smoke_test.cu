#include "test/endpoint_persistent_smoke_test.h"

#include "comm/collective/operation.h"
#include "comm/endpoint_persistent_kernel.h"
#include "comm/endpoint_runtime.h"
#include "comm/group.h"
#include "comm/transport/buffer.h"
#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ooverlap {
namespace {

struct MappedHostU32Buffer {
    uint32_t* host_ptr = nullptr;
    uint32_t* device_ptr = nullptr;
    size_t count = 0;

    void allocate(size_t n) {
        if (n == 0) {
            throw std::invalid_argument("MappedHostU32Buffer::allocate: n must be > 0");
        }
        release();

        count = n;
        cudaError_t err = cudaHostAlloc(
            reinterpret_cast<void**>(&host_ptr),
            count * sizeof(uint32_t),
            cudaHostAllocMapped | cudaHostAllocPortable);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaHostAlloc(mapped u32 buffer) failed: ") +
                cudaGetErrorString(err));
        }

        std::memset(host_ptr, 0, count * sizeof(uint32_t));

        err = cudaHostGetDevicePointer(
            reinterpret_cast<void**>(&device_ptr),
            host_ptr,
            0);
        if (err != cudaSuccess) {
            cudaFreeHost(host_ptr);
            host_ptr = nullptr;
            device_ptr = nullptr;
            count = 0;
            throw std::runtime_error(
                std::string("cudaHostGetDevicePointer(mapped u32 buffer) failed: ") +
                cudaGetErrorString(err));
        }
    }

    void reset_zero() {
        if (host_ptr == nullptr) {
            throw std::invalid_argument("MappedHostU32Buffer::reset_zero: buffer not allocated");
        }
        std::memset(host_ptr, 0, count * sizeof(uint32_t));
    }

    void release() {
        if (host_ptr != nullptr) {
            cudaError_t err = cudaFreeHost(host_ptr);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaFreeHost(mapped u32 buffer) failed: ") +
                    cudaGetErrorString(err));
            }
        }
        host_ptr = nullptr;
        device_ptr = nullptr;
        count = 0;
    }
};

std::vector<int> normalize_devices(
    const std::vector<int64_t>& devices64) {
    std::vector<int> out;

    if (devices64.empty()) {
        out = {0, 1};
    } else {
        out.reserve(devices64.size());
        for (int64_t d64 : devices64) {
            if (d64 < 0 || d64 > static_cast<int64_t>(std::numeric_limits<int>::max())) {
                throw std::invalid_argument("endpoint_persistent_smoke_test: invalid device id");
            }
            out.push_back(static_cast<int>(d64));
        }
    }

    if (out.size() < 2) {
        throw std::invalid_argument("endpoint_persistent_smoke_test: need at least 2 devices");
    }

    for (size_t i = 0; i < out.size(); ++i) {
        for (size_t j = i + 1; j < out.size(); ++j) {
            if (out[i] == out[j]) {
                throw std::invalid_argument("endpoint_persistent_smoke_test: duplicate devices are not allowed");
            }
        }
    }

    return out;
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

std::vector<half> sum_host_vectors(
    const std::vector<std::vector<half>>& inputs) {
    if (inputs.empty()) {
        return {};
    }

    const size_t n = inputs[0].size();
    std::vector<float> accum(n, 0.0f);

    for (const auto& vec : inputs) {
        if (vec.size() != n) {
            throw std::invalid_argument("sum_host_vectors: size mismatch");
        }
        for (size_t i = 0; i < n; ++i) {
            accum[i] += __half2float(vec[i]);
        }
    }

    std::vector<half> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = __float2half_rn(accum[i]);
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

bool wait_until_all_chunks_done(
    const MappedHostU32Buffer& chunk_done,
    int timeout_ms) {
    const auto start = std::chrono::steady_clock::now();

    while (true) {
        if (all_u32_equal_to_one(chunk_done.host_ptr, chunk_done.count)) {
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

std::string build_progress_debug_string(
    const MappedHostU32Buffer& chunk_steps,
    const MappedHostU32Buffer& chunk_done,
    uint32_t expected_final_step) {
    size_t first_bad = chunk_done.count;
    for (size_t i = 0; i < chunk_done.count; ++i) {
        if (chunk_done.host_ptr[i] != 1u ||
            chunk_steps.host_ptr[i] != expected_final_step) {
            first_bad = i;
            break;
        }
    }

    if (first_bad == chunk_done.count) {
        return "all chunks complete";
    }

    return std::string("first incomplete chunk idx=") +
           std::to_string(first_bad) +
           " step=" + std::to_string(chunk_steps.host_ptr[first_bad]) +
           " done=" + std::to_string(chunk_done.host_ptr[first_bad]) +
           " expected_step=" + std::to_string(expected_final_step);
}

} // namespace

bool endpoint_persistent_smoke_test(
    int64_t numel,
    const std::vector<int64_t>& devices64,
    int timeout_ms) {
    if (numel <= 0) {
        throw std::invalid_argument("endpoint_persistent_smoke_test: numel must be > 0");
    }
    if (timeout_ms <= 0) {
        throw std::invalid_argument("endpoint_persistent_smoke_test: timeout_ms must be > 0");
    }

    const std::vector<int> devices = normalize_devices(devices64);
    const int world_size = static_cast<int>(devices.size());

    comm::Group group{};
    std::vector<comm::EndpointRuntime> runtimes(static_cast<size_t>(world_size));
    std::vector<comm::EndpointPersistentControl> controls(static_cast<size_t>(world_size));

    std::vector<comm::transport::CommBuffer> accums(static_cast<size_t>(world_size));
    MappedHostU32Buffer chunk_steps{};
    MappedHostU32Buffer chunk_done{};

    std::vector<comm::collective::ChunkStateTable> chunk_states(static_cast<size_t>(world_size));
    std::vector<comm::collective::DeviceOperationDesc> op_devs(static_cast<size_t>(world_size));
    std::vector<comm::collective::OperationDesc> ops(static_cast<size_t>(world_size));

    std::vector<bool> control_initialized(static_cast<size_t>(world_size), false);
    std::vector<bool> kernel_launched(static_cast<size_t>(world_size), false);

    try {
        const size_t bytes = static_cast<size_t>(numel) * sizeof(half);
        const size_t chunk_bytes = comm::kEndpointPersistentChunkBytes;
        const uint32_t num_chunks =
            comm::collective::operation_desc_compute_num_chunks(bytes, chunk_bytes);

        std::vector<std::vector<half>> host_srcs(static_cast<size_t>(world_size));
        for (int r = 0; r < world_size; ++r) {
            host_srcs[static_cast<size_t>(r)] = make_host_pattern(numel, r);
        }
        const auto host_ref = sum_host_vectors(host_srcs);

        std::printf("[smoke] before group_init\n"); std::fflush(stdout);
        comm::group_init(&group, devices, comm::kEndpointPersistentChunkBytes);
        std::printf("[smoke] after group_init\n"); std::fflush(stdout);

        for (int r = 0; r < world_size; ++r) {
            comm::endpoint_runtime_init(
                &runtimes[static_cast<size_t>(r)],
                &group,
                r);
        }
        std::printf("[smoke] after endpoint_runtime_init\n"); std::fflush(stdout);

        for (int r = 0; r < world_size; ++r) {
            accums[static_cast<size_t>(r)] =
                comm::transport::alloc_peer_visible_buffer_for_rank(
                    group.devices,
                    r,
                    bytes);
        }

        chunk_steps.allocate(num_chunks);
        chunk_done.allocate(num_chunks);
        chunk_steps.reset_zero();
        chunk_done.reset_zero();

        for (int r = 0; r < world_size; ++r) {
            system::runtime::set_device(group.devices[static_cast<size_t>(r)]);
            system::runtime::check_cuda(
                cudaMemcpy(
                    accums[static_cast<size_t>(r)].ptr,
                    host_srcs[static_cast<size_t>(r)].data(),
                    bytes,
                    cudaMemcpyHostToDevice),
                "cudaMemcpy(host_src -> accum)");
        }

        for (int r = 0; r < world_size; ++r) {
            comm::collective::chunk_state_table_init(
                &chunk_states[static_cast<size_t>(r)],
                group.devices[static_cast<size_t>(r)],
                num_chunks);
        }

        for (int r = 0; r < world_size; ++r) {
            const int next_rank = (r + 1) % world_size;

            comm::endpoint_runtime_build_ring_allreduce_operation(
                &runtimes[static_cast<size_t>(r)],
                &ops[static_cast<size_t>(r)],
                accums[static_cast<size_t>(r)].ptr,
                accums[static_cast<size_t>(next_rank)].ptr,
                bytes,
                chunk_bytes,
                chunk_steps.device_ptr,
                chunk_done.device_ptr,
                chunk_states[static_cast<size_t>(r)].records,
                1);
        }

        for (int r = 0; r < world_size; ++r) {
            comm::collective::operation_desc_reset_local_chunk_state(
                group.devices[static_cast<size_t>(r)],
                &ops[static_cast<size_t>(r)]);
        }

        for (int r = 0; r < world_size; ++r) {
            comm::collective::device_operation_desc_create(
                &op_devs[static_cast<size_t>(r)],
                group.devices[static_cast<size_t>(r)],
                &ops[static_cast<size_t>(r)]);
        }

        for (int r = 0; r < world_size; ++r) {
            const int next_rank = (r + 1) % world_size;
            comm::endpoint_runtime_configure_submission(
                &runtimes[static_cast<size_t>(r)],
                accums[static_cast<size_t>(r)].ptr,
                accums[static_cast<size_t>(r)].ptr,
                bytes,
                comm::exec::ChunkOpKind::kReduceAddNoFtzF16,
                1,
                next_rank,
                0);
        }

        std::printf("[smoke] after operation setup\n"); std::fflush(stdout);

        for (int r = 0; r < world_size; ++r) {
            comm::endpoint_persistent_control_init(
                &controls[static_cast<size_t>(r)],
                group.devices[static_cast<size_t>(r)]);
            control_initialized[static_cast<size_t>(r)] = true;
        }

        std::printf("[smoke] before persistent launch\n"); std::fflush(stdout);
        for (int r = 0; r < world_size; ++r) {
            system::runtime::check_cuda(
                comm::launch_endpoint_persistent_kernel_sm90(
                    comm::endpoint_runtime_device_handle(&runtimes[static_cast<size_t>(r)]),
                    op_devs[static_cast<size_t>(r)].ptr,
                    &controls[static_cast<size_t>(r)],
                    runtimes[static_cast<size_t>(r)].endpoint.stream),
                "launch_endpoint_persistent_kernel_sm90");
            kernel_launched[static_cast<size_t>(r)] = true;
        }
        std::printf("[smoke] after persistent launch\n"); std::fflush(stdout);

        std::printf("[smoke] waiting for chunk completion\n"); std::fflush(stdout);
        const bool all_done = wait_until_all_chunks_done(chunk_done, timeout_ms);

        if (!all_done) {
            const uint32_t expected_final_step =
                comm::collective::operation_desc_total_ring_steps(&ops[0]);

            throw std::runtime_error(
                std::string("endpoint_persistent_smoke_test: timeout waiting for chunk_done; ") +
                build_progress_debug_string(chunk_steps, chunk_done, expected_final_step));
        }

        std::printf("[smoke] all chunks done\n"); std::fflush(stdout);

        std::printf("[smoke] requesting stop\n"); std::fflush(stdout);
        for (int r = 0; r < world_size; ++r) {
            comm::endpoint_persistent_control_request_stop(
                &controls[static_cast<size_t>(r)]);
        }

        for (int r = 0; r < world_size; ++r) {
            system::runtime::check_cuda(
                cudaStreamSynchronize(runtimes[static_cast<size_t>(r)].endpoint.stream),
                "cudaStreamSynchronize(persistent stream)");
        }
        std::printf("[smoke] persistent streams joined\n"); std::fflush(stdout);

        const uint32_t expected_final_step =
            comm::collective::operation_desc_total_ring_steps(&ops[0]);

        for (uint32_t i = 0; i < num_chunks; ++i) {
            if (chunk_steps.host_ptr[static_cast<size_t>(i)] != expected_final_step) {
                throw std::runtime_error(
                    "endpoint_persistent_smoke_test: chunk step did not reach final value");
            }
            if (chunk_done.host_ptr[static_cast<size_t>(i)] != 1u) {
                throw std::runtime_error(
                    "endpoint_persistent_smoke_test: chunk done flag was not set");
            }
        }

        for (int r = 0; r < world_size; ++r) {
            std::vector<half> host_out(static_cast<size_t>(numel));
            system::runtime::set_device(group.devices[static_cast<size_t>(r)]);
            system::runtime::check_cuda(
                cudaMemcpy(
                    host_out.data(),
                    accums[static_cast<size_t>(r)].ptr,
                    bytes,
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(accum -> host)");

            expect_half_vectors_close(
                host_out,
                host_ref,
                "endpoint_persistent_smoke_test");
        }

        for (auto& op_dev : op_devs) {
            comm::collective::device_operation_desc_destroy(&op_dev);
        }

        for (auto& table : chunk_states) {
            comm::collective::chunk_state_table_destroy(&table);
        }

        chunk_done.release();
        chunk_steps.release();

        for (auto& accum : accums) {
            comm::transport::free_comm_buffer(group.devices, accum);
        }

        for (size_t i = 0; i < controls.size(); ++i) {
            if (control_initialized[i]) {
                comm::endpoint_persistent_control_destroy(&controls[i]);
                control_initialized[i] = false;
            }
        }

        for (auto& rt : runtimes) {
            comm::endpoint_runtime_destroy(&rt);
        }
        comm::group_destroy(&group);
        return true;
    } catch (...) {
        for (size_t i = 0; i < controls.size(); ++i) {
            try {
                if (kernel_launched[i] && control_initialized[i]) {
                    comm::endpoint_persistent_control_request_stop(&controls[i]);
                    cudaStreamSynchronize(runtimes[i].endpoint.stream);
                }
            } catch (...) {
            }
        }

        for (auto& op_dev : op_devs) {
            comm::collective::device_operation_desc_destroy(&op_dev);
        }

        for (auto& table : chunk_states) {
            comm::collective::chunk_state_table_destroy(&table);
        }

        try { chunk_done.release(); } catch (...) {}
        try { chunk_steps.release(); } catch (...) {}

        for (auto& accum : accums) {
            try {
                comm::transport::free_comm_buffer(group.devices, accum);
            } catch (...) {
            }
        }

        for (size_t i = 0; i < controls.size(); ++i) {
            try {
                if (control_initialized[i]) {
                    comm::endpoint_persistent_control_destroy(&controls[i]);
                }
            } catch (...) {
            }
        }

        for (auto& rt : runtimes) {
            try {
                comm::endpoint_runtime_destroy(&rt);
            } catch (...) {
            }
        }

        try {
            comm::group_destroy(&group);
        } catch (...) {
        }

        throw;
    }
}

} // namespace ooverlap
