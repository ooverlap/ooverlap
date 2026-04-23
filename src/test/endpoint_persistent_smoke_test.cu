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
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ooverlap {
namespace {

int prev_rank_of(
    int rank,
    int world_size) {
    return (rank - 1 + world_size) % world_size;
}

int next_rank_of(
    int rank,
    int world_size) {
    return (rank + 1) % world_size;
}

void ensure_ring_peer_access_or_throw(
    const std::vector<int>& devices) {
    const int world_size = static_cast<int>(devices.size());

    for (int r = 0; r < world_size; ++r) {
        const int src_dev = devices[static_cast<size_t>(r)];
        const int dst_dev = devices[static_cast<size_t>(next_rank_of(r, world_size))];

        int can_access = 0;
        system::runtime::check_cuda(
            cudaDeviceCanAccessPeer(&can_access, src_dev, dst_dev),
            "cudaDeviceCanAccessPeer(smoke ring edge)");

        if (can_access == 0) {
            throw std::runtime_error(
                "endpoint_persistent_smoke_test: ring edge is not peer-accessible: " +
                std::to_string(src_dev) + " -> " + std::to_string(dst_dev));
        }

        system::runtime::set_device(src_dev);
        cudaError_t enable_err = cudaDeviceEnablePeerAccess(dst_dev, 0);
        if (enable_err == cudaErrorPeerAccessAlreadyEnabled) {
            cudaGetLastError();
        } else {
            system::runtime::check_cuda(
                enable_err,
                "cudaDeviceEnablePeerAccess(smoke)");
        }
    }
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

        if (n == 0) {
            throw std::invalid_argument("DeviceMailbox::init: n must be > 0");
        }

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
        if (owner_rank < 0) {
            throw std::invalid_argument("DeviceMailbox::copy_owner_to_host: mailbox not initialized");
        }

        system::runtime::set_device(devices[static_cast<size_t>(owner_rank)]);
        system::runtime::check_cuda(
            cudaMemcpy(
                host_cache.data(),
                owner_device_ptr(),
                count * sizeof(uint32_t),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(DeviceMailbox owner -> host)");
    }

    void destroy(
        const std::vector<int>& devices) {
        if (owner_rank >= 0 && buf.bytes != 0) {
            comm::transport::free_comm_buffer(devices, buf);
        }
        buf = comm::transport::CommBuffer{};
        host_cache.clear();
        count = 0;
        owner_rank = -1;
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
            if (d64 < 0 ||
                d64 > static_cast<int64_t>(std::numeric_limits<int>::max())) {
                throw std::invalid_argument(
                    "endpoint_persistent_smoke_test: invalid device id");
            }
            out.push_back(static_cast<int>(d64));
        }
    }

    if (out.size() < 2) {
        throw std::invalid_argument(
            "endpoint_persistent_smoke_test: need at least 2 devices");
    }

    for (size_t i = 0; i < out.size(); ++i) {
        for (size_t j = i + 1; j < out.size(); ++j) {
            if (out[i] == out[j]) {
                throw std::invalid_argument(
                    "endpoint_persistent_smoke_test: duplicate devices are not allowed");
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

bool wait_until_all_ranks_operation_done(
    const std::vector<int>& devices,
    const std::vector<uint32_t*>& completion_flags,
    std::vector<uint32_t>& host_completion_flags,
    int timeout_ms) {
    const auto start = std::chrono::steady_clock::now();

    while (true) {
        bool all_done = true;

        for (size_t r = 0; r < completion_flags.size(); ++r) {
            system::runtime::set_device(devices[r]);
            system::runtime::check_cuda(
                cudaMemcpy(
                    &host_completion_flags[r],
                    completion_flags[r],
                    sizeof(uint32_t),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(smoke completion flag -> host)");

            if (host_completion_flags[r] != 1u) {
                all_done = false;
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

std::string build_progress_debug_string(
    const std::vector<int>& devices,
    std::vector<DeviceMailbox>& progress_mailboxes,
    const std::vector<comm::collective::ChunkStateTable>& chunk_states,
    const std::vector<comm::collective::OperationDesc>& ops) {
    std::string out;

    for (size_t r = 0; r < progress_mailboxes.size(); ++r) {
        progress_mailboxes[r].copy_owner_to_host(devices);

        const uint32_t total_steps =
            comm::collective::operation_desc_total_ring_steps(&ops[r]);

        size_t first_bad = progress_mailboxes[r].count;
        for (size_t i = 0; i < progress_mailboxes[r].count; ++i) {
            if (progress_mailboxes[r].host_cache[i] < total_steps) {
                first_bad = i;
                break;
            }
        }

        comm::collective::ChunkState st{};
        system::runtime::set_device(devices[r]);
        system::runtime::check_cuda(
            cudaMemcpy(
                &st,
                chunk_states[r].records,
                sizeof(comm::collective::ChunkState),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(smoke chunk state -> host)");

        out += " rank=" + std::to_string(r) +
               " total_steps=" + std::to_string(total_steps);

        if (first_bad == progress_mailboxes[r].count) {
            out += " progress=complete";
        } else {
            out += " first_incomplete_chunk=" + std::to_string(first_bad) +
                   " progress=" + std::to_string(progress_mailboxes[r].host_cache[first_bad]);
        }

        out += " chunk0(flags=" + std::to_string(st.flags) +
               ", started=" + std::to_string(st.last_step_started) +
               ", completed=" + std::to_string(st.last_step_completed) + ")";
    }

    return out;
}

struct SmokeRunState {
    std::vector<int> devices{};

    comm::Group group{};
    std::vector<comm::EndpointRuntime> runtimes;
    std::vector<comm::EndpointPersistentControl> controls;

    std::vector<comm::transport::CommBuffer> accums;
    std::vector<DeviceMailbox> progress_mailboxes;

    std::vector<comm::collective::ChunkStateTable> chunk_states;
    std::vector<comm::collective::OperationDesc> ops;
    std::vector<comm::collective::DeviceOperationDesc> op_devs;

    std::vector<uint32_t*> completion_counts;
    std::vector<uint32_t*> completion_flags;
    std::vector<uint32_t> host_completion_flags;

    bool kernels_running = false;
};

void destroy_smoke_run_state(
    SmokeRunState* st) {
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
                    "cudaStreamSynchronize(smoke persistent stream)");
            } catch (...) {
            }
        }
        st->kernels_running = false;
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

    for (auto& box : st->progress_mailboxes) {
        try {
            box.destroy(st->devices);
        } catch (...) {
        }
    }

    for (auto& accum : st->accums) {
        try {
            comm::transport::free_comm_buffer(st->devices, accum);
        } catch (...) {
        }
    }

    for (size_t r = 0; r < st->completion_counts.size(); ++r) {
        try {
            if (st->completion_counts[r] != nullptr) {
                system::runtime::set_device(st->devices[r]);
                cudaFree(st->completion_counts[r]);
            }
        } catch (...) {
        }
    }

    for (size_t r = 0; r < st->completion_flags.size(); ++r) {
        try {
            if (st->completion_flags[r] != nullptr) {
                system::runtime::set_device(st->devices[r]);
                cudaFree(st->completion_flags[r]);
            }
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

    try {
        comm::group_destroy(&st->group);
    } catch (...) {
    }

    st->devices.clear();
    st->runtimes.clear();
    st->controls.clear();
    st->accums.clear();
    st->progress_mailboxes.clear();
    st->chunk_states.clear();
    st->ops.clear();
    st->op_devs.clear();
    st->completion_counts.clear();
    st->completion_flags.clear();
    st->host_completion_flags.clear();
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

    SmokeRunState st;
    st.devices = normalize_devices(devices64);

    const int world_size = static_cast<int>(st.devices.size());
    const size_t bytes = static_cast<size_t>(numel) * sizeof(half);
    const size_t chunk_bytes = comm::kEndpointPersistentChunkBytes;
    const uint32_t num_chunks =
        comm::collective::operation_desc_compute_num_chunks(bytes, chunk_bytes);

    try {
        ensure_ring_peer_access_or_throw(st.devices);

        st.runtimes.resize(static_cast<size_t>(world_size));
        st.controls.resize(static_cast<size_t>(world_size));
        st.accums.resize(static_cast<size_t>(world_size));
        st.progress_mailboxes.resize(static_cast<size_t>(world_size));
        st.chunk_states.resize(static_cast<size_t>(world_size));
        st.ops.resize(static_cast<size_t>(world_size));
        st.op_devs.resize(static_cast<size_t>(world_size));
        st.completion_counts.assign(static_cast<size_t>(world_size), nullptr);
        st.completion_flags.assign(static_cast<size_t>(world_size), nullptr);
        st.host_completion_flags.assign(static_cast<size_t>(world_size), 0u);

        std::vector<std::vector<half>> host_srcs(static_cast<size_t>(world_size));
        for (int r = 0; r < world_size; ++r) {
            host_srcs[static_cast<size_t>(r)] = make_host_pattern(numel, r);
        }
        const auto host_ref = sum_host_vectors(host_srcs);

        comm::group_init(&st.group, st.devices, comm::kEndpointPersistentChunkBytes);

        for (int r = 0; r < world_size; ++r) {
            comm::endpoint_runtime_init(
                &st.runtimes[static_cast<size_t>(r)],
                &st.group,
                r);
        }

        for (int r = 0; r < world_size; ++r) {
            const int prev_rank = prev_rank_of(r, world_size);
            const int next_rank = next_rank_of(r, world_size);

            // Accum owner r must be visible to:
            // - r itself
            // - prev rank, which writes into r's buffer
            st.accums[static_cast<size_t>(r)] =
                comm::transport::alloc_peer_visible_buffer_for_rank_with_access_ranks(
                    st.group.devices,
                    r,
                    {r, prev_rank},
                    bytes);

            // Progress owner r must be visible to:
            // - r itself, which publishes local progress
            // - next rank, which polls it as prev_progress
            st.progress_mailboxes[static_cast<size_t>(r)].init(
                st.group.devices,
                r,
                {r, next_rank},
                num_chunks);
        }

        for (int r = 0; r < world_size; ++r) {
            system::runtime::set_device(st.group.devices[static_cast<size_t>(r)]);
            system::runtime::check_cuda(
                cudaMemcpy(
                    st.accums[static_cast<size_t>(r)].device_ptr_for_rank(static_cast<size_t>(r)),
                    host_srcs[static_cast<size_t>(r)].data(),
                    bytes,
                    cudaMemcpyHostToDevice),
                "cudaMemcpy(smoke host_src -> accum)");
        }

        for (int r = 0; r < world_size; ++r) {
            comm::collective::chunk_state_table_init(
                &st.chunk_states[static_cast<size_t>(r)],
                st.group.devices[static_cast<size_t>(r)],
                num_chunks);
        }

        for (int r = 0; r < world_size; ++r) {
            const int next_rank = next_rank_of(r, world_size);
            const int prev_rank = prev_rank_of(r, world_size);

            comm::endpoint_runtime_build_ring_allreduce_operation(
                &st.runtimes[static_cast<size_t>(r)],
                &st.ops[static_cast<size_t>(r)],
                st.accums[static_cast<size_t>(r)].device_ptr_for_rank(static_cast<size_t>(r)),
                st.accums[static_cast<size_t>(next_rank)].device_ptr_for_rank(static_cast<size_t>(r)),
                bytes,
                chunk_bytes,
                nullptr,
                nullptr,
                st.progress_mailboxes[static_cast<size_t>(r)].device_ptr_for_rank(static_cast<size_t>(r)),
                st.progress_mailboxes[static_cast<size_t>(prev_rank)].device_ptr_for_rank(static_cast<size_t>(r)),
                st.chunk_states[static_cast<size_t>(r)].records,
                1);

            system::runtime::set_device(st.group.devices[static_cast<size_t>(r)]);
            system::runtime::check_cuda(
                cudaMalloc(&st.completion_counts[static_cast<size_t>(r)], sizeof(uint32_t)),
                "cudaMalloc(smoke completion count)");
            system::runtime::check_cuda(
                cudaMalloc(&st.completion_flags[static_cast<size_t>(r)], sizeof(uint32_t)),
                "cudaMalloc(smoke completion flag)");

            st.ops[static_cast<size_t>(r)].completion_count_ptr =
                reinterpret_cast<uint64_t>(st.completion_counts[static_cast<size_t>(r)]);
            st.ops[static_cast<size_t>(r)].completion_flag_ptr =
                reinterpret_cast<uint64_t>(st.completion_flags[static_cast<size_t>(r)]);
            st.ops[static_cast<size_t>(r)].completion_target =
                comm::collective::operation_desc_local_completion_target(
                    &st.ops[static_cast<size_t>(r)]);

            comm::collective::device_operation_desc_create(
                &st.op_devs[static_cast<size_t>(r)],
                st.group.devices[static_cast<size_t>(r)],
                &st.ops[static_cast<size_t>(r)]);
        }

        for (int r = 0; r < world_size; ++r) {
            comm::collective::operation_desc_reset_local_state(
                st.group.devices[static_cast<size_t>(r)],
                &st.ops[static_cast<size_t>(r)]);
        }

        for (int r = 0; r < world_size; ++r) {
            comm::endpoint_persistent_control_init(
                &st.controls[static_cast<size_t>(r)],
                st.group.devices[static_cast<size_t>(r)]);
        }

        for (int r = 0; r < world_size; ++r) {
            system::runtime::set_device(st.group.devices[static_cast<size_t>(r)]);
            system::runtime::check_cuda(
                cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 8 * 1024 * 1024),
                "cudaDeviceSetLimit(cudaLimitPrintfFifoSize)");
        }

        for (int r = 0; r < world_size; ++r) {
            system::runtime::check_cuda(
                comm::launch_endpoint_persistent_kernel_sm90(
                    comm::endpoint_runtime_device_handle(
                        &st.runtimes[static_cast<size_t>(r)]),
                    st.op_devs[static_cast<size_t>(r)].ptr,
                    &st.controls[static_cast<size_t>(r)],
                    st.runtimes[static_cast<size_t>(r)].endpoint.stream),
                "launch_endpoint_persistent_kernel_sm90(smoke)");
        }
        st.kernels_running = true;

        const bool all_done = wait_until_all_ranks_operation_done(
            st.devices,
            st.completion_flags,
            st.host_completion_flags,
            timeout_ms);

        if (!all_done) {
            throw std::runtime_error(
                "endpoint_persistent_smoke_test: timeout waiting for completion;" +
                build_progress_debug_string(
                    st.devices,
                    st.progress_mailboxes,
                    st.chunk_states,
                    st.ops));
        }

        for (int r = 0; r < world_size; ++r) {
            comm::endpoint_persistent_control_request_stop(
                &st.controls[static_cast<size_t>(r)]);
        }

        for (int r = 0; r < world_size; ++r) {
            system::runtime::check_cuda(
                cudaStreamSynchronize(st.runtimes[static_cast<size_t>(r)].endpoint.stream),
                "cudaStreamSynchronize(smoke persistent stream)");
        }
        st.kernels_running = false;

        for (int r = 0; r < world_size; ++r) {
            std::vector<half> host_out(static_cast<size_t>(numel));
            system::runtime::set_device(st.group.devices[static_cast<size_t>(r)]);
            system::runtime::check_cuda(
                cudaMemcpy(
                    host_out.data(),
                    st.accums[static_cast<size_t>(r)].device_ptr_for_rank(static_cast<size_t>(r)),
                    bytes,
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(smoke accum -> host)");

            expect_half_vectors_close(
                host_out,
                host_ref,
                "endpoint_persistent_smoke_test");
        }

        destroy_smoke_run_state(&st);
        return true;
    } catch (...) {
        destroy_smoke_run_state(&st);
        throw;
    }
}

} // namespace ooverlap
