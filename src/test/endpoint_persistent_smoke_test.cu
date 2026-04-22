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
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

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

std::string build_progress_debug_string(
    const std::vector<HostMappedMailbox>& inbound_steps,
    const std::vector<HostMappedMailbox>& done_flags) {
    std::string out;

    for (size_t r = 0; r < done_flags.size(); ++r) {
        for (size_t c = 0; c < done_flags[r].count; ++c) {
            if (done_flags[r].host_ptr[c] != 1u) {
                out += " first incomplete rank=" + std::to_string(r) +
                       " chunk=" + std::to_string(c) +
                       " done=" + std::to_string(done_flags[r].host_ptr[c]) +
                       " inbound_step=" + std::to_string(inbound_steps[r].host_ptr[c]);
                break;
            }
        }
    }

    if (out.empty()) {
        out = " all ranks complete";
    }
    return out;
}

static void dump_rank_accum_samples(
    const std::vector<int>& devices,
    const std::vector<comm::transport::CommBuffer>& accums,
    int64_t numel,
    int sample_count = 8) {
    const int64_t n = std::min<int64_t>(numel, sample_count);

    for (size_t r = 0; r < accums.size(); ++r) {
        std::vector<half> host(static_cast<size_t>(n));

        system::runtime::set_device(devices[r]);
        system::runtime::check_cuda(
            cudaMemcpy(
    host.data(),
    accums[r].device_ptr_for_rank(r),
                static_cast<size_t>(n) * sizeof(half),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(accum sample -> host)");

        std::printf("[debug] rank=%zu accum[0:%lld] =", r, static_cast<long long>(n));
        for (int64_t i = 0; i < n; ++i) {
            std::printf(" %.6f", __half2float(host[static_cast<size_t>(i)]));
        }
        std::printf("\n");
    }
    std::fflush(stdout);
}

static void dump_rank_chunk0_state(
    const std::vector<int>& devices,
    const std::vector<comm::collective::ChunkStateTable>& chunk_states) {
    for (size_t r = 0; r < chunk_states.size(); ++r) {
        comm::collective::ChunkState host_state{};

        system::runtime::set_device(devices[r]);
        system::runtime::check_cuda(
            cudaMemcpy(
                &host_state,
                chunk_states[r].records,
                sizeof(comm::collective::ChunkState),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy(chunk state -> host)");

        std::printf(
            "[debug] rank=%zu chunk0_state flags=%u last_started=%u last_completed=%u bytes=%zu offset=%zu\n",
            r,
            host_state.flags,
            host_state.last_step_started,
            host_state.last_step_completed,
            host_state.bytes,
            host_state.offset_bytes);
    }
    std::fflush(stdout);
}

static void print_peer_access_matrix_and_validate_ring(
    const std::vector<int>& devices) {
    const int n = static_cast<int>(devices.size());

    std::printf("[smoke] cudaDeviceCanAccessPeer matrix (rows=src, cols=dst)\n");
    std::printf("[smoke]        ");
    for (int j = 0; j < n; ++j) {
        std::printf(" %4d", devices[static_cast<size_t>(j)]);
    }
    std::printf("\n");

    for (int i = 0; i < n; ++i) {
        std::printf("[smoke] src=%2d:", devices[static_cast<size_t>(i)]);
        for (int j = 0; j < n; ++j) {
            int can_access = 0;
            system::runtime::check_cuda(
                cudaDeviceCanAccessPeer(
                    &can_access,
                    devices[static_cast<size_t>(i)],
                    devices[static_cast<size_t>(j)]),
                "cudaDeviceCanAccessPeer");
            std::printf(" %4d", can_access);
        }
        std::printf("\n");
    }
    std::fflush(stdout);

    // Validate the actual ring edges you are about to use:
    // devices[0] -> devices[1] -> ... -> devices[n-1] -> devices[0]
    for (int i = 0; i < n; ++i) {
        const int src_dev = devices[static_cast<size_t>(i)];
        const int dst_dev = devices[static_cast<size_t>((i + 1) % n)];

        int can_access = 0;
        system::runtime::check_cuda(
            cudaDeviceCanAccessPeer(&can_access, src_dev, dst_dev),
            "cudaDeviceCanAccessPeer(ring edge)");

        std::printf(
            "[smoke] ring edge %d -> %d peer_access=%d\n",
            src_dev,
            dst_dev,
            can_access);
        std::fflush(stdout);

        if (can_access == 0) {
            throw std::runtime_error(
                std::string("endpoint_persistent_smoke_test: chosen ring edge is not peer-accessible: ") +
                std::to_string(src_dev) + " -> " + std::to_string(dst_dev));
        }
    }
}

static void dump_accum_peer_views(
    const std::vector<int>& devices,
    const std::vector<comm::transport::CommBuffer>& accums,
    const std::vector<std::vector<half>>& host_srcs,
    int64_t numel,
    int sample_count = 8) {
    const int64_t n = std::min<int64_t>(numel, sample_count);

    std::printf("[debug] prelaunch accum peer-view dump\n");
    for (size_t owner = 0; owner < accums.size(); ++owner) {
        std::printf("[debug] owner=%zu expected =", owner);
        for (int64_t i = 0; i < n; ++i) {
            std::printf(" %.6f", __half2float(host_srcs[owner][static_cast<size_t>(i)]));
        }
        std::printf("\n");

        for (size_t view_rank = 0; view_rank < devices.size(); ++view_rank) {
            std::vector<half> host(static_cast<size_t>(n));

            system::runtime::set_device(devices[view_rank]);
            system::runtime::check_cuda(
                cudaMemcpy(
                    host.data(),
                    accums[owner].device_ptr_for_rank(view_rank),
                    static_cast<size_t>(n) * sizeof(half),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(accum peer-view -> host)");

            std::printf(
                "[debug] owner=%zu view_rank=%zu dev=%d ptr=0x%llx values =",
                owner,
                view_rank,
                devices[view_rank],
                static_cast<unsigned long long>(
                    reinterpret_cast<uintptr_t>(
                        accums[owner].device_ptr_for_rank(view_rank))));

            for (int64_t i = 0; i < n; ++i) {
                std::printf(" %.6f", __half2float(host[static_cast<size_t>(i)]));
            }
            std::printf("\n");
        }
    }
    std::fflush(stdout);
}

static void validate_accum_peer_views_or_throw(
    const std::vector<int>& devices,
    const std::vector<comm::transport::CommBuffer>& accums,
    const std::vector<std::vector<half>>& host_srcs,
    int64_t numel,
    int sample_count = 8) {
    const int64_t n = std::min<int64_t>(numel, sample_count);

    for (size_t owner = 0; owner < accums.size(); ++owner) {
        for (size_t view_rank = 0; view_rank < devices.size(); ++view_rank) {
            std::vector<half> host(static_cast<size_t>(n));

            system::runtime::set_device(devices[view_rank]);
            system::runtime::check_cuda(
                cudaMemcpy(
                    host.data(),
                    accums[owner].device_ptr_for_rank(view_rank),
                    static_cast<size_t>(n) * sizeof(half),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(accum peer-view -> host)");

            for (int64_t i = 0; i < n; ++i) {
                const float got = __half2float(host[static_cast<size_t>(i)]);
                const float ref = __half2float(host_srcs[owner][static_cast<size_t>(i)]);
                const float err = std::fabs(got - ref);

                if (err > 1.0e-3f) {
                    throw std::runtime_error(
                        std::string("prelaunch accum peer-view mismatch: owner=") +
                        std::to_string(owner) +
                        " view_rank=" + std::to_string(view_rank) +
                        " idx=" + std::to_string(i) +
                        " got=" + std::to_string(got) +
                        " ref=" + std::to_string(ref));
                }
            }
        }
    }
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
    print_peer_access_matrix_and_validate_ring(devices);

    comm::Group group{};
    std::vector<comm::EndpointRuntime> runtimes(static_cast<size_t>(world_size));
    std::vector<comm::EndpointPersistentControl> controls(static_cast<size_t>(world_size));

    std::vector<comm::transport::CommBuffer> accums(static_cast<size_t>(world_size));
    std::vector<HostMappedMailbox> inbound_steps(static_cast<size_t>(world_size));
    std::vector<HostMappedMailbox> done_flags(static_cast<size_t>(world_size));

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

            inbound_steps[static_cast<size_t>(r)].init(num_chunks, devices);
            done_flags[static_cast<size_t>(r)].init(num_chunks, devices);
        }

        for (int r = 0; r < world_size; ++r) {
            system::runtime::set_device(group.devices[static_cast<size_t>(r)]);
            system::runtime::check_cuda(
                    cudaMemcpy(
    accums[static_cast<size_t>(r)].device_ptr_for_rank(static_cast<size_t>(r)),
    host_srcs[static_cast<size_t>(r)].data(),
                    bytes,
                    cudaMemcpyHostToDevice),
                "cudaMemcpy(host_src -> accum)");
        }

        dump_accum_peer_views(
    devices,
    accums,
    host_srcs,
    numel,
    8);

validate_accum_peer_views_or_throw(
    devices,
    accums,
    host_srcs,
    numel,
    8);

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
              accums[static_cast<size_t>(r)].device_ptr_for_rank(static_cast<size_t>(r)),
              accums[static_cast<size_t>(next_rank)].device_ptr_for_rank(static_cast<size_t>(r)),
              bytes,
              chunk_bytes,
              inbound_steps[static_cast<size_t>(r)].device_ptr_for_rank(static_cast<size_t>(r)),
              inbound_steps[static_cast<size_t>(next_rank)].device_ptr_for_rank(static_cast<size_t>(r)),
              done_flags[static_cast<size_t>(r)].device_ptr_for_rank(static_cast<size_t>(r)),
              done_flags[static_cast<size_t>(next_rank)].device_ptr_for_rank(static_cast<size_t>(r)),
              chunk_states[static_cast<size_t>(r)].records,
              1);

            std::printf(
    "[smoke] op rank=%d accum=0x%llx next_accum=0x%llx inbound=0x%llx next_inbound=0x%llx done=0x%llx next_done=0x%llx\n",
    r,
    static_cast<unsigned long long>(ops[static_cast<size_t>(r)].accum_ptr),
    static_cast<unsigned long long>(ops[static_cast<size_t>(r)].next_accum_ptr),
    static_cast<unsigned long long>(ops[static_cast<size_t>(r)].inbound_steps_ptr),
    static_cast<unsigned long long>(ops[static_cast<size_t>(r)].next_inbound_steps_ptr),
    static_cast<unsigned long long>(ops[static_cast<size_t>(r)].done_ptr),
    static_cast<unsigned long long>(ops[static_cast<size_t>(r)].next_done_ptr));
std::fflush(stdout);
            }

        for (int r = 0; r < world_size; ++r) {
            inbound_steps[static_cast<size_t>(r)].reset(
                comm::collective::kOperationInboundStepInvalid);
            done_flags[static_cast<size_t>(r)].reset(0u);
        }

        for (int r = 0; r < world_size; ++r) {
            for (uint32_t idx = 0; idx < num_chunks; ++idx) {
                if (comm::collective::operation_desc_actor_rank_for_step(
                        &ops[static_cast<size_t>(r)],
                        idx,
                        0u) == r) {
                    inbound_steps[static_cast<size_t>(r)].host_ptr[idx] = 0u;
                }
            }
        }

        for (int r = 0; r < world_size; ++r) {
            std::printf("[smoke] init mailbox rank=%d chunk0 inbound=%u done=%u\n",
                        r,
                        inbound_steps[static_cast<size_t>(r)].host_ptr[0],
                        done_flags[static_cast<size_t>(r)].host_ptr[0]);
        }
        std::fflush(stdout);

        for (int r = 0; r < world_size; ++r) {
            comm::collective::chunk_state_table_reset(
                &chunk_states[static_cast<size_t>(r)]);
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
    accums[static_cast<size_t>(r)].device_ptr_for_rank(static_cast<size_t>(r)),
    accums[static_cast<size_t>(r)].device_ptr_for_rank(static_cast<size_t>(r)),
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

        for (int r = 0; r < world_size; ++r) {
            system::runtime::set_device(group.devices[static_cast<size_t>(r)]);
            system::runtime::check_cuda(
                cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 8 * 1024 * 1024),
                "cudaDeviceSetLimit(cudaLimitPrintfFifoSize)");
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
        const bool all_done = wait_until_all_ranks_done(done_flags, timeout_ms);

        if (!all_done) {
            dump_rank_chunk0_state(devices, chunk_states);
            dump_rank_accum_samples(devices, accums, numel);
        
            throw std::runtime_error(
                std::string("endpoint_persistent_smoke_test: timeout waiting for done flags; ") +
                build_progress_debug_string(
                    inbound_steps,
                    done_flags));
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

        for (int r = 0; r < world_size; ++r) {
            for (size_t i = 0; i < done_flags[static_cast<size_t>(r)].count; ++i) {
                if (done_flags[static_cast<size_t>(r)].host_ptr[i] != 1u) {
                    throw std::runtime_error(
                        "endpoint_persistent_smoke_test: done flag was not set");
                }
            }
        }

        for (int r = 0; r < world_size; ++r) {
            std::vector<half> host_out(static_cast<size_t>(numel));
            system::runtime::set_device(group.devices[static_cast<size_t>(r)]);
            system::runtime::check_cuda(
                cudaMemcpy(
    host_out.data(),
    accums[static_cast<size_t>(r)].device_ptr_for_rank(static_cast<size_t>(r)),
                    bytes,
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(accum -> host)");

            try {
                expect_half_vectors_close(
                    host_out,
                    host_ref,
                    "endpoint_persistent_smoke_test");
            } catch (...) {
                dump_rank_chunk0_state(devices, chunk_states);
                dump_rank_accum_samples(devices, accums, numel);
                throw;
            }
        }

        for (auto& op_dev : op_devs) {
            comm::collective::device_operation_desc_destroy(&op_dev);
        }

        for (auto& table : chunk_states) {
            comm::collective::chunk_state_table_destroy(&table);
        }

        for (auto& box : done_flags) {
            box.destroy();
        }
        for (auto& box : inbound_steps) {
            box.destroy();
        }
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
            try { comm::collective::device_operation_desc_destroy(&op_dev); } catch (...) {}
        }

        for (auto& table : chunk_states) {
            try { comm::collective::chunk_state_table_destroy(&table); } catch (...) {}
        }

        for (auto& box : done_flags) {
            try { box.destroy(); } catch (...) {}
        }
        for (auto& box : inbound_steps) {
            try { box.destroy(); } catch (...) {}
        }
        for (auto& accum : accums) {
            try { comm::transport::free_comm_buffer(group.devices, accum); } catch (...) {}
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
            try { comm::endpoint_runtime_destroy(&rt); } catch (...) {}
        }

        try { comm::group_destroy(&group); } catch (...) {}
        throw;
    }
}

} // namespace ooverlap
