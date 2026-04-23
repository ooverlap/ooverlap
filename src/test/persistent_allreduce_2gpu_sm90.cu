#include "test/persistent_allreduce_2gpu_sm90.h"

#include "comm/collective/operation.h"
#include "comm/endpoint_persistent_kernel.h"
#include "comm/endpoint_runtime.h"
#include "comm/group.h"
#include "comm/transport/buffer.h"
#include "ooverlap/system/runtime_utils.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
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

void ensure_bidirectional_peer_access_or_throw(
    int dev0,
    int dev1) {
    int can01 = 0;
    int can10 = 0;

    system::runtime::check_cuda(
        cudaDeviceCanAccessPeer(&can01, dev0, dev1),
        "cudaDeviceCanAccessPeer(dev0 -> dev1)");
    system::runtime::check_cuda(
        cudaDeviceCanAccessPeer(&can10, dev1, dev0),
        "cudaDeviceCanAccessPeer(dev1 -> dev0)");

    if (can01 == 0 || can10 == 0) {
        throw std::runtime_error(
            "benchmark_persistent_two_gpu_allreduce_sm90: selected devices do not support peer access");
    }

    system::runtime::set_device(dev0);
    cudaError_t err01 = cudaDeviceEnablePeerAccess(dev1, 0);
    if (err01 == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
    } else {
        system::runtime::check_cuda(err01, "cudaDeviceEnablePeerAccess(dev0 -> dev1)");
    }

    system::runtime::set_device(dev1);
    cudaError_t err10 = cudaDeviceEnablePeerAccess(dev0, 0);
    if (err10 == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
    } else {
        system::runtime::check_cuda(err10, "cudaDeviceEnablePeerAccess(dev1 -> dev0)");
    }
}

template <typename Fn>
double measure_host_ms(Fn&& fn) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
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

struct WaitPollStats {
    uint64_t polls = 0;
    double memcpy_ms = 0.0;
};

bool wait_until_all_ranks_done_no_sleep(
    const std::vector<int>& devices,
    const std::vector<uint32_t*>& completion_flags,
    std::vector<uint32_t>& host_completion_flags,
    int timeout_ms,
    WaitPollStats* stats) {
    const auto start = std::chrono::steady_clock::now();

    while (true) {
        bool all_done = true;

        const auto copy_begin = std::chrono::steady_clock::now();
        for (size_t r = 0; r < devices.size(); ++r) {
            system::runtime::set_device(devices[r]);
            system::runtime::check_cuda(
                cudaMemcpy(
                    &host_completion_flags[r],
                    completion_flags[r],
                    sizeof(uint32_t),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy(persistent completion flag -> host)");

            if (host_completion_flags[r] != 1u) {
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

std::string build_persistent_progress_debug_string(
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
            "cudaMemcpy(persistent chunk state -> host)");

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

struct PersistentTimingBreakdown {
    double host_wait_done_ms = 0.0;
    double launch_to_done_ms = 0.0;
    double stop_join_ms = 0.0;
    double total_ms = 0.0;
};

struct PersistentTwoGpuState {
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

    size_t bytes = 0;
    uint32_t num_chunks = 0;
    bool kernels_running = false;
};

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
    st->bytes = 0;
    st->num_chunks = 0;
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
    st->bytes = numel * sizeof(uint16_t);
    st->num_chunks = comm::collective::operation_desc_compute_num_chunks(
        st->bytes,
        comm::kEndpointPersistentChunkBytes);

    st->runtimes.resize(2);
    st->controls.resize(2);
    st->accums.resize(2);
    st->progress_mailboxes.resize(2);
    st->chunk_states.resize(2);
    st->ops.resize(2);
    st->op_devs.resize(2);
    st->completion_counts.assign(2, nullptr);
    st->completion_flags.assign(2, nullptr);
    st->host_completion_flags.assign(2, 0u);

    comm::group_init(&st->group, st->devices, comm::kEndpointPersistentChunkBytes);

    for (int r = 0; r < 2; ++r) {
        comm::endpoint_runtime_init(
            &st->runtimes[static_cast<size_t>(r)],
            &st->group,
            r);
    }

    for (int r = 0; r < 2; ++r) {
        const int prev_rank = prev_rank_of(r, 2);
        const int next_rank = next_rank_of(r, 2);

        st->accums[static_cast<size_t>(r)] =
            comm::transport::alloc_peer_visible_buffer_for_rank_with_access_ranks(
                st->group.devices,
                r,
                {r, prev_rank},
                st->bytes);

        st->progress_mailboxes[static_cast<size_t>(r)].init(
            st->group.devices,
            r,
            {r, next_rank},
            st->num_chunks);
    }

    for (int r = 0; r < 2; ++r) {
        comm::collective::chunk_state_table_init(
            &st->chunk_states[static_cast<size_t>(r)],
            st->group.devices[static_cast<size_t>(r)],
            st->num_chunks);
    }

    for (int r = 0; r < 2; ++r) {
        const int next_rank = next_rank_of(r, 2);
        const int prev_rank = prev_rank_of(r, 2);

        comm::endpoint_runtime_build_ring_allreduce_operation(
            &st->runtimes[static_cast<size_t>(r)],
            &st->ops[static_cast<size_t>(r)],
            st->accums[static_cast<size_t>(r)].device_ptr_for_rank(static_cast<size_t>(r)),
            st->accums[static_cast<size_t>(next_rank)].device_ptr_for_rank(static_cast<size_t>(r)),
            st->bytes,
            comm::kEndpointPersistentChunkBytes,
            nullptr,
            nullptr,
            st->progress_mailboxes[static_cast<size_t>(r)].device_ptr_for_rank(static_cast<size_t>(r)),
            st->progress_mailboxes[static_cast<size_t>(prev_rank)].device_ptr_for_rank(static_cast<size_t>(r)),
            st->chunk_states[static_cast<size_t>(r)].records,
            1);

        system::runtime::set_device(st->group.devices[static_cast<size_t>(r)]);
        system::runtime::check_cuda(
            cudaMalloc(&st->completion_counts[static_cast<size_t>(r)], sizeof(uint32_t)),
            "cudaMalloc(persistent completion count)");
        system::runtime::check_cuda(
            cudaMalloc(&st->completion_flags[static_cast<size_t>(r)], sizeof(uint32_t)),
            "cudaMalloc(persistent completion flag)");

        st->ops[static_cast<size_t>(r)].completion_count_ptr =
            reinterpret_cast<uint64_t>(st->completion_counts[static_cast<size_t>(r)]);
        st->ops[static_cast<size_t>(r)].completion_flag_ptr =
            reinterpret_cast<uint64_t>(st->completion_flags[static_cast<size_t>(r)]);
        st->ops[static_cast<size_t>(r)].completion_target =
            comm::collective::operation_desc_local_completion_target(
                &st->ops[static_cast<size_t>(r)]);

        comm::collective::device_operation_desc_create(
            &st->op_devs[static_cast<size_t>(r)],
            st->group.devices[static_cast<size_t>(r)],
            &st->ops[static_cast<size_t>(r)]);
    }

    for (int r = 0; r < 2; ++r) {
        comm::endpoint_persistent_control_init(
            &st->controls[static_cast<size_t>(r)],
            st->group.devices[static_cast<size_t>(r)]);
    }
}

void reset_persistent_two_gpu_state(
    PersistentTwoGpuState* st) {
    if (st == nullptr) {
        throw std::invalid_argument("reset_persistent_two_gpu_state: st is null");
    }

    for (int r = 0; r < 2; ++r) {
        system::runtime::set_device(st->devices[static_cast<size_t>(r)]);
        system::runtime::check_cuda(
            cudaMemset(
                st->accums[static_cast<size_t>(r)].device_ptr_for_rank(static_cast<size_t>(r)),
                0,
                st->bytes),
            "cudaMemset(persistent accum)");
    }

    for (int r = 0; r < 2; ++r) {
        comm::collective::operation_desc_reset_local_state(
            st->devices[static_cast<size_t>(r)],
            &st->ops[static_cast<size_t>(r)]);
    }
}

void launch_persistent_kernels(
    PersistentTwoGpuState* st) {
    if (st == nullptr) {
        throw std::invalid_argument("launch_persistent_kernels: st is null");
    }

    for (int r = 0; r < 2; ++r) {
        system::runtime::check_cuda(
            comm::launch_endpoint_persistent_kernel_sm90(
                comm::endpoint_runtime_device_handle(&st->runtimes[static_cast<size_t>(r)]),
                st->op_devs[static_cast<size_t>(r)].ptr,
                &st->controls[static_cast<size_t>(r)],
                st->runtimes[static_cast<size_t>(r)].endpoint.stream),
            "launch_endpoint_persistent_kernel_sm90(bench)");

        system::runtime::set_device(st->devices[static_cast<size_t>(r)]);
        const cudaError_t q =
            cudaStreamQuery(st->runtimes[static_cast<size_t>(r)].endpoint.stream);
        if (q == cudaErrorNotReady) {
            cudaGetLastError();
        } else {
            system::runtime::check_cuda(
                q,
                "cudaStreamQuery(after persistent launch)");
        }
    }

    st->kernels_running = true;
}

PersistentTimingBreakdown run_persistent_once(
    int dev0,
    int dev1,
    int64_t numel,
    int timeout_ms = 5000) {
    PersistentTimingBreakdown out{};
    PersistentTwoGpuState st;

    try {
        init_persistent_two_gpu_state(&st, dev0, dev1, static_cast<size_t>(numel));
        reset_persistent_two_gpu_state(&st);

        const auto launch_begin = std::chrono::steady_clock::now();
        launch_persistent_kernels(&st);

        WaitPollStats wait_stats{};
        out.host_wait_done_ms = measure_host_ms([&]() {
            const bool all_done = wait_until_all_ranks_done_no_sleep(
                st.devices,
                st.completion_flags,
                st.host_completion_flags,
                timeout_ms,
                &wait_stats);

            if (!all_done) {
                throw std::runtime_error(
                    "persistent control-only timeout;" +
                    build_persistent_progress_debug_string(
                        st.devices,
                        st.progress_mailboxes,
                        st.chunk_states,
                        st.ops));
            }
        });

        const auto done_time = std::chrono::steady_clock::now();
        out.launch_to_done_ms =
            std::chrono::duration<double, std::milli>(done_time - launch_begin).count();

        out.stop_join_ms = measure_host_ms([&]() {
            for (int r = 0; r < 2; ++r) {
                comm::endpoint_persistent_control_request_stop(
                    &st.controls[static_cast<size_t>(r)]);
            }

            for (int r = 0; r < 2; ++r) {
                system::runtime::check_cuda(
                    cudaStreamSynchronize(st.runtimes[static_cast<size_t>(r)].endpoint.stream),
                    "cudaStreamSynchronize(persistent bench stream)");
            }

            st.kernels_running = false;
        });

        out.total_ms = out.launch_to_done_ms;

        std::printf(
            "[wait-prof] polls=%llu memcpy_ms=%.6f avg_copy_us=%.3f\n",
            static_cast<unsigned long long>(wait_stats.polls),
            wait_stats.memcpy_ms,
            (wait_stats.polls == 0)
                ? 0.0
                : (1000.0 * wait_stats.memcpy_ms / static_cast<double>(wait_stats.polls)));
        std::fflush(stdout);

        destroy_persistent_two_gpu_state(&st);
        return out;
    } catch (...) {
        destroy_persistent_two_gpu_state(&st);
        throw;
    }
}

} // namespace

std::map<std::string, double> benchmark_persistent_two_gpu_allreduce_sm90(
    int64_t numel,
    int iters,
    int warmup,
    int dev0,
    int dev1) {
    if (numel <= 0) {
        throw std::invalid_argument(
            "benchmark_persistent_two_gpu_allreduce_sm90: numel must be > 0");
    }
    if (iters <= 0) {
        throw std::invalid_argument(
            "benchmark_persistent_two_gpu_allreduce_sm90: iters must be > 0");
    }
    if (warmup < 0) {
        throw std::invalid_argument(
            "benchmark_persistent_two_gpu_allreduce_sm90: warmup must be >= 0");
    }
    if (dev0 == dev1) {
        throw std::invalid_argument(
            "benchmark_persistent_two_gpu_allreduce_sm90: devices must be distinct");
    }

    ensure_bidirectional_peer_access_or_throw(dev0, dev1);

    double sum_persistent_wait_ms = 0.0;
    double sum_persistent_launch_to_done_ms = 0.0;
    double sum_persistent_stop_join_ms = 0.0;
    double sum_persistent_total_ms = 0.0;

    for (int iter = 0; iter < warmup + iters; ++iter) {
        const bool measure = iter >= warmup;

        const PersistentTimingBreakdown persistent =
            run_persistent_once(
                dev0,
                dev1,
                numel,
                5000);

        if (measure) {
            sum_persistent_wait_ms += persistent.host_wait_done_ms;
            sum_persistent_launch_to_done_ms += persistent.launch_to_done_ms;
            sum_persistent_stop_join_ms += persistent.stop_join_ms;
            sum_persistent_total_ms += persistent.total_ms;
        }
    }

    const double denom = static_cast<double>(iters);
    const double avg_persistent_wait_ms = sum_persistent_wait_ms / denom;
    const double avg_persistent_launch_to_done_ms =
        sum_persistent_launch_to_done_ms / denom;
    const double avg_persistent_stop_join_ms =
        sum_persistent_stop_join_ms / denom;
    const double avg_persistent_total_ms =
        sum_persistent_total_ms / denom;

    std::map<std::string, double> out;
    out["avg_ms_persistent_host_wait_done"] = avg_persistent_wait_ms;
    out["avg_ms_persistent_launch_to_done"] = avg_persistent_launch_to_done_ms;
    out["avg_ms_persistent_stop_join"] = avg_persistent_stop_join_ms;
    out["avg_ms_persistent_total"] = avg_persistent_total_ms;
    out["avg_ms_cuda_memcpy"] = 0.0;
    out["avg_ms_nccl"] = 0.0;
    out["numel"] = static_cast<double>(numel);
    out["iters"] = static_cast<double>(iters);
    out["warmup"] = static_cast<double>(warmup);

    return out;
}

} // namespace ooverlap
