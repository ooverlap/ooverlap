#include "test/external_p2p_collective_sweep_sm90.h"

#include "comm/ooverlap_comm_private.h"
#include "ooverlap/comm.h"
#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/testing/checks.cuh"
#include "ooverlap/testing/collective_test_utils.cuh"
#include "ooverlap/testing/two_gpu_test_utils.cuh"
#include "ooverlap/testing/nccl_utils.cuh"

#include "test/internal_comm_test_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifndef OOVERLAP_BENCH_VERIFY_RESULTS
#define OOVERLAP_BENCH_VERIFY_RESULTS 0
#endif

namespace ooverlap {
namespace {

using testing::TestCollective;

int ring_size_from_env() {
    constexpr int kDefaultRingSize = 16;
    constexpr int kMaxReasonableRingSize = 4096;

    const char* env = std::getenv("OOVERLAP_BENCH_RING_SIZE");
    if (env == nullptr || env[0] == '\0') {
        return kDefaultRingSize;
    }

    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end == env || parsed <= 0) {
        return kDefaultRingSize;
    }

    return static_cast<int>(
        std::min<long>(parsed, kMaxReasonableRingSize));
}

struct SweepContext {
    std::vector<int> devices;
    oo_group_t* group = nullptr;
    std::vector<oo_node_t*> nodes;
    std::vector<cudaStream_t> streams;
    std::vector<ncclComm_t> nccl_comms;
};

struct SizeBuffers {
    std::vector<half*> sources;
    std::vector<half*> ooverlap_work;
    std::vector<oo_buffer_t*> ooverlap_buffers;
    std::vector<half*> nccl_work;
    std::vector<half*> nccl_symmetric_work;
    std::vector<ncclWindow_t> nccl_symmetric_windows;
};

struct TimingEvents {
    std::vector<cudaEvent_t> starts;
    std::vector<cudaEvent_t> stops;
};

/*
 * Benchmark-only persistent host workers for same-process multi-GPU launch.
 * Workers spin between submissions so CUDA event timing does not absorb an OS
 * condition-variable wakeup on every collective. Stop them before NCCL timing.
 */
class ParallelRankSubmission {
public:
    explicit ParallelRankSubmission(std::size_t rank_count)
        : errors_(rank_count) {
        if (rank_count == 0) {
            throw std::invalid_argument(
                "ParallelRankSubmission requires at least one rank");
        }

        workers_.reserve(rank_count);
        try {
            for (std::size_t rank = 0; rank < rank_count; ++rank) {
                workers_.emplace_back(
                    [this, rank]() { worker_loop(rank); });
            }
        } catch (...) {
            stop();
            throw;
        }
    }

    ~ParallelRankSubmission() {
        stop();
    }

    ParallelRankSubmission(const ParallelRankSubmission&) = delete;
    ParallelRankSubmission& operator=(const ParallelRankSubmission&) = delete;

    void run(std::function<void(std::size_t)> job) {
        if (!job) {
            throw std::invalid_argument(
                "ParallelRankSubmission job must be non-empty");
        }
        if (stop_.load(std::memory_order_acquire)) {
            throw std::logic_error(
                "ParallelRankSubmission has already been stopped");
        }

        for (std::exception_ptr& error : errors_) {
            error = nullptr;
        }

        job_ = std::move(job);
        completed_.store(0, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);

        while (completed_.load(std::memory_order_acquire) != workers_.size()) {
        }

        for (const std::exception_ptr& error : errors_) {
            if (error != nullptr) {
                std::rethrow_exception(error);
            }
        }
    }

    void stop() noexcept {
        stop_.store(true, std::memory_order_release);
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

private:
    void worker_loop(std::size_t rank) {
        std::uint64_t seen_generation = 0;

        for (;;) {
            if (stop_.load(std::memory_order_acquire)) {
                return;
            }

            const std::uint64_t generation =
                generation_.load(std::memory_order_acquire);
            if (generation == seen_generation) {
                continue;
            }

            seen_generation = generation;

            try {
                job_(rank);
            } catch (...) {
                errors_[rank] = std::current_exception();
            }

            completed_.fetch_add(1, std::memory_order_acq_rel);
        }
    }

    std::vector<std::thread> workers_;
    std::vector<std::exception_ptr> errors_;
    std::function<void(std::size_t)> job_;
    std::atomic<std::uint64_t> generation_{0};
    std::atomic<std::size_t> completed_{0};
    std::atomic<bool> stop_{false};
};

void validate_devices(const std::vector<int>& devices) {
    if (devices.size() < 2) {
        throw std::invalid_argument(
            "external P2P sweep requires at least two devices");
    }

    std::set<int> unique;
    for (int device : devices) {
        if (device < 0) {
            throw std::invalid_argument("device ids must be non-negative");
        }
        if (!unique.insert(device).second) {
            throw std::invalid_argument("device ids must be unique");
        }
    }
}

void validate_sizes(
    TestCollective collective,
    const std::vector<int64_t>& sizes,
    int world_size) {
    if (sizes.empty()) {
        throw std::invalid_argument("sizes must be non-empty");
    }

    for (int64_t numel : sizes) {
        if (numel <= 0) {
            throw std::invalid_argument("all sizes must be > 0");
        }
        testing::validate_numel_for_collective(
            collective,
            numel,
            world_size);
    }
}

void sync_streams(
    const std::vector<int>& devices,
    const std::vector<cudaStream_t>& streams,
    const char* label) {
    if (devices.size() != streams.size()) {
        throw std::invalid_argument("sync_streams: size mismatch");
    }

    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        system::runtime::sync_stream_on_device(
            devices[rank],
            streams[rank],
            label);
    }
}

void destroy_timing_events_best_effort(
    const std::vector<int>& devices,
    TimingEvents& events) {
    const std::size_t start_count =
        std::min(devices.size(), events.starts.size());
    for (std::size_t rank = 0; rank < start_count; ++rank) {
        if (events.starts[rank] != nullptr) {
            (void)cudaSetDevice(devices[rank]);
            (void)cudaEventDestroy(events.starts[rank]);
            events.starts[rank] = nullptr;
        }
    }

    const std::size_t stop_count =
        std::min(devices.size(), events.stops.size());
    for (std::size_t rank = 0; rank < stop_count; ++rank) {
        if (events.stops[rank] != nullptr) {
            (void)cudaSetDevice(devices[rank]);
            (void)cudaEventDestroy(events.stops[rank]);
            events.stops[rank] = nullptr;
        }
    }
}

TimingEvents create_timing_events(const std::vector<int>& devices) {
    TimingEvents events;
    events.starts.assign(devices.size(), nullptr);
    events.stops.assign(devices.size(), nullptr);

    try {
        for (std::size_t rank = 0; rank < devices.size(); ++rank) {
            system::runtime::set_device(devices[rank]);
            testing::check_cuda(
                cudaEventCreate(&events.starts[rank]),
                "cudaEventCreate(external sweep start)");
            testing::check_cuda(
                cudaEventCreate(&events.stops[rank]),
                "cudaEventCreate(external sweep stop)");
        }
    } catch (...) {
        destroy_timing_events_best_effort(devices, events);
        throw;
    }

    return events;
}

template <typename LaunchOnce>
double elapsed_collective_once_ms(
    const std::vector<int>& devices,
    const std::vector<cudaStream_t>& streams,
    TimingEvents& events,
    LaunchOnce&& launch_once) {
    if (devices.empty() ||
        devices.size() != streams.size() ||
        devices.size() != events.starts.size() ||
        devices.size() != events.stops.size()) {
        throw std::invalid_argument(
            "elapsed_collective_once_ms: size mismatch");
    }

    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        system::runtime::set_device(devices[rank]);
        testing::check_cuda(
            cudaEventRecord(events.starts[rank], streams[rank]),
            "cudaEventRecord(external sweep start)");
    }

    launch_once();

    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        system::runtime::set_device(devices[rank]);
        testing::check_cuda(
            cudaEventRecord(events.stops[rank], streams[rank]),
            "cudaEventRecord(external sweep stop)");
    }

    double max_ms = 0.0;
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        system::runtime::set_device(devices[rank]);
        testing::check_cuda(
            cudaEventSynchronize(events.stops[rank]),
            "cudaEventSynchronize(external sweep stop)");

        float rank_ms = 0.0f;
        testing::check_cuda(
            cudaEventElapsedTime(
                &rank_ms,
                events.starts[rank],
                events.stops[rank]),
            "cudaEventElapsedTime(external sweep)");
        max_ms = std::max(max_ms, static_cast<double>(rank_ms));
    }

    return max_ms;
}

void nccl_mem_alloc_half_on_device(
    int device,
    half** ptr,
    std::size_t bytes,
    const char* label) {
    if (ptr == nullptr) {
        throw std::invalid_argument(
            "nccl_mem_alloc_half_on_device: ptr must not be null");
    }

    *ptr = nullptr;
    system::runtime::set_device(device);

    void* raw = nullptr;
    OOVERLAP_TEST_NCCL_CHECK(ncclMemAlloc(&raw, bytes));
    if (raw == nullptr) {
        throw std::runtime_error(
            std::string(label != nullptr ? label : "ncclMemAlloc") +
            ": returned nullptr");
    }

    *ptr = reinterpret_cast<half*>(raw);
}

void nccl_mem_free_on_device(int device, half*& ptr) {
    if (ptr == nullptr) {
        return;
    }

    (void)cudaSetDevice(device);
    (void)ncclMemFree(static_cast<void*>(ptr));
    ptr = nullptr;
}

void deregister_nccl_window_best_effort(
    ncclComm_t comm,
    ncclWindow_t& window) {
    if (comm == nullptr || window == nullptr) {
        return;
    }

    (void)ncclCommWindowDeregister(comm, window);
    window = nullptr;
}

void register_nccl_symmetric_windows(
    const std::vector<ncclComm_t>& comms,
    const std::vector<half*>& buffers,
    std::size_t bytes,
    std::vector<ncclWindow_t>& windows) {
    if (comms.empty() || comms.size() != buffers.size()) {
        throw std::invalid_argument(
            "register_nccl_symmetric_windows: size mismatch");
    }

    windows.assign(comms.size(), nullptr);

    OOVERLAP_TEST_NCCL_CHECK(ncclGroupStart());
    for (std::size_t rank = 0; rank < comms.size(); ++rank) {
        OOVERLAP_TEST_NCCL_CHECK(
            ncclCommWindowRegister(
                comms[rank],
                buffers[rank],
                bytes,
                &windows[rank],
                NCCL_WIN_COLL_SYMMETRIC));
    }
    OOVERLAP_TEST_NCCL_CHECK(ncclGroupEnd());
}

void launch_ooverlap_once_for_rank(
    TestCollective collective,
    oo_node_t* node,
    oo_buffer_t* buffer,
    std::size_t numel,
    cudaStream_t stream,
    const char* label) {
    if (collective == TestCollective::AllReduce) {
        testing::check_oo(
            oo_allreduce_tuned(
                node,
                buffer,
                numel,
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                OO_TUNING_BEST_PERFORMANCE,
                stream),
            label);
        return;
    }

    if (collective == TestCollective::ReduceScatter) {
        oo_tensor_slice_t slice{};
        testing::check_oo(
            oo_reduce_scatter_tuned(
                node,
                buffer,
                numel,
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                OO_TUNING_BEST_PERFORMANCE,
                &slice,
                stream),
            label);
        return;
    }

    if (collective == TestCollective::AllGather) {
        testing::check_oo(
            oo_all_gather_tuned(
                node,
                buffer,
                numel,
                OO_DTYPE_FLOAT16,
                OO_TUNING_BEST_PERFORMANCE,
                stream),
            label);
        return;
    }

    throw std::invalid_argument("unsupported collective");
}

void launch_ooverlap_once(
    TestCollective collective,
    const SweepContext& ctx,
    const SizeBuffers& buffers,
    std::size_t numel) {
    if (ctx.nodes.size() != buffers.ooverlap_buffers.size() ||
        ctx.nodes.size() != ctx.streams.size()) {
        throw std::invalid_argument("launch_ooverlap_once: size mismatch");
    }

    for (std::size_t rank = 0; rank < ctx.nodes.size(); ++rank) {
        const std::string label =
            "external sweep ooverlap rank" + std::to_string(rank);
        launch_ooverlap_once_for_rank(
            collective,
            ctx.nodes[rank],
            buffers.ooverlap_buffers[rank],
            numel,
            ctx.streams[rank],
            label.c_str());
    }
}

void launch_ooverlap_prebound_once_for_rank(
    TestCollective collective,
    oo_node_t* node,
    oo_buffer_t* const* rank_buffers,
    int rank_buffer_count,
    std::size_t numel,
    cudaStream_t stream,
    const char* label) {
    if (collective == TestCollective::AllReduce) {
        testing::check_oo(
            comm::api::allreduce_prebound_tuned(
                node,
                rank_buffers,
                rank_buffer_count,
                numel,
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                OO_TUNING_BEST_PERFORMANCE,
                stream),
            label);
        return;
    }

    if (collective == TestCollective::ReduceScatter) {
        oo_tensor_slice_t slice{};
        testing::check_oo(
            comm::api::reduce_scatter_prebound_tuned(
                node,
                rank_buffers,
                rank_buffer_count,
                numel,
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                OO_TUNING_BEST_PERFORMANCE,
                &slice,
                stream),
            label);
        return;
    }

    if (collective == TestCollective::AllGather) {
        testing::check_oo(
            comm::api::all_gather_prebound_tuned(
                node,
                rank_buffers,
                rank_buffer_count,
                numel,
                OO_DTYPE_FLOAT16,
                OO_TUNING_BEST_PERFORMANCE,
                stream),
            label);
        return;
    }

    throw std::invalid_argument("unsupported collective");
}

void launch_ooverlap_prebound_once(
    TestCollective collective,
    const SweepContext& ctx,
    const SizeBuffers& buffers,
    std::size_t numel,
    ParallelRankSubmission& rank_submission) {
    if (ctx.nodes.size() != buffers.ooverlap_buffers.size() ||
        ctx.nodes.size() != ctx.streams.size()) {
        throw std::invalid_argument(
            "launch_ooverlap_prebound_once: size mismatch");
    }

    oo_buffer_t* const* rank_buffers =
        buffers.ooverlap_buffers.data();
    const int rank_buffer_count =
        static_cast<int>(buffers.ooverlap_buffers.size());

    rank_submission.run([&](std::size_t rank) {
        const std::string label =
            "external ring ooverlap rank" + std::to_string(rank);
        launch_ooverlap_prebound_once_for_rank(
            collective,
            ctx.nodes[rank],
            rank_buffers,
            rank_buffer_count,
            numel,
            ctx.streams[rank],
            label.c_str());
    });
}

void launch_nccl_once(
    TestCollective collective,
    const SweepContext& ctx,
    const std::vector<half*>& work,
    std::size_t numel) {
    const std::size_t world_size = ctx.devices.size();
    if (ctx.nccl_comms.size() != world_size ||
        ctx.streams.size() != world_size ||
        work.size() != world_size) {
        throw std::invalid_argument("launch_nccl_once: size mismatch");
    }

    OOVERLAP_TEST_NCCL_CHECK(ncclGroupStart());
    for (std::size_t rank = 0; rank < world_size; ++rank) {
        testing::launch_nccl_collective_fp16(
            collective,
            ctx.nccl_comms[rank],
            work[rank],
            numel,
            static_cast<int>(rank),
            static_cast<int>(world_size),
            ctx.streams[rank]);
    }
    OOVERLAP_TEST_NCCL_CHECK(ncclGroupEnd());
}

void prepare_work_buffers(
    const SweepContext& ctx,
    const std::vector<half*>& sources,
    const std::vector<half*>& work,
    std::size_t bytes,
    const char* label) {
    const std::size_t world_size = ctx.devices.size();
    if (sources.size() != world_size ||
        work.size() != world_size ||
        ctx.streams.size() != world_size) {
        throw std::invalid_argument("prepare_work_buffers: size mismatch");
    }

    for (std::size_t rank = 0; rank < world_size; ++rank) {
        testing::reset_work_buffer_async(
            work[rank],
            sources[rank],
            bytes,
            ctx.devices[rank],
            ctx.streams[rank]);
    }

    sync_streams(ctx.devices, ctx.streams, label);
}

template <typename PrepareOnce, typename LaunchOnce>
void run_warmup(
    int warmup,
    const SweepContext& ctx,
    PrepareOnce&& prepare_once,
    LaunchOnce&& launch_once,
    const char* sync_label) {
    for (int i = 0; i < warmup; ++i) {
        prepare_once();
        launch_once();
        sync_streams(ctx.devices, ctx.streams, sync_label);
    }
}

template <typename PrepareOnce, typename LaunchOnce>
double benchmark_total_ms(
    int iters,
    const SweepContext& ctx,
    TimingEvents& events,
    PrepareOnce&& prepare_once,
    LaunchOnce&& launch_once) {
    double total_ms = 0.0;
    for (int i = 0; i < iters; ++i) {
        prepare_once();
        total_ms += elapsed_collective_once_ms(
            ctx.devices,
            ctx.streams,
            events,
            launch_once);
    }
    return total_ms;
}

bool verification_enabled(bool verify_runtime) {
#if OOVERLAP_BENCH_VERIFY_RESULTS
    return true;
#else
    return verify_runtime;
#endif
}

void verify_result(
    TestCollective collective,
    const char* label,
    const SweepContext& ctx,
    const std::vector<half*>& work,
    int64_t numel,
    bool verify_runtime) {
    if (!verification_enabled(verify_runtime)) {
        return;
    }

    const int world_size = static_cast<int>(ctx.devices.size());
    for (int rank = 0; rank < world_size; ++rank) {
        testing::verify_collective_fp16(
            collective,
            label,
            work[static_cast<std::size_t>(rank)],
            numel,
            rank,
            world_size,
            ctx.devices[static_cast<std::size_t>(rank)]);
    }
}

void destroy_size_buffers_best_effort(
    const SweepContext& ctx,
    SizeBuffers& buffers) {
    const std::size_t world_size = ctx.devices.size();

    for (oo_buffer_t*& buffer : buffers.ooverlap_buffers) {
        testing::destroy_oo_buffer(buffer);
    }

    const std::size_t window_count =
        std::min(ctx.nccl_comms.size(), buffers.nccl_symmetric_windows.size());
    for (std::size_t rank = 0; rank < window_count; ++rank) {
        deregister_nccl_window_best_effort(
            ctx.nccl_comms[rank],
            buffers.nccl_symmetric_windows[rank]);
    }

    for (std::size_t rank = 0; rank < world_size; ++rank) {
        if (rank < buffers.sources.size()) {
            testing::cuda_free_on_device(
                ctx.devices[rank],
                buffers.sources[rank]);
        }
        if (rank < buffers.ooverlap_work.size()) {
            testing::cuda_free_on_device(
                ctx.devices[rank],
                buffers.ooverlap_work[rank]);
        }
        if (rank < buffers.nccl_work.size()) {
            testing::cuda_free_on_device(
                ctx.devices[rank],
                buffers.nccl_work[rank]);
        }
        if (rank < buffers.nccl_symmetric_work.size()) {
            nccl_mem_free_on_device(
                ctx.devices[rank],
                buffers.nccl_symmetric_work[rank]);
        }
    }

    buffers = SizeBuffers{};
}

SizeBuffers allocate_size_buffers(
    const SweepContext& ctx,
    int64_t numel,
    std::size_t bytes) {
    const std::size_t world_size = ctx.devices.size();
    SizeBuffers buffers;
    buffers.sources.assign(world_size, nullptr);
    buffers.ooverlap_work.assign(world_size, nullptr);
    buffers.ooverlap_buffers.assign(world_size, nullptr);
    buffers.nccl_work.assign(world_size, nullptr);
    buffers.nccl_symmetric_work.assign(world_size, nullptr);
    buffers.nccl_symmetric_windows.assign(world_size, nullptr);

    try {
        for (std::size_t rank = 0; rank < world_size; ++rank) {
            const int device = ctx.devices[rank];

            const std::string source_label =
                "cudaMalloc(external sweep source rank" +
                std::to_string(rank) + ")";
            testing::cuda_malloc_half_on_device(
                device,
                &buffers.sources[rank],
                bytes,
                source_label.c_str());

            const std::string oo_label =
                "cudaMalloc(external sweep ooverlap rank" +
                std::to_string(rank) + ")";
            testing::cuda_malloc_half_on_device(
                device,
                &buffers.ooverlap_work[rank],
                bytes,
                oo_label.c_str());

            const std::string wrap_label =
                "oo_buffer_wrap(external sweep rank" +
                std::to_string(rank) + ")";
            testing::check_oo(
                oo_buffer_wrap(
                    ctx.nodes[rank],
                    buffers.ooverlap_work[rank],
                    bytes,
                    &buffers.ooverlap_buffers[rank]),
                wrap_label.c_str());

            const std::string nccl_label =
                "cudaMalloc(external sweep nccl rank" +
                std::to_string(rank) + ")";
            testing::cuda_malloc_half_on_device(
                device,
                &buffers.nccl_work[rank],
                bytes,
                nccl_label.c_str());

            const std::string symmetric_label =
                "ncclMemAlloc(external sweep rank" +
                std::to_string(rank) + ")";
            nccl_mem_alloc_half_on_device(
                device,
                &buffers.nccl_symmetric_work[rank],
                bytes,
                symmetric_label.c_str());

            testing::fill_rank_source_fp16(
                buffers.sources[rank],
                numel,
                static_cast<int>(rank),
                device,
                ctx.streams[rank]);
        }

        sync_streams(
            ctx.devices,
            ctx.streams,
            "sync external sweep source initialization");

        register_nccl_symmetric_windows(
            ctx.nccl_comms,
            buffers.nccl_symmetric_work,
            bytes,
            buffers.nccl_symmetric_windows);

        return buffers;
    } catch (...) {
        destroy_size_buffers_best_effort(ctx, buffers);
        throw;
    }
}

void destroy_size_buffer_ring_best_effort(
    const SweepContext& ctx,
    std::vector<SizeBuffers>& ring) {
    for (SizeBuffers& slot : ring) {
        destroy_size_buffers_best_effort(ctx, slot);
    }
    ring.clear();
}

std::vector<SizeBuffers> allocate_size_buffer_ring(
    const SweepContext& ctx,
    int64_t numel,
    std::size_t bytes,
    int ring_size) {
    if (ring_size <= 0) {
        throw std::invalid_argument("ring_size must be > 0");
    }

    std::vector<SizeBuffers> ring;
    ring.reserve(static_cast<std::size_t>(ring_size));

    try {
        for (int slot = 0; slot < ring_size; ++slot) {
            ring.push_back(allocate_size_buffers(ctx, numel, bytes));
        }
        return ring;
    } catch (...) {
        destroy_size_buffer_ring_best_effort(ctx, ring);
        throw;
    }
}

void destroy_context_best_effort(SweepContext& ctx) {
    if (!ctx.nccl_comms.empty()) {
        for (ncclComm_t& comm : ctx.nccl_comms) {
            if (comm != nullptr) {
                (void)ncclCommDestroy(comm);
                comm = nullptr;
            }
        }
    }

    for (oo_node_t*& node : ctx.nodes) {
        testing::destroy_oo_node(node);
    }
    testing::destroy_oo_group(ctx.group);

    const std::size_t stream_count =
        std::min(ctx.devices.size(), ctx.streams.size());
    for (std::size_t rank = 0; rank < stream_count; ++rank) {
        if (ctx.streams[rank] != nullptr) {
            system::runtime::destroy_stream_on_device(
                ctx.devices[rank],
                ctx.streams[rank]);
            ctx.streams[rank] = nullptr;
        }
    }

    ctx = SweepContext{};
}

SweepContext create_context(const std::vector<int>& devices) {
    SweepContext ctx;
    ctx.devices = devices;
    ctx.nodes.assign(devices.size(), nullptr);
    ctx.streams.assign(devices.size(), nullptr);
    ctx.nccl_comms.assign(devices.size(), nullptr);

    try {
        testing::check_oo(
            oo_group_create_p2p(
                ctx.devices.data(),
                static_cast<int>(ctx.devices.size()),
                &ctx.group),
            "oo_group_create_p2p(external sweep)");

        for (std::size_t rank = 0; rank < ctx.devices.size(); ++rank) {
            const std::string node_label =
                "oo_node_create(external sweep rank" +
                std::to_string(rank) + ")";
            testing::check_oo(
                oo_node_create(
                    ctx.group,
                    static_cast<int>(rank),
                    &ctx.nodes[rank]),
                node_label.c_str());

            const int node_device = oo_node_device(ctx.nodes[rank]);
            if (node_device != ctx.devices[rank]) {
                throw std::runtime_error(
                    "oo_node_device did not match requested device");
            }

            ctx.streams[rank] =
                system::runtime::create_stream_on_device(node_device);
        }

        OOVERLAP_TEST_NCCL_CHECK(
            ncclCommInitAll(
                ctx.nccl_comms.data(),
                static_cast<int>(ctx.devices.size()),
                ctx.devices.data()));

        return ctx;
    } catch (...) {
        destroy_context_best_effort(ctx);
        throw;
    }
}

std::size_t bandwidth_bytes_per_rank(
    TestCollective /*collective*/,
    std::size_t numel,
    int /*world_size*/) {
    // Match nccl-tests AlgoBW: logical full-message bytes divided by time.
    // AllReduce uses count * typesize. ReduceScatter and AllGather use
    // local_count * typesize * nranks, which is the same full message size.
    // Keep this helper name for output compatibility with existing results.
    return numel * sizeof(half);
}

std::map<std::string, double> run_one_size_ring(
    TestCollective collective,
    int64_t numel_arg,
    int iters,
    int warmup,
    bool verify,
    const SweepContext& ctx,
    TimingEvents& events) {
    const std::size_t numel = static_cast<std::size_t>(numel_arg);
    const std::size_t bytes = numel * sizeof(half);
    const int world_size = static_cast<int>(ctx.devices.size());
    const int ring_size = ring_size_from_env();

    std::vector<SizeBuffers> ring =
        allocate_size_buffer_ring(ctx, numel_arg, bytes, ring_size);

    try {
        ParallelRankSubmission rank_submission(ctx.devices.size());
        rank_submission.run([&](std::size_t rank) {
            system::runtime::set_device(ctx.devices[rank]);
        });

        const auto prepare_ooverlap_ring = [&]() {
            for (const SizeBuffers& slot : ring) {
                prepare_work_buffers(
                    ctx,
                    slot.sources,
                    slot.ooverlap_work,
                    bytes,
                    "sync external ring ooverlap reset");
            }
        };
        const auto launch_ooverlap_iters = [&](int count) {
            for (int i = 0; i < count; ++i) {
                const SizeBuffers& slot =
                    ring[static_cast<std::size_t>(i) % ring.size()];
                launch_ooverlap_prebound_once(
                    collective,
                    ctx,
                    slot,
                    numel,
                    rank_submission);
            }
        };

        prepare_ooverlap_ring();
        testing::reset_ready_signals(ctx.group);
        launch_ooverlap_iters(warmup);
        sync_streams(
            ctx.devices,
            ctx.streams,
            "sync external ring ooverlap warmup");

        prepare_ooverlap_ring();
        testing::reset_ready_signals(ctx.group);
        const double oo_total_ms = elapsed_collective_once_ms(
            ctx.devices,
            ctx.streams,
            events,
            [&]() { launch_ooverlap_iters(iters); });

        if (verification_enabled(verify)) {
            SizeBuffers& slot = ring.front();
            prepare_work_buffers(
                ctx,
                slot.sources,
                slot.ooverlap_work,
                bytes,
                "sync external ring ooverlap verification reset");
            testing::reset_ready_signals(ctx.group);
            launch_ooverlap_prebound_once(
                collective,
                ctx,
                slot,
                numel,
                rank_submission);
            sync_streams(
                ctx.devices,
                ctx.streams,
                "sync external ring ooverlap verification");
            verify_result(
                collective,
                "ooverlap external ring",
                ctx,
                slot.ooverlap_work,
                numel_arg,
                verify);
        }

        rank_submission.stop();

        const auto prepare_nccl_ring = [&]() {
            for (const SizeBuffers& slot : ring) {
                prepare_work_buffers(
                    ctx,
                    slot.sources,
                    slot.nccl_work,
                    bytes,
                    "sync external ring nccl reset");
            }
        };
        const auto launch_nccl_iters = [&](int count) {
            for (int i = 0; i < count; ++i) {
                const SizeBuffers& slot =
                    ring[static_cast<std::size_t>(i) % ring.size()];
                launch_nccl_once(
                    collective,
                    ctx,
                    slot.nccl_work,
                    numel);
            }
        };

        prepare_nccl_ring();
        launch_nccl_iters(warmup);
        sync_streams(
            ctx.devices,
            ctx.streams,
            "sync external ring nccl warmup");

        prepare_nccl_ring();
        const double nccl_total_ms = elapsed_collective_once_ms(
            ctx.devices,
            ctx.streams,
            events,
            [&]() { launch_nccl_iters(iters); });

        if (verification_enabled(verify)) {
            SizeBuffers& slot = ring.front();
            prepare_work_buffers(
                ctx,
                slot.sources,
                slot.nccl_work,
                bytes,
                "sync external ring nccl verification reset");
            launch_nccl_once(collective, ctx, slot.nccl_work, numel);
            sync_streams(
                ctx.devices,
                ctx.streams,
                "sync external ring nccl verification");
            verify_result(
                collective,
                "NCCL external ring",
                ctx,
                slot.nccl_work,
                numel_arg,
                verify);
        }

        const auto prepare_nccl_symmetric_ring = [&]() {
            for (const SizeBuffers& slot : ring) {
                prepare_work_buffers(
                    ctx,
                    slot.sources,
                    slot.nccl_symmetric_work,
                    bytes,
                    "sync external ring symmetric NCCL reset");
            }
        };
        const auto launch_nccl_symmetric_iters = [&](int count) {
            for (int i = 0; i < count; ++i) {
                const SizeBuffers& slot =
                    ring[static_cast<std::size_t>(i) % ring.size()];
                launch_nccl_once(
                    collective,
                    ctx,
                    slot.nccl_symmetric_work,
                    numel);
            }
        };

        prepare_nccl_symmetric_ring();
        launch_nccl_symmetric_iters(warmup);
        sync_streams(
            ctx.devices,
            ctx.streams,
            "sync external ring symmetric NCCL warmup");

        prepare_nccl_symmetric_ring();
        const double nccl_symmetric_total_ms = elapsed_collective_once_ms(
            ctx.devices,
            ctx.streams,
            events,
            [&]() { launch_nccl_symmetric_iters(iters); });

        if (verification_enabled(verify)) {
            SizeBuffers& slot = ring.front();
            prepare_work_buffers(
                ctx,
                slot.sources,
                slot.nccl_symmetric_work,
                bytes,
                "sync external ring symmetric NCCL verification reset");
            launch_nccl_once(
                collective,
                ctx,
                slot.nccl_symmetric_work,
                numel);
            sync_streams(
                ctx.devices,
                ctx.streams,
                "sync external ring symmetric NCCL verification");
            verify_result(
                collective,
                "symmetric NCCL external ring",
                ctx,
                slot.nccl_symmetric_work,
                numel_arg,
                verify);
        }

        std::size_t max_local_shard_numel = 0;
        for (int rank = 0; rank < world_size; ++rank) {
            max_local_shard_numel = std::max(
                max_local_shard_numel,
                testing::rank_partition_count(
                    numel,
                    rank,
                    world_size));
        }

        const std::size_t normalized_bytes =
            bandwidth_bytes_per_rank(
                collective,
                numel,
                world_size);

        std::map<std::string, double> row = {
            {"collective", testing::collective_code(collective)},
            {"world_size", static_cast<double>(world_size)},
            {"numel", static_cast<double>(numel)},
            {"bytes", static_cast<double>(bytes)},
            {"local_shard_numel", static_cast<double>(max_local_shard_numel)},
            {"local_shard_bytes",
             static_cast<double>(max_local_shard_numel * sizeof(half))},
            {"bandwidth_bytes_per_rank", static_cast<double>(normalized_bytes)},
            {"iters", static_cast<double>(iters)},
            {"warmup", static_cast<double>(warmup)},
            {"verify", verify ? 1.0 : 0.0},
            {"ring_size", static_cast<double>(ring_size)},
            {"oo_total_ms", oo_total_ms},
            {"nccl_total_ms", nccl_total_ms},
            {"nccl_symmetric_total_ms", nccl_symmetric_total_ms},
            {"ooverlap_ms", oo_total_ms / static_cast<double>(iters)},
            {"nccl_ms", nccl_total_ms / static_cast<double>(iters)},
            {"nccl_symmetric_ms",
             nccl_symmetric_total_ms / static_cast<double>(iters)},
        };

        destroy_size_buffer_ring_best_effort(ctx, ring);
        return row;
    } catch (...) {
        destroy_size_buffer_ring_best_effort(ctx, ring);
        throw;
    }
}

std::map<std::string, double> run_one_size(
    TestCollective collective,
    int64_t numel_arg,
    int iters,
    int warmup,
    bool verify,
    const SweepContext& ctx,
    TimingEvents& events) {
    const std::size_t numel = static_cast<std::size_t>(numel_arg);
    const std::size_t bytes = numel * sizeof(half);
    const int world_size = static_cast<int>(ctx.devices.size());

    SizeBuffers buffers = allocate_size_buffers(ctx, numel_arg, bytes);

    try {
        const auto prepare_ooverlap = [&]() {
            prepare_work_buffers(
                ctx,
                buffers.sources,
                buffers.ooverlap_work,
                bytes,
                "sync external sweep ooverlap reset");
        };
        const auto launch_ooverlap = [&]() {
            launch_ooverlap_once(
                collective,
                ctx,
                buffers,
                numel);
        };

        testing::reset_ready_signals(ctx.group);
        run_warmup(
            warmup,
            ctx,
            prepare_ooverlap,
            launch_ooverlap,
            "sync external sweep ooverlap warmup");

        testing::reset_ready_signals(ctx.group);
        const double oo_total_ms =
            benchmark_total_ms(
                iters,
                ctx,
                events,
                prepare_ooverlap,
                launch_ooverlap);

        if (verification_enabled(verify)) {
            prepare_ooverlap();
            testing::reset_ready_signals(ctx.group);
            launch_ooverlap();
            sync_streams(
                ctx.devices,
                ctx.streams,
                "sync external sweep ooverlap verification");
            verify_result(
                collective,
                "ooverlap external sweep",
                ctx,
                buffers.ooverlap_work,
                numel_arg,
                verify);
        }

        const auto prepare_nccl = [&]() {
            prepare_work_buffers(
                ctx,
                buffers.sources,
                buffers.nccl_work,
                bytes,
                "sync external sweep nccl reset");
        };
        const auto launch_nccl = [&]() {
            launch_nccl_once(
                collective,
                ctx,
                buffers.nccl_work,
                numel);
        };

        run_warmup(
            warmup,
            ctx,
            prepare_nccl,
            launch_nccl,
            "sync external sweep nccl warmup");
        const double nccl_total_ms =
            benchmark_total_ms(
                iters,
                ctx,
                events,
                prepare_nccl,
                launch_nccl);

        if (verification_enabled(verify)) {
            prepare_nccl();
            launch_nccl();
            sync_streams(
                ctx.devices,
                ctx.streams,
                "sync external sweep nccl verification");
            verify_result(
                collective,
                "NCCL external sweep",
                ctx,
                buffers.nccl_work,
                numel_arg,
                verify);
        }

        const auto prepare_nccl_symmetric = [&]() {
            prepare_work_buffers(
                ctx,
                buffers.sources,
                buffers.nccl_symmetric_work,
                bytes,
                "sync external sweep symmetric NCCL reset");
        };
        const auto launch_nccl_symmetric = [&]() {
            launch_nccl_once(
                collective,
                ctx,
                buffers.nccl_symmetric_work,
                numel);
        };

        run_warmup(
            warmup,
            ctx,
            prepare_nccl_symmetric,
            launch_nccl_symmetric,
            "sync external sweep symmetric NCCL warmup");
        const double nccl_symmetric_total_ms =
            benchmark_total_ms(
                iters,
                ctx,
                events,
                prepare_nccl_symmetric,
                launch_nccl_symmetric);

        if (verification_enabled(verify)) {
            prepare_nccl_symmetric();
            launch_nccl_symmetric();
            sync_streams(
                ctx.devices,
                ctx.streams,
                "sync external sweep symmetric NCCL verification");
            verify_result(
                collective,
                "symmetric NCCL external sweep",
                ctx,
                buffers.nccl_symmetric_work,
                numel_arg,
                verify);
        }

        std::size_t max_local_shard_numel = 0;
        for (int rank = 0; rank < world_size; ++rank) {
            max_local_shard_numel = std::max(
                max_local_shard_numel,
                testing::rank_partition_count(
                    numel,
                    rank,
                    world_size));
        }

        const std::size_t normalized_bytes =
            bandwidth_bytes_per_rank(
                collective,
                numel,
                world_size);

        std::map<std::string, double> row = {
            {"collective", testing::collective_code(collective)},
            {"world_size", static_cast<double>(world_size)},
            {"numel", static_cast<double>(numel)},
            {"bytes", static_cast<double>(bytes)},
            {"local_shard_numel", static_cast<double>(max_local_shard_numel)},
            {"local_shard_bytes",
             static_cast<double>(max_local_shard_numel * sizeof(half))},
            {"bandwidth_bytes_per_rank", static_cast<double>(normalized_bytes)},
            {"iters", static_cast<double>(iters)},
            {"warmup", static_cast<double>(warmup)},
            {"verify", verify ? 1.0 : 0.0},
            {"ring_size", 0.0},
            {"oo_total_ms", oo_total_ms},
            {"nccl_total_ms", nccl_total_ms},
            {"nccl_symmetric_total_ms", nccl_symmetric_total_ms},
            {"ooverlap_ms", oo_total_ms / static_cast<double>(iters)},
            {"nccl_ms", nccl_total_ms / static_cast<double>(iters)},
            {"nccl_symmetric_ms",
             nccl_symmetric_total_ms / static_cast<double>(iters)},
        };

        destroy_size_buffers_best_effort(ctx, buffers);
        return row;
    } catch (...) {
        destroy_size_buffers_best_effort(ctx, buffers);
        throw;
    }
}

} // namespace

std::vector<std::map<std::string, double>>
benchmark_external_p2p_collective_sweep_sm90(
    const std::string& collective_name_arg,
    const std::vector<int64_t>& sizes,
    int iters,
    int warmup,
    const std::vector<int>& devices,
    bool verify,
    bool use_ring) {
    if (iters <= 0) {
        throw std::invalid_argument("iters must be > 0");
    }
    if (warmup < 0) {
        throw std::invalid_argument("warmup must be >= 0");
    }

    validate_devices(devices);

    const TestCollective collective =
        testing::parse_collective(collective_name_arg);
    validate_sizes(
        collective,
        sizes,
        static_cast<int>(devices.size()));

    SweepContext ctx = create_context(devices);
    TimingEvents events;

    try {
        events = create_timing_events(devices);

        std::vector<std::map<std::string, double>> rows;
        rows.reserve(sizes.size());

        for (int64_t numel : sizes) {
            rows.push_back(
                use_ring
                    ? run_one_size_ring(
                          collective,
                          numel,
                          iters,
                          warmup,
                          verify,
                          ctx,
                          events)
                    : run_one_size(
                          collective,
                          numel,
                          iters,
                          warmup,
                          verify,
                          ctx,
                          events));
        }

        destroy_timing_events_best_effort(devices, events);
        destroy_context_best_effort(ctx);
        return rows;
    } catch (...) {
        destroy_timing_events_best_effort(devices, events);
        destroy_context_best_effort(ctx);
        throw;
    }
}

} // namespace ooverlap
