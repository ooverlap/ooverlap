#include "test/tma_collective_sweep_2gpu.h"

#include "comm/launch_config.h"
#include "comm/ooverlap_comm_internal.h"
#include "comm/ooverlap_comm_private.h"
#include "comm/plan/transfer_plan_distribution.h"
#include "comm/tma_multi_gpu_all_gather_sm90.h"
#include "comm/tma_multi_gpu_allreduce_sm90.h"
#include "comm/tma_multi_gpu_reduce_scatter_sm90.h"

#include "ooverlap/comm.h"
#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/testing/checks.cuh"
#include "ooverlap/testing/collective_test_utils.cuh"
#include "ooverlap/testing/two_gpu_test_utils.cuh"

#include "test/internal_comm_test_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace ooverlap {
namespace {

using testing::TestCollective;

constexpr const char* kMaxCtasEnv = "OOVERLAP_MAX_CTAS";
constexpr const char* kMaxCtasPerReduceTaskEnv =
    "OOVERLAP_MAX_CTAS_PER_REDUCE_TASK";

struct SweepContext {
    std::vector<int> devices;
    oo_group_t* group = nullptr;
    std::vector<oo_node_t*> nodes;
    std::vector<cudaStream_t> streams;
};

struct SizeBuffers {
    std::vector<half*> sources;
    std::vector<half*> work;
    std::vector<oo_buffer_t*> buffers;
};

struct TimingEvents {
    std::vector<cudaEvent_t> starts;
    std::vector<cudaEvent_t> stops;
};

int required_positive_env(const char* name) {
    if (name == nullptr || name[0] == '\0') {
        throw std::invalid_argument("environment variable name is empty");
    }

    const char* text = std::getenv(name);
    if (text == nullptr || text[0] == '\0') {
        throw std::invalid_argument(
            std::string(name) + " must be set for the CTA tuning sweep");
    }

    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);

    if (errno != 0 ||
        end == text ||
        *end != '\0' ||
        parsed <= 0 ||
        parsed > INT_MAX) {
        throw std::invalid_argument(
            std::string(name) + " must be a positive integer");
    }

    return static_cast<int>(parsed);
}

comm::CollectivePlanFor collective_plan_for(TestCollective collective) {
    switch (collective) {
        case TestCollective::AllReduce:
            return comm::CollectivePlanFor::AllReduce;
        case TestCollective::ReduceScatter:
            return comm::CollectivePlanFor::ReduceScatter;
        case TestCollective::AllGather:
            return comm::CollectivePlanFor::AllGather;
        default:
            throw std::invalid_argument("unsupported collective");
    }
}

comm::LaunchConfig explicit_tma_launch_config(TestCollective collective) {
    comm::LaunchConfig config{};

    switch (collective) {
        case TestCollective::AllReduce:
            config = comm::make_allreduce_launch_config(
                comm::AllReducePlanKind::TmaCopy);
            break;
        case TestCollective::ReduceScatter:
            config = comm::make_reduce_scatter_launch_config(
                comm::ReduceScatterPlanKind::TmaReduce);
            break;
        case TestCollective::AllGather:
            config = comm::make_all_gather_launch_config(
                comm::AllGatherPlanKind::TmaCopy);
            break;
        default:
            throw std::invalid_argument("unsupported collective");
    }

    config.max_ctas = required_positive_env(kMaxCtasEnv);
    config.max_ctas_per_reduce_task =
        required_positive_env(kMaxCtasPerReduceTaskEnv);

    if (config.max_ctas_per_reduce_task > config.max_ctas) {
        throw std::invalid_argument(
            "OOVERLAP_MAX_CTAS_PER_REDUCE_TASK must not exceed "
            "OOVERLAP_MAX_CTAS");
    }

    if (!comm::launch_config_valid(config)) {
        throw std::invalid_argument(
            "CTA environment values produce an invalid LaunchConfig");
    }

    return config;
}

void validate_devices(const std::vector<int>& devices) {
    if (devices.size() < 2) {
        throw std::invalid_argument(
            "CTA tuning sweep requires at least two CUDA devices");
    }

    if (devices.size() > static_cast<std::size_t>(kOoMaxLocalDevices)) {
        throw std::invalid_argument(
            "CTA tuning sweep exceeds Ooverlap's local-rank limit");
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

void validate_numels(
    TestCollective collective,
    const std::vector<int64_t>& numels,
    int world_size) {
    if (numels.empty()) {
        throw std::invalid_argument("numels must be non-empty");
    }

    for (int64_t numel : numels) {
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

void destroy_context_best_effort(SweepContext& context) {
    for (oo_node_t*& node : context.nodes) {
        testing::destroy_oo_node(node);
    }

    testing::destroy_oo_group(context.group);

    const std::size_t stream_count =
        std::min(context.devices.size(), context.streams.size());
    for (std::size_t rank = 0; rank < stream_count; ++rank) {
        if (context.streams[rank] != nullptr) {
            system::runtime::destroy_stream_on_device(
                context.devices[rank],
                context.streams[rank]);
            context.streams[rank] = nullptr;
        }
    }

    context = SweepContext{};
}

SweepContext create_context(const std::vector<int>& devices) {
    SweepContext context;
    context.devices = devices;
    context.nodes.assign(devices.size(), nullptr);
    context.streams.assign(devices.size(), nullptr);

    try {
        testing::check_oo(
            oo_group_create_p2p(
                context.devices.data(),
                static_cast<int>(context.devices.size()),
                &context.group),
            "oo_group_create_p2p(CTA sweep)");

        for (std::size_t rank = 0; rank < devices.size(); ++rank) {
            testing::check_oo(
                oo_node_create(
                    context.group,
                    static_cast<int>(rank),
                    &context.nodes[rank]),
                "oo_node_create(CTA sweep)");

            const int node_device = oo_node_device(context.nodes[rank]);
            if (node_device != devices[rank]) {
                throw std::runtime_error(
                    "oo_node_device did not match requested device");
            }

            context.streams[rank] =
                system::runtime::create_stream_on_device(node_device);
        }

        return context;
    } catch (...) {
        destroy_context_best_effort(context);
        throw;
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
                "cudaEventCreate(CTA sweep start)");
            testing::check_cuda(
                cudaEventCreate(&events.stops[rank]),
                "cudaEventCreate(CTA sweep stop)");
        }

        return events;
    } catch (...) {
        destroy_timing_events_best_effort(devices, events);
        throw;
    }
}

void destroy_size_buffers_best_effort(
    const SweepContext& context,
    SizeBuffers& buffers) {
    for (oo_buffer_t*& buffer : buffers.buffers) {
        testing::destroy_oo_buffer(buffer);
    }

    const std::size_t source_count =
        std::min(context.devices.size(), buffers.sources.size());
    for (std::size_t rank = 0; rank < source_count; ++rank) {
        testing::cuda_free_on_device(
            context.devices[rank],
            buffers.sources[rank]);
    }

    buffers = SizeBuffers{};
}

SizeBuffers allocate_size_buffers(
    const SweepContext& context,
    int64_t numel,
    std::size_t bytes) {
    const std::size_t world_size = context.devices.size();

    SizeBuffers buffers;
    buffers.sources.assign(world_size, nullptr);
    buffers.work.assign(world_size, nullptr);
    buffers.buffers.assign(world_size, nullptr);

    try {
        for (std::size_t rank = 0; rank < world_size; ++rank) {
            testing::cuda_malloc_half_on_device(
                context.devices[rank],
                &buffers.sources[rank],
                bytes,
                "cudaMalloc(CTA sweep source)");

            testing::check_oo(
                oo_buffer_alloc(
                    context.nodes[rank],
                    bytes,
                    &buffers.buffers[rank]),
                "oo_buffer_alloc(CTA sweep work)");

            buffers.work[rank] = reinterpret_cast<half*>(
                oo_buffer_ptr(buffers.buffers[rank]));

            if (buffers.work[rank] == nullptr) {
                throw std::runtime_error(
                    "oo_buffer_ptr returned nullptr in CTA sweep");
            }

            testing::fill_rank_source_fp16(
                buffers.sources[rank],
                numel,
                static_cast<int>(rank),
                context.devices[rank],
                context.streams[rank]);
        }

        return buffers;
    } catch (...) {
        destroy_size_buffers_best_effort(context, buffers);
        throw;
    }
}

void prepare_work_buffers(
    const SweepContext& context,
    const SizeBuffers& buffers,
    std::size_t bytes) {
    const std::size_t world_size = context.devices.size();

    if (buffers.sources.size() != world_size ||
        buffers.work.size() != world_size ||
        context.streams.size() != world_size) {
        throw std::invalid_argument(
            "prepare_work_buffers: size mismatch");
    }

    for (std::size_t rank = 0; rank < world_size; ++rank) {
        testing::reset_work_buffer_async(
            buffers.work[rank],
            buffers.sources[rank],
            bytes,
            context.devices[rank],
            context.streams[rank]);
    }

    sync_streams(
        context.devices,
        context.streams,
        "sync CTA sweep work reset");
}

void launch_rank_once(
    TestCollective collective,
    const SweepContext& context,
    const SizeBuffers& buffers,
    std::size_t numel,
    std::size_t rank,
    const comm::LaunchConfig& config) {
    oo_node_t* node = context.nodes[rank];
    oo_buffer_t* buffer = buffers.buffers[rank];
    cudaStream_t stream = context.streams[rank];

    comm::api::CollectiveLaunchState launch{};
    testing::check_oo(
        comm::api::prepare_collective_launch(
            node,
            buffer,
            collective_plan_for(collective),
            0,
            numel,
            OO_DTYPE_FLOAT16,
            &launch),
        "prepare_collective_launch(CTA sweep)");

    if (collective == TestCollective::AllReduce) {
        comm::plan::AllreduceTransferPlan* transfer_plan = nullptr;
        testing::check_oo(
            context.group->transfer_plan_distribution->
                get_allreduce_transfer_plan(
                    node,
                    launch,
                    numel,
                    OO_DTYPE_FLOAT16,
                    OO_REDUCE_SUM,
                    config,
                    &transfer_plan),
            "get_allreduce_transfer_plan(CTA sweep)");

        testing::check_cuda(
            enqueue_tma_multi_gpu_allreduce_rank_sm90(
                launch,
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                stream,
                config,
                *transfer_plan),
            "enqueue allreduce CTA sweep");
        return;
    }

    if (collective == TestCollective::ReduceScatter) {
        comm::plan::ReduceScatterTransferPlan* transfer_plan = nullptr;
        testing::check_oo(
            context.group->transfer_plan_distribution->
                get_reduce_scatter_transfer_plan(
                    node,
                    launch,
                    numel,
                    OO_DTYPE_FLOAT16,
                    OO_REDUCE_SUM,
                    config,
                    &transfer_plan),
            "get_reduce_scatter_transfer_plan(CTA sweep)");

        testing::check_cuda(
            enqueue_tma_multi_gpu_reduce_scatter_rank_sm90(
                launch,
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                stream,
                config,
                *transfer_plan),
            "enqueue reduce-scatter CTA sweep");
        return;
    }

    if (collective == TestCollective::AllGather) {
        comm::plan::AllGatherTransferPlan* transfer_plan = nullptr;
        testing::check_oo(
            context.group->transfer_plan_distribution->
                get_all_gather_transfer_plan(
                    node,
                    launch,
                    numel,
                    OO_DTYPE_FLOAT16,
                    config,
                    &transfer_plan),
            "get_all_gather_transfer_plan(CTA sweep)");

        testing::check_cuda(
            enqueue_tma_multi_gpu_all_gather_rank_sm90(
                launch,
                OO_DTYPE_FLOAT16,
                stream,
                config,
                *transfer_plan),
            "enqueue all-gather CTA sweep");
        return;
    }

    throw std::invalid_argument("unsupported collective");
}

void launch_collective_once(
    TestCollective collective,
    const SweepContext& context,
    const SizeBuffers& buffers,
    std::size_t numel,
    const comm::LaunchConfig& config) {
    for (std::size_t rank = 0; rank < context.devices.size(); ++rank) {
        launch_rank_once(
            collective,
            context,
            buffers,
            numel,
            rank,
            config);
    }
}

void run_warmup(
    TestCollective collective,
    const SweepContext& context,
    const SizeBuffers& buffers,
    std::size_t numel,
    int warmup,
    const comm::LaunchConfig& config) {
    testing::reset_ready_signals(context.group);

    for (int iteration = 0; iteration < warmup; ++iteration) {
        launch_collective_once(
            collective,
            context,
            buffers,
            numel,
            config);
    }

    sync_streams(
        context.devices,
        context.streams,
        "sync CTA sweep warmup");
}

double elapsed_collective_ms(
    TestCollective collective,
    const SweepContext& context,
    const SizeBuffers& buffers,
    std::size_t numel,
    int iters,
    const comm::LaunchConfig& config,
    TimingEvents& events) {
    const std::size_t world_size = context.devices.size();

    for (std::size_t rank = 0; rank < world_size; ++rank) {
        system::runtime::set_device(context.devices[rank]);
        testing::check_cuda(
            cudaEventRecord(events.starts[rank], context.streams[rank]),
            "cudaEventRecord(CTA sweep start)");
    }

    for (int iteration = 0; iteration < iters; ++iteration) {
        launch_collective_once(
            collective,
            context,
            buffers,
            numel,
            config);
    }

    for (std::size_t rank = 0; rank < world_size; ++rank) {
        system::runtime::set_device(context.devices[rank]);
        testing::check_cuda(
            cudaEventRecord(events.stops[rank], context.streams[rank]),
            "cudaEventRecord(CTA sweep stop)");
    }

    double max_ms = 0.0;
    for (std::size_t rank = 0; rank < world_size; ++rank) {
        system::runtime::set_device(context.devices[rank]);
        testing::check_cuda(
            cudaEventSynchronize(events.stops[rank]),
            "cudaEventSynchronize(CTA sweep stop)");

        float rank_ms = 0.0f;
        testing::check_cuda(
            cudaEventElapsedTime(
                &rank_ms,
                events.starts[rank],
                events.stops[rank]),
            "cudaEventElapsedTime(CTA sweep)");

        max_ms = std::max(max_ms, static_cast<double>(rank_ms));
    }

    return max_ms;
}

void verify_result(
    TestCollective collective,
    const SweepContext& context,
    const SizeBuffers& buffers,
    int64_t numel,
    const comm::LaunchConfig& config) {
    prepare_work_buffers(
        context,
        buffers,
        static_cast<std::size_t>(numel) * sizeof(half));

    testing::reset_ready_signals(context.group);
    launch_collective_once(
        collective,
        context,
        buffers,
        static_cast<std::size_t>(numel),
        config);

    sync_streams(
        context.devices,
        context.streams,
        "sync CTA sweep verification");

    for (std::size_t rank = 0; rank < context.devices.size(); ++rank) {
        testing::verify_collective_fp16(
            collective,
            "TMA CTA sweep",
            buffers.work[rank],
            numel,
            static_cast<int>(rank),
            static_cast<int>(context.devices.size()),
            context.devices[rank]);
    }
}

std::map<std::string, double> benchmark_one_size(
    TestCollective collective,
    const SweepContext& context,
    int64_t numel_arg,
    int iters,
    int warmup,
    bool verify,
    const comm::LaunchConfig& config,
    TimingEvents& events) {
    const std::size_t numel = static_cast<std::size_t>(numel_arg);
    const std::size_t bytes = numel * sizeof(half);

    SizeBuffers buffers = allocate_size_buffers(context, numel_arg, bytes);

    try {
        prepare_work_buffers(context, buffers, bytes);
        run_warmup(
            collective,
            context,
            buffers,
            numel,
            warmup,
            config);

        prepare_work_buffers(context, buffers, bytes);
        testing::reset_ready_signals(context.group);

        const double total_ms =
            elapsed_collective_ms(
                collective,
                context,
                buffers,
                numel,
                iters,
                config,
                events);

        if (verify) {
            verify_result(
                collective,
                context,
                buffers,
                numel_arg,
                config);
        }

        const double avg_ms =
            total_ms / static_cast<double>(iters);

        std::map<std::string, double> row = {
            {"collective", testing::collective_code(collective)},
            {"world_size", static_cast<double>(context.devices.size())},
            {"numel", static_cast<double>(numel)},
            {"bytes", static_cast<double>(bytes)},
            {"iters", static_cast<double>(iters)},
            {"warmup", static_cast<double>(warmup)},
            {"max_ctas", static_cast<double>(config.max_ctas)},
            {"max_ctas_per_reduce_task",
             static_cast<double>(config.max_ctas_per_reduce_task)},
            {"total_ms", total_ms},
            {"avg_ms", avg_ms},
            {"latency_us", avg_ms * 1000.0},
        };

        destroy_size_buffers_best_effort(context, buffers);
        return row;
    } catch (...) {
        destroy_size_buffers_best_effort(context, buffers);
        throw;
    }
}

} // namespace

std::vector<std::map<std::string, double>>
benchmark_tma_collective_cta_sweep_sm90(
    const std::string& collective_name,
    const std::vector<int64_t>& numels,
    int iters,
    int warmup,
    const std::vector<int>& devices,
    bool verify) {
    if (iters <= 0) {
        throw std::invalid_argument("iters must be > 0");
    }

    if (warmup < 0) {
        throw std::invalid_argument("warmup must be >= 0");
    }

    validate_devices(devices);

    const TestCollective collective =
        testing::parse_collective(collective_name);

    validate_numels(
        collective,
        numels,
        static_cast<int>(devices.size()));

    const comm::LaunchConfig config =
        explicit_tma_launch_config(collective);

    SweepContext context = create_context(devices);
    TimingEvents events;

    try {
        events = create_timing_events(devices);

        std::vector<std::map<std::string, double>> rows;
        rows.reserve(numels.size());

        for (int64_t numel : numels) {
            rows.push_back(
                benchmark_one_size(
                    collective,
                    context,
                    numel,
                    iters,
                    warmup,
                    verify,
                    config,
                    events));
        }

        destroy_timing_events_best_effort(devices, events);
        destroy_context_best_effort(context);
        return rows;
    } catch (...) {
        destroy_timing_events_best_effort(devices, events);
        destroy_context_best_effort(context);
        throw;
    }
}

} // namespace ooverlap
