#include "test/ipc_collective_sm90.h"

#include "comm/ooverlap_comm_internal.h"
#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/testing/test_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define OOVERLAP_IPC_COLLECTIVE_NCCL_CHECK(cmd)                              \
    do {                                                                      \
        ncclResult_t result__ = (cmd);                                        \
        if (result__ != ncclSuccess) {                                        \
            throw std::runtime_error(                                         \
                std::string("NCCL error at ") + __FILE__ + ":" +              \
                std::to_string(__LINE__) + " " +                              \
                ncclGetErrorString(result__));                                \
        }                                                                     \
    } while (0)

namespace ooverlap {
namespace {

enum class IpcCollective {
    AllReduce = 0,
    ReduceScatter = 1,
    AllGather = 2,
};

IpcCollective parse_collective(const std::string& value) {
    if (value == "allreduce" ||
        value == "all_reduce" ||
        value == "all-reduce" ||
        value == "ar") {
        return IpcCollective::AllReduce;
    }

    if (value == "reduce_scatter" ||
        value == "reduce-scatter" ||
        value == "reducescatter" ||
        value == "rs") {
        return IpcCollective::ReduceScatter;
    }

    if (value == "all_gather" ||
        value == "all-gather" ||
        value == "allgather" ||
        value == "ag") {
        return IpcCollective::AllGather;
    }

    throw std::invalid_argument(
        "unknown collective '" + value +
        "'; expected allreduce, reduce_scatter, or all_gather");
}

const char* collective_name(IpcCollective collective) {
    switch (collective) {
        case IpcCollective::AllReduce:
            return "allreduce";
        case IpcCollective::ReduceScatter:
            return "reduce_scatter";
        case IpcCollective::AllGather:
            return "all_gather";
        default:
            return "unknown";
    }
}

double collective_code(IpcCollective collective) {
    return static_cast<double>(static_cast<int>(collective));
}

const char* oo_status_string(oo_status_t status) {
    switch (status) {
        case OO_SUCCESS:
            return "OO_SUCCESS";
        case OO_ERROR_INVALID_ARGUMENT:
            return "OO_ERROR_INVALID_ARGUMENT";
        case OO_ERROR_INVALID_DEVICE:
            return "OO_ERROR_INVALID_DEVICE";
        case OO_ERROR_UNSUPPORTED:
            return "OO_ERROR_UNSUPPORTED";
        case OO_ERROR_CUDA:
            return "OO_ERROR_CUDA";
        case OO_ERROR_INTERNAL:
            return "OO_ERROR_INTERNAL";
        default:
            return "OO_ERROR_UNKNOWN";
    }
}

void check_oo(oo_status_t status, const char* what) {
    if (status != OO_SUCCESS) {
        throw std::runtime_error(
            std::string(what) + " failed: " + oo_status_string(status));
    }
}

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(what) + " failed: " + cudaGetErrorString(err));
    }
}

float rank_scale(int rank) {
    return rank == 0 ? 0.25f : 0.50f;
}

float rank_offset(int rank) {
    return rank == 0 ? 1.0f : 2.0f;
}

size_t rank_partition_begin(
    size_t count,
    int rank,
    int world_size) {
    const size_t world = static_cast<size_t>(world_size);
    const size_t r = static_cast<size_t>(rank);
    const size_t base = count / world;
    const size_t rem = count % world;

    return r * base + ((r < rem) ? r : rem);
}

size_t rank_partition_count(
    size_t count,
    int rank,
    int world_size) {
    const size_t world = static_cast<size_t>(world_size);
    const size_t r = static_cast<size_t>(rank);
    const size_t base = count / world;
    const size_t rem = count % world;

    return base + ((r < rem) ? 1 : 0);
}

std::vector<float> reference_rank_fp16(
    int64_t numel,
    int rank) {
    return testing::host_reference_pattern_fp16(
        numel,
        rank_scale(rank),
        rank_offset(rank));
}

std::vector<float> reference_sum_fp16(int64_t numel) {
    std::vector<float> ref0 = reference_rank_fp16(numel, 0);
    std::vector<float> ref1 = reference_rank_fp16(numel, 1);

    std::vector<float> out(static_cast<size_t>(numel));

    for (int64_t i = 0; i < numel; ++i) {
        float acc = ref0[static_cast<size_t>(i)];
        acc = testing::round_to_half(acc + ref1[static_cast<size_t>(i)]);
        out[static_cast<size_t>(i)] = acc;
    }

    return out;
}

std::vector<float> reference_all_gather_fp16(int64_t numel) {
    std::vector<float> ref0 = reference_rank_fp16(numel, 0);
    std::vector<float> ref1 = reference_rank_fp16(numel, 1);

    std::vector<float> out(static_cast<size_t>(numel), 0.0f);

    const size_t begin0 =
        rank_partition_begin(static_cast<size_t>(numel), 0, 2);
    const size_t count0 =
        rank_partition_count(static_cast<size_t>(numel), 0, 2);

    const size_t begin1 =
        rank_partition_begin(static_cast<size_t>(numel), 1, 2);
    const size_t count1 =
        rank_partition_count(static_cast<size_t>(numel), 1, 2);

    for (size_t i = 0; i < count0; ++i) {
        const size_t idx = begin0 + i;
        out[idx] = ref0[idx];
    }

    for (size_t i = 0; i < count1; ++i) {
        const size_t idx = begin1 + i;
        out[idx] = ref1[idx];
    }

    return out;
}

std::vector<float> slice_vector(
    const std::vector<float>& values,
    size_t begin,
    size_t count) {
    if (begin > values.size() || count > values.size() - begin) {
        throw std::invalid_argument("slice_vector: invalid slice");
    }

    return std::vector<float>(
        values.begin() + static_cast<std::ptrdiff_t>(begin),
        values.begin() + static_cast<std::ptrdiff_t>(begin + count));
}

void validate_numel_for_collective(
    IpcCollective collective,
    int64_t numel) {
    if (numel <= 0) {
        throw std::invalid_argument("numel must be > 0");
    }

    if ((collective == IpcCollective::ReduceScatter ||
         collective == IpcCollective::AllGather) &&
        (numel % 2) != 0) {
        throw std::invalid_argument(
            std::string(collective_name(collective)) +
            " requires numel divisible by 2 for NCCL comparison");
    }
}

void fill_local_source(
    half* local_src,
    int64_t numel,
    int local_rank,
    int device,
    cudaStream_t stream) {
    system::runtime::set_device(device);

    testing::fill_pattern(
        local_src,
        numel,
        rank_scale(local_rank),
        rank_offset(local_rank),
        stream);

    system::runtime::sync_stream_on_device(
        device,
        stream,
        "sync fill_local_source");
}

void reset_work_buffer_async(
    half* local_work,
    const half* local_src,
    size_t bytes,
    int device,
    cudaStream_t stream) {
    system::runtime::set_device(device);

    check_cuda(
        cudaMemcpyAsync(
            local_work,
            local_src,
            bytes,
            cudaMemcpyDeviceToDevice,
            stream),
        "cudaMemcpyAsync(local_src -> local_work)");
}

void sync_stream(
    int device,
    cudaStream_t stream,
    const char* label) {
    system::runtime::sync_stream_on_device(device, stream, label);
}

void verify_collective_result(
    IpcCollective collective,
    const char* label,
    half* local_work,
    int64_t numel,
    int local_rank,
    int device) {
    const std::vector<float> got =
        testing::copy_half_device_to_host_float(
            local_work,
            numel,
            device);

    if (collective == IpcCollective::AllReduce) {
        const std::vector<float> ref = reference_sum_fp16(numel);

        testing::expect_allclose(
            got,
            ref,
            (std::string(label) + " rank" +
             std::to_string(local_rank) + " allreduce").c_str());

        return;
    }

    if (collective == IpcCollective::ReduceScatter) {
        const std::vector<float> ref = reference_sum_fp16(numel);

        const size_t begin =
            rank_partition_begin(static_cast<size_t>(numel), local_rank, 2);

        const size_t count =
            rank_partition_count(static_cast<size_t>(numel), local_rank, 2);

        testing::expect_allclose(
            slice_vector(got, begin, count),
            slice_vector(ref, begin, count),
            (std::string(label) + " rank" +
             std::to_string(local_rank) + " reduce_scatter").c_str());

        return;
    }

    if (collective == IpcCollective::AllGather) {
        const std::vector<float> ref = reference_all_gather_fp16(numel);

        testing::expect_allclose(
            got,
            ref,
            (std::string(label) + " rank" +
             std::to_string(local_rank) + " all_gather").c_str());

        return;
    }

    throw std::invalid_argument("verify_collective_result: unknown collective");
}

ncclUniqueId make_nccl_unique_id(
    const std::vector<int64_t>& encoded) {
    ncclUniqueId id;
    std::memset(&id, 0, sizeof(id));

    const size_t expected_bytes = sizeof(id.internal);

    if (encoded.size() * sizeof(int64_t) == expected_bytes) {
        std::memcpy(id.internal, encoded.data(), expected_bytes);
        return id;
    }

    if (encoded.size() == expected_bytes) {
        for (size_t i = 0; i < encoded.size(); ++i) {
            if (encoded[i] < 0 || encoded[i] > 255) {
                throw std::invalid_argument(
                    "NCCL unique ID byte out of range");
            }

            id.internal[i] = static_cast<char>(encoded[i]);
        }

        return id;
    }

    throw std::invalid_argument(
        "NCCL unique ID has wrong encoded size: got " +
        std::to_string(encoded.size()) +
        " int64 values; expected either " +
        std::to_string(expected_bytes / sizeof(int64_t)) +
        " packed int64 values or " +
        std::to_string(expected_bytes) +
        " byte values");
}

double elapsed_one_rank_ms(
    int device,
    cudaStream_t stream,
    const std::function<void()>& launch_once) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    system::runtime::set_device(device);

    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");

    launch_once();

    check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float ms = 0.0f;

    check_cuda(
        cudaEventElapsedTime(&ms, start, stop),
        "cudaEventElapsedTime");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return static_cast<double>(ms);
}

struct IpcOoContext {
    oo_group_t* group = nullptr;
    oo_node_t* node = nullptr;
    oo_buffer_t* local_buf = nullptr;
    oo_buffer_t* peer_buf = nullptr;
    half* local_work = nullptr;
};

void destroy_ipc_oo_context(IpcOoContext& ctx) {
    if (ctx.group != nullptr && ctx.group->broker) {
        try {
            ctx.group->broker->sync();
        } catch (...) {
        }
    }

    if (ctx.peer_buf != nullptr) {
        oo_buffer_destroy(ctx.peer_buf);
        ctx.peer_buf = nullptr;
    }

    if (ctx.group != nullptr && ctx.group->broker) {
        try {
            ctx.group->broker->sync();
        } catch (...) {
        }
    }

    if (ctx.local_buf != nullptr) {
        oo_buffer_destroy(ctx.local_buf);
        ctx.local_buf = nullptr;
        ctx.local_work = nullptr;
    }

    if (ctx.node != nullptr) {
        oo_node_destroy(ctx.node);
        ctx.node = nullptr;
    }

    if (ctx.group != nullptr) {
        oo_group_destroy(ctx.group);
        ctx.group = nullptr;
    }
}

IpcOoContext create_ipc_oo_context(
    int64_t numel,
    int local_rank,
    int dev0,
    int dev1,
    const std::string& broker_key) {
    if (local_rank != 0 && local_rank != 1) {
        throw std::invalid_argument("local_rank must be 0 or 1");
    }

    if (broker_key.empty()) {
        throw std::invalid_argument("broker_key must be non-empty");
    }

    int devices[2] = {dev0, dev1};
    const int peer_rank = local_rank ^ 1;
    const size_t bytes = static_cast<size_t>(numel) * sizeof(half);

    IpcOoContext ctx;

    check_oo(
        oo_group_create_ipc(
            devices,
            2,
            local_rank,
            broker_key.c_str(),
            &ctx.group),
        "oo_group_create_ipc");

    check_oo(
        oo_node_create(
            ctx.group,
            local_rank,
            &ctx.node),
        "oo_node_create(local)");

    check_oo(
        oo_buffer_alloc(
            ctx.node,
            bytes,
            &ctx.local_buf),
        "oo_buffer_alloc(local VMM)");

    ctx.local_work =
        reinterpret_cast<half*>(oo_buffer_ptr(ctx.local_buf));

    if (ctx.local_work == nullptr) {
        throw std::runtime_error("oo_buffer_ptr(local_buf) returned null");
    }

    std::vector<int> access_devices = {dev0, dev1};

    ooverlap::system::vmm_peer_buffer_descriptor local_desc =
        ooverlap::system::make_vmm_peer_buffer_descriptor(
            ctx.local_buf->mapped,
            bytes);

    ooverlap::system::ipc::vmm_handle local_fd =
        ooverlap::system::export_vmm_peer_buffer_fd(
            ctx.local_buf->mapped);

    std::vector<ooverlap::system::vmm_peer_buffer_descriptor> all_desc(2);

    ctx.group->broker->exchange_data(
        all_desc.data(),
        &local_desc,
        sizeof(local_desc));

    std::vector<int> all_fds(2, -1);

    ctx.group->broker->exchange_fds(
        all_fds.data(),
        local_fd.value);

    local_fd.value = -1;

    if (all_fds[peer_rank] < 0) {
        throw std::runtime_error("exchange_fds did not return peer fd");
    }

    ooverlap::system::imported_peer_buffer imported_peer =
        ooverlap::system::import_vmm_peer_buffer(
            all_fds[peer_rank],
            all_desc[peer_rank],
            access_devices);

    check_oo(
        oo_buffer_adopt_imported_peer_buffer(
            ctx.node,
            std::move(imported_peer),
            &ctx.peer_buf),
        "oo_buffer_adopt_imported_peer_buffer(peer VMM)");

    ctx.peer_buf->owner_rank = peer_rank;
    ctx.group->broker->sync();

    return ctx;
}

void launch_ooverlap_collective(
    IpcCollective collective,
    IpcOoContext& ctx,
    size_t numel,
    cudaStream_t stream) {
    oo_buffer_t* peer_bufs[] = {
        ctx.peer_buf
    };

    if (collective == IpcCollective::AllReduce) {
        check_oo(
            oo_allreduce(
                ctx.node,
                ctx.local_buf,
                peer_bufs,
                1,
                numel,
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                stream),
            "oo_allreduce");
        return;
    }

    if (collective == IpcCollective::ReduceScatter) {
        oo_tensor_slice_t slice{};

        check_oo(
            oo_reduce_scatter(
                ctx.node,
                ctx.local_buf,
                peer_bufs,
                1,
                numel,
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                &slice,
                stream),
            "oo_reduce_scatter");
        return;
    }

    if (collective == IpcCollective::AllGather) {
        check_oo(
            oo_all_gather(
                ctx.node,
                ctx.local_buf,
                peer_bufs,
                1,
                numel,
                OO_DTYPE_FLOAT16,
                stream),
            "oo_all_gather");
        return;
    }

    throw std::invalid_argument("launch_ooverlap_collective: unknown collective");
}

void launch_nccl_collective(
    IpcCollective collective,
    ncclComm_t comm,
    half* work,
    size_t numel,
    int local_rank,
    cudaStream_t stream) {
    if (collective == IpcCollective::AllReduce) {
        OOVERLAP_IPC_COLLECTIVE_NCCL_CHECK(
            ncclAllReduce(
                work,
                work,
                numel,
                ncclFloat16,
                ncclSum,
                comm,
                stream));
        return;
    }

    if (collective == IpcCollective::ReduceScatter) {
        const size_t shard_begin =
            rank_partition_begin(numel, local_rank, 2);

        const size_t shard_count =
            rank_partition_count(numel, local_rank, 2);

        OOVERLAP_IPC_COLLECTIVE_NCCL_CHECK(
            ncclReduceScatter(
                work,
                work + shard_begin,
                shard_count,
                ncclFloat16,
                ncclSum,
                comm,
                stream));
        return;
    }

    if (collective == IpcCollective::AllGather) {
        const size_t shard_begin =
            rank_partition_begin(numel, local_rank, 2);

        const size_t shard_count =
            rank_partition_count(numel, local_rank, 2);

        OOVERLAP_IPC_COLLECTIVE_NCCL_CHECK(
            ncclAllGather(
                work + shard_begin,
                work,
                shard_count,
                ncclFloat16,
                comm,
                stream));
        return;
    }

    throw std::invalid_argument("launch_nccl_collective: unknown collective");
}

void warmup_ooverlap(
    IpcCollective collective,
    IpcOoContext& ctx,
    const half* local_src,
    size_t numel,
    int local_device,
    cudaStream_t stream,
    int warmup) {
    const size_t bytes = numel * sizeof(half);

    for (int i = 0; i < warmup; ++i) {
        reset_work_buffer_async(
            ctx.local_work,
            local_src,
            bytes,
            local_device,
            stream);

        sync_stream(local_device, stream, "sync oo warmup reset");

        ctx.group->broker->sync();

        launch_ooverlap_collective(
            collective,
            ctx,
            numel,
            stream);

        sync_stream(local_device, stream, "sync oo warmup");

        ctx.group->broker->sync();
    }
}

double benchmark_ooverlap_total_ms(
    IpcCollective collective,
    IpcOoContext& ctx,
    const half* local_src,
    size_t numel,
    int local_device,
    cudaStream_t stream,
    int iters) {
    const size_t bytes = numel * sizeof(half);
    double total_ms = 0.0;

    for (int i = 0; i < iters; ++i) {
        reset_work_buffer_async(
            ctx.local_work,
            local_src,
            bytes,
            local_device,
            stream);

        sync_stream(local_device, stream, "sync oo timed reset");

        ctx.group->broker->sync();

        total_ms += elapsed_one_rank_ms(
            local_device,
            stream,
            [&]() {
                launch_ooverlap_collective(
                    collective,
                    ctx,
                    numel,
                    stream);
            });

        ctx.group->broker->sync();
    }

    return total_ms;
}

void warmup_nccl(
    IpcCollective collective,
    ncclComm_t comm,
    half* work,
    const half* local_src,
    size_t numel,
    int local_rank,
    int local_device,
    cudaStream_t stream,
    IpcOoContext& barrier_ctx,
    int warmup) {
    const size_t bytes = numel * sizeof(half);

    for (int i = 0; i < warmup; ++i) {
        reset_work_buffer_async(
            work,
            local_src,
            bytes,
            local_device,
            stream);

        sync_stream(local_device, stream, "sync nccl warmup reset");

        barrier_ctx.group->broker->sync();

        launch_nccl_collective(
            collective,
            comm,
            work,
            numel,
            local_rank,
            stream);

        sync_stream(local_device, stream, "sync nccl warmup");

        barrier_ctx.group->broker->sync();
    }
}

double benchmark_nccl_total_ms(
    IpcCollective collective,
    ncclComm_t comm,
    half* work,
    const half* local_src,
    size_t numel,
    int local_rank,
    int local_device,
    cudaStream_t stream,
    IpcOoContext& barrier_ctx,
    int iters) {
    const size_t bytes = numel * sizeof(half);
    double total_ms = 0.0;

    for (int i = 0; i < iters; ++i) {
        reset_work_buffer_async(
            work,
            local_src,
            bytes,
            local_device,
            stream);

        sync_stream(local_device, stream, "sync nccl timed reset");

        barrier_ctx.group->broker->sync();

        total_ms += elapsed_one_rank_ms(
            local_device,
            stream,
            [&]() {
                launch_nccl_collective(
                    collective,
                    comm,
                    work,
                    numel,
                    local_rank,
                    stream);
            });

        barrier_ctx.group->broker->sync();
    }

    return total_ms;
}

void verify_ooverlap_once(
    IpcCollective collective,
    IpcOoContext& ctx,
    const half* local_src,
    int64_t numel,
    int local_rank,
    int local_device,
    cudaStream_t stream) {
    const size_t bytes = static_cast<size_t>(numel) * sizeof(half);

    reset_work_buffer_async(
        ctx.local_work,
        local_src,
        bytes,
        local_device,
        stream);

    sync_stream(local_device, stream, "sync oo verify reset");

    ctx.group->broker->sync();

    launch_ooverlap_collective(
        collective,
        ctx,
        static_cast<size_t>(numel),
        stream);

    sync_stream(local_device, stream, "sync oo verify");

    verify_collective_result(
        collective,
        "ooverlap IPC verify",
        ctx.local_work,
        numel,
        local_rank,
        local_device);

    ctx.group->broker->sync();
}

void verify_nccl_once(
    IpcCollective collective,
    ncclComm_t comm,
    half* work,
    const half* local_src,
    int64_t numel,
    int local_rank,
    int local_device,
    cudaStream_t stream,
    IpcOoContext& barrier_ctx) {
    const size_t bytes = static_cast<size_t>(numel) * sizeof(half);

    reset_work_buffer_async(
        work,
        local_src,
        bytes,
        local_device,
        stream);

    sync_stream(local_device, stream, "sync nccl verify reset");

    barrier_ctx.group->broker->sync();

    launch_nccl_collective(
        collective,
        comm,
        work,
        static_cast<size_t>(numel),
        local_rank,
        stream);

    sync_stream(local_device, stream, "sync nccl verify");

    verify_collective_result(
        collective,
        "NCCL IPC verify",
        work,
        numel,
        local_rank,
        local_device);

    barrier_ctx.group->broker->sync();
}

std::map<std::string, double> run_one_size(
    IpcCollective collective,
    int64_t numel,
    int local_rank,
    int dev0,
    int dev1,
    const std::string& broker_key,
    ncclComm_t nccl_comm,
    int iters,
    int warmup,
    bool verify) {
    validate_numel_for_collective(collective, numel);

    const int local_device = local_rank == 0 ? dev0 : dev1;
    const size_t bytes = static_cast<size_t>(numel) * sizeof(half);

    IpcOoContext ctx;

    half* local_src = nullptr;
    half* nccl_work = nullptr;
    cudaStream_t stream = nullptr;

    try {
        system::runtime::set_device(local_device);

        stream =
            system::runtime::create_stream_on_device(local_device);

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&local_src),
                bytes),
            "cudaMalloc(local_src)");

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&nccl_work),
                bytes),
            "cudaMalloc(nccl_work)");

        fill_local_source(
            local_src,
            numel,
            local_rank,
            local_device,
            stream);

        ctx =
            create_ipc_oo_context(
                numel,
                local_rank,
                dev0,
                dev1,
                broker_key);

        ctx.group->broker->sync();

        system::runtime::set_device(local_device);

        warmup_ooverlap(
            collective,
            ctx,
            local_src,
            static_cast<size_t>(numel),
            local_device,
            stream,
            warmup);

        const double oo_total_ms =
            benchmark_ooverlap_total_ms(
                collective,
                ctx,
                local_src,
                static_cast<size_t>(numel),
                local_device,
                stream,
                iters);

        warmup_nccl(
            collective,
            nccl_comm,
            nccl_work,
            local_src,
            static_cast<size_t>(numel),
            local_rank,
            local_device,
            stream,
            ctx,
            warmup);

        const double nccl_total_ms =
            benchmark_nccl_total_ms(
                collective,
                nccl_comm,
                nccl_work,
                local_src,
                static_cast<size_t>(numel),
                local_rank,
                local_device,
                stream,
                ctx,
                iters);

        if (verify) {
            verify_ooverlap_once(
                collective,
                ctx,
                local_src,
                numel,
                local_rank,
                local_device,
                stream);

            verify_nccl_once(
                collective,
                nccl_comm,
                nccl_work,
                local_src,
                numel,
                local_rank,
                local_device,
                stream,
                ctx);
        }

        ctx.group->broker->sync();

        ctx.group->broker->sync();

        if (local_src != nullptr) {
            system::runtime::set_device(local_device);
            cudaFree(local_src);
            local_src = nullptr;
        }

        if (nccl_work != nullptr) {
            system::runtime::set_device(local_device);
            cudaFree(nccl_work);
            nccl_work = nullptr;
        }

        destroy_ipc_oo_context(ctx);

        if (stream != nullptr) {
            system::runtime::destroy_stream_on_device(
                local_device,
                stream);
            stream = nullptr;
        }

        const size_t local_shard_count =
            rank_partition_count(static_cast<size_t>(numel), local_rank, 2);

        return {
            {"collective", collective_code(collective)},
            {"rank", static_cast<double>(local_rank)},
            {"world_size", 2.0},
            {"numel", static_cast<double>(numel)},
            {"bytes", static_cast<double>(bytes)},
            {"local_shard_numel", static_cast<double>(local_shard_count)},
            {"local_shard_bytes",
             static_cast<double>(local_shard_count * sizeof(half))},
            {"oo_total_ms", oo_total_ms},
            {"nccl_total_ms", nccl_total_ms},
            {"iters", static_cast<double>(iters)},
            {"warmup", static_cast<double>(warmup)},
            {"verify", verify ? 1.0 : 0.0}
        };
    } catch (...) {
        if (local_src != nullptr) {
            system::runtime::set_device(local_device);
            cudaFree(local_src);
            local_src = nullptr;
        }

        if (nccl_work != nullptr) {
            system::runtime::set_device(local_device);
            cudaFree(nccl_work);
            nccl_work = nullptr;
        }

        destroy_ipc_oo_context(ctx);

        if (stream != nullptr) {
            system::runtime::destroy_stream_on_device(
                local_device,
                stream);
            stream = nullptr;
        }

        throw;
    }
}

} // namespace

bool smoke_ipc_collective_rank_sm90(
    const std::string& collective,
    int64_t numel,
    int local_rank,
    int dev0,
    int dev1,
    const std::string& broker_key,
    const std::vector<int64_t>& nccl_unique_id_bytes,
    bool verify) {
    std::vector<std::map<std::string, double>> rows =
        benchmark_ipc_collective_rank_sm90(
            collective,
            std::vector<int64_t>{numel},
            local_rank,
            dev0,
            dev1,
            broker_key,
            nccl_unique_id_bytes,
            1,
            0,
            verify);

    return rows.size() == 1;
}

std::vector<std::map<std::string, double>> benchmark_ipc_collective_rank_sm90(
    const std::string& collective_name_arg,
    const std::vector<int64_t>& sizes,
    int local_rank,
    int dev0,
    int dev1,
    const std::string& broker_key,
    const std::vector<int64_t>& nccl_unique_id_bytes,
    int iters,
    int warmup,
    bool verify) {
    if (sizes.empty()) {
        throw std::invalid_argument("sizes must be non-empty");
    }

    if (iters <= 0) {
        throw std::invalid_argument("iters must be > 0");
    }

    if (warmup < 0) {
        throw std::invalid_argument("warmup must be >= 0");
    }

    if (dev0 == dev1) {
        throw std::invalid_argument("dev0 and dev1 must differ");
    }

    if (local_rank != 0 && local_rank != 1) {
        throw std::invalid_argument("local_rank must be 0 or 1");
    }

    if (broker_key.empty()) {
        throw std::invalid_argument("broker_key must be non-empty");
    }

    const IpcCollective collective =
        parse_collective(collective_name_arg);

    const ncclUniqueId nccl_id =
        make_nccl_unique_id(nccl_unique_id_bytes);

    const int local_device = local_rank == 0 ? dev0 : dev1;

    ncclComm_t nccl_comm = nullptr;

    try {
        system::runtime::set_device(local_device);

        OOVERLAP_IPC_COLLECTIVE_NCCL_CHECK(
            ncclCommInitRank(
                &nccl_comm,
                2,
                nccl_id,
                local_rank));

        std::vector<std::map<std::string, double>> rows;
        rows.reserve(sizes.size());

        for (int64_t numel : sizes) {
            rows.push_back(
                run_one_size(
                    collective,
                    numel,
                    local_rank,
                    dev0,
                    dev1,
                    broker_key,
                    nccl_comm,
                    iters,
                    warmup,
                    verify));
        }

        if (nccl_comm != nullptr) {
            ncclCommDestroy(nccl_comm);
            nccl_comm = nullptr;
        }

        return rows;
    } catch (...) {
        if (nccl_comm != nullptr) {
            ncclCommDestroy(nccl_comm);
            nccl_comm = nullptr;
        }

        throw;
    }
}

} // namespace ooverlap
