#include "test/ipc_collective_sm90.h"

#include "comm/ooverlap_comm_internal.h"
#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/testing/checks.cuh"
#include "ooverlap/testing/collective_test_utils.cuh"
#include "ooverlap/testing/nccl_utils.cuh"
#include "ooverlap/testing/timing.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ooverlap {
namespace {

using testing::TestCollective;

struct IpcOoContext {
    oo_group_t* group = nullptr;
    oo_node_t* node = nullptr;
    oo_buffer_t* local_buf = nullptr;
    oo_buffer_t* peer_buf = nullptr;
    half* local_work = nullptr;
};

void broker_sync(IpcOoContext& ctx) {
    if (ctx.group != nullptr && ctx.group->broker) {
        ctx.group->broker->sync();
    }
}

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

    testing::check_oo(
        oo_group_create_ipc(
            devices,
            2,
            local_rank,
            broker_key.c_str(),
            &ctx.group),
        "oo_group_create_ipc");

    testing::check_oo(
        oo_node_create(
            ctx.group,
            local_rank,
            &ctx.node),
        "oo_node_create(local)");

    testing::check_oo(
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

    testing::check_oo(
        oo_buffer_adopt_imported_peer_buffer(
            ctx.node,
            std::move(imported_peer),
            &ctx.peer_buf),
        "oo_buffer_adopt_imported_peer_buffer(peer VMM)");

    ctx.peer_buf->owner_rank = peer_rank;
    broker_sync(ctx);

    return ctx;
}

void launch_ooverlap_collective(
    TestCollective collective,
    IpcOoContext& ctx,
    size_t numel,
    cudaStream_t stream) {
    oo_buffer_t* peer_bufs[] = {
        ctx.peer_buf,
    };

    if (collective == TestCollective::AllReduce) {
        testing::check_oo(
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

    if (collective == TestCollective::ReduceScatter) {
        oo_tensor_slice_t slice{};

        testing::check_oo(
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

    if (collective == TestCollective::AllGather) {
        testing::check_oo(
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

void sync_device_stream(
    int device,
    cudaStream_t stream,
    const char* label) {
    system::runtime::sync_stream_on_device(device, stream, label);
}

void reset_and_sync(
    half* work,
    const half* src,
    size_t bytes,
    int device,
    cudaStream_t stream,
    const char* label) {
    testing::reset_work_buffer_async(
        work,
        src,
        bytes,
        device,
        stream);

    sync_device_stream(device, stream, label);
}

void warmup_ooverlap(
    TestCollective collective,
    IpcOoContext& ctx,
    const half* local_src,
    size_t numel,
    int local_device,
    cudaStream_t stream,
    int warmup) {
    const size_t bytes = numel * sizeof(half);

    for (int i = 0; i < warmup; ++i) {
        reset_and_sync(
            ctx.local_work,
            local_src,
            bytes,
            local_device,
            stream,
            "sync oo warmup reset");

        broker_sync(ctx);

        launch_ooverlap_collective(
            collective,
            ctx,
            numel,
            stream);

        sync_device_stream(local_device, stream, "sync oo warmup");
        broker_sync(ctx);
    }
}

double benchmark_ooverlap_total_ms(
    TestCollective collective,
    IpcOoContext& ctx,
    const half* local_src,
    size_t numel,
    int local_device,
    cudaStream_t stream,
    int iters) {
    const size_t bytes = numel * sizeof(half);
    double total_ms = 0.0;

    for (int i = 0; i < iters; ++i) {
        reset_and_sync(
            ctx.local_work,
            local_src,
            bytes,
            local_device,
            stream,
            "sync oo timed reset");

        broker_sync(ctx);

        total_ms +=
            testing::elapsed_one_rank_ms(
                local_device,
                stream,
                [&]() {
                    launch_ooverlap_collective(
                        collective,
                        ctx,
                        numel,
                        stream);
                });

        broker_sync(ctx);
    }

    return total_ms;
}

void warmup_nccl(
    TestCollective collective,
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
        reset_and_sync(
            work,
            local_src,
            bytes,
            local_device,
            stream,
            "sync nccl warmup reset");

        broker_sync(barrier_ctx);

        testing::launch_nccl_collective_fp16(
            collective,
            comm,
            work,
            numel,
            local_rank,
            2,
            stream);

        sync_device_stream(local_device, stream, "sync nccl warmup");
        broker_sync(barrier_ctx);
    }
}

double benchmark_nccl_total_ms(
    TestCollective collective,
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
        reset_and_sync(
            work,
            local_src,
            bytes,
            local_device,
            stream,
            "sync nccl timed reset");

        broker_sync(barrier_ctx);

        total_ms +=
            testing::elapsed_one_rank_ms(
                local_device,
                stream,
                [&]() {
                    testing::launch_nccl_collective_fp16(
                        collective,
                        comm,
                        work,
                        numel,
                        local_rank,
                        2,
                        stream);
                });

        broker_sync(barrier_ctx);
    }

    return total_ms;
}

void verify_ooverlap_once(
    TestCollective collective,
    IpcOoContext& ctx,
    const half* local_src,
    int64_t numel,
    int local_rank,
    int local_device,
    cudaStream_t stream) {
    const size_t bytes = static_cast<size_t>(numel) * sizeof(half);

    reset_and_sync(
        ctx.local_work,
        local_src,
        bytes,
        local_device,
        stream,
        "sync oo verify reset");

    broker_sync(ctx);

    launch_ooverlap_collective(
        collective,
        ctx,
        static_cast<size_t>(numel),
        stream);

    sync_device_stream(local_device, stream, "sync oo verify");

    testing::verify_collective_fp16(
        collective,
        "ooverlap IPC verify",
        ctx.local_work,
        numel,
        local_rank,
        2,
        local_device);

    broker_sync(ctx);
}

void verify_nccl_once(
    TestCollective collective,
    ncclComm_t comm,
    half* work,
    const half* local_src,
    int64_t numel,
    int local_rank,
    int local_device,
    cudaStream_t stream,
    IpcOoContext& barrier_ctx) {
    const size_t bytes = static_cast<size_t>(numel) * sizeof(half);

    reset_and_sync(
        work,
        local_src,
        bytes,
        local_device,
        stream,
        "sync nccl verify reset");

    broker_sync(barrier_ctx);

    testing::launch_nccl_collective_fp16(
        collective,
        comm,
        work,
        static_cast<size_t>(numel),
        local_rank,
        2,
        stream);

    sync_device_stream(local_device, stream, "sync nccl verify");

    testing::verify_collective_fp16(
        collective,
        "NCCL IPC verify",
        work,
        numel,
        local_rank,
        2,
        local_device);

    broker_sync(barrier_ctx);
}

std::map<std::string, double> run_one_size(
    TestCollective collective,
    int64_t numel,
    int local_rank,
    int dev0,
    int dev1,
    const std::string& broker_key,
    ncclComm_t nccl_comm,
    int iters,
    int warmup,
    bool verify) {
    testing::validate_numel_for_collective(
        collective,
        numel,
        2);

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

        testing::check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&local_src),
                bytes),
            "cudaMalloc(local_src)");

        testing::check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&nccl_work),
                bytes),
            "cudaMalloc(nccl_work)");

        testing::fill_rank_source_fp16(
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

        broker_sync(ctx);

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

        broker_sync(ctx);
        broker_sync(ctx);

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
            testing::rank_partition_count(
                static_cast<size_t>(numel),
                local_rank,
                2);

        return {
            {"collective", testing::collective_code(collective)},
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
            {"verify", verify ? 1.0 : 0.0},
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

    const TestCollective collective =
        testing::parse_collective(collective_name_arg);

    const ncclUniqueId nccl_id =
        testing::make_nccl_unique_id(nccl_unique_id_bytes);

    const int local_device = local_rank == 0 ? dev0 : dev1;

    ncclComm_t nccl_comm = nullptr;

    try {
        system::runtime::set_device(local_device);

        OOVERLAP_TEST_NCCL_CHECK(
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
