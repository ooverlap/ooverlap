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
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ooverlap {
namespace {

using testing::TestCollective;

/* OOVERLAP_IPC_COLLECTIVE_MULTI_GPU_V1 */
void validate_devices(
    const std::vector<int>& devices,
    int local_rank) {
    if (devices.size() < 2) {
        throw std::invalid_argument(
            "IPC collective requires at least two devices");
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

    if (local_rank < 0 ||
        local_rank >= static_cast<int>(devices.size())) {
        throw std::invalid_argument("local_rank is outside the device list");
    }
}

void nccl_mem_alloc_half_on_device(
    int device,
    half** ptr,
    size_t bytes,
    const char* label) {
    if (ptr == nullptr) {
        throw std::invalid_argument(
            "nccl_mem_alloc_half_on_device: ptr must not be null");
    }

    *ptr = nullptr;

    system::runtime::set_device(device);

    void* raw = nullptr;

    OOVERLAP_TEST_NCCL_CHECK(
        ncclMemAlloc(
            &raw,
            bytes));

    if (raw == nullptr) {
        throw std::runtime_error(
            std::string(label != nullptr ? label : "ncclMemAlloc") +
            ": ncclMemAlloc returned nullptr");
    }

    *ptr =
        reinterpret_cast<half*>(raw);
}

void nccl_mem_free_on_device(
    int device,
    half*& ptr) {
    if (ptr == nullptr) {
        return;
    }

    (void)cudaSetDevice(device);
    (void)ncclMemFree(static_cast<void*>(ptr));
    ptr = nullptr;
}

void register_nccl_symmetric_window(
    ncclComm_t comm,
    half* buf,
    size_t bytes,
    ncclWindow_t& win) {
    win = nullptr;

    if (comm == nullptr || buf == nullptr || bytes == 0) {
        throw std::invalid_argument(
            "register_nccl_symmetric_window: invalid argument");
    }

    OOVERLAP_TEST_NCCL_CHECK(
        ncclCommWindowRegister(
            comm,
            buf,
            bytes,
            &win,
            NCCL_WIN_COLL_SYMMETRIC));
}

void deregister_nccl_window_best_effort(
    ncclComm_t comm,
    ncclWindow_t& win) {
    if (comm == nullptr || win == nullptr) {
        return;
    }

    (void)ncclCommWindowDeregister(comm, win);
    win = nullptr;
}

struct IpcOoContext {
    oo_group_t* group = nullptr;
    oo_node_t* node = nullptr;
    oo_buffer_t* local_buf = nullptr;

    /*
     * OOVERLAP_IPC_COLLECTIVE_LEGACY_TEST_CONTEXT_PATCH:
     *
     * For the new IPC path, the test owns a normal cudaMalloc/framework-style
     * pointer and registers it with oo_buffer_wrap(...).  The library then
     * exports/imports peer pointers internally via legacy CUDA IPC during
     * prepare_collective_launch().
     */
    half* local_work = nullptr;
    int local_device = -1;
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

    if (ctx.local_buf != nullptr) {
        oo_buffer_destroy(ctx.local_buf);
        ctx.local_buf = nullptr;
    }

    if (ctx.local_work != nullptr) {
        try {
            if (ctx.local_device >= 0) {
                system::runtime::set_device(ctx.local_device);
            }

            cudaFree(ctx.local_work);
        } catch (...) {
        }

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

    ctx.local_device = -1;
}


IpcOoContext create_ipc_oo_context(
    int64_t numel,
    int local_rank,
    const std::vector<int>& devices,
    const std::string& broker_key) {
    /*
     * OOVERLAP_IPC_COLLECTIVE_LEGACY_TEST_CREATE_PATCH:
     *
     * Do not manually exchange/import peer buffers here.  This test now exercises
     * the real public IPC path:
     *
     *   cudaMalloc local pointer
     *   oo_buffer_wrap(local pointer)
     *   oo_allreduce / oo_reduce_scatter / oo_all_gather
     *
     * The comm layer exports/imports peer pointers internally every collective.
     */
    validate_devices(devices, local_rank);

    if (broker_key.empty()) {
        throw std::invalid_argument("broker_key must be non-empty");
    }

    const int world_size = static_cast<int>(devices.size());
    const size_t bytes = static_cast<size_t>(numel) * sizeof(half);

    IpcOoContext ctx;
    ctx.local_device = devices[static_cast<std::size_t>(local_rank)];

    testing::check_oo(
        oo_group_create_ipc(
            devices.data(),
            world_size,
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

    system::runtime::set_device(ctx.local_device);

    testing::check_cuda(
        cudaMalloc(
            reinterpret_cast<void**>(&ctx.local_work),
            bytes),
        "cudaMalloc(ipc local_work)");

    testing::check_oo(
        oo_buffer_wrap(
            ctx.node,
            ctx.local_work,
            bytes,
            &ctx.local_buf),
        "oo_buffer_wrap(ipc local cudaMalloc)");

    if (ctx.local_work == nullptr) {
        throw std::runtime_error("cudaMalloc(ipc local_work) returned null");
    }

    broker_sync(ctx);
    return ctx;
}


void launch_ooverlap_collective(
    TestCollective collective,
    IpcOoContext& ctx,
    size_t numel,
    cudaStream_t stream) {
    /*
     * OOVERLAP_IPC_COLLECTIVE_PUBLIC_API_LAUNCH_PATCH:
     *
     * Use the current clean public collective API.  Peer buffers are resolved by
     * prepare_collective_launch() through the IPC export/import backend.
     */
    if (collective == TestCollective::AllReduce) {
        testing::check_oo(
            oo_allreduce(
                ctx.node,
                ctx.local_buf,
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
    int world_size,
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
            world_size,
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
    int world_size,
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
                        world_size,
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
    int world_size,
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
        world_size,
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
    int world_size,
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
        world_size,
        stream);

    sync_device_stream(local_device, stream, "sync nccl verify");

    testing::verify_collective_fp16(
        collective,
        "NCCL IPC verify",
        work,
        numel,
        local_rank,
        world_size,
        local_device);

    broker_sync(barrier_ctx);
}

std::map<std::string, double> run_one_size(
    TestCollective collective,
    int64_t numel,
    int local_rank,
    const std::vector<int>& devices,
    const std::string& broker_key,
    ncclComm_t nccl_comm,
    int iters,
    int warmup,
    bool verify) {
    validate_devices(devices, local_rank);
    const int world_size = static_cast<int>(devices.size());

    testing::validate_numel_for_collective(
        collective,
        numel,
        world_size);

    const int local_device = devices[static_cast<std::size_t>(local_rank)];
    const size_t bytes = static_cast<size_t>(numel) * sizeof(half);

    IpcOoContext ctx;

    half* local_src = nullptr;
    half* nccl_work = nullptr;
    half* nccl_symmetric_work = nullptr;
    ncclWindow_t nccl_symmetric_win = nullptr;
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

        nccl_mem_alloc_half_on_device(
            local_device,
            &nccl_symmetric_work,
            bytes,
            "ncclMemAlloc(ipc nccl_symmetric_work)");

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
                devices,
                broker_key);

        broker_sync(ctx);

        /*
         * Symmetric NCCL baseline for the IPC benchmark.
         *
         * Each OS process owns one rank and registers only its local
         * ncclMemAlloc buffer.  All ranks execute this same registration point
         * before any benchmarked collectives, with Broker syncs around it to keep
         * the host-side sequence matched.
         */
        register_nccl_symmetric_window(
            nccl_comm,
            nccl_symmetric_work,
            bytes,
            nccl_symmetric_win);

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
            world_size,
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
                world_size,
                local_device,
                stream,
                ctx,
                iters);

        warmup_nccl(
            collective,
            nccl_comm,
            nccl_symmetric_work,
            local_src,
            static_cast<size_t>(numel),
            local_rank,
            world_size,
            local_device,
            stream,
            ctx,
            warmup);

        const double nccl_symmetric_total_ms =
            benchmark_nccl_total_ms(
                collective,
                nccl_comm,
                nccl_symmetric_work,
                local_src,
                static_cast<size_t>(numel),
                local_rank,
                world_size,
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
                world_size,
                local_device,
                stream);

            verify_nccl_once(
                collective,
                nccl_comm,
                nccl_work,
                local_src,
                numel,
                local_rank,
                world_size,
                local_device,
                stream,
                ctx);

            verify_nccl_once(
                collective,
                nccl_comm,
                nccl_symmetric_work,
                local_src,
                numel,
                local_rank,
                world_size,
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

        deregister_nccl_window_best_effort(
            nccl_comm,
            nccl_symmetric_win);

        if (nccl_work != nullptr) {
            system::runtime::set_device(local_device);
            cudaFree(nccl_work);
            nccl_work = nullptr;
        }

        if (nccl_symmetric_work != nullptr) {
            nccl_mem_free_on_device(
                local_device,
                nccl_symmetric_work);
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
                world_size);

        return {
            {"collective", testing::collective_code(collective)},
            {"rank", static_cast<double>(local_rank)},
            {"world_size", static_cast<double>(world_size)},
            {"numel", static_cast<double>(numel)},
            {"bytes", static_cast<double>(bytes)},
            {"local_shard_numel", static_cast<double>(local_shard_count)},
            {"local_shard_bytes",
             static_cast<double>(local_shard_count * sizeof(half))},
            {"oo_total_ms", oo_total_ms},
            {"nccl_total_ms", nccl_total_ms},
            {"nccl_symmetric_total_ms", nccl_symmetric_total_ms},
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
    const std::vector<int>& devices,
    const std::string& broker_key,
    const std::vector<int64_t>& nccl_unique_id_bytes,
    bool verify) {
    std::vector<std::map<std::string, double>> rows =
        benchmark_ipc_collective_rank_sm90(
            collective,
            std::vector<int64_t>{numel},
            local_rank,
            devices,
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
    const std::vector<int>& devices,
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

    validate_devices(devices, local_rank);

    if (broker_key.empty()) {
        throw std::invalid_argument("broker_key must be non-empty");
    }

    const TestCollective collective =
        testing::parse_collective(collective_name_arg);

    const ncclUniqueId nccl_id =
        testing::make_nccl_unique_id(nccl_unique_id_bytes);

    const int world_size = static_cast<int>(devices.size());
    const int local_device = devices[static_cast<std::size_t>(local_rank)];

    ncclComm_t nccl_comm = nullptr;

    try {
        system::runtime::set_device(local_device);

        OOVERLAP_TEST_NCCL_CHECK(
            ncclCommInitRank(
                &nccl_comm,
                world_size,
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
                    devices,
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

// Backward-compatible two-GPU overloads.
bool smoke_ipc_collective_rank_sm90(
    const std::string& collective,
    int64_t numel,
    int local_rank,
    int dev0,
    int dev1,
    const std::string& broker_key,
    const std::vector<int64_t>& nccl_unique_id_bytes,
    bool verify) {
    return smoke_ipc_collective_rank_sm90(
        collective,
        numel,
        local_rank,
        std::vector<int>{dev0, dev1},
        broker_key,
        nccl_unique_id_bytes,
        verify);
}

std::vector<std::map<std::string, double>> benchmark_ipc_collective_rank_sm90(
    const std::string& collective,
    const std::vector<int64_t>& sizes,
    int local_rank,
    int dev0,
    int dev1,
    const std::string& broker_key,
    const std::vector<int64_t>& nccl_unique_id_bytes,
    int iters,
    int warmup,
    bool verify) {
    return benchmark_ipc_collective_rank_sm90(
        collective,
        sizes,
        local_rank,
        std::vector<int>{dev0, dev1},
        broker_key,
        nccl_unique_id_bytes,
        iters,
        warmup,
        verify);
}

} // namespace ooverlap
