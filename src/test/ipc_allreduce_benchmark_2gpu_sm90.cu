#include "test/ipc_allreduce_benchmark_2gpu_sm90.h"

#include "comm/ooverlap_comm_internal.h"
#include "ooverlap/system/logging.h"
#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/testing/test_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define OOVERLAP_IPC_BENCH_NCCL_CHECK(cmd)                                                \
    do {                                                                                   \
        ncclResult_t result__ = (cmd);                                                     \
        if (result__ != ncclSuccess) {                                                     \
            throw std::runtime_error(                                                      \
                std::string("NCCL error at ") + __FILE__ + ":" +                           \
                std::to_string(__LINE__) + " " + ncclGetErrorString(result__));             \
        }                                                                                  \
    } while (0)

namespace ooverlap {
namespace {

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

std::vector<float> reference_two_gpu_sum_fp16(int64_t numel) {
    auto ref0 = testing::host_reference_pattern_fp16(
        numel,
        rank_scale(0),
        rank_offset(0));

    auto ref1 = testing::host_reference_pattern_fp16(
        numel,
        rank_scale(1),
        rank_offset(1));

    std::vector<float> ref(static_cast<size_t>(numel));
    for (int64_t i = 0; i < numel; ++i) {
        float acc = ref0[static_cast<size_t>(i)];
        acc = testing::round_to_half(acc + ref1[static_cast<size_t>(i)]);
        ref[static_cast<size_t>(i)] = acc;
    }

    return ref;
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

void verify_local_result(
    const char* label,
    half* local_work,
    int64_t numel,
    int local_rank,
    int device) {
    auto got = testing::copy_half_device_to_host_float(
        local_work,
        numel,
        device);

    auto ref = reference_two_gpu_sum_fp16(numel);

    testing::expect_allclose(
        got,
        ref,
        (std::string(label) + " rank" + std::to_string(local_rank)).c_str());
}

ncclUniqueId make_nccl_unique_id(
    const std::vector<int64_t>& encoded) {
    ncclUniqueId id;
    std::memset(&id, 0, sizeof(id));

    const size_t expected_bytes = sizeof(id.internal);

    /*
     * Existing ooverlap generate_nccl_id() returns packed int64 words:
     *
     *   ret.resize(NCCL_UNIQUE_ID_BYTES / sizeof(int64_t));
     *   memcpy(ret.data(), nccl_id.internal, NCCL_UNIQUE_ID_BYTES);
     *
     * So the normal path is 16 int64 values for a 128-byte NCCL ID.
     */
    if (encoded.size() * sizeof(int64_t) == expected_bytes) {
        std::memcpy(
            id.internal,
            encoded.data(),
            expected_bytes);
        return id;
    }

    /*
     * Also allow byte-expanded form in case a future Python helper passes
     * 128 integer byte values.
     */
    if (encoded.size() == expected_bytes) {
        for (size_t i = 0; i < encoded.size(); ++i) {
            if (encoded[i] < 0 || encoded[i] > 255) {
                throw std::invalid_argument("NCCL unique ID byte out of range");
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

void destroy_ipc_oo_context(
    IpcOoContext& ctx,
    int local_device,
    cudaStream_t stream) {
    if (ctx.group && ctx.group->broker) {
        try {
            ctx.group->broker->sync();
        } catch (...) {
        }
    }

    if (ctx.peer_buf != nullptr) {
        oo_buffer_destroy(ctx.peer_buf);
        ctx.peer_buf = nullptr;
    }

    if (ctx.group && ctx.group->broker) {
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

    (void)local_device;
    (void)stream;
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

    ctx.local_work = reinterpret_cast<half*>(oo_buffer_ptr(ctx.local_buf));
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

    ctx.group->broker->sync();

    return ctx;
}

double benchmark_ooverlap_ipc_rank(
    IpcOoContext& ctx,
    const half* local_src,
    int64_t numel,
    int local_rank,
    int local_device,
    cudaStream_t stream,
    int iters,
    int warmup) {
    const size_t bytes = static_cast<size_t>(numel) * sizeof(half);

    for (int i = 0; i < warmup; ++i) {
        reset_work_buffer_async(
            ctx.local_work,
            local_src,
            bytes,
            local_device,
            stream);

        system::runtime::sync_stream_on_device(
            local_device,
            stream,
            "sync oo warmup reset");

        ctx.group->broker->sync();

        oo_buffer_t* peer_bufs[] = {
            ctx.peer_buf
        };

        check_oo(
            oo_allreduce(
                ctx.node,
                ctx.local_buf,
                peer_bufs,
                1,
                static_cast<size_t>(numel),
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                stream),
            "oo_allreduce warmup");

        system::runtime::sync_stream_on_device(
            local_device,
            stream,
            "sync oo warmup");

        ctx.group->broker->sync();
    }

    double total_ms = 0.0;

    for (int i = 0; i < iters; ++i) {
        reset_work_buffer_async(
            ctx.local_work,
            local_src,
            bytes,
            local_device,
            stream);

        system::runtime::sync_stream_on_device(
            local_device,
            stream,
            "sync oo timed reset");

        ctx.group->broker->sync();

        const double ms = elapsed_one_rank_ms(
            local_device,
            stream,
            [&]() {
                oo_buffer_t* peer_bufs[] = {
                    ctx.peer_buf
                };

                check_oo(
                    oo_allreduce(
                        ctx.node,
                        ctx.local_buf,
                        peer_bufs,
                        1,
                        static_cast<size_t>(numel),
                        OO_DTYPE_FLOAT16,
                        OO_REDUCE_SUM,
                        stream),
                    "oo_allreduce timed");
            });

        total_ms += ms;

        ctx.group->broker->sync();
    }

    return total_ms / static_cast<double>(iters);
}

double benchmark_nccl_ipc_rank(
    ncclComm_t comm,
    const half* local_src,
    half* local_out,
    int64_t numel,
    int local_device,
    cudaStream_t stream,
    IpcOoContext& barrier_ctx,
    int iters,
    int warmup) {
    for (int i = 0; i < warmup; ++i) {
        barrier_ctx.group->broker->sync();

        OOVERLAP_IPC_BENCH_NCCL_CHECK(
            ncclAllReduce(
                local_src,
                local_out,
                static_cast<size_t>(numel),
                ncclFloat16,
                ncclSum,
                comm,
                stream));

        system::runtime::sync_stream_on_device(
            local_device,
            stream,
            "sync NCCL warmup");

        barrier_ctx.group->broker->sync();
    }

    double total_ms = 0.0;

    for (int i = 0; i < iters; ++i) {
        barrier_ctx.group->broker->sync();

        const double ms = elapsed_one_rank_ms(
            local_device,
            stream,
            [&]() {
                OOVERLAP_IPC_BENCH_NCCL_CHECK(
                    ncclAllReduce(
                        local_src,
                        local_out,
                        static_cast<size_t>(numel),
                        ncclFloat16,
                        ncclSum,
                        comm,
                        stream));
            });

        total_ms += ms;

        barrier_ctx.group->broker->sync();
    }

    return total_ms / static_cast<double>(iters);
}

} // namespace

std::map<std::string, double> benchmark_ipc_two_gpu_allreduce_rank_sm90(
    int64_t numel,
    int local_rank,
    int dev0,
    int dev1,
    const std::string& broker_key,
    const std::vector<int64_t>& nccl_unique_id_bytes,
    int iters,
    int warmup,
    bool verify) {
    if (numel <= 0) {
        throw std::invalid_argument("numel must be > 0");
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

    const int local_device = local_rank == 0 ? dev0 : dev1;
    const size_t bytes = static_cast<size_t>(numel) * sizeof(half);

    IpcOoContext oo_ctx;

    half* local_src = nullptr;
    half* nccl_out = nullptr;
    cudaStream_t stream = nullptr;
    ncclComm_t nccl_comm = nullptr;

    try {
        system::runtime::set_device(local_device);

        stream = system::runtime::create_stream_on_device(local_device);

        check_cuda(
            cudaMalloc(&local_src, bytes),
            "cudaMalloc(local_src)");

        check_cuda(
            cudaMalloc(&nccl_out, bytes),
            "cudaMalloc(nccl_out)");

        fill_local_source(
            local_src,
            numel,
            local_rank,
            local_device,
            stream);

        oo_ctx = create_ipc_oo_context(
            numel,
            local_rank,
            dev0,
            dev1,
            broker_key);

        const ncclUniqueId nccl_id = make_nccl_unique_id(nccl_unique_id_bytes);

        oo_ctx.group->broker->sync();

        system::runtime::set_device(local_device);

        OOVERLAP_IPC_BENCH_NCCL_CHECK(
            ncclCommInitRank(
                &nccl_comm,
                2,
                nccl_id,
                local_rank));

        oo_ctx.group->broker->sync();

        const double avg_oo_ms = benchmark_ooverlap_ipc_rank(
            oo_ctx,
            local_src,
            numel,
            local_rank,
            local_device,
            stream,
            iters,
            warmup);

        const double avg_nccl_ms = benchmark_nccl_ipc_rank(
            nccl_comm,
            local_src,
            nccl_out,
            numel,
            local_device,
            stream,
            oo_ctx,
            iters,
            warmup);

        if (verify) {
            reset_work_buffer_async(
                oo_ctx.local_work,
                local_src,
                bytes,
                local_device,
                stream);

            system::runtime::sync_stream_on_device(
                local_device,
                stream,
                "sync oo verify reset");

            oo_ctx.group->broker->sync();

            oo_buffer_t* peer_bufs[] = {
                oo_ctx.peer_buf
            };

            check_oo(
                oo_allreduce(
                    oo_ctx.node,
                    oo_ctx.local_buf,
                    peer_bufs,
                    1,
                    static_cast<size_t>(numel),
                    OO_DTYPE_FLOAT16,
                    OO_REDUCE_SUM,
                    stream),
                "oo_allreduce verify");

            system::runtime::sync_stream_on_device(
                local_device,
                stream,
                "sync oo verify");

            verify_local_result(
                "ooverlap IPC verify",
                oo_ctx.local_work,
                numel,
                local_rank,
                local_device);

            oo_ctx.group->broker->sync();

            OOVERLAP_IPC_BENCH_NCCL_CHECK(
                ncclAllReduce(
                    local_src,
                    nccl_out,
                    static_cast<size_t>(numel),
                    ncclFloat16,
                    ncclSum,
                    nccl_comm,
                    stream));

            system::runtime::sync_stream_on_device(
                local_device,
                stream,
                "sync NCCL verify");

            verify_local_result(
                "NCCL IPC verify",
                nccl_out,
                numel,
                local_rank,
                local_device);

            oo_ctx.group->broker->sync();
        }

        oo_ctx.group->broker->sync();

        if (nccl_comm != nullptr) {
            ncclCommDestroy(nccl_comm);
            nccl_comm = nullptr;
        }

        oo_ctx.group->broker->sync();

        if (local_src != nullptr) {
            system::runtime::set_device(local_device);
            cudaFree(local_src);
            local_src = nullptr;
        }

        if (nccl_out != nullptr) {
            system::runtime::set_device(local_device);
            cudaFree(nccl_out);
            nccl_out = nullptr;
        }

        destroy_ipc_oo_context(
            oo_ctx,
            local_device,
            stream);

        if (stream != nullptr) {
            system::runtime::destroy_stream_on_device(
                local_device,
                stream);
            stream = nullptr;
        }

        return {
            {"rank", static_cast<double>(local_rank)},
            {"numel", static_cast<double>(numel)},
            {"avg_ms_oo_ipc", avg_oo_ms},
            {"avg_ms_nccl_ipc", avg_nccl_ms},
            {"speedup_nccl_over_oo_ipc", avg_nccl_ms / avg_oo_ms},
            {"iters", static_cast<double>(iters)},
            {"warmup", static_cast<double>(warmup)},
            {"verify", verify ? 1.0 : 0.0}
        };

    } catch (...) {
        if (nccl_comm != nullptr) {
            ncclCommDestroy(nccl_comm);
            nccl_comm = nullptr;
        }

        if (local_src != nullptr) {
            system::runtime::set_device(local_device);
            cudaFree(local_src);
            local_src = nullptr;
        }

        if (nccl_out != nullptr) {
            system::runtime::set_device(local_device);
            cudaFree(nccl_out);
            nccl_out = nullptr;
        }

        destroy_ipc_oo_context(
            oo_ctx,
            local_device,
            stream);

        if (stream != nullptr) {
            system::runtime::destroy_stream_on_device(
                local_device,
                stream);
            stream = nullptr;
        }

        throw;
    }
}

} // namespace ooverlap
