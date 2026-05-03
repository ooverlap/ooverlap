#include "test/ipc_allreduce_2gpu_sm90.h"

#include "comm/ooverlap_comm_internal.h"
#include "ooverlap/system/logging.h"
#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/testing/test_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <execinfo.h>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ooverlap {
namespace {

static volatile sig_atomic_t g_local_rank = -1;
static volatile sig_atomic_t g_stage_id = 0;
static const char* g_stage_name = "unset";

void debug_log_impl(
    int rank,
    const char* file,
    int line,
    const char* func,
    const char* fmt,
    ...) {
#if OOVERLAP_LOG_LEVEL >= OOVERLAP_LOG_LEVEL_DEBUG
    std::fprintf(
        stderr,
        "[ipc-test][pid=%ld][rank=%d][stage=%d:%s][%s:%d:%s] ",
        static_cast<long>(getpid()),
        rank,
        static_cast<int>(g_stage_id),
        g_stage_name ? g_stage_name : "null",
        file,
        line,
        func);

    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);

    std::fprintf(stderr, "\n");
    std::fflush(stderr);
#else
    (void)rank;
    (void)file;
    (void)line;
    (void)func;
    (void)fmt;
#endif
}

#define IPC_LOG(rank, fmt, ...) \
    debug_log_impl((rank), __FILE__, __LINE__, __func__, (fmt), ##__VA_ARGS__)

#define IPC_STAGE(rank, id, name)                                      \
    do {                                                               \
        g_stage_id = (id);                                             \
        g_stage_name = (name);                                         \
        IPC_LOG((rank), "ENTER %s", (name));                           \
    } while (0)

void segv_handler(int sig) {
    void* trace[64];
    int n = backtrace(trace, 64);

    std::fprintf(
        stderr,
        "\n[ipc-test][pid=%ld][rank=%d] SIGNAL %d at stage=%d:%s\n",
        static_cast<long>(getpid()),
        static_cast<int>(g_local_rank),
        sig,
        static_cast<int>(g_stage_id),
        g_stage_name ? g_stage_name : "null");

    backtrace_symbols_fd(trace, n, STDERR_FILENO);
    std::fflush(stderr);

    std::_Exit(128 + sig);
}

void install_signal_handlers(int local_rank) {
    g_local_rank = local_rank;

    struct sigaction sa {};
    sa.sa_handler = segv_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESETHAND;

    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGABRT, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);
    sigaction(SIGILL, &sa, nullptr);
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

void check_oo(oo_status_t status, const char* what, int rank) {
    IPC_LOG(rank, "%s returned %s", what, oo_status_string(status));
    if (status != OO_SUCCESS) {
        throw std::runtime_error(
            std::string(what) + " failed: " + oo_status_string(status));
    }
}

void check_cuda(cudaError_t err, const char* what, int rank) {
    IPC_LOG(rank, "%s returned %s", what, cudaGetErrorString(err));
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
    IPC_STAGE(local_rank, 100, "fill_local_source");

    system::runtime::set_device(device);

    testing::fill_pattern(
        local_src,
        numel,
        rank_scale(local_rank),
        rank_offset(local_rank),
        stream);

    IPC_LOG(local_rank, "fill_pattern launched local_src=%p numel=%ld",
            static_cast<void*>(local_src),
            static_cast<long>(numel));

    system::runtime::sync_stream_on_device(
        device,
        stream,
        "sync fill_local_source");

    IPC_LOG(local_rank, "fill_local_source done");
}

void reset_work_buffer(
    half* local_work,
    const half* local_src,
    size_t bytes,
    int device,
    cudaStream_t stream,
    int local_rank,
    int iter) {
    IPC_STAGE(local_rank, 200 + iter, "reset_work_buffer");

    IPC_LOG(local_rank,
            "reset_work_buffer iter=%d local_work=%p local_src=%p bytes=%zu",
            iter,
            static_cast<void*>(local_work),
            static_cast<const void*>(local_src),
            bytes);

    system::runtime::set_device(device);

    system::runtime::check_cuda(
        cudaMemcpyAsync(
            local_work,
            local_src,
            bytes,
            cudaMemcpyDeviceToDevice,
            stream),
        "cudaMemcpyAsync(local_src -> local_work)");

    system::runtime::sync_stream_on_device(
        device,
        stream,
        "sync reset_work_buffer");

    IPC_LOG(local_rank, "reset_work_buffer done iter=%d", iter);
}

void verify_local_result(
    const char* label,
    half* local_work,
    int64_t numel,
    int local_rank,
    int device) {
    IPC_STAGE(local_rank, 800, "verify_local_result");

    IPC_LOG(local_rank,
            "verify_local_result local_work=%p numel=%ld device=%d",
            static_cast<void*>(local_work),
            static_cast<long>(numel),
            device);

    auto got = testing::copy_half_device_to_host_float(
        local_work,
        numel,
        device);

    auto ref = reference_two_gpu_sum_fp16(numel);

    testing::expect_allclose(
        got,
        ref,
        (std::string(label) + " rank" + std::to_string(local_rank)).c_str());

    IPC_LOG(local_rank, "verify_local_result done");
}

void dump_buffer_state(
    int rank,
    const char* name,
    oo_buffer_t* buf) {
    if (buf == nullptr) {
        IPC_LOG(rank, "%s=null", name);
        return;
    }

    IPC_LOG(rank,
            "%s ptr=%p bytes=%zu mapped_bytes=%zu kind=%d owner_rank=%d owner_device=%d system_kind=%d mapped.ptr=%p mapped.size=%zu imported.ptr=%p imported.bytes=%zu",
            name,
            buf->ptr,
            buf->bytes,
            buf->mapped_bytes,
            static_cast<int>(buf->kind),
            buf->owner_rank,
            buf->owner_device,
            static_cast<int>(buf->system_kind),
            buf->mapped.ptr,
            buf->mapped.mapped_size,
            buf->imported.ptr,
            buf->imported.bytes);
}

void broker_sync_logged(oo_group_t* group, int rank, const char* label) {
    if (group == nullptr || !group->broker) {
        throw std::runtime_error("broker_sync_logged: group/broker is null");
    }

    IPC_LOG(rank, "broker sync BEGIN: %s", label);
    group->broker->sync();
    IPC_LOG(rank, "broker sync END: %s", label);
}

} // namespace

bool tma_ipc_two_gpu_allreduce_rank_smoke_test(
    int64_t numel,
    int local_rank,
    int dev0,
    int dev1,
    const std::string& broker_key,
    int iters) {
    install_signal_handlers(local_rank);

    IPC_STAGE(local_rank, 1, "entry");
    IPC_LOG(local_rank,
            "args numel=%ld local_rank=%d dev0=%d dev1=%d broker_key=%s iters=%d",
            static_cast<long>(numel),
            local_rank,
            dev0,
            dev1,
            broker_key.c_str(),
            iters);

    if (numel <= 0) {
        throw std::invalid_argument(
            "tma_ipc_two_gpu_allreduce_rank_smoke_test: numel must be > 0");
    }
    if (iters <= 0) {
        throw std::invalid_argument(
            "tma_ipc_two_gpu_allreduce_rank_smoke_test: iters must be > 0");
    }
    if (dev0 == dev1) {
        throw std::invalid_argument(
            "tma_ipc_two_gpu_allreduce_rank_smoke_test: dev0 and dev1 must differ");
    }
    if (local_rank != 0 && local_rank != 1) {
        throw std::invalid_argument(
            "tma_ipc_two_gpu_allreduce_rank_smoke_test: local_rank must be 0 or 1");
    }
    if (broker_key.empty()) {
        throw std::invalid_argument(
            "tma_ipc_two_gpu_allreduce_rank_smoke_test: broker_key must be non-empty");
    }

    int devices[2] = {dev0, dev1};
    const int peer_rank = local_rank ^ 1;
    const size_t bytes = static_cast<size_t>(numel) * sizeof(half);

    oo_group_t* group = nullptr;
    oo_node_t* node = nullptr;
    oo_buffer_t* local_buf = nullptr;
    oo_buffer_t* peer_buf = nullptr;

    half* local_src = nullptr;
    half* local_work = nullptr;

    cudaStream_t stream = nullptr;

    try {
        IPC_STAGE(local_rank, 10, "oo_group_create_ipc begin");
        oo_status_t st = oo_group_create_ipc(
            devices,
            2,
            local_rank,
            broker_key.c_str(),
            &group);
        check_oo(st, "oo_group_create_ipc", local_rank);

        IPC_LOG(local_rank,
                "group created group=%p broker=%p num_devices=%d local_rank=%d",
                static_cast<void*>(group),
                group ? static_cast<void*>(group->broker.get()) : nullptr,
                group ? group->num_devices : -1,
                group ? group->local_rank : -1);

        IPC_STAGE(local_rank, 20, "oo_node_create");
        st = oo_node_create(
            group,
            local_rank,
            &node);
        check_oo(st, "oo_node_create(local)", local_rank);

        const int local_device = oo_node_device(node);
        IPC_LOG(local_rank,
                "node=%p local_device=%d peer_rank=%d peer_device=%d",
                static_cast<void*>(node),
                local_device,
                peer_rank,
                devices[peer_rank]);

        IPC_STAGE(local_rank, 30, "create stream");
        stream = system::runtime::create_stream_on_device(local_device);
        IPC_LOG(local_rank, "stream=%p", reinterpret_cast<void*>(stream));

        IPC_STAGE(local_rank, 40, "allocate local_src cudaMalloc");
        system::runtime::set_device(local_device);
        check_cuda(cudaMalloc(&local_src, bytes), "cudaMalloc(local_src)", local_rank);
        IPC_LOG(local_rank, "local_src=%p bytes=%zu", static_cast<void*>(local_src), bytes);

        IPC_STAGE(local_rank, 50, "oo_buffer_alloc local VMM buffer");
        st = oo_buffer_alloc(
            node,
            bytes,
            &local_buf);
        check_oo(st, "oo_buffer_alloc(local VMM work buffer)", local_rank);

        local_work = reinterpret_cast<half*>(oo_buffer_ptr(local_buf));
        if (local_work == nullptr) {
            throw std::runtime_error("oo_buffer_ptr(local_buf) returned null");
        }

        dump_buffer_state(local_rank, "local_buf after alloc", local_buf);

        IPC_STAGE(local_rank, 60, "fill local source");
        fill_local_source(
            local_src,
            numel,
            local_rank,
            local_device,
            stream);

        IPC_STAGE(local_rank, 70, "make local VMM descriptor");
        std::vector<int> access_devices = {dev0, dev1};

        ooverlap::system::vmm_peer_buffer_descriptor local_desc =
            ooverlap::system::make_vmm_peer_buffer_descriptor(
                local_buf->mapped,
                bytes);

        IPC_LOG(local_rank,
                "local_desc bytes=%llu mapped_size=%llu owner_device=%d",
                static_cast<unsigned long long>(local_desc.bytes),
                static_cast<unsigned long long>(local_desc.mapped_size),
                local_desc.owner_device);

        IPC_STAGE(local_rank, 80, "export local VMM fd");
        ooverlap::system::ipc::vmm_handle local_fd =
            ooverlap::system::export_vmm_peer_buffer_fd(
                local_buf->mapped);

        IPC_LOG(local_rank, "local_fd.value=%d", local_fd.value);

        IPC_STAGE(local_rank, 90, "broker exchange descriptors");
        std::vector<ooverlap::system::vmm_peer_buffer_descriptor> all_desc(2);

        IPC_LOG(local_rank, "exchange_data desc BEGIN");
        group->broker->exchange_data(
            all_desc.data(),
            &local_desc,
            sizeof(local_desc));
        IPC_LOG(local_rank, "exchange_data desc END");

        for (int r = 0; r < 2; ++r) {
            IPC_LOG(local_rank,
                    "all_desc[%d] bytes=%llu mapped_size=%llu owner_device=%d",
                    r,
                    static_cast<unsigned long long>(all_desc[r].bytes),
                    static_cast<unsigned long long>(all_desc[r].mapped_size),
                    all_desc[r].owner_device);
        }

        IPC_STAGE(local_rank, 95, "broker exchange fds");
        std::vector<int> all_fds(2, -1);

        IPC_LOG(local_rank, "exchange_fds BEGIN src_fd=%d", local_fd.value);
        group->broker->exchange_fds(
            all_fds.data(),
            local_fd.value);
        IPC_LOG(local_rank,
                "exchange_fds END all_fds[0]=%d all_fds[1]=%d",
                all_fds[0],
                all_fds[1]);

        local_fd.value = -1;

        if (all_fds[peer_rank] < 0) {
            throw std::runtime_error("exchange_fds did not return peer fd");
        }

        IPC_STAGE(local_rank, 110, "import peer VMM buffer");
        ooverlap::system::imported_peer_buffer imported_peer =
            ooverlap::system::import_vmm_peer_buffer(
                all_fds[peer_rank],
                all_desc[peer_rank],
                access_devices);

        IPC_LOG(local_rank,
                "imported_peer ptr=%p bytes=%zu mapped_size=%zu owner_device=%d kind=%d",
                imported_peer.ptr,
                imported_peer.bytes,
                imported_peer.mapped_size,
                imported_peer.owner_device,
                static_cast<int>(imported_peer.kind));

        IPC_STAGE(local_rank, 120, "adopt imported peer buffer");
        st = oo_buffer_adopt_imported_peer_buffer(
            node,
            std::move(imported_peer),
            &peer_buf);
        check_oo(st, "oo_buffer_adopt_imported_peer_buffer(peer VMM)", local_rank);

        dump_buffer_state(local_rank, "peer_buf after adopt", peer_buf);

        IPC_STAGE(local_rank, 130, "post peer import broker sync");
        broker_sync_logged(group, local_rank, "after peer import");

        for (int iter = 0; iter < iters; ++iter) {
            IPC_STAGE(local_rank, 2000 + iter * 100, "iteration begin");
            IPC_LOG(local_rank, "ITER %d begin", iter);

            reset_work_buffer(
                local_work,
                local_src,
                bytes,
                local_device,
                stream,
                local_rank,
                iter);

            IPC_STAGE(local_rank, 2010 + iter * 100, "pre allreduce broker sync");
            broker_sync_logged(group, local_rank, "before oo_allreduce");

            IPC_STAGE(local_rank, 2020 + iter * 100, "oo_allreduce launch");
            IPC_LOG(local_rank,
                    "calling oo_allreduce iter=%d local_buf=%p peer_buf=%p local_ptr=%p peer_ptr=%p",
                    iter,
                    static_cast<void*>(local_buf),
                    static_cast<void*>(peer_buf),
                    oo_buffer_ptr(local_buf),
                    oo_buffer_ptr(peer_buf));

            oo_buffer_t* peer_bufs[] = {
                peer_buf
            };

            st = oo_allreduce(
                node,
                local_buf,
                peer_bufs,
                1,
                static_cast<size_t>(numel),
                OO_DTYPE_FLOAT16,
                OO_REDUCE_SUM,
                stream);

            check_oo(st, "oo_allreduce(ipc)", local_rank);

            cudaError_t peek = cudaPeekAtLastError();
            check_cuda(peek, "cudaPeekAtLastError(after oo_allreduce)", local_rank);

            IPC_STAGE(local_rank, 2030 + iter * 100, "sync oo_allreduce stream");
            IPC_LOG(local_rank, "stream sync begin iter=%d", iter);

            system::runtime::sync_stream_on_device(
                local_device,
                stream,
                "sync oo_allreduce(ipc)");

            IPC_LOG(local_rank, "stream sync end iter=%d", iter);

            IPC_STAGE(local_rank, 2040 + iter * 100, "verify result");
            verify_local_result(
                "ipc oo_allreduce",
                local_work,
                numel,
                local_rank,
                local_device);

            IPC_STAGE(local_rank, 2050 + iter * 100, "post verify broker sync");
            broker_sync_logged(group, local_rank, "after verify");

            IPC_LOG(local_rank, "ITER %d end", iter);
        }

        IPC_STAGE(local_rank, 9000, "cleanup begin");
        broker_sync_logged(group, local_rank, "cleanup before peer destroy");

        IPC_LOG(local_rank, "destroy peer_buf");
        oo_buffer_destroy(peer_buf);
        peer_buf = nullptr;

        broker_sync_logged(group, local_rank, "cleanup before local destroy");

        IPC_LOG(local_rank, "destroy local_buf");
        oo_buffer_destroy(local_buf);
        local_buf = nullptr;
        local_work = nullptr;

        IPC_LOG(local_rank, "free local_src");
        system::runtime::set_device(local_device);
        check_cuda(cudaFree(local_src), "cudaFree(local_src)", local_rank);
        local_src = nullptr;

        IPC_LOG(local_rank, "destroy node");
        oo_node_destroy(node);
        node = nullptr;

        IPC_LOG(local_rank, "destroy group");
        oo_group_destroy(group);
        group = nullptr;

        IPC_LOG(local_rank, "destroy stream");
        system::runtime::destroy_stream_on_device(
            local_device,
            stream);
        stream = nullptr;

        IPC_STAGE(local_rank, 9999, "success");
        IPC_LOG(local_rank, "SUCCESS");
        return true;

    } catch (...) {
        IPC_LOG(local_rank, "exception path cleanup begin");

        if (peer_buf != nullptr) {
            IPC_LOG(local_rank, "exception cleanup destroy peer_buf");
            oo_buffer_destroy(peer_buf);
            peer_buf = nullptr;
        }

        if (local_buf != nullptr) {
            IPC_LOG(local_rank, "exception cleanup destroy local_buf");
            oo_buffer_destroy(local_buf);
            local_buf = nullptr;
            local_work = nullptr;
        }

        if (local_src != nullptr) {
            int device = (node != nullptr) ? oo_node_device(node) : devices[local_rank];
            cudaSetDevice(device);
            IPC_LOG(local_rank, "exception cleanup cudaFree local_src=%p", static_cast<void*>(local_src));
            cudaFree(local_src);
            local_src = nullptr;
        }

        if (node != nullptr) {
            IPC_LOG(local_rank, "exception cleanup destroy node");
            oo_node_destroy(node);
            node = nullptr;
        }

        if (group != nullptr) {
            IPC_LOG(local_rank, "exception cleanup destroy group");
            oo_group_destroy(group);
            group = nullptr;
        }

        if (stream != nullptr) {
            int device = devices[local_rank];
            IPC_LOG(local_rank, "exception cleanup destroy stream");
            system::runtime::destroy_stream_on_device(device, stream);
            stream = nullptr;
        }

        IPC_LOG(local_rank, "exception path cleanup end");
        throw;
    }
}

} // namespace ooverlap
