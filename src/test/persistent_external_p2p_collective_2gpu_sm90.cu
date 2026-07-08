#include "test/persistent_external_p2p_collective_2gpu_sm90.h"

#include "ooverlap/comm.h"
#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/testing/checks.cuh"
#include "ooverlap/testing/collective_test_utils.cuh"
#include "ooverlap/testing/two_gpu_test_utils.cuh"

#include "test/internal_comm_test_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#include <map>
#include <stdexcept>
#include <string>

#ifndef OOVERLAP_BENCH_VERIFY_RESULTS
#define OOVERLAP_BENCH_VERIFY_RESULTS 0
#endif

#ifndef OOVERLAP_BENCH_TRY_NCCL_REGISTERED
#define OOVERLAP_BENCH_TRY_NCCL_REGISTERED 1
#endif

namespace ooverlap {
namespace {

using testing::TestCollective;

struct NcclRegistrationApi {
    using MemAllocFn = ncclResult_t (*)(void**, size_t);
    using MemFreeFn = ncclResult_t (*)(void*);
    using CommRegisterFn = ncclResult_t (*)(const ncclComm_t, void*, size_t, void**);
    using CommDeregisterFn = ncclResult_t (*)(const ncclComm_t, void*);

    MemAllocFn mem_alloc = nullptr;
    MemFreeFn mem_free = nullptr;
    CommRegisterFn comm_register = nullptr;
    CommDeregisterFn comm_deregister = nullptr;
};

struct NcclRegisteredBuffers {
    half* rank0 = nullptr;
    half* rank1 = nullptr;
    void* rank0_handle = nullptr;
    void* rank1_handle = nullptr;
    bool enabled = false;
};

NcclRegistrationApi load_nccl_registration_api() {
    NcclRegistrationApi api{};

#if OOVERLAP_BENCH_TRY_NCCL_REGISTERED
    api.mem_alloc =
        reinterpret_cast<NcclRegistrationApi::MemAllocFn>(
            dlsym(RTLD_DEFAULT, "ncclMemAlloc"));
    api.mem_free =
        reinterpret_cast<NcclRegistrationApi::MemFreeFn>(
            dlsym(RTLD_DEFAULT, "ncclMemFree"));
    api.comm_register =
        reinterpret_cast<NcclRegistrationApi::CommRegisterFn>(
            dlsym(RTLD_DEFAULT, "ncclCommRegister"));
    api.comm_deregister =
        reinterpret_cast<NcclRegistrationApi::CommDeregisterFn>(
            dlsym(RTLD_DEFAULT, "ncclCommDeregister"));
#endif

    return api;
}

bool nccl_registration_api_available(
    const NcclRegistrationApi& api) {
    return api.mem_alloc != nullptr &&
           api.mem_free != nullptr &&
           api.comm_register != nullptr &&
           api.comm_deregister != nullptr;
}

bool optional_nccl_ok(
    ncclResult_t result,
    const char* what) {
    if (result == ncclSuccess) {
        return true;
    }

    std::fprintf(
        stderr,
        "[warn] %s failed: %s\n",
        what,
        ncclGetErrorString(result));
    std::fflush(stderr);
    return false;
}

void free_nccl_registered_buffers(
    const NcclRegistrationApi& api,
    int dev0,
    int dev1,
    ncclComm_t* comms,
    NcclRegisteredBuffers& buffers) {
    if (api.comm_deregister != nullptr && comms != nullptr) {
        if (buffers.rank0_handle != nullptr && comms[0] != nullptr) {
            optional_nccl_ok(
                api.comm_deregister(comms[0], buffers.rank0_handle),
                "ncclCommDeregister(rank0 registered)");
            buffers.rank0_handle = nullptr;
        }

        if (buffers.rank1_handle != nullptr && comms[1] != nullptr) {
            optional_nccl_ok(
                api.comm_deregister(comms[1], buffers.rank1_handle),
                "ncclCommDeregister(rank1 registered)");
            buffers.rank1_handle = nullptr;
        }
    }

    if (api.mem_free != nullptr) {
        if (buffers.rank0 != nullptr) {
            system::runtime::set_device(dev0);
            optional_nccl_ok(
                api.mem_free(buffers.rank0),
                "ncclMemFree(rank0 registered)");
            buffers.rank0 = nullptr;
        }

        if (buffers.rank1 != nullptr) {
            system::runtime::set_device(dev1);
            optional_nccl_ok(
                api.mem_free(buffers.rank1),
                "ncclMemFree(rank1 registered)");
            buffers.rank1 = nullptr;
        }
    }

    buffers.enabled = false;
}

bool try_create_nccl_registered_buffers(
    const NcclRegistrationApi& api,
    int dev0,
    int dev1,
    size_t bytes,
    ncclComm_t* comms,
    NcclRegisteredBuffers* out) {
    if (out == nullptr) {
        return false;
    }

    *out = NcclRegisteredBuffers{};

    if (!nccl_registration_api_available(api)) {
        std::fprintf(
            stderr,
            "[warn] NCCL registered-buffer path unavailable: missing "
            "ncclMemAlloc/ncclMemFree/ncclCommRegister/ncclCommDeregister. "
            "Skipping nccl_registered_ms.\n");
        std::fflush(stderr);
        return false;
    }

    if (comms == nullptr || comms[0] == nullptr || comms[1] == nullptr) {
        std::fprintf(
            stderr,
            "[warn] NCCL registered-buffer path unavailable: communicators are null. "
            "Skipping nccl_registered_ms.\n");
        std::fflush(stderr);
        return false;
    }

    system::runtime::set_device(dev0);
    if (!optional_nccl_ok(
            api.mem_alloc(reinterpret_cast<void**>(&out->rank0), bytes),
            "ncclMemAlloc(rank0 registered)")) {
        free_nccl_registered_buffers(api, dev0, dev1, comms, *out);
        return false;
    }

    system::runtime::set_device(dev1);
    if (!optional_nccl_ok(
            api.mem_alloc(reinterpret_cast<void**>(&out->rank1), bytes),
            "ncclMemAlloc(rank1 registered)")) {
        free_nccl_registered_buffers(api, dev0, dev1, comms, *out);
        return false;
    }

    if (!optional_nccl_ok(
            api.comm_register(
                comms[0],
                out->rank0,
                bytes,
                &out->rank0_handle),
            "ncclCommRegister(rank0 registered)")) {
        free_nccl_registered_buffers(api, dev0, dev1, comms, *out);
        return false;
    }

    if (!optional_nccl_ok(
            api.comm_register(
                comms[1],
                out->rank1,
                bytes,
                &out->rank1_handle),
            "ncclCommRegister(rank1 registered)")) {
        free_nccl_registered_buffers(api, dev0, dev1, comms, *out);
        return false;
    }

    out->enabled = true;
    return true;
}

void launch_ooverlap_public_once_for_rank(
    TestCollective collective,
    oo_node_t* node,
    oo_buffer_t* local,
    size_t numel,
    cudaStream_t stream,
    const char* label) {
    if (collective == TestCollective::AllReduce) {
        testing::check_oo(
            oo_allreduce_tuned(
                node,
                local,
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
                local,
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
                local,
                numel,
                OO_DTYPE_FLOAT16,
                OO_TUNING_BEST_PERFORMANCE,
                stream),
            label);
        return;
    }

    throw std::invalid_argument("unsupported collective");
}

void launch_ooverlap_public_once(
    TestCollective collective,
    oo_node_t* node0,
    oo_node_t* node1,
    oo_buffer_t* rank0_buf,
    oo_buffer_t* rank1_buf,
    size_t numel,
    cudaStream_t stream0,
    cudaStream_t stream1) {
    launch_ooverlap_public_once_for_rank(
        collective,
        node0,
        rank0_buf,
        numel,
        stream0,
        "ooverlap rank0");

    launch_ooverlap_public_once_for_rank(
        collective,
        node1,
        rank1_buf,
        numel,
        stream1,
        "ooverlap rank1");
}

void run_ooverlap_public_iters(
    TestCollective collective,
    oo_group_t* group,
    oo_node_t* node0,
    oo_node_t* node1,
    oo_buffer_t* rank0_buf,
    oo_buffer_t* rank1_buf,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int iters) {
    if (iters <= 0) {
        return;
    }

    testing::reset_ready_signals(group);

    for (int i = 0; i < iters; ++i) {
        launch_ooverlap_public_once(
            collective,
            node0,
            node1,
            rank0_buf,
            rank1_buf,
            numel,
            stream0,
            stream1);
    }

    testing::sync_two_streams(
        dev0,
        stream0,
        dev1,
        stream1,
        "sync ooverlap external p2p warmup");
}

double elapsed_ms_ooverlap_public(
    TestCollective collective,
    oo_group_t* group,
    oo_node_t* node0,
    oo_node_t* node1,
    oo_buffer_t* rank0_buf,
    oo_buffer_t* rank1_buf,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int iters) {
    testing::reset_ready_signals(group);

    return testing::elapsed_ms_two_stream_max(
        dev0,
        stream0,
        dev1,
        stream1,
        iters,
        [&](int) {
            launch_ooverlap_public_once(
                collective,
                node0,
                node1,
                rank0_buf,
                rank1_buf,
                numel,
                stream0,
                stream1);
        });
}

void launch_nccl_once(
    TestCollective collective,
    half* rank0_buf,
    half* rank1_buf,
    size_t numel,
    cudaStream_t stream0,
    cudaStream_t stream1,
    ncclComm_t* comms) {
    OOVERLAP_TEST_NCCL_CHECK(ncclGroupStart());

    testing::launch_nccl_collective_fp16(
        collective,
        comms[0],
        rank0_buf,
        numel,
        0,
        2,
        stream0);

    testing::launch_nccl_collective_fp16(
        collective,
        comms[1],
        rank1_buf,
        numel,
        1,
        2,
        stream1);

    OOVERLAP_TEST_NCCL_CHECK(ncclGroupEnd());
}

void run_nccl_iters(
    TestCollective collective,
    half* rank0_buf,
    half* rank1_buf,
    size_t numel,
    cudaStream_t stream0,
    cudaStream_t stream1,
    ncclComm_t* comms,
    int iters) {
    if (iters <= 0) {
        return;
    }

    for (int i = 0; i < iters; ++i) {
        launch_nccl_once(
            collective,
            rank0_buf,
            rank1_buf,
            numel,
            stream0,
            stream1,
            comms);
    }
}

double elapsed_ms_nccl(
    TestCollective collective,
    half* rank0_buf,
    half* rank1_buf,
    size_t numel,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    ncclComm_t* comms,
    int iters) {
    return testing::elapsed_ms_two_stream_max(
        dev0,
        stream0,
        dev1,
        stream1,
        iters,
        [&](int) {
            launch_nccl_once(
                collective,
                rank0_buf,
                rank1_buf,
                numel,
                stream0,
                stream1,
                comms);
        });
}

void verify_collective_result(
    TestCollective collective,
    const char* label,
    half* rank0,
    half* rank1,
    int64_t numel,
    int dev0,
    int dev1) {
#if OOVERLAP_BENCH_VERIFY_RESULTS
    testing::verify_collective_fp16(
        collective,
        label,
        rank0,
        numel,
        0,
        2,
        dev0);

    testing::verify_collective_fp16(
        collective,
        label,
        rank1,
        numel,
        1,
        2,
        dev1);
#else
    (void)collective;
    (void)label;
    (void)rank0;
    (void)rank1;
    (void)numel;
    (void)dev0;
    (void)dev1;
#endif
}

void bench_ooverlap_external(
    std::map<std::string, double>& results,
    TestCollective collective,
    oo_group_t* group,
    oo_node_t* node0,
    oo_node_t* node1,
    oo_buffer_t* rank0_buf,
    oo_buffer_t* rank1_buf,
    const half* rank0_src,
    const half* rank1_src,
    half* rank0_work,
    half* rank1_work,
    size_t numel,
    size_t bytes,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    int iters,
    int warmup) {
    testing::prepare_two_work_buffers(
        rank0_src,
        rank1_src,
        rank0_work,
        rank1_work,
        bytes,
        dev0,
        dev1,
        stream0,
        stream1);

    run_ooverlap_public_iters(
        collective,
        group,
        node0,
        node1,
        rank0_buf,
        rank1_buf,
        numel,
        dev0,
        dev1,
        stream0,
        stream1,
        warmup);

    testing::prepare_two_work_buffers(
        rank0_src,
        rank1_src,
        rank0_work,
        rank1_work,
        bytes,
        dev0,
        dev1,
        stream0,
        stream1);

    const double total_ms =
        elapsed_ms_ooverlap_public(
            collective,
            group,
            node0,
            node1,
            rank0_buf,
            rank1_buf,
            numel,
            dev0,
            dev1,
            stream0,
            stream1,
            iters);

    testing::sync_two_streams(
        dev0,
        stream0,
        dev1,
        stream1,
        "sync ooverlap external p2p measured");

    verify_collective_result(
        collective,
        "ooverlap",
        rank0_work,
        rank1_work,
        static_cast<int64_t>(numel),
        dev0,
        dev1);

    results["ooverlap_ms"] =
        total_ms / static_cast<double>(iters);
}

void bench_nccl_external(
    std::map<std::string, double>& results,
    const char* result_key,
    TestCollective collective,
    const half* rank0_src,
    const half* rank1_src,
    half* rank0_work,
    half* rank1_work,
    size_t numel,
    size_t bytes,
    int dev0,
    int dev1,
    cudaStream_t stream0,
    cudaStream_t stream1,
    ncclComm_t* comms,
    int iters,
    int warmup) {
    testing::prepare_two_work_buffers(
        rank0_src,
        rank1_src,
        rank0_work,
        rank1_work,
        bytes,
        dev0,
        dev1,
        stream0,
        stream1);

    run_nccl_iters(
        collective,
        rank0_work,
        rank1_work,
        numel,
        stream0,
        stream1,
        comms,
        warmup);

    testing::sync_two_streams(
        dev0,
        stream0,
        dev1,
        stream1,
        "sync nccl external p2p warmup");

    testing::prepare_two_work_buffers(
        rank0_src,
        rank1_src,
        rank0_work,
        rank1_work,
        bytes,
        dev0,
        dev1,
        stream0,
        stream1);

    const double total_ms =
        elapsed_ms_nccl(
            collective,
            rank0_work,
            rank1_work,
            numel,
            dev0,
            dev1,
            stream0,
            stream1,
            comms,
            iters);

    testing::sync_two_streams(
        dev0,
        stream0,
        dev1,
        stream1,
        "sync nccl external p2p measured");

    verify_collective_result(
        collective,
        result_key,
        rank0_work,
        rank1_work,
        static_cast<int64_t>(numel),
        dev0,
        dev1);

    results[result_key] =
        total_ms / static_cast<double>(iters);
}

void cleanup(
    const NcclRegistrationApi& nccl_registration_api,
    int dev0,
    int dev1,
    half*& rank0_src,
    half*& rank1_src,
    half*& ooverlap_rank0,
    half*& ooverlap_rank1,
    half*& nccl_rank0,
    half*& nccl_rank1,
    NcclRegisteredBuffers& nccl_registered,
    oo_buffer_t*& ooverlap_rank0_buf,
    oo_buffer_t*& ooverlap_rank1_buf,
    oo_node_t*& node0,
    oo_node_t*& node1,
    oo_group_t*& group,
    cudaStream_t& stream0,
    cudaStream_t& stream1,
    ncclComm_t* comms) {
    /*
     * Registered buffers must be deregistered before communicators are destroyed.
     */
    free_nccl_registered_buffers(
        nccl_registration_api,
        dev0,
        dev1,
        comms,
        nccl_registered);

    testing::destroy_nccl_comms(comms, 2);

    /*
     * These are non-owning wrappers around external cudaMalloc buffers.
     * Destroy wrappers before freeing the external memory.
     */
    testing::destroy_oo_buffer(ooverlap_rank0_buf);
    testing::destroy_oo_buffer(ooverlap_rank1_buf);

    testing::cuda_free_on_device(dev0, rank0_src);
    testing::cuda_free_on_device(dev1, rank1_src);

    testing::cuda_free_on_device(dev0, ooverlap_rank0);
    testing::cuda_free_on_device(dev1, ooverlap_rank1);

    testing::cuda_free_on_device(dev0, nccl_rank0);
    testing::cuda_free_on_device(dev1, nccl_rank1);

    testing::destroy_oo_node(node0);
    testing::destroy_oo_node(node1);
    testing::destroy_oo_group(group);

    testing::destroy_stream_on_device(dev0, stream0);
    testing::destroy_stream_on_device(dev1, stream1);
}

} // namespace

bool external_p2p_two_gpu_collective_smoke_test(
    const std::string& collective_name_arg,
    int64_t numel,
    int dev0,
    int dev1) {
    std::map<std::string, double> result =
        benchmark_external_p2p_two_gpu_collective_sm90(
            collective_name_arg,
            numel,
            1,
            0,
            dev0,
            dev1);

    return !result.empty();
}

bool external_p2p_two_gpu_allreduce_smoke_test(
    int64_t numel,
    int dev0,
    int dev1) {
    return external_p2p_two_gpu_collective_smoke_test(
        "allreduce",
        numel,
        dev0,
        dev1);
}

std::map<std::string, double> benchmark_external_p2p_two_gpu_collective_sm90(
    const std::string& collective_name_arg,
    int64_t numel_arg,
    int iters,
    int warmup,
    int dev0,
    int dev1) {
    if (numel_arg <= 0 || iters <= 0 || warmup < 0) {
        throw std::invalid_argument(
            "benchmark_external_p2p_two_gpu_collective_sm90: invalid args");
    }

    if (dev0 == dev1) {
        throw std::invalid_argument(
            "benchmark_external_p2p_two_gpu_collective_sm90: dev0 and dev1 must differ");
    }

    const TestCollective collective =
        testing::parse_collective(collective_name_arg);

    testing::validate_numel_for_collective(
        collective,
        numel_arg,
        2);

    const size_t numel =
        static_cast<size_t>(numel_arg);

    const size_t bytes =
        numel * sizeof(half);

    const NcclRegistrationApi nccl_registration_api =
        load_nccl_registration_api();

    oo_group_t* group = nullptr;
    oo_node_t* node0 = nullptr;
    oo_node_t* node1 = nullptr;

    oo_buffer_t* ooverlap_rank0_buf = nullptr;
    oo_buffer_t* ooverlap_rank1_buf = nullptr;

    half* rank0_src = nullptr;
    half* rank1_src = nullptr;

    /*
     * External/user-owned work buffers.
     * ooverlap only wraps these pointers; it does not allocate or free them.
     */
    half* ooverlap_rank0 = nullptr;
    half* ooverlap_rank1 = nullptr;

    /*
     * Standard NCCL comparison buffers.
     * These use plain cudaMalloc.
     */
    half* nccl_rank0 = nullptr;
    half* nccl_rank1 = nullptr;

    /*
     * Optional NCCL symmetric/registered comparison buffers.
     * These use ncclMemAlloc + ncclCommRegister when the runtime exposes those
     * symbols and registration succeeds for the current communicator/topology.
     */
    NcclRegisteredBuffers nccl_registered{};

    cudaStream_t stream0 = nullptr;
    cudaStream_t stream1 = nullptr;

    ncclComm_t comms[2] = {
        nullptr,
        nullptr,
    };

    try {
        int devices[2] = {dev0, dev1};

        testing::check_oo(
            oo_group_create_p2p(
                devices,
                2,
                &group),
            "oo_group_create_p2p");

        testing::check_oo(
            oo_node_create(
                group,
                0,
                &node0),
            "oo_node_create(rank0)");

        testing::check_oo(
            oo_node_create(
                group,
                1,
                &node1),
            "oo_node_create(rank1)");

        const int node0_dev =
            oo_node_device(node0);

        const int node1_dev =
            oo_node_device(node1);

        stream0 =
            system::runtime::create_stream_on_device(node0_dev);

        stream1 =
            system::runtime::create_stream_on_device(node1_dev);

        testing::cuda_malloc_half_on_device(
            node0_dev,
            &rank0_src,
            bytes,
            "cudaMalloc(rank0_src)");

        testing::cuda_malloc_half_on_device(
            node1_dev,
            &rank1_src,
            bytes,
            "cudaMalloc(rank1_src)");

        testing::cuda_malloc_half_on_device(
            node0_dev,
            &ooverlap_rank0,
            bytes,
            "cudaMalloc(ooverlap_rank0 external)");

        testing::cuda_malloc_half_on_device(
            node1_dev,
            &ooverlap_rank1,
            bytes,
            "cudaMalloc(ooverlap_rank1 external)");

        testing::cuda_malloc_half_on_device(
            node0_dev,
            &nccl_rank0,
            bytes,
            "cudaMalloc(nccl_rank0)");

        testing::cuda_malloc_half_on_device(
            node1_dev,
            &nccl_rank1,
            bytes,
            "cudaMalloc(nccl_rank1)");

        testing::fill_two_rank_sources_fp16(
            rank0_src,
            rank1_src,
            numel_arg,
            node0_dev,
            node1_dev,
            stream0,
            stream1);

        OOVERLAP_TEST_NCCL_CHECK(
            ncclCommInitAll(
                comms,
                2,
                devices));

        const bool have_nccl_registered =
            try_create_nccl_registered_buffers(
                nccl_registration_api,
                node0_dev,
                node1_dev,
                bytes,
                comms,
                &nccl_registered);

        std::map<std::string, double> results;

        testing::check_oo(
            oo_buffer_wrap(
                node0,
                ooverlap_rank0,
                bytes,
                &ooverlap_rank0_buf),
            "oo_buffer_wrap(ooverlap rank0)");

        testing::check_oo(
            oo_buffer_wrap(
                node1,
                ooverlap_rank1,
                bytes,
                &ooverlap_rank1_buf),
            "oo_buffer_wrap(ooverlap rank1)");

        bench_ooverlap_external(
            results,
            collective,
            group,
            node0,
            node1,
            ooverlap_rank0_buf,
            ooverlap_rank1_buf,
            rank0_src,
            rank1_src,
            ooverlap_rank0,
            ooverlap_rank1,
            numel,
            bytes,
            node0_dev,
            node1_dev,
            stream0,
            stream1,
            iters,
            warmup);

        testing::destroy_oo_buffer(ooverlap_rank0_buf);
        testing::destroy_oo_buffer(ooverlap_rank1_buf);

        bench_nccl_external(
            results,
            "nccl_ms",
            collective,
            rank0_src,
            rank1_src,
            nccl_rank0,
            nccl_rank1,
            numel,
            bytes,
            node0_dev,
            node1_dev,
            stream0,
            stream1,
            comms,
            iters,
            warmup);

        if (have_nccl_registered) {
            bench_nccl_external(
                results,
                "nccl_registered_ms",
                collective,
                rank0_src,
                rank1_src,
                nccl_registered.rank0,
                nccl_registered.rank1,
                numel,
                bytes,
                node0_dev,
                node1_dev,
                stream0,
                stream1,
                comms,
                iters,
                warmup);
        }

        results["collective"] =
            testing::collective_code(collective);

        results["numel"] =
            static_cast<double>(numel);

        results["bytes"] =
            static_cast<double>(bytes);

        results["iters"] =
            static_cast<double>(iters);

        results["warmup"] =
            static_cast<double>(warmup);

        cleanup(
            nccl_registration_api,
            node0_dev,
            node1_dev,
            rank0_src,
            rank1_src,
            ooverlap_rank0,
            ooverlap_rank1,
            nccl_rank0,
            nccl_rank1,
            nccl_registered,
            ooverlap_rank0_buf,
            ooverlap_rank1_buf,
            node0,
            node1,
            group,
            stream0,
            stream1,
            comms);

        return results;
    } catch (...) {
        const int node0_dev =
            node0 != nullptr ? oo_node_device(node0) : dev0;

        const int node1_dev =
            node1 != nullptr ? oo_node_device(node1) : dev1;

        cleanup(
            nccl_registration_api,
            node0_dev,
            node1_dev,
            rank0_src,
            rank1_src,
            ooverlap_rank0,
            ooverlap_rank1,
            nccl_rank0,
            nccl_rank1,
            nccl_registered,
            ooverlap_rank0_buf,
            ooverlap_rank1_buf,
            node0,
            node1,
            group,
            stream0,
            stream1,
            comms);

        throw;
    }
}

std::map<std::string, double> benchmark_external_p2p_two_gpu_allreduce_sm90(
    int64_t numel,
    int iters,
    int warmup,
    int dev0,
    int dev1) {
    return benchmark_external_p2p_two_gpu_collective_sm90(
        "allreduce",
        numel,
        iters,
        warmup,
        dev0,
        dev1);
}

} // namespace ooverlap
