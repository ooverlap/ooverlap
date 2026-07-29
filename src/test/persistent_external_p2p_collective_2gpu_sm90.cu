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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
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

void validate_devices(const std::vector<int>& devices) {
    if (devices.size() < 2) {
        throw std::invalid_argument(
            "external P2P collective requires at least two devices");
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

void destroy_events_best_effort(
    const std::vector<int>& devices,
    std::vector<cudaEvent_t>& events) {
    const std::size_t count = std::min(devices.size(), events.size());
    for (std::size_t rank = 0; rank < count; ++rank) {
        if (events[rank] != nullptr) {
            (void)cudaSetDevice(devices[rank]);
            (void)cudaEventDestroy(events[rank]);
            events[rank] = nullptr;
        }
    }
}

template <typename LaunchOnce>
double elapsed_ms_stream_max(
    const std::vector<int>& devices,
    const std::vector<cudaStream_t>& streams,
    int iters,
    LaunchOnce&& launch_once) {
    if (devices.size() != streams.size() || devices.empty()) {
        throw std::invalid_argument("elapsed_ms_stream_max: invalid streams");
    }
    if (iters <= 0) {
        throw std::invalid_argument("elapsed_ms_stream_max: iters must be > 0");
    }

    const std::size_t world_size = devices.size();
    std::vector<cudaEvent_t> starts(world_size, nullptr);
    std::vector<cudaEvent_t> stops(world_size, nullptr);

    try {
        for (std::size_t rank = 0; rank < world_size; ++rank) {
            system::runtime::set_device(devices[rank]);
            testing::check_cuda(
                cudaEventCreate(&starts[rank]),
                "cudaEventCreate(start)");
            testing::check_cuda(
                cudaEventCreate(&stops[rank]),
                "cudaEventCreate(stop)");
            testing::check_cuda(
                cudaEventRecord(starts[rank], streams[rank]),
                "cudaEventRecord(start)");
        }

        for (int i = 0; i < iters; ++i) {
            launch_once(i);
        }

        for (std::size_t rank = 0; rank < world_size; ++rank) {
            system::runtime::set_device(devices[rank]);
            testing::check_cuda(
                cudaEventRecord(stops[rank], streams[rank]),
                "cudaEventRecord(stop)");
        }

        double max_ms = 0.0;
        for (std::size_t rank = 0; rank < world_size; ++rank) {
            system::runtime::set_device(devices[rank]);
            testing::check_cuda(
                cudaEventSynchronize(stops[rank]),
                "cudaEventSynchronize(stop)");

            float rank_ms = 0.0f;
            testing::check_cuda(
                cudaEventElapsedTime(
                    &rank_ms,
                    starts[rank],
                    stops[rank]),
                "cudaEventElapsedTime");
            max_ms = std::max(max_ms, static_cast<double>(rank_ms));
        }

        destroy_events_best_effort(devices, starts);
        destroy_events_best_effort(devices, stops);
        return max_ms;
    } catch (...) {
        destroy_events_best_effort(devices, starts);
        destroy_events_best_effort(devices, stops);
        throw;
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
    OOVERLAP_TEST_NCCL_CHECK(ncclMemAlloc(&raw, bytes));

    if (raw == nullptr) {
        throw std::runtime_error(
            std::string(label) + ": ncclMemAlloc returned nullptr");
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

void register_nccl_symmetric_windows(
    const std::vector<ncclComm_t>& comms,
    const std::vector<half*>& buffers,
    size_t bytes,
    std::vector<ncclWindow_t>& windows) {
    if (comms.size() != buffers.size() || comms.empty()) {
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

void deregister_nccl_window_best_effort(
    ncclComm_t comm,
    ncclWindow_t& window) {
    if (comm == nullptr || window == nullptr) {
        return;
    }

    (void)ncclCommWindowDeregister(comm, window);
    window = nullptr;
}

struct OoverlapRingSlot {
    std::vector<half*> work;
    std::vector<oo_buffer_t*> buffers;
};

struct NcclRingSlot {
    std::vector<half*> work;
    std::vector<ncclWindow_t> windows;
};

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
    const std::vector<oo_node_t*>& nodes,
    const OoverlapRingSlot& slot,
    size_t numel,
    const std::vector<cudaStream_t>& streams) {
    if (nodes.size() != slot.buffers.size() ||
        nodes.size() != streams.size()) {
        throw std::invalid_argument(
            "launch_ooverlap_public_once: size mismatch");
    }

    for (std::size_t rank = 0; rank < nodes.size(); ++rank) {
        const std::string label =
            "ooverlap rank" + std::to_string(rank);
        launch_ooverlap_public_once_for_rank(
            collective,
            nodes[rank],
            slot.buffers[rank],
            numel,
            streams[rank],
            label.c_str());
    }
}

void launch_nccl_once(
    TestCollective collective,
    const NcclRingSlot& slot,
    size_t numel,
    const std::vector<cudaStream_t>& streams,
    const std::vector<ncclComm_t>& comms) {
    const std::size_t world_size = comms.size();
    if (slot.work.size() != world_size || streams.size() != world_size) {
        throw std::invalid_argument("launch_nccl_once: size mismatch");
    }

    OOVERLAP_TEST_NCCL_CHECK(ncclGroupStart());
    for (std::size_t rank = 0; rank < world_size; ++rank) {
        testing::launch_nccl_collective_fp16(
            collective,
            comms[rank],
            slot.work[rank],
            numel,
            static_cast<int>(rank),
            static_cast<int>(world_size),
            streams[rank]);
    }
    OOVERLAP_TEST_NCCL_CHECK(ncclGroupEnd());
}

void prepare_work_buffers(
    const std::vector<half*>& sources,
    const std::vector<half*>& work,
    size_t bytes,
    const std::vector<int>& devices,
    const std::vector<cudaStream_t>& streams) {
    const std::size_t world_size = devices.size();
    if (sources.size() != world_size ||
        work.size() != world_size ||
        streams.size() != world_size) {
        throw std::invalid_argument("prepare_work_buffers: size mismatch");
    }

    for (std::size_t rank = 0; rank < world_size; ++rank) {
        testing::reset_work_buffer_async(
            work[rank],
            sources[rank],
            bytes,
            devices[rank],
            streams[rank]);
    }

    sync_streams(devices, streams, "sync prepare_work_buffers");
}

void prepare_ooverlap_ring(
    const std::vector<OoverlapRingSlot>& ring,
    const std::vector<half*>& sources,
    size_t bytes,
    const std::vector<int>& devices,
    const std::vector<cudaStream_t>& streams) {
    for (const OoverlapRingSlot& slot : ring) {
        prepare_work_buffers(
            sources,
            slot.work,
            bytes,
            devices,
            streams);
    }
}

void prepare_nccl_ring(
    const std::vector<NcclRingSlot>& ring,
    const std::vector<half*>& sources,
    size_t bytes,
    const std::vector<int>& devices,
    const std::vector<cudaStream_t>& streams) {
    for (const NcclRingSlot& slot : ring) {
        prepare_work_buffers(
            sources,
            slot.work,
            bytes,
            devices,
            streams);
    }
}

void run_ooverlap_ring_iters(
    TestCollective collective,
    oo_group_t* group,
    const std::vector<oo_node_t*>& nodes,
    const std::vector<OoverlapRingSlot>& ring,
    size_t numel,
    const std::vector<int>& devices,
    const std::vector<cudaStream_t>& streams,
    int iters,
    const char* sync_label) {
    if (iters <= 0) {
        return;
    }
    if (ring.empty()) {
        throw std::invalid_argument("run_ooverlap_ring_iters: ring is empty");
    }

    testing::reset_ready_signals(group);
    for (int i = 0; i < iters; ++i) {
        const OoverlapRingSlot& slot =
            ring[static_cast<std::size_t>(i) % ring.size()];
        launch_ooverlap_public_once(
            collective,
            nodes,
            slot,
            numel,
            streams);
    }

    sync_streams(devices, streams, sync_label);
}

double elapsed_ms_ooverlap_ring(
    TestCollective collective,
    oo_group_t* group,
    const std::vector<oo_node_t*>& nodes,
    const std::vector<OoverlapRingSlot>& ring,
    size_t numel,
    const std::vector<int>& devices,
    const std::vector<cudaStream_t>& streams,
    int iters) {
    if (ring.empty()) {
        throw std::invalid_argument("elapsed_ms_ooverlap_ring: ring is empty");
    }

    testing::reset_ready_signals(group);
    return elapsed_ms_stream_max(
        devices,
        streams,
        iters,
        [&](int i) {
            const OoverlapRingSlot& slot =
                ring[static_cast<std::size_t>(i) % ring.size()];
            launch_ooverlap_public_once(
                collective,
                nodes,
                slot,
                numel,
                streams);
        });
}

void run_nccl_ring_iters(
    TestCollective collective,
    const std::vector<NcclRingSlot>& ring,
    size_t numel,
    const std::vector<int>& devices,
    const std::vector<cudaStream_t>& streams,
    const std::vector<ncclComm_t>& comms,
    int iters,
    const char* sync_label) {
    if (iters <= 0) {
        return;
    }
    if (ring.empty()) {
        throw std::invalid_argument("run_nccl_ring_iters: ring is empty");
    }

    for (int i = 0; i < iters; ++i) {
        const NcclRingSlot& slot =
            ring[static_cast<std::size_t>(i) % ring.size()];
        launch_nccl_once(
            collective,
            slot,
            numel,
            streams,
            comms);
    }

    sync_streams(devices, streams, sync_label);
}

double elapsed_ms_nccl_ring(
    TestCollective collective,
    const std::vector<NcclRingSlot>& ring,
    size_t numel,
    const std::vector<int>& devices,
    const std::vector<cudaStream_t>& streams,
    const std::vector<ncclComm_t>& comms,
    int iters) {
    if (ring.empty()) {
        throw std::invalid_argument("elapsed_ms_nccl_ring: ring is empty");
    }

    return elapsed_ms_stream_max(
        devices,
        streams,
        iters,
        [&](int i) {
            const NcclRingSlot& slot =
                ring[static_cast<std::size_t>(i) % ring.size()];
            launch_nccl_once(
                collective,
                slot,
                numel,
                streams,
                comms);
        });
}

void allocate_ooverlap_ring(
    std::vector<OoverlapRingSlot>& ring,
    int ring_size,
    const std::vector<oo_node_t*>& nodes,
    size_t bytes,
    const std::vector<int>& devices) {
    const std::size_t world_size = devices.size();
    if (nodes.size() != world_size) {
        throw std::invalid_argument("allocate_ooverlap_ring: size mismatch");
    }

    ring.clear();
    ring.resize(static_cast<std::size_t>(ring_size));

    for (int i = 0; i < ring_size; ++i) {
        OoverlapRingSlot& slot = ring[static_cast<std::size_t>(i)];
        slot.work.assign(world_size, nullptr);
        slot.buffers.assign(world_size, nullptr);

        for (std::size_t rank = 0; rank < world_size; ++rank) {
            const std::string malloc_label =
                "cudaMalloc(ooverlap ring rank" +
                std::to_string(rank) + ")";
            testing::cuda_malloc_half_on_device(
                devices[rank],
                &slot.work[rank],
                bytes,
                malloc_label.c_str());

            const std::string wrap_label =
                "oo_buffer_wrap(ooverlap ring rank" +
                std::to_string(rank) + ")";
            testing::check_oo(
                oo_buffer_wrap(
                    nodes[rank],
                    slot.work[rank],
                    bytes,
                    &slot.buffers[rank]),
                wrap_label.c_str());
        }
    }
}

void allocate_nccl_ring(
    std::vector<NcclRingSlot>& ring,
    int ring_size,
    size_t bytes,
    const std::vector<int>& devices,
    bool symmetric,
    const std::vector<ncclComm_t>& comms) {
    const std::size_t world_size = devices.size();
    if (comms.size() != world_size) {
        throw std::invalid_argument("allocate_nccl_ring: size mismatch");
    }

    ring.clear();
    ring.resize(static_cast<std::size_t>(ring_size));

    for (int i = 0; i < ring_size; ++i) {
        NcclRingSlot& slot = ring[static_cast<std::size_t>(i)];
        slot.work.assign(world_size, nullptr);
        slot.windows.assign(world_size, nullptr);

        for (std::size_t rank = 0; rank < world_size; ++rank) {
            const std::string label =
                std::string(symmetric ? "ncclMemAlloc" : "cudaMalloc") +
                "(nccl ring rank" + std::to_string(rank) + ")";

            if (symmetric) {
                nccl_mem_alloc_half_on_device(
                    devices[rank],
                    &slot.work[rank],
                    bytes,
                    label.c_str());
            } else {
                testing::cuda_malloc_half_on_device(
                    devices[rank],
                    &slot.work[rank],
                    bytes,
                    label.c_str());
            }
        }

        if (symmetric) {
            register_nccl_symmetric_windows(
                comms,
                slot.work,
                bytes,
                slot.windows);
        }
    }
}

void destroy_ooverlap_ring(
    std::vector<OoverlapRingSlot>& ring,
    const std::vector<int>& devices) {
    for (OoverlapRingSlot& slot : ring) {
        const std::size_t world_size =
            std::min(devices.size(), slot.work.size());

        for (std::size_t rank = 0; rank < slot.buffers.size(); ++rank) {
            if (slot.buffers[rank] != nullptr) {
                oo_buffer_destroy(slot.buffers[rank]);
                slot.buffers[rank] = nullptr;
            }
        }

        for (std::size_t rank = 0; rank < world_size; ++rank) {
            testing::cuda_free_on_device(
                devices[rank],
                slot.work[rank]);
        }
    }

    ring.clear();
}

void destroy_nccl_ring(
    std::vector<NcclRingSlot>& ring,
    const std::vector<int>& devices,
    bool symmetric,
    const std::vector<ncclComm_t>& comms) {
    for (NcclRingSlot& slot : ring) {
        const std::size_t world_size =
            std::min(devices.size(), slot.work.size());

        for (std::size_t rank = 0; rank < world_size; ++rank) {
            if (symmetric) {
                if (rank < slot.windows.size() && rank < comms.size()) {
                    deregister_nccl_window_best_effort(
                        comms[rank],
                        slot.windows[rank]);
                }
                nccl_mem_free_on_device(
                    devices[rank],
                    slot.work[rank]);
            } else {
                testing::cuda_free_on_device(
                    devices[rank],
                    slot.work[rank]);
            }
        }
    }

    ring.clear();
}

void verify_collective_result(
    TestCollective collective,
    const char* label,
    const std::vector<half*>& work,
    int64_t numel,
    const std::vector<int>& devices) {
#if OOVERLAP_BENCH_VERIFY_RESULTS
    if (work.size() != devices.size()) {
        throw std::invalid_argument("verify_collective_result: size mismatch");
    }

    const int world_size = static_cast<int>(devices.size());
    for (int rank = 0; rank < world_size; ++rank) {
        testing::verify_collective_fp16(
            collective,
            label,
            work[static_cast<std::size_t>(rank)],
            numel,
            rank,
            world_size,
            devices[static_cast<std::size_t>(rank)]);
    }
#else
    (void)collective;
    (void)label;
    (void)work;
    (void)numel;
    (void)devices;
#endif
}

void verify_ooverlap_once(
    TestCollective collective,
    oo_group_t* group,
    const std::vector<oo_node_t*>& nodes,
    const std::vector<OoverlapRingSlot>& ring,
    const std::vector<half*>& sources,
    size_t numel,
    size_t bytes,
    const std::vector<int>& devices,
    const std::vector<cudaStream_t>& streams) {
#if OOVERLAP_BENCH_VERIFY_RESULTS
    if (ring.empty()) {
        return;
    }

    const OoverlapRingSlot& slot = ring[0];
    prepare_work_buffers(sources, slot.work, bytes, devices, streams);
    testing::reset_ready_signals(group);
    launch_ooverlap_public_once(
        collective,
        nodes,
        slot,
        numel,
        streams);
    sync_streams(devices, streams, "sync ooverlap ring verification");
    verify_collective_result(
        collective,
        "ooverlap",
        slot.work,
        static_cast<int64_t>(numel),
        devices);
#else
    (void)collective;
    (void)group;
    (void)nodes;
    (void)ring;
    (void)sources;
    (void)numel;
    (void)bytes;
    (void)devices;
    (void)streams;
#endif
}

void verify_nccl_once(
    TestCollective collective,
    const char* label,
    const std::vector<NcclRingSlot>& ring,
    const std::vector<half*>& sources,
    size_t numel,
    size_t bytes,
    const std::vector<int>& devices,
    const std::vector<cudaStream_t>& streams,
    const std::vector<ncclComm_t>& comms) {
#if OOVERLAP_BENCH_VERIFY_RESULTS
    if (ring.empty()) {
        return;
    }

    const NcclRingSlot& slot = ring[0];
    prepare_work_buffers(sources, slot.work, bytes, devices, streams);
    launch_nccl_once(
        collective,
        slot,
        numel,
        streams,
        comms);
    sync_streams(devices, streams, "sync nccl ring verification");
    verify_collective_result(
        collective,
        label,
        slot.work,
        static_cast<int64_t>(numel),
        devices);
#else
    (void)collective;
    (void)label;
    (void)ring;
    (void)sources;
    (void)numel;
    (void)bytes;
    (void)devices;
    (void)streams;
    (void)comms;
#endif
}

void bench_ooverlap_external_ring(
    std::map<std::string, double>& results,
    TestCollective collective,
    oo_group_t* group,
    const std::vector<oo_node_t*>& nodes,
    const std::vector<OoverlapRingSlot>& ring,
    const std::vector<half*>& sources,
    size_t numel,
    size_t bytes,
    const std::vector<int>& devices,
    const std::vector<cudaStream_t>& streams,
    int iters,
    int warmup) {
    prepare_ooverlap_ring(ring, sources, bytes, devices, streams);

    run_ooverlap_ring_iters(
        collective,
        group,
        nodes,
        ring,
        numel,
        devices,
        streams,
        warmup,
        "sync ooverlap ring warmup");

    prepare_ooverlap_ring(ring, sources, bytes, devices, streams);

    const double total_ms =
        elapsed_ms_ooverlap_ring(
            collective,
            group,
            nodes,
            ring,
            numel,
            devices,
            streams,
            iters);

    sync_streams(devices, streams, "sync ooverlap ring measured");

    verify_ooverlap_once(
        collective,
        group,
        nodes,
        ring,
        sources,
        numel,
        bytes,
        devices,
        streams);

    results["ooverlap_ms"] =
        total_ms / static_cast<double>(iters);
}

void bench_nccl_external_ring(
    std::map<std::string, double>& results,
    const char* result_key,
    TestCollective collective,
    const std::vector<NcclRingSlot>& ring,
    const std::vector<half*>& sources,
    size_t numel,
    size_t bytes,
    const std::vector<int>& devices,
    const std::vector<cudaStream_t>& streams,
    const std::vector<ncclComm_t>& comms,
    int iters,
    int warmup) {
    prepare_nccl_ring(ring, sources, bytes, devices, streams);

    run_nccl_ring_iters(
        collective,
        ring,
        numel,
        devices,
        streams,
        comms,
        warmup,
        "sync nccl ring warmup");

    prepare_nccl_ring(ring, sources, bytes, devices, streams);

    const double total_ms =
        elapsed_ms_nccl_ring(
            collective,
            ring,
            numel,
            devices,
            streams,
            comms,
            iters);

    sync_streams(devices, streams, "sync nccl ring measured");

    verify_nccl_once(
        collective,
        result_key,
        ring,
        sources,
        numel,
        bytes,
        devices,
        streams,
        comms);

    results[result_key] =
        total_ms / static_cast<double>(iters);
}

void cleanup(
    const std::vector<int>& devices,
    std::vector<half*>& sources,
    std::vector<OoverlapRingSlot>& ooverlap_ring,
    std::vector<NcclRingSlot>& nccl_ring,
    std::vector<NcclRingSlot>& nccl_symmetric_ring,
    std::vector<oo_node_t*>& nodes,
    oo_group_t*& group,
    std::vector<cudaStream_t>& streams,
    std::vector<ncclComm_t>& comms) {
    destroy_ooverlap_ring(ooverlap_ring, devices);
    destroy_nccl_ring(nccl_ring, devices, false, comms);
    destroy_nccl_ring(nccl_symmetric_ring, devices, true, comms);

    if (!comms.empty()) {
        testing::destroy_nccl_comms(
            comms.data(),
            static_cast<int>(comms.size()));
    }

    const std::size_t world_size = devices.size();
    for (std::size_t rank = 0;
         rank < world_size && rank < sources.size();
         ++rank) {
        testing::cuda_free_on_device(devices[rank], sources[rank]);
    }

    for (oo_node_t*& node : nodes) {
        testing::destroy_oo_node(node);
    }
    testing::destroy_oo_group(group);

    for (std::size_t rank = 0;
         rank < world_size && rank < streams.size();
         ++rank) {
        testing::destroy_stream_on_device(
            devices[rank],
            streams[rank]);
    }
}

} // namespace

bool external_p2p_collective_smoke_test(
    const std::string& collective_name_arg,
    int64_t numel,
    const std::vector<int>& devices) {
    const std::map<std::string, double> result =
        benchmark_external_p2p_collective_sm90(
            collective_name_arg,
            numel,
            1,
            0,
            devices);

    return !result.empty();
}

bool external_p2p_allreduce_smoke_test(
    int64_t numel,
    const std::vector<int>& devices) {
    return external_p2p_collective_smoke_test(
        "allreduce",
        numel,
        devices);
}

std::map<std::string, double> benchmark_external_p2p_collective_sm90(
    const std::string& collective_name_arg,
    int64_t numel_arg,
    int iters,
    int warmup,
    const std::vector<int>& devices) {
    if (numel_arg <= 0 || iters <= 0 || warmup < 0) {
        throw std::invalid_argument(
            "benchmark_external_p2p_collective_sm90: invalid args");
    }

    validate_devices(devices);

    const TestCollective collective =
        testing::parse_collective(collective_name_arg);
    const int world_size = static_cast<int>(devices.size());

    testing::validate_numel_for_collective(
        collective,
        numel_arg,
        world_size);

    const size_t numel = static_cast<size_t>(numel_arg);
    const size_t bytes = numel * sizeof(half);
    const int ring_size = ring_size_from_env();

    oo_group_t* group = nullptr;
    std::vector<oo_node_t*> nodes(
        static_cast<std::size_t>(world_size), nullptr);
    std::vector<half*> sources(
        static_cast<std::size_t>(world_size), nullptr);
    std::vector<cudaStream_t> streams(
        static_cast<std::size_t>(world_size), nullptr);
    std::vector<ncclComm_t> comms(
        static_cast<std::size_t>(world_size), nullptr);

    std::vector<OoverlapRingSlot> ooverlap_ring;
    std::vector<NcclRingSlot> nccl_ring;
    std::vector<NcclRingSlot> nccl_symmetric_ring;

    try {
        testing::check_oo(
            oo_group_create_p2p(
                devices.data(),
                world_size,
                &group),
            "oo_group_create_p2p");

        for (int rank = 0; rank < world_size; ++rank) {
            const std::string node_label =
                "oo_node_create(rank" + std::to_string(rank) + ")";
            testing::check_oo(
                oo_node_create(
                    group,
                    rank,
                    &nodes[static_cast<std::size_t>(rank)]),
                node_label.c_str());

            const int node_device =
                oo_node_device(nodes[static_cast<std::size_t>(rank)]);

            streams[static_cast<std::size_t>(rank)] =
                system::runtime::create_stream_on_device(node_device);

            const std::string source_label =
                "cudaMalloc(rank" + std::to_string(rank) + "_src)";
            testing::cuda_malloc_half_on_device(
                node_device,
                &sources[static_cast<std::size_t>(rank)],
                bytes,
                source_label.c_str());

            testing::fill_rank_source_fp16(
                sources[static_cast<std::size_t>(rank)],
                numel_arg,
                rank,
                node_device,
                streams[static_cast<std::size_t>(rank)]);
        }

        OOVERLAP_TEST_NCCL_CHECK(
            ncclCommInitAll(
                comms.data(),
                world_size,
                devices.data()));

        allocate_ooverlap_ring(
            ooverlap_ring,
            ring_size,
            nodes,
            bytes,
            devices);

        allocate_nccl_ring(
            nccl_ring,
            ring_size,
            bytes,
            devices,
            false,
            comms);

        allocate_nccl_ring(
            nccl_symmetric_ring,
            ring_size,
            bytes,
            devices,
            true,
            comms);

        std::map<std::string, double> results;

        // OOVERLAP_EXTERNAL_P2P_BACKEND_SELECT_V1
        const char* backend_env = std::getenv("OOVERLAP_BENCH_ONLY");
        const std::string selected_backend =
            backend_env != nullptr && backend_env[0] != '\0'
                ? std::string(backend_env)
                : std::string("all");

        if (selected_backend != "all" &&
            selected_backend != "ooverlap" &&
            selected_backend != "nccl" &&
            selected_backend != "nccl_symmetric") {
            throw std::invalid_argument(
                "OOVERLAP_BENCH_ONLY must be all, ooverlap, nccl, or "
                "nccl_symmetric");
        }

        const auto backend_enabled = [&](const char* backend) {
            return selected_backend == "all" || selected_backend == backend;
        };

        if (backend_enabled("ooverlap")) {
            bench_ooverlap_external_ring(
                results,
                collective,
                group,
                nodes,
                ooverlap_ring,
                sources,
                numel,
                bytes,
                devices,
                streams,
                iters,
                warmup);
        }

        if (backend_enabled("nccl")) {
            bench_nccl_external_ring(
                results,
                "nccl_ms",
                collective,
                nccl_ring,
                sources,
                numel,
                bytes,
                devices,
                streams,
                comms,
                iters,
                warmup);
        }

        if (backend_enabled("nccl_symmetric")) {
            bench_nccl_external_ring(
                results,
                "nccl_symmetric_ms",
                collective,
                nccl_symmetric_ring,
                sources,
                numel,
                bytes,
                devices,
                streams,
                comms,
                iters,
                warmup);
        }

        results["collective"] = testing::collective_code(collective);
        results["world_size"] = static_cast<double>(world_size);
        results["numel"] = static_cast<double>(numel);
        results["bytes"] = static_cast<double>(bytes);
        results["iters"] = static_cast<double>(iters);
        results["warmup"] = static_cast<double>(warmup);
        results["ring_size"] = static_cast<double>(ring_size);

        cleanup(
            devices,
            sources,
            ooverlap_ring,
            nccl_ring,
            nccl_symmetric_ring,
            nodes,
            group,
            streams,
            comms);

        return results;
    } catch (...) {
        cleanup(
            devices,
            sources,
            ooverlap_ring,
            nccl_ring,
            nccl_symmetric_ring,
            nodes,
            group,
            streams,
            comms);
        throw;
    }
}

std::map<std::string, double> benchmark_external_p2p_allreduce_sm90(
    int64_t numel,
    int iters,
    int warmup,
    const std::vector<int>& devices) {
    return benchmark_external_p2p_collective_sm90(
        "allreduce",
        numel,
        iters,
        warmup,
        devices);
}

bool external_p2p_two_gpu_collective_smoke_test(
    const std::string& collective,
    int64_t numel,
    int dev0,
    int dev1) {
    return external_p2p_collective_smoke_test(
        collective,
        numel,
        std::vector<int>{dev0, dev1});
}

bool external_p2p_two_gpu_allreduce_smoke_test(
    int64_t numel,
    int dev0,
    int dev1) {
    return external_p2p_allreduce_smoke_test(
        numel,
        std::vector<int>{dev0, dev1});
}

std::map<std::string, double> benchmark_external_p2p_two_gpu_collective_sm90(
    const std::string& collective,
    int64_t numel,
    int iters,
    int warmup,
    int dev0,
    int dev1) {
    return benchmark_external_p2p_collective_sm90(
        collective,
        numel,
        iters,
        warmup,
        std::vector<int>{dev0, dev1});
}

std::map<std::string, double> benchmark_external_p2p_two_gpu_allreduce_sm90(
    int64_t numel,
    int iters,
    int warmup,
    int dev0,
    int dev1) {
    return benchmark_external_p2p_allreduce_sm90(
        numel,
        iters,
        warmup,
        std::vector<int>{dev0, dev1});
}

} // namespace ooverlap
