#include "overlap/tma_benchmark_sm90.h"

#include "overlap/bulk_tma_copy_sm90.cuh"
#include "overlap/tma_basic_collective_sm90.h"
#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/system/peer_buffer.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <nccl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#define OOVERLAP_BENCH_NCCL_CHECK(cmd)                                                              \
    do {                                                                                            \
        ncclResult_t result__ = (cmd);                                                              \
        if (result__ != ncclSuccess) {                                                              \
            throw std::runtime_error(std::string("NCCL error: ") + ncclGetErrorString(result__));  \
        }                                                                                           \
    } while (0)

namespace ooverlap {
namespace {

std::vector<int> normalize_devices(const std::vector<int64_t>& requested) {
    int ndev = 0;
    system::runtime::check_cuda(cudaGetDeviceCount(&ndev), "cudaGetDeviceCount");
    if (ndev < 2) {
        throw std::runtime_error("Need at least 2 CUDA devices");
    }

    std::vector<int> out;
    if (requested.empty()) {
        out.reserve(static_cast<size_t>(ndev));
        for (int d = 0; d < ndev; ++d) out.push_back(d);
        return out;
    }

    out.reserve(requested.size());
    for (int64_t d64 : requested) {
        if (d64 < 0 || d64 >= ndev) {
            throw std::invalid_argument("Invalid device id in device list");
        }
        int d = static_cast<int>(d64);
        for (int seen : out) {
            if (seen == d) {
                throw std::invalid_argument("Duplicate device id in device list");
            }
        }
        out.push_back(d);
    }

    if (out.size() < 2) {
        throw std::invalid_argument("Need at least 2 devices");
    }
    return out;
}

std::vector<half*> allocate_local_buffers(
    const std::vector<int>& devices,
    size_t numel) {
    std::vector<half*> bufs(devices.size(), nullptr);
    const size_t bytes = numel * sizeof(half);

    for (size_t i = 0; i < devices.size(); ++i) {
        system::runtime::set_device(devices[i]);
        system::runtime::check_cuda(cudaMalloc(&bufs[i], bytes), "cudaMalloc(local buffer)");
    }
    return bufs;
}

void free_local_buffers(
    const std::vector<int>& devices,
    std::vector<half*>& bufs) {
    for (size_t i = 0; i < bufs.size(); ++i) {
        if (bufs[i] != nullptr) {
            system::runtime::set_device(devices[i]);
            system::runtime::check_cuda(cudaFree(bufs[i]), "cudaFree(local buffer)");
            bufs[i] = nullptr;
        }
    }
}

std::vector<uint16_t> make_host_pattern(
    size_t numel,
    uint16_t seed) {
    std::vector<uint16_t> host(numel);
    for (size_t i = 0; i < numel; ++i) {
        host[i] = static_cast<uint16_t>((seed + 17 * i) & 0xffffu);
    }
    return host;
}

void fill_one_buffer_from_host_pattern(
    int dev,
    cudaStream_t stream,
    half* dst,
    size_t numel,
    uint16_t seed) {
    std::vector<uint16_t> host = make_host_pattern(numel, seed);
    system::runtime::set_device(dev);
    system::runtime::check_cuda(
        cudaMemcpyAsync(
            dst,
            host.data(),
            numel * sizeof(uint16_t),
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync(host pattern -> device)");
}

void fill_local_buffers(
    const std::vector<int>& devices,
    const std::vector<cudaStream_t>& streams,
    std::vector<half*>& bufs,
    size_t numel) {
    for (size_t i = 0; i < devices.size(); ++i) {
        fill_one_buffer_from_host_pattern(
            devices[i],
            streams[i],
            bufs[i],
            numel,
            static_cast<uint16_t>(0x100u + i * 0x31u));
    }

    for (size_t i = 0; i < devices.size(); ++i) {
        system::runtime::sync_stream_on_device(devices[i], streams[i], "sync fill");
    }
}

double elapsed_ms_single_stream(
    int dev,
    cudaStream_t stream,
    int iters,
    const std::function<void(int)>& launch_once) {

    system::runtime::set_device(dev);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    system::runtime::check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    system::runtime::check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    system::runtime::check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");
    for (int i = 0; i < iters; ++i) {
        launch_once(i);
    }
    system::runtime::check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
    system::runtime::check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float ms = 0.0f;
    system::runtime::check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<double>(ms);
}

double elapsed_ms_multi_stream_max(
    const std::vector<int>& devices,
    const std::vector<cudaStream_t>& streams,
    int iters,
    const std::function<void(int)>& launch_once) {

    std::vector<cudaEvent_t> starts(devices.size(), nullptr);
    std::vector<cudaEvent_t> stops(devices.size(), nullptr);

    for (size_t i = 0; i < devices.size(); ++i) {
        system::runtime::set_device(devices[i]);
        system::runtime::check_cuda(cudaEventCreate(&starts[i]), "cudaEventCreate(start)");
        system::runtime::check_cuda(cudaEventCreate(&stops[i]), "cudaEventCreate(stop)");
        system::runtime::check_cuda(cudaEventRecord(starts[i], streams[i]), "cudaEventRecord(start)");
    }

    for (int iter = 0; iter < iters; ++iter) {
        launch_once(iter);
    }

    for (size_t i = 0; i < devices.size(); ++i) {
        system::runtime::set_device(devices[i]);
        system::runtime::check_cuda(cudaEventRecord(stops[i], streams[i]), "cudaEventRecord(stop)");
    }

    double max_ms = 0.0;
    for (size_t i = 0; i < devices.size(); ++i) {
        system::runtime::set_device(devices[i]);
        system::runtime::check_cuda(cudaEventSynchronize(stops[i]), "cudaEventSynchronize(stop)");
        float ms = 0.0f;
        system::runtime::check_cuda(cudaEventElapsedTime(&ms, starts[i], stops[i]), "cudaEventElapsedTime");
        max_ms = std::max(max_ms, static_cast<double>(ms));
        cudaEventDestroy(starts[i]);
        cudaEventDestroy(stops[i]);
    }

    return max_ms;
}

void nccl_init_all(
    const std::vector<int>& devices,
    std::vector<ncclComm_t>& comms) {
    comms.resize(devices.size(), nullptr);
    OOVERLAP_BENCH_NCCL_CHECK(
        ncclCommInitAll(comms.data(), static_cast<int>(devices.size()), devices.data()));
}

void nccl_destroy_all(std::vector<ncclComm_t>& comms) {
    for (auto& c : comms) {
        if (c != nullptr) {
            ncclCommDestroy(c);
            c = nullptr;
        }
    }
}

void run_nccl_collective_once(
    const std::string& op,
    const std::vector<int>& devices,
    const std::vector<cudaStream_t>& streams,
    const std::vector<ncclComm_t>& comms,
    std::vector<half*>& full_bufs,
    std::vector<half*>& shard_bufs,
    size_t full_numel) {

    const int world_size = static_cast<int>(devices.size());
    const size_t shard_numel = full_numel / static_cast<size_t>(world_size);

    OOVERLAP_BENCH_NCCL_CHECK(ncclGroupStart());
    for (int r = 0; r < world_size; ++r) {
        system::runtime::set_device(devices[r]);

        if (op == "allreduce") {
            OOVERLAP_BENCH_NCCL_CHECK(
                ncclAllReduce(
                    full_bufs[r],
                    full_bufs[r],
                    full_numel,
                    ncclFloat16,
                    ncclSum,
                    comms[r],
                    streams[r]));
        } else if (op == "reducescatter") {
            OOVERLAP_BENCH_NCCL_CHECK(
                ncclReduceScatter(
                    full_bufs[r],
                    shard_bufs[r],
                    shard_numel,
                    ncclFloat16,
                    ncclSum,
                    comms[r],
                    streams[r]));
        } else if (op == "allgather") {
            OOVERLAP_BENCH_NCCL_CHECK(
                ncclAllGather(
                    shard_bufs[r],
                    full_bufs[r],
                    shard_numel,
                    ncclFloat16,
                    comms[r],
                    streams[r]));
        } else {
            OOVERLAP_BENCH_NCCL_CHECK(ncclGroupEnd());
            throw std::invalid_argument("Unsupported NCCL op");
        }
    }
    OOVERLAP_BENCH_NCCL_CHECK(ncclGroupEnd());
}

} // namespace

std::map<std::string, double> benchmark_2gpu_copy_sm90(
    int64_t numel,
    int iters,
    int warmup,
    int dev0,
    int dev1) {

    if (numel <= 0 || iters <= 0 || warmup < 0) {
        throw std::invalid_argument("Invalid benchmark arguments");
    }
    if (dev0 == dev1) {
        throw std::invalid_argument("dev0 and dev1 must differ");
    }

    const size_t numel_sz = static_cast<size_t>(numel);
    const size_t bytes = numel_sz * sizeof(half);

    system::runtime::ensure_context_on_device(dev0);
    system::runtime::ensure_context_on_device(dev1);

    cudaStream_t stream0 = system::runtime::create_stream_on_device(dev0);

    half* src = nullptr;
    system::mapped_peer_buffer peer_dst{};

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(cudaMalloc(&src, bytes), "cudaMalloc(src)");
    fill_one_buffer_from_host_pattern(dev0, stream0, src, numel_sz, static_cast<uint16_t>(0x1234u));
    system::runtime::sync_stream_on_device(dev0, stream0, "sync fill src");

    peer_dst = system::alloc_peer_visible_buffer(bytes, dev1, {dev0, dev1});

    system::runtime::set_device(dev1);
    system::runtime::check_cuda(cudaMemset(peer_dst.ptr, 0, peer_dst.mapped_size), "cudaMemset(peer dst)");

    for (int i = 0; i < warmup; ++i) {
        system::runtime::set_device(dev0);
        system::runtime::check_cuda(
            enqueue_bulk_tma_copy_sm90(src, reinterpret_cast<half*>(peer_dst.ptr), numel_sz, stream0),
            "enqueue_bulk_tma_copy_sm90 warmup");
        system::runtime::sync_stream_on_device(dev0, stream0, "sync bulk warmup");
    }

    const double tma_total_ms = elapsed_ms_single_stream(
        dev0, stream0, iters,
        [&](int) {
            system::runtime::check_cuda(
                enqueue_bulk_tma_copy_sm90(src, reinterpret_cast<half*>(peer_dst.ptr), numel_sz, stream0),
                "enqueue_bulk_tma_copy_sm90");
        });

    for (int i = 0; i < warmup; ++i) {
        system::runtime::set_device(dev0);
        system::runtime::check_cuda(
            cudaMemcpyPeerAsync(peer_dst.ptr, dev1, src, dev0, bytes, stream0),
            "cudaMemcpyPeerAsync warmup");
        system::runtime::sync_stream_on_device(dev0, stream0, "sync memcpypeer warmup");
    }

    const double memcpy_total_ms = elapsed_ms_single_stream(
        dev0, stream0, iters,
        [&](int) {
            system::runtime::check_cuda(
                cudaMemcpyPeerAsync(peer_dst.ptr, dev1, src, dev0, bytes, stream0),
                "cudaMemcpyPeerAsync");
        });

    const double avg_tma_ms = tma_total_ms / static_cast<double>(iters);
    const double avg_memcpy_ms = memcpy_total_ms / static_cast<double>(iters);
    const double gb = static_cast<double>(bytes) / 1.0e9;

    system::runtime::set_device(dev0);
    system::runtime::check_cuda(cudaFree(src), "cudaFree(src)");
    system::runtime::destroy_stream_on_device(dev0, stream0);
    system::free_peer_visible_buffer(peer_dst);

    return {
        {"numel", static_cast<double>(numel)},
        {"bytes", static_cast<double>(bytes)},
        {"avg_ms_tma", avg_tma_ms},
        {"avg_ms_memcpy_peer", avg_memcpy_ms},
        {"gbps_tma", gb / (avg_tma_ms / 1.0e3)},
        {"gbps_memcpy_peer", gb / (avg_memcpy_ms / 1.0e3)},
        {"speedup_memcpy_over_tma", avg_memcpy_ms / avg_tma_ms}
    };
}

std::map<std::string, double> benchmark_basic_ngpu_collective_sm90(
    const std::string& op,
    int64_t numel,
    const std::vector<int64_t>& devices64,
    int iters,
    int warmup) {

    if (numel <= 0 || iters <= 0 || warmup < 0) {
        throw std::invalid_argument("Invalid benchmark arguments");
    }

    const bool is_allgather = (op == "allgather");
    const bool is_rs = (op == "reducescatter");
    const bool is_ar = (op == "allreduce");
    if (!is_allgather && !is_rs && !is_ar) {
        throw std::invalid_argument("op must be one of: allgather, reducescatter, allreduce");
    }

    std::vector<int> devices = normalize_devices(devices64);
    const int world_size = static_cast<int>(devices.size());

    const size_t user_numel = static_cast<size_t>(numel);
    const size_t full_numel = is_allgather ? user_numel * static_cast<size_t>(world_size)
                                           : user_numel;
    if ((is_rs || is_ar) && full_numel % static_cast<size_t>(world_size) != 0) {
        throw std::invalid_argument("For reducescatter/allreduce, numel must be divisible by world_size");
    }
    const size_t shard_numel = full_numel / static_cast<size_t>(world_size);

    BasicCollectiveState st{};
    init_basic_collective_same_process(&st, devices, full_numel);

    auto full_bufs = allocate_local_buffers(devices, full_numel);
    auto shard_bufs = allocate_local_buffers(devices, shard_numel);

    fill_local_buffers(devices, st.streams, full_bufs, full_numel);
    fill_local_buffers(devices, st.streams, shard_bufs, shard_numel);

    for (int i = 0; i < warmup; ++i) {
        if (is_ar) {
            system::runtime::check_cuda(
                enqueue_basic_all_reduce_tma_sm90(&st, full_bufs, full_numel),
                "enqueue_basic_all_reduce_tma_sm90 warmup");
        } else if (is_rs) {
            system::runtime::check_cuda(
                enqueue_basic_reduce_scatter_tma_sm90(&st, full_bufs, full_numel),
                "enqueue_basic_reduce_scatter_tma_sm90 warmup");
        } else {
            system::runtime::check_cuda(
                enqueue_basic_all_gather_tma_sm90(&st, shard_bufs, shard_numel),
                "enqueue_basic_all_gather_tma_sm90 warmup");
        }
    }

    const double tma_total_ms = elapsed_ms_multi_stream_max(
        devices, st.streams, iters,
        [&](int) {
            if (is_ar) {
                system::runtime::check_cuda(
                    enqueue_basic_all_reduce_tma_sm90(&st, full_bufs, full_numel),
                    "enqueue_basic_all_reduce_tma_sm90");
            } else if (is_rs) {
                system::runtime::check_cuda(
                    enqueue_basic_reduce_scatter_tma_sm90(&st, full_bufs, full_numel),
                    "enqueue_basic_reduce_scatter_tma_sm90");
            } else {
                system::runtime::check_cuda(
                    enqueue_basic_all_gather_tma_sm90(&st, shard_bufs, shard_numel),
                    "enqueue_basic_all_gather_tma_sm90");
            }
        });

    std::vector<ncclComm_t> comms;
    nccl_init_all(devices, comms);

    for (int i = 0; i < warmup; ++i) {
        run_nccl_collective_once(op, devices, st.streams, comms, full_bufs, shard_bufs, full_numel);
        for (int r = 0; r < world_size; ++r) {
            system::runtime::sync_stream_on_device(devices[r], st.streams[r], "sync nccl warmup");
        }
    }

    const double nccl_total_ms = elapsed_ms_multi_stream_max(
        devices, st.streams, iters,
        [&](int) {
            run_nccl_collective_once(op, devices, st.streams, comms, full_bufs, shard_bufs, full_numel);
        });

    nccl_destroy_all(comms);
    free_local_buffers(devices, full_bufs);
    free_local_buffers(devices, shard_bufs);
    destroy_basic_collective_same_process(&st);

    const double avg_tma_ms = tma_total_ms / static_cast<double>(iters);
    const double avg_nccl_ms = nccl_total_ms / static_cast<double>(iters);

    return {
        {"world_size", static_cast<double>(world_size)},
        {"numel", static_cast<double>(numel)},
        {"avg_ms_tma", avg_tma_ms},
        {"avg_ms_nccl", avg_nccl_ms},
        {"speedup_nccl_over_tma", avg_nccl_ms / avg_tma_ms}
    };
}

} // namespace ooverlap
