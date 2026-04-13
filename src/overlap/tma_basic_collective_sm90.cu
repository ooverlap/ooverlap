#include "overlap/tma_basic_collective_sm90.h"
#include "overlap/bulk_tma_copy_sm90.cuh"
#include "overlap/tma_collective_sm90.h"

#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/system/peer_buffer.cuh"
#include "ooverlap/testing/test_utils.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

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

inline void validate_state(BasicCollectiveState* st) {
    if (st == nullptr) {
        throw std::invalid_argument("BasicCollectiveState pointer is null");
    }
    if (st->world_size < 2) {
        throw std::runtime_error("BasicCollectiveState is not initialized");
    }
}

inline void zero_peer_buffer_on_owner(
    const system::mapped_peer_buffer& buf,
    int owner_dev) {
    system::runtime::set_device(owner_dev);
    system::runtime::check_cuda(cudaMemset(buf.ptr, 0, buf.mapped_size), "cudaMemset(peer buffer)");
}

inline void sync_all_streams(const BasicCollectiveState* st, const char* what) {
    for (int r = 0; r < st->world_size; ++r) {
        system::runtime::sync_stream_on_device(st->devices[r], st->streams[r], what);
    }
}

inline std::vector<half*> allocate_local_full_buffers(
    const std::vector<int>& devices,
    size_t full_numel) {
    std::vector<half*> bufs(devices.size(), nullptr);
    const size_t bytes = full_numel * sizeof(half);

    for (size_t r = 0; r < devices.size(); ++r) {
        system::runtime::set_device(devices[r]);
        system::runtime::check_cuda(cudaMalloc(&bufs[r], bytes), "cudaMalloc(local full buffer)");
    }
    return bufs;
}

inline std::vector<half*> allocate_local_shard_buffers(
    const std::vector<int>& devices,
    size_t shard_numel) {
    std::vector<half*> bufs(devices.size(), nullptr);
    const size_t bytes = shard_numel * sizeof(half);

    for (size_t r = 0; r < devices.size(); ++r) {
        system::runtime::set_device(devices[r]);
        system::runtime::check_cuda(cudaMalloc(&bufs[r], bytes), "cudaMalloc(local shard buffer)");
    }
    return bufs;
}

inline void free_local_buffers(
    const std::vector<int>& devices,
    std::vector<half*>& bufs) {
    for (size_t r = 0; r < bufs.size(); ++r) {
        if (bufs[r] != nullptr) {
            system::runtime::set_device(devices[r]);
            system::runtime::check_cuda(cudaFree(bufs[r]), "cudaFree(local buffer)");
            bufs[r] = nullptr;
        }
    }
}

inline std::vector<float> reference_full_sum_fp16(
    size_t full_numel,
    int world_size) {
    std::vector<std::vector<float>> refs;
    refs.reserve(static_cast<size_t>(world_size));
    for (int r = 0; r < world_size; ++r) {
        refs.push_back(
            testing::host_reference_pattern_fp16(
                static_cast<int64_t>(full_numel),
                0.10f * static_cast<float>(r + 1),
                10.0f * static_cast<float>(r + 1)));
    }

    std::vector<float> sum(full_numel, 0.0f);
    for (size_t i = 0; i < full_numel; ++i) {
        float acc = refs[0][i];
        for (int r = 1; r < world_size; ++r) {
            acc = testing::round_to_half(acc + refs[static_cast<size_t>(r)][i]);
        }
        sum[i] = acc;
    }
    return sum;
}

inline std::vector<float> reference_all_gather_full_fp16(
    size_t shard_numel,
    int world_size) {
    std::vector<float> full(shard_numel * static_cast<size_t>(world_size), 0.0f);
    for (int r = 0; r < world_size; ++r) {
        auto ref = testing::host_reference_pattern_fp16(
            static_cast<int64_t>(shard_numel),
            1.0f,
            100.0f * static_cast<float>(r + 1));
        for (size_t i = 0; i < shard_numel; ++i) {
            full[static_cast<size_t>(r) * shard_numel + i] = ref[i];
        }
    }
    return full;
}

} // namespace

bool init_basic_collective_same_process(
    BasicCollectiveState* st,
    const std::vector<int>& devices,
    size_t max_full_numel) {

    if (st == nullptr) {
        throw std::invalid_argument("BasicCollectiveState pointer is null");
    }
    if (devices.size() < 2) {
        throw std::invalid_argument("Need at least 2 devices");
    }
    if (max_full_numel == 0) {
        throw std::invalid_argument("max_full_numel must be > 0");
    }
    if (max_full_numel % devices.size() != 0) {
        throw std::invalid_argument("max_full_numel must be divisible by world_size");
    }

    destroy_basic_collective_same_process(st);

    st->world_size = static_cast<int>(devices.size());
    st->devices = devices;
    st->streams.resize(devices.size(), nullptr);
    st->max_full_numel = max_full_numel;
    st->max_shard_numel = max_full_numel / devices.size();
    st->rs_inboxes.resize(devices.size());
    st->full_outputs.resize(devices.size());
    st->shard_outputs.resize(devices.size(), nullptr);

    std::vector<int> access_devices = devices;

    const size_t shard_bytes = st->max_shard_numel * sizeof(half);
    const size_t full_bytes = st->max_full_numel * sizeof(half);

    for (int r = 0; r < st->world_size; ++r) {
        system::runtime::ensure_context_on_device(st->devices[r]);
    }

    for (int r = 0; r < st->world_size; ++r) {
        st->streams[r] = system::runtime::create_stream_on_device(st->devices[r]);

        st->rs_inboxes[r] = system::alloc_peer_visible_buffer(
            shard_bytes * static_cast<size_t>(st->world_size),
            st->devices[r],
            access_devices);

        st->full_outputs[r] = system::alloc_peer_visible_buffer(
            full_bytes,
            st->devices[r],
            access_devices);

        system::runtime::set_device(st->devices[r]);
        system::runtime::check_cuda(
            cudaMalloc(&st->shard_outputs[r], shard_bytes),
            "cudaMalloc(shard_output)");
    }

    return true;
}

void destroy_basic_collective_same_process(BasicCollectiveState* st) {
    if (st == nullptr) return;

    for (size_t r = 0; r < st->shard_outputs.size(); ++r) {
        if (st->shard_outputs[r] != nullptr) {
            system::runtime::set_device(st->devices[r]);
            system::runtime::check_cuda(cudaFree(st->shard_outputs[r]), "cudaFree(shard_output)");
            st->shard_outputs[r] = nullptr;
        }
    }

    for (auto& buf : st->rs_inboxes) {
        system::free_peer_visible_buffer(buf);
    }
    for (auto& buf : st->full_outputs) {
        system::free_peer_visible_buffer(buf);
    }

    for (size_t r = 0; r < st->streams.size(); ++r) {
        if (st->streams[r] != nullptr) {
            system::runtime::destroy_stream_on_device(st->devices[r], st->streams[r]);
        }
    }

    st->world_size = 0;
    st->devices.clear();
    st->streams.clear();
    st->max_full_numel = 0;
    st->max_shard_numel = 0;
    st->rs_inboxes.clear();
    st->full_outputs.clear();
    st->shard_outputs.clear();
}

half* basic_collective_shard_output_ptr(BasicCollectiveState* st, int rank) {
    validate_state(st);
    if (rank < 0 || rank >= st->world_size) {
        throw std::invalid_argument("Invalid rank in basic_collective_shard_output_ptr");
    }
    return st->shard_outputs[rank];
}

half* basic_collective_full_output_ptr(BasicCollectiveState* st, int rank) {
    validate_state(st);
    if (rank < 0 || rank >= st->world_size) {
        throw std::invalid_argument("Invalid rank in basic_collective_full_output_ptr");
    }
    return reinterpret_cast<half*>(st->full_outputs[rank].ptr);
}

cudaError_t enqueue_basic_reduce_scatter_tma_sm90(
    BasicCollectiveState* st,
    const std::vector<half*>& local_full_buffers,
    size_t full_numel) {

    validate_state(st);

    if (local_full_buffers.size() != static_cast<size_t>(st->world_size)) {
        return cudaErrorInvalidValue;
    }
    if (full_numel == 0 || full_numel > st->max_full_numel) {
        return cudaErrorInvalidValue;
    }
    if (full_numel % static_cast<size_t>(st->world_size) != 0) {
        return cudaErrorInvalidValue;
    }

    const size_t shard_numel = full_numel / static_cast<size_t>(st->world_size);
    const size_t shard_bytes = shard_numel * sizeof(half);

    for (int owner = 0; owner < st->world_size; ++owner) {
        zero_peer_buffer_on_owner(st->rs_inboxes[owner], st->devices[owner]);

        system::runtime::set_device(st->devices[owner]);
        const half* src_local_shard = local_full_buffers[owner] + static_cast<size_t>(owner) * shard_numel;
        system::runtime::check_cuda(
            cudaMemcpyAsync(
                st->shard_outputs[owner],
                src_local_shard,
                shard_bytes,
                cudaMemcpyDeviceToDevice,
                st->streams[owner]),
            "cudaMemcpyAsync(local shard -> shard_output)");
    }

    for (int sender = 0; sender < st->world_size; ++sender) {
        system::runtime::set_device(st->devices[sender]);

        for (int owner = 0; owner < st->world_size; ++owner) {
            if (owner == sender) continue;

            const half* src = local_full_buffers[sender] + static_cast<size_t>(owner) * shard_numel;
            half* dst = reinterpret_cast<half*>(st->rs_inboxes[owner].ptr) +
                        static_cast<size_t>(sender) * shard_numel;

            system::runtime::check_cuda(
                enqueue_bulk_tma_copy_sm90(src, dst, shard_numel, st->streams[sender]),
                "enqueue_bulk_tma_copy_sm90 RS");
        }
    }

    sync_all_streams(st, "cudaStreamSynchronize(RS sends)");

    for (int owner = 0; owner < st->world_size; ++owner) {
        system::runtime::set_device(st->devices[owner]);

        for (int sender = 0; sender < st->world_size; ++sender) {
            if (sender == owner) continue;

            const half* src = reinterpret_cast<const half*>(st->rs_inboxes[owner].ptr) +
                              static_cast<size_t>(sender) * shard_numel;

            system::runtime::check_cuda(
                enqueue_fp16_add_inplace_sm90(
                    st->shard_outputs[owner],
                    src,
                    shard_numel,
                    st->streams[owner]),
                "enqueue_fp16_add_inplace_sm90 RS");
        }
    }

    sync_all_streams(st, "cudaStreamSynchronize(RS reductions)");
    return cudaSuccess;
}

cudaError_t enqueue_basic_all_gather_tma_sm90(
    BasicCollectiveState* st,
    const std::vector<half*>& local_shard_buffers,
    size_t shard_numel) {

    validate_state(st);

    if (local_shard_buffers.size() != static_cast<size_t>(st->world_size)) {
        return cudaErrorInvalidValue;
    }
    if (shard_numel == 0 || shard_numel > st->max_shard_numel) {
        return cudaErrorInvalidValue;
    }

    const size_t shard_bytes = shard_numel * sizeof(half);

    for (int recv = 0; recv < st->world_size; ++recv) {
        zero_peer_buffer_on_owner(st->full_outputs[recv], st->devices[recv]);

        system::runtime::set_device(st->devices[recv]);
        half* local_slot = reinterpret_cast<half*>(st->full_outputs[recv].ptr) +
                           static_cast<size_t>(recv) * shard_numel;

        system::runtime::check_cuda(
            cudaMemcpyAsync(
                local_slot,
                local_shard_buffers[recv],
                shard_bytes,
                cudaMemcpyDeviceToDevice,
                st->streams[recv]),
            "cudaMemcpyAsync(local shard -> full_output local slot)");
    }

    for (int sender = 0; sender < st->world_size; ++sender) {
        system::runtime::set_device(st->devices[sender]);

        for (int recv = 0; recv < st->world_size; ++recv) {
            if (recv == sender) continue;

            half* dst = reinterpret_cast<half*>(st->full_outputs[recv].ptr) +
                        static_cast<size_t>(sender) * shard_numel;

            system::runtime::check_cuda(
                enqueue_bulk_tma_copy_sm90(
                    local_shard_buffers[sender],
                    dst,
                    shard_numel,
                    st->streams[sender]),
                "enqueue_bulk_tma_copy_sm90 AG");
        }
    }

    sync_all_streams(st, "cudaStreamSynchronize(AG)");
    return cudaSuccess;
}

cudaError_t enqueue_basic_all_reduce_tma_sm90(
    BasicCollectiveState* st,
    const std::vector<half*>& local_full_buffers,
    size_t full_numel) {

    validate_state(st);

    cudaError_t err = enqueue_basic_reduce_scatter_tma_sm90(st, local_full_buffers, full_numel);
    if (err != cudaSuccess) {
        return err;
    }

    std::vector<half*> local_shards(static_cast<size_t>(st->world_size), nullptr);
    for (int r = 0; r < st->world_size; ++r) {
        local_shards[static_cast<size_t>(r)] = st->shard_outputs[r];
    }

    return enqueue_basic_all_gather_tma_sm90(
        st,
        local_shards,
        full_numel / static_cast<size_t>(st->world_size));
}

bool tma_basic_ngpu_reduce_scatter_smoke_test(
    int64_t full_numel,
    const std::vector<int64_t>& devices64) {

    if (full_numel <= 0) {
        throw std::invalid_argument("full_numel must be > 0");
    }

    std::vector<int> devices = normalize_devices(devices64);
    const size_t full_numel_sz = static_cast<size_t>(full_numel);
    const int world_size = static_cast<int>(devices.size());

    if (full_numel_sz % static_cast<size_t>(world_size) != 0) {
        throw std::invalid_argument("full_numel must be divisible by world_size");
    }

    BasicCollectiveState st{};
    init_basic_collective_same_process(&st, devices, full_numel_sz);

    auto local_full = allocate_local_full_buffers(devices, full_numel_sz);

    for (int r = 0; r < world_size; ++r) {
        system::runtime::set_device(devices[r]);
        testing::fill_pattern(
            local_full[static_cast<size_t>(r)],
            static_cast<int64_t>(full_numel_sz),
            0.10f * static_cast<float>(r + 1),
            10.0f * static_cast<float>(r + 1),
            st.streams[static_cast<size_t>(r)]);
    }

    sync_all_streams(&st, "cudaStreamSynchronize(fill full buffers)");

    system::runtime::check_cuda(
        enqueue_basic_reduce_scatter_tma_sm90(&st, local_full, full_numel_sz),
        "enqueue_basic_reduce_scatter_tma_sm90");

    std::vector<float> full_ref = reference_full_sum_fp16(full_numel_sz, world_size);
    const size_t shard_numel = full_numel_sz / static_cast<size_t>(world_size);

    for (int r = 0; r < world_size; ++r) {
        auto got = testing::copy_half_device_to_host_float(
            st.shard_outputs[r], static_cast<int64_t>(shard_numel), devices[r]);

        std::vector<float> ref(shard_numel);
        for (size_t i = 0; i < shard_numel; ++i) {
            ref[i] = full_ref[static_cast<size_t>(r) * shard_numel + i];
        }

        testing::expect_allclose(got, ref, "basic_ngpu_reduce_scatter");
    }

    free_local_buffers(devices, local_full);
    destroy_basic_collective_same_process(&st);
    return true;
}

bool tma_basic_ngpu_all_gather_smoke_test(
    int64_t shard_numel,
    const std::vector<int64_t>& devices64) {

    if (shard_numel <= 0) {
        throw std::invalid_argument("shard_numel must be > 0");
    }

    std::vector<int> devices = normalize_devices(devices64);
    const size_t shard_numel_sz = static_cast<size_t>(shard_numel);
    const int world_size = static_cast<int>(devices.size());
    const size_t full_numel_sz = shard_numel_sz * static_cast<size_t>(world_size);

    BasicCollectiveState st{};
    init_basic_collective_same_process(&st, devices, full_numel_sz);

    auto local_shards = allocate_local_shard_buffers(devices, shard_numel_sz);

    for (int r = 0; r < world_size; ++r) {
        system::runtime::set_device(devices[r]);
        testing::fill_pattern(
            local_shards[static_cast<size_t>(r)],
            static_cast<int64_t>(shard_numel_sz),
            1.0f,
            100.0f * static_cast<float>(r + 1),
            st.streams[static_cast<size_t>(r)]);
    }

    sync_all_streams(&st, "cudaStreamSynchronize(fill shards)");

    system::runtime::check_cuda(
        enqueue_basic_all_gather_tma_sm90(&st, local_shards, shard_numel_sz),
        "enqueue_basic_all_gather_tma_sm90");

    std::vector<float> ref_full = reference_all_gather_full_fp16(shard_numel_sz, world_size);

    for (int r = 0; r < world_size; ++r) {
        auto got = testing::copy_half_device_to_host_float(
            reinterpret_cast<half*>(st.full_outputs[static_cast<size_t>(r)].ptr),
            static_cast<int64_t>(full_numel_sz),
            devices[r]);

        testing::expect_allclose(got, ref_full, "basic_ngpu_all_gather");
    }

    free_local_buffers(devices, local_shards);
    destroy_basic_collective_same_process(&st);
    return true;
}

bool tma_basic_ngpu_all_reduce_smoke_test(
    int64_t full_numel,
    const std::vector<int64_t>& devices64) {

    if (full_numel <= 0) {
        throw std::invalid_argument("full_numel must be > 0");
    }

    std::vector<int> devices = normalize_devices(devices64);
    const size_t full_numel_sz = static_cast<size_t>(full_numel);
    const int world_size = static_cast<int>(devices.size());

    if (full_numel_sz % static_cast<size_t>(world_size) != 0) {
        throw std::invalid_argument("full_numel must be divisible by world_size");
    }

    BasicCollectiveState st{};
    init_basic_collective_same_process(&st, devices, full_numel_sz);

    auto local_full = allocate_local_full_buffers(devices, full_numel_sz);

    for (int r = 0; r < world_size; ++r) {
        system::runtime::set_device(devices[r]);
        testing::fill_pattern(
            local_full[static_cast<size_t>(r)],
            static_cast<int64_t>(full_numel_sz),
            0.10f * static_cast<float>(r + 1),
            10.0f * static_cast<float>(r + 1),
            st.streams[static_cast<size_t>(r)]);
    }

    sync_all_streams(&st, "cudaStreamSynchronize(fill full buffers)");

    system::runtime::check_cuda(
        enqueue_basic_all_reduce_tma_sm90(&st, local_full, full_numel_sz),
        "enqueue_basic_all_reduce_tma_sm90");

    std::vector<float> ref = reference_full_sum_fp16(full_numel_sz, world_size);

    for (int r = 0; r < world_size; ++r) {
        auto got = testing::copy_half_device_to_host_float(
            reinterpret_cast<half*>(st.full_outputs[static_cast<size_t>(r)].ptr),
            static_cast<int64_t>(full_numel_sz),
            devices[r]);

        testing::expect_allclose(got, ref, "basic_ngpu_all_reduce");
    }

    free_local_buffers(devices, local_full);
    destroy_basic_collective_same_process(&st);
    return true;
}

} // namespace ooverlap
