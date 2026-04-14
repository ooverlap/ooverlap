#include "overlap/tma_basic_collective_sm90.h"
#include "overlap/tma_collective_sm90.h"

#include "ooverlap/system/runtime_utils.cuh"
#include "ooverlap/testing/test_utils.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstddef>
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

inline void zero_buffer_on_owner(
    const BasicCollectiveState* st,
    const Buffer& buf) {
    system::runtime::set_device(st->devices[static_cast<size_t>(buf.owner_rank)]);
    system::runtime::check_cuda(cudaMemset(buf.ptr, 0, buf.bytes), "cudaMemset(buffer)");
}

inline void sync_all_streams(const BasicCollectiveState* st, const char* what) {
    for (int r = 0; r < st->world_size; ++r) {
        system::runtime::sync_stream_on_device(st->devices[static_cast<size_t>(r)],
                                               st->streams[static_cast<size_t>(r)],
                                               what);
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
    return communicator_init(st, devices, max_full_numel, 1);
}

void destroy_basic_collective_same_process(BasicCollectiveState* st) {
    communicator_destroy(st);
}

half* basic_collective_shard_output_ptr(BasicCollectiveState* st, int rank) {
    validate_state(st);
    return buffer_as_half(communicator_get_local_shard_buffer(st, rank));
}

half* basic_collective_full_output_ptr(BasicCollectiveState* st, int rank) {
    validate_state(st);
    return buffer_as_half(communicator_get_local_full_buffer(st, rank));
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

    // Each owner starts from its local shard contribution.
    for (int owner = 0; owner < st->world_size; ++owner) {
        Buffer* shard_out = communicator_get_local_shard_buffer(st, owner);

        system::runtime::set_device(st->devices[static_cast<size_t>(owner)]);
        const half* src_local_shard =
            local_full_buffers[static_cast<size_t>(owner)] + static_cast<size_t>(owner) * shard_numel;

        system::runtime::check_cuda(
            cudaMemcpyAsync(
                shard_out->ptr,
                src_local_shard,
                shard_bytes,
                cudaMemcpyDeviceToDevice,
                st->streams[static_cast<size_t>(owner)]),
            "cudaMemcpyAsync(local shard -> local_shard_buffer)");
    }

    // Each sender pushes shard[owner] into channel(sender -> owner, slot 0).
    for (int sender = 0; sender < st->world_size; ++sender) {
        system::runtime::set_device(st->devices[static_cast<size_t>(sender)]);

        for (int owner = 0; owner < st->world_size; ++owner) {
            if (owner == sender) continue;

            Buffer* slot_buf = channel_get_slot_buffer(st, sender, owner, 0);
            zero_buffer_on_owner(st, *slot_buf);

            const half* src =
                local_full_buffers[static_cast<size_t>(sender)] + static_cast<size_t>(owner) * shard_numel;

            system::runtime::check_cuda(
                channel_send_bulk_tma(
                    st,
                    sender,
                    owner,
                    0,
                    src,
                    shard_numel,
                    st->streams[static_cast<size_t>(sender)]),
                "channel_send_bulk_tma RS");
        }
    }

    sync_all_streams(st, "cudaStreamSynchronize(RS sends)");

    // Each owner accumulates received shards from channel(sender -> owner, slot 0).
    for (int owner = 0; owner < st->world_size; ++owner) {
        Buffer* shard_out = communicator_get_local_shard_buffer(st, owner);

        system::runtime::set_device(st->devices[static_cast<size_t>(owner)]);
        for (int sender = 0; sender < st->world_size; ++sender) {
            if (sender == owner) continue;

            const Buffer* slot_buf = channel_get_slot_buffer(st, sender, owner, 0);

            system::runtime::check_cuda(
                enqueue_fp16_add_inplace_sm90(
                    buffer_as_half(shard_out),
                    buffer_as_half(slot_buf),
                    shard_numel,
                    st->streams[static_cast<size_t>(owner)]),
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

    // Zero full outputs and place local shard into its local slot.
    for (int recv = 0; recv < st->world_size; ++recv) {
        Buffer* full_out = communicator_get_local_full_buffer(st, recv);
        zero_buffer_on_owner(st, *full_out);

        system::runtime::set_device(st->devices[static_cast<size_t>(recv)]);
        half* local_slot = buffer_as_half(full_out) + static_cast<size_t>(recv) * shard_numel;

        system::runtime::check_cuda(
            cudaMemcpyAsync(
                local_slot,
                local_shard_buffers[static_cast<size_t>(recv)],
                shard_bytes,
                cudaMemcpyDeviceToDevice,
                st->streams[static_cast<size_t>(recv)]),
            "cudaMemcpyAsync(local shard -> full_output local slot)");
    }

    // Each sender pushes its shard into channel(sender -> recv, slot 0).
    for (int sender = 0; sender < st->world_size; ++sender) {
        system::runtime::set_device(st->devices[static_cast<size_t>(sender)]);

        for (int recv = 0; recv < st->world_size; ++recv) {
            if (recv == sender) continue;

            Buffer* slot_buf = channel_get_slot_buffer(st, sender, recv, 0);
            zero_buffer_on_owner(st, *slot_buf);

            system::runtime::check_cuda(
                channel_send_bulk_tma(
                    st,
                    sender,
                    recv,
                    0,
                    local_shard_buffers[static_cast<size_t>(sender)],
                    shard_numel,
                    st->streams[static_cast<size_t>(sender)]),
                "channel_send_bulk_tma AG");
        }
    }

    sync_all_streams(st, "cudaStreamSynchronize(AG sends)");

    // Each receiver copies received shards into its full output slots.
    for (int recv = 0; recv < st->world_size; ++recv) {
        Buffer* full_out = communicator_get_local_full_buffer(st, recv);

        system::runtime::set_device(st->devices[static_cast<size_t>(recv)]);
        for (int sender = 0; sender < st->world_size; ++sender) {
            if (sender == recv) continue;

            const Buffer* slot_buf = channel_get_slot_buffer(st, sender, recv, 0);
            half* dst_slot = buffer_as_half(full_out) + static_cast<size_t>(sender) * shard_numel;

            system::runtime::check_cuda(
                cudaMemcpyAsync(
                    dst_slot,
                    slot_buf->ptr,
                    shard_bytes,
                    cudaMemcpyDeviceToDevice,
                    st->streams[static_cast<size_t>(recv)]),
                "cudaMemcpyAsync(channel slot -> full_output slot)");
        }
    }

    sync_all_streams(st, "cudaStreamSynchronize(AG assemble)");
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
        local_shards[static_cast<size_t>(r)] = basic_collective_shard_output_ptr(st, r);
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
        system::runtime::set_device(devices[static_cast<size_t>(r)]);
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
            basic_collective_shard_output_ptr(&st, r),
            static_cast<int64_t>(shard_numel),
            devices[static_cast<size_t>(r)]);

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
        system::runtime::set_device(devices[static_cast<size_t>(r)]);
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
            basic_collective_full_output_ptr(&st, r),
            static_cast<int64_t>(full_numel_sz),
            devices[static_cast<size_t>(r)]);

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
        system::runtime::set_device(devices[static_cast<size_t>(r)]);
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
            basic_collective_full_output_ptr(&st, r),
            static_cast<int64_t>(full_numel_sz),
            devices[static_cast<size_t>(r)]);

        testing::expect_allclose(got, ref, "basic_ngpu_all_reduce");
    }

    free_local_buffers(devices, local_full);
    destroy_basic_collective_same_process(&st);
    return true;
}

} // namespace ooverlap
