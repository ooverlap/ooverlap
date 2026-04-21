#include "comm/group.h"

#include "ooverlap/system/runtime_utils.cuh"

#include <stdexcept>
#include <vector>

namespace ooverlap {
namespace comm {
namespace {

void channel_reset_metadata(Channel* ch) {
    if (ch == nullptr) {
        return;
    }

    ch->src_rank = -1;
    ch->dst_rank = -1;
    ch->src_device = -1;
    ch->dst_device = -1;
    ch->bytes = 0;
    ch->buffer = transport::CommBuffer{};
}

} // namespace

bool group_init(
    Group* group,
    const std::vector<int>& devices,
    size_t channel_bytes) {
    if (group == nullptr) {
        throw std::invalid_argument("group_init: group is null");
    }
    if (devices.size() < 2) {
        throw std::invalid_argument("group_init: need at least 2 devices");
    }
    if (channel_bytes == 0) {
        throw std::invalid_argument("group_init: channel_bytes must be > 0");
    }

    group_destroy(group);

    group->world_size = static_cast<int>(devices.size());
    group->devices = devices;
    group->channel_bytes = channel_bytes;

    group->endpoints.resize(devices.size());
    group->channels.resize(static_cast<size_t>(group->world_size * group->world_size));

    for (int rank = 0; rank < group->world_size; ++rank) {
        system::runtime::ensure_context_on_device(group->devices[static_cast<size_t>(rank)]);

        Endpoint ep{};
        ep.rank = rank;
        ep.device = group->devices[static_cast<size_t>(rank)];
        ep.stream = system::runtime::create_stream_on_device(ep.device);

        group->endpoints[static_cast<size_t>(rank)] = ep;
    }

    for (int src = 0; src < group->world_size; ++src) {
        for (int dst = 0; dst < group->world_size; ++dst) {
            Channel& ch =
                group->channels[static_cast<size_t>(
                    channel_index(group->world_size, src, dst))];

            ch.src_rank = src;
            ch.dst_rank = dst;
            ch.src_device = group->devices[static_cast<size_t>(src)];
            ch.dst_device = group->devices[static_cast<size_t>(dst)];

            if (src == dst) {
                ch.bytes = 0;
                ch.buffer = transport::CommBuffer{};
                continue;
            }

            ch.bytes = channel_bytes;
            ch.buffer = transport::alloc_peer_visible_buffer_for_rank(
                group->devices,
                dst,
                channel_bytes);
        }
    }

    return true;
}

void group_destroy(
    Group* group) {
    if (group == nullptr) {
        return;
    }

    for (auto& ch : group->channels) {
        transport::free_comm_buffer(group->devices, ch.buffer);
        channel_reset_metadata(&ch);
    }

    for (auto& ep : group->endpoints) {
        if (ep.stream != nullptr) {
            system::runtime::destroy_stream_on_device(ep.device, ep.stream);
        }
        ep = Endpoint{};
    }

    group->world_size = 0;
    group->devices.clear();
    group->endpoints.clear();
    group->channels.clear();
    group->channel_bytes = 0;
}

Endpoint* group_get_endpoint(
    Group* group,
    int rank) {
    if (group == nullptr) {
        throw std::invalid_argument("group_get_endpoint: group is null");
    }
    validate_rank_or_throw(group->world_size, rank, "group_get_endpoint: invalid rank");
    return &group->endpoints[static_cast<size_t>(rank)];
}

const Endpoint* group_get_endpoint(
    const Group* group,
    int rank) {
    if (group == nullptr) {
        throw std::invalid_argument("group_get_endpoint: group is null");
    }
    validate_rank_or_throw(group->world_size, rank, "group_get_endpoint: invalid rank");
    return &group->endpoints[static_cast<size_t>(rank)];
}

Channel* group_get_channel(
    Group* group,
    int src_rank,
    int dst_rank) {
    if (group == nullptr) {
        throw std::invalid_argument("group_get_channel: group is null");
    }
    validate_rank_or_throw(group->world_size, src_rank, "group_get_channel: invalid src_rank");
    validate_rank_or_throw(group->world_size, dst_rank, "group_get_channel: invalid dst_rank");

    if (src_rank == dst_rank) {
        return nullptr;
    }

    return &group->channels[static_cast<size_t>(
        channel_index(group->world_size, src_rank, dst_rank))];
}

const Channel* group_get_channel(
    const Group* group,
    int src_rank,
    int dst_rank) {
    if (group == nullptr) {
        throw std::invalid_argument("group_get_channel: group is null");
    }
    validate_rank_or_throw(group->world_size, src_rank, "group_get_channel: invalid src_rank");
    validate_rank_or_throw(group->world_size, dst_rank, "group_get_channel: invalid dst_rank");

    if (src_rank == dst_rank) {
        return nullptr;
    }

    return &group->channels[static_cast<size_t>(
        channel_index(group->world_size, src_rank, dst_rank))];
}

} // namespace comm
} // namespace ooverlap
