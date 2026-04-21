#pragma once

#include <cstddef>
#include <vector>

#include "comm/channel.h"
#include "comm/endpoint.h"

namespace ooverlap {
namespace comm {

struct Group {
    int world_size = 0;

    std::vector<int> devices;
    std::vector<Endpoint> endpoints;
    std::vector<Channel> channels;

    size_t channel_bytes = 0;
};

bool group_init(
    Group* group,
    const std::vector<int>& devices,
    size_t channel_bytes);

void group_destroy(
    Group* group);

Endpoint* group_get_endpoint(
    Group* group,
    int rank);

const Endpoint* group_get_endpoint(
    const Group* group,
    int rank);

Channel* group_get_channel(
    Group* group,
    int src_rank,
    int dst_rank);

const Channel* group_get_channel(
    const Group* group,
    int src_rank,
    int dst_rank);

} // namespace comm
} // namespace ooverlap
