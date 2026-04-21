#pragma once

#include "comm/channel.h"
#include "comm/endpoint.h"

namespace ooverlap {
namespace comm {

struct ChannelWorker {
    int owner_rank = -1;
    int src_rank = -1;
    int dst_rank = -1;

    Endpoint* endpoint = nullptr;
    Channel* channel = nullptr;

    transport::DeviceDispatchQueueHandle dispatch_queue{};
    transport::DeviceDirectReduceControlHandle direct_control{};
};

void channel_worker_reset(
    ChannelWorker* worker);

bool channel_worker_bind(
    ChannelWorker* worker,
    Endpoint* endpoint,
    Channel* channel);

inline bool channel_worker_is_valid(
    const ChannelWorker* worker) {
    return worker != nullptr &&
           worker->owner_rank >= 0 &&
           worker->src_rank >= 0 &&
           worker->dst_rank >= 0 &&
           worker->endpoint != nullptr &&
           worker->channel != nullptr &&
           worker->dispatch_queue.records != nullptr;
}

} // namespace comm
} // namespace ooverlap
