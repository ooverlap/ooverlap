#include "comm/channel_worker.h"

namespace ooverlap {
namespace comm {

void channel_worker_reset(
    ChannelWorker* worker) {
    if (worker == nullptr) {
        return;
    }

    worker->owner_rank = -1;
    worker->src_rank = -1;
    worker->dst_rank = -1;
    worker->endpoint = nullptr;
    worker->channel = nullptr;
    worker->dispatch_queue = transport::DeviceDispatchQueueHandle{};
    worker->direct_control = transport::DeviceDirectReduceControlHandle{};
}

bool channel_worker_bind(
    ChannelWorker* worker,
    Endpoint* endpoint,
    Channel* channel) {
    if (worker == nullptr || endpoint == nullptr || channel == nullptr) {
        return false;
    }

    worker->owner_rank = endpoint->rank;
    worker->src_rank = channel->src_rank;
    worker->dst_rank = channel->dst_rank;
    worker->endpoint = endpoint;
    worker->channel = channel;
    worker->dispatch_queue =
        transport::dispatch_queue_get_device_handle(&channel->dispatch_queue);

    if (channel_is_direct_reduce(channel)) {
        worker->direct_control =
            transport::direct_reduce_control_get_device_handle(&channel->direct_control);
    } else {
        worker->direct_control = transport::DeviceDirectReduceControlHandle{};
    }

    return true;
}

} // namespace comm
} // namespace ooverlap
