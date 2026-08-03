#pragma once

#include <cuda_runtime.h>

namespace ooverlap {
namespace comm {
namespace kernels {

/*
 * Ready-signal protocols.
 *
 * DeviceMemoryStoreRelease:
 *   Fast path for one-writer ready slots in peer-visible device memory.
 *   Fixed prologue/epilogue rendezvous uses receiver-local inbox slots:
 *   senders write remotely once, receivers poll local memory.
 *
 * DeviceMemoryAtomicMax:
 *   Compatibility/debug path.  More robust if the same rank can publish epochs
 *   out of order from multiple streams, but slower than single-writer store.
 *
 * HostMappedStoreRelease:
 *   SYS / cross-island fallback.  Ready slots live in mapped pinned host memory.
 *   Do not use GPU atomic RMW operations on host mapped memory for multi-GPU
 *   synchronization.  Use one writer per slot plus plain store/poll.
 */
enum class MultiGpuReadySignalProtocol : int {
    DeviceMemoryStoreRelease = 0,
    DeviceMemoryAtomicMax = 1,
    HostMappedStoreRelease = 2,
    Disabled = 3,
};

__host__ __device__ __forceinline__ int default_ready_signal_poll_sleep_cycles(
    MultiGpuReadySignalProtocol protocol) {
    switch (protocol) {
        case MultiGpuReadySignalProtocol::HostMappedStoreRelease:
            return 256;

        case MultiGpuReadySignalProtocol::DeviceMemoryStoreRelease:
        case MultiGpuReadySignalProtocol::DeviceMemoryAtomicMax:
            return 64;

        case MultiGpuReadySignalProtocol::Disabled:
        default:
            return 0;
    }
}

template <int MaxPeers>
struct MultiGpuReadySignalPlan {
    int peer_count = 0;

    /* Slot-0 pointers retained for planner-generated ready tasks. */
    const int* peer_ready_signals[MaxPeers] = {};

    /* Fixed prologue/epilogue directional inbox pointers. */
    int* peer_publish_signals[MaxPeers] = {};
    const int* local_wait_signals[MaxPeers] = {};

    MultiGpuReadySignalProtocol protocol =
        MultiGpuReadySignalProtocol::DeviceMemoryStoreRelease;

    int poll_sleep_cycles = 64;
};

template <int MaxPeers>
inline MultiGpuReadySignalPlan<MaxPeers> make_multi_gpu_ready_signal_plan(
    int peer_count,
    const int* const* peer_ready_signals,
    int* const* peer_publish_signals,
    const int* const* local_wait_signals,
    MultiGpuReadySignalProtocol protocol =
        MultiGpuReadySignalProtocol::DeviceMemoryStoreRelease,
    int poll_sleep_cycles = 0) {
    MultiGpuReadySignalPlan<MaxPeers> plan{};

    if (peer_count < 0) {
        peer_count = 0;
    }

    if (peer_count > MaxPeers) {
        peer_count = MaxPeers;
    }

    plan.peer_count = peer_count;
    plan.protocol = protocol;
    plan.poll_sleep_cycles =
        poll_sleep_cycles > 0
            ? poll_sleep_cycles
            : default_ready_signal_poll_sleep_cycles(protocol);

    for (int i = 0; i < peer_count; ++i) {
        plan.peer_ready_signals[i] =
            peer_ready_signals != nullptr ? peer_ready_signals[i] : nullptr;
        plan.peer_publish_signals[i] =
            peer_publish_signals != nullptr ? peer_publish_signals[i] : nullptr;
        plan.local_wait_signals[i] =
            local_wait_signals != nullptr ? local_wait_signals[i] : nullptr;
    }

    return plan;
}

__device__ __forceinline__ void publish_ready_signal_store_release(
    int* ready_signal,
    int collective_epoch) {
    if (ready_signal == nullptr) {
        return;
    }

    /*
     * Single-writer slot protocol:
     *   owner rank writes ready[owner_rank]
     *   all peers only read that slot
     *
     * This avoids atomic RMW on the fast path.  The system fence after the
     * volatile store keeps the published epoch visible outside the writer GPU.
     */
#if defined(__CUDA_ARCH__)
    __threadfence_system();
#endif

    volatile int* ready =
        reinterpret_cast<volatile int*>(ready_signal);

    ready[0] = collective_epoch;
}

__device__ __forceinline__ void publish_ready_signal_atomic_max(
    int* ready_signal,
    int collective_epoch) {
    if (ready_signal == nullptr) {
        return;
    }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
    atomicMax_system(ready_signal, collective_epoch);
#else
    atomicMax(ready_signal, collective_epoch);
#endif

#if defined(__CUDA_ARCH__)
    __threadfence_system();
#endif
}

__device__ __forceinline__ void publish_ready_signal(
    int* ready_signal,
    int collective_epoch,
    MultiGpuReadySignalProtocol protocol) {
    switch (protocol) {
        case MultiGpuReadySignalProtocol::DeviceMemoryAtomicMax:
            publish_ready_signal_atomic_max(
                ready_signal,
                collective_epoch);
            return;

        case MultiGpuReadySignalProtocol::DeviceMemoryStoreRelease:
        case MultiGpuReadySignalProtocol::HostMappedStoreRelease:
            publish_ready_signal_store_release(
                ready_signal,
                collective_epoch);
            return;

        case MultiGpuReadySignalProtocol::Disabled:
        default:
            return;
    }
}

__device__ __forceinline__ int load_ready_signal(
    const int* ready_signal) {

    const volatile int* ready =
        reinterpret_cast<const volatile int*>(ready_signal);

    return ready[0];
}

__device__ __forceinline__ void wait_until_ready_signal_at_least(
    const int* ready_signal,
    int collective_epoch,
    int poll_sleep_cycles) {

    while (load_ready_signal(ready_signal) < collective_epoch) {
#if defined(__CUDA_ARCH__)
        __nanosleep(
            poll_sleep_cycles > 0 ?
                static_cast<unsigned int>(poll_sleep_cycles) :
                16u);
#endif
    }
}

/*
 * Distribute fixed rank rendezvous edges over CTAs. Each CTA first publishes
 * every peer inbox assigned to it, then polls the matching receiver-local
 * inboxes. Publishing all assigned edges before waiting avoids ordering cycles
 * when one CTA owns multiple peers.
 */
template <int MaxPeers>
__device__ __forceinline__ void distributed_ready_rendezvous_for_cta(
    MultiGpuReadySignalPlan<MaxPeers> ready_plan,
    int ready_value) {

    if (threadIdx.x != 0) {
        return;
    }

    const int first_peer = static_cast<int>(blockIdx.x);
    const int peer_stride = static_cast<int>(gridDim.x);

    for (int peer_idx = first_peer;
         peer_idx < ready_plan.peer_count;
         peer_idx += peer_stride) {
        publish_ready_signal(
            ready_plan.peer_publish_signals[peer_idx],
            ready_value,
            ready_plan.protocol);
    }

    for (int peer_idx = first_peer;
         peer_idx < ready_plan.peer_count;
         peer_idx += peer_stride) {
        wait_until_ready_signal_at_least(
            ready_plan.local_wait_signals[peer_idx],
            ready_value,
            ready_plan.poll_sleep_cycles);
    }
}


/* OOVERLAP_READY_PUBLISH_WAIT_MERGE_PATCH: combined publish+wait helper.
 *
 * Exactly one CTA, selected by owner_cta, performs the publish. Every CTA
 * waits on the peer signal. This preserves the "all CTAs wait before later
 * work" property while avoiding duplicate publishes.
 */
__device__ __forceinline__ void publish_then_wait_ready_signal_for_cta(
    int cta_idx,
    int owner_cta,
    int* publish_signal,
    int publish_epoch,
    MultiGpuReadySignalProtocol publish_protocol,
    const int* wait_signal,
    int wait_epoch,
    int wait_poll_sleep_cycles) {
    if (wait_signal == nullptr || wait_epoch <= 0) {
        return;
    }

    if (cta_idx == owner_cta &&
        publish_signal != nullptr &&
        publish_epoch > 0) {
        publish_ready_signal(
            publish_signal,
            publish_epoch,
            publish_protocol);
    }

    wait_until_ready_signal_at_least(
        wait_signal,
        wait_epoch,
        wait_poll_sleep_cycles);
}

template <int MaxPeers>
__device__ __forceinline__ void wait_for_multi_gpu_collective_ready(
    int* local_ready_signal,
    MultiGpuReadySignalPlan<MaxPeers> ready_plan,
    int collective_epoch) {
    if (collective_epoch <= 0 ||
        ready_plan.protocol == MultiGpuReadySignalProtocol::Disabled) {
        return;
    }

    if (local_ready_signal == nullptr) {
        return;
    }

    if (threadIdx.x == 0) {
        publish_ready_signal(
            local_ready_signal,
            collective_epoch,
            ready_plan.protocol);

        for (int peer_idx = 0; peer_idx < ready_plan.peer_count; ++peer_idx) {
            wait_until_ready_signal_at_least(
                ready_plan.peer_ready_signals[peer_idx],
                collective_epoch,
                ready_plan.poll_sleep_cycles);
        }
    }

    __syncthreads();
}

} // namespace kernels
} // namespace comm
} // namespace ooverlap
