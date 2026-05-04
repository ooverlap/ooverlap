#pragma once

#include <cuda_runtime.h>

namespace ooverlap {
namespace comm {
namespace kernels {

template <int MaxPeers>
struct MultiGpuReadySignalPlan {
    int peer_count = 0;
    const int* peer_ready_signals[MaxPeers] = {};
};

template <int MaxPeers>
inline MultiGpuReadySignalPlan<MaxPeers> make_multi_gpu_ready_signal_plan(
    int peer_count,
    const int* const* peer_ready_signals) {
    MultiGpuReadySignalPlan<MaxPeers> plan{};
    plan.peer_count = peer_count;

    for (int i = 0; i < peer_count && i < MaxPeers; ++i) {
        plan.peer_ready_signals[i] =
            peer_ready_signals != nullptr ? peer_ready_signals[i] : nullptr;
    }

    return plan;
}

template <int MaxPeers>
__device__ __forceinline__ void wait_for_multi_gpu_collective_ready(
    int* local_ready_signal,
    MultiGpuReadySignalPlan<MaxPeers> ready_plan,
    int collective_epoch) {
    if (local_ready_signal == nullptr || collective_epoch <= 0) {
        return;
    }

    if (threadIdx.x == 0) {
        atomicMax(local_ready_signal, collective_epoch);
        __threadfence_system();

        for (int peer_idx = 0; peer_idx < ready_plan.peer_count; ++peer_idx) {
            const int* peer_ready_signal =
                ready_plan.peer_ready_signals[peer_idx];

            if (peer_ready_signal == nullptr) {
                continue;
            }

            const volatile int* peer_ready =
                reinterpret_cast<const volatile int*>(peer_ready_signal);

            while (peer_ready[0] < collective_epoch) {
#if defined(__CUDA_ARCH__)
                __nanosleep(64);
#endif
            }
        }
    }

    __syncthreads();
}

} // namespace kernels
} // namespace comm
} // namespace ooverlap
