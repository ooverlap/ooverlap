#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace ooverlap {
namespace sync {

struct semaphore {
private:
    uint64_t value;
};

__device__ __forceinline__ void init_semaphore(
    semaphore& bar,
    int thread_count,
    int transaction_count = 0) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :
        : "r"(bar_ptr), "r"(thread_count + transaction_count));
}

__device__ __forceinline__ void wait(semaphore& bar, int phase_bit) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1 bra.uni DONE;\n"
        "bra.uni LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :
        : "r"(bar_ptr), "r"(phase_bit));
}

} // namespace hopper
} // namespace ooverlap
