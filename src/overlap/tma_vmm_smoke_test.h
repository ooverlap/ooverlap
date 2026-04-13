#pragma once

#include <cstddef>
#include <cstdint>

namespace ooverlap {

bool tma_vmm_smoke_test(
    int64_t num_elements,
    int src_device = 0,
    int dst_device = 1);

} // namespace ooverlap
