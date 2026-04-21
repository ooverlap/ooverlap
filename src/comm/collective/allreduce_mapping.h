#pragma once

#include <cstddef>
#include <cstdint>

namespace ooverlap {
namespace comm {
namespace collective {

static constexpr uint32_t kInvalidWindowIndex = 0xffffffffu;

enum class AllReducePhysicalDstKind : uint8_t {
    kInvalid = 0,
    kDirectFinal = 1,
    kIntermediateAccum = 2,
};

struct AllReducePhysicalTileMapping {
    uint32_t window_idx = kInvalidWindowIndex;
    AllReducePhysicalDstKind dst_kind = AllReducePhysicalDstKind::kInvalid;

    int dst_rank = -1;
    unsigned char* dst_base = nullptr;

    size_t bytes = 0;
    uint64_t logical_dst_offset_bytes = 0;
};

} // namespace collective
} // namespace comm
} // namespace ooverlap
