#pragma once

namespace ooverlap {
namespace comm {
namespace plan {

constexpr int kTmaMultiGpuAllReduceMaxPeers = 15;
constexpr int kTmaMultiGpuReduceScatterMaxPeers = 15;
constexpr int kTmaMultiGpuAllGatherMaxPeers = 15;

constexpr int kTmaMultiGpuAllReduceMaxWindowTasks = 256;
constexpr int kTmaMultiGpuReduceScatterMaxWindowTasks = 256;
constexpr int kTmaMultiGpuAllGatherMaxWindowTasks = 256;

/* OOVERLAP_BY_VALUE_CAPACITY_DISPATCH_V1 */
constexpr int kTmaMultiGpuByValueCapacity16 = 16;
constexpr int kTmaMultiGpuByValueCapacity32 = 32;
constexpr int kTmaMultiGpuByValueCapacity64 = 64;
constexpr int kTmaMultiGpuByValueCapacity128 = 112;
constexpr int kTmaMultiGpuByValueMaxWindowTasks =
    kTmaMultiGpuByValueCapacity128;

} // namespace plan
} // namespace comm
} // namespace ooverlap
