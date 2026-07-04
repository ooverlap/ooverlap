#pragma once

#include "topology/topology.h"

namespace ooverlap {
namespace topology {
namespace detail {

ProbeResult probe_direct_load_store(int src_device, int dst_device);
ProbeResult probe_direct_atomic_add_i32(int src_device, int dst_device);
ProbeResult probe_direct_tma_load_f16(int src_device, int dst_device);
ProbeResult probe_direct_tma_store_f16(int src_device, int dst_device);
ProbeResult probe_direct_tma_reduce_f16(int src_device, int dst_device);

ProbeResult probe_shm_load_store(int src_device, int dst_device);
ProbeResult probe_shm_atomic_add_i32(int src_device, int dst_device);
ProbeResult probe_shm_tma_load_f16(int src_device, int dst_device);
ProbeResult probe_shm_tma_store_f16(int src_device, int dst_device);
ProbeResult probe_shm_tma_reduce_f16(int src_device, int dst_device);

} // namespace detail
} // namespace topology
} // namespace ooverlap
