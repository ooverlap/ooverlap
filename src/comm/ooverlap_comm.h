#pragma once

#include <cuda_runtime.h>
#include <stddef.h>

#include "ooverlap/system/peer_buffer.cuh"

#ifdef __cplusplus
extern "C" {
#endif

constexpr int kOoMaxLocalDevices = 16;

struct oo_group {
    int num_devices = 0;
    int devices[kOoMaxLocalDevices] = {};

    // One peer-visible ready signal per rank/device.
    // ready_signals[r] is physically owned by devices[r], visible to all
    // devices in this group.
    ooverlap::system::mapped_peer_buffer ready_signals[kOoMaxLocalDevices] = {};
};

typedef struct oo_group oo_group_t;

struct oo_node {
    oo_group_t* group = nullptr;
    int rank = -1;
    int device = -1;

    // Monotonic per-node collective sequence.
    // Rank-local calls must be issued in matching order, same as NCCL.
    int collective_epoch = 0;
};

typedef struct oo_node oo_node_t;

typedef enum {
    OO_SUCCESS = 0,
    OO_ERROR_INVALID_ARGUMENT,
    OO_ERROR_INVALID_DEVICE,
    OO_ERROR_UNSUPPORTED,
    OO_ERROR_CUDA,
    OO_ERROR_INTERNAL
} oo_status_t;

typedef enum {
    OO_DTYPE_FLOAT16 = 0
} oo_dtype_t;

typedef enum {
    OO_REDUCE_SUM = 0
} oo_reduce_op_t;

typedef enum {
    OO_BUFFER_KIND_VMM = 0,
    OO_BUFFER_KIND_WRAPPED = 1
} oo_buffer_kind_t;

struct oo_buffer {
    void* ptr = nullptr;
    size_t bytes = 0;
    size_t mapped_bytes = 0;
    oo_buffer_kind_t kind = OO_BUFFER_KIND_WRAPPED;

    // Internal validation/debug metadata. Public semantics should still treat
    // Buffer as pointer + size + kind.
    oo_group_t* group = nullptr;
    int owner_device = -1;

    // Valid only for OO_BUFFER_KIND_VMM.
    ooverlap::system::mapped_peer_buffer mapped{};
};

typedef struct oo_buffer oo_buffer_t;


/* Group */
oo_status_t oo_group_create(
    const int* devices,
    int num_devices,
    oo_group_t** out_group);

void oo_group_destroy(
    oo_group_t* group);

int oo_group_size(
    const oo_group_t* group);

int oo_group_device(
    const oo_group_t* group,
    int rank);

/* Node */
oo_status_t oo_node_create(
    oo_group_t* group,
    int rank,
    oo_node_t** out_node);

void oo_node_destroy(
    oo_node_t* node);

oo_group_t* oo_node_group(
    const oo_node_t* node);

int oo_node_rank(
    const oo_node_t* node);

int oo_node_device(
    const oo_node_t* node);

/* Buffer */
oo_status_t oo_buffer_alloc(
    oo_node_t* node,
    size_t bytes,
    oo_buffer_t** out_buffer);

oo_status_t oo_buffer_wrap(
    oo_node_t* node,
    void* ptr,
    size_t bytes,
    oo_buffer_t** out_buffer);

void oo_buffer_destroy(
    oo_buffer_t* buffer);

void* oo_buffer_ptr(
    const oo_buffer_t* buffer);

size_t oo_buffer_bytes(
    const oo_buffer_t* buffer);

size_t oo_buffer_mapped_bytes(
    const oo_buffer_t* buffer);

oo_buffer_kind_t oo_buffer_kind(
    const oo_buffer_t* buffer);

/* Collectives */
oo_status_t oo_allreduce(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* peer,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
