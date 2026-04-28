#pragma once

#include <cuda_runtime_api.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct oo_group oo_group_t;
typedef struct oo_node oo_node_t;
typedef struct oo_buffer oo_buffer_t;

typedef enum {
    OO_SUCCESS = 0,
    OO_ERROR_INVALID_ARGUMENT,
    OO_ERROR_INVALID_DEVICE,
    OO_ERROR_UNSUPPORTED,
    OO_ERROR_CUDA,
    OO_ERROR_INTERNAL
} oo_status_t;

typedef enum {
    OO_DTYPE_FLOAT16 = 0,
    OO_DTYPE_BFLOAT16 = 1,
    OO_DTYPE_FLOAT32 = 2
} oo_dtype_t;

typedef enum {
    OO_REDUCE_ADD = 0,

    // Alias for users coming from NCCL-style naming.
    // Sum and add are the same operation for this collective.
    OO_REDUCE_SUM = OO_REDUCE_ADD,

    OO_REDUCE_MIN = 1,
    OO_REDUCE_MAX = 2
} oo_reduce_op_t;

typedef enum {
    OO_BUFFER_KIND_VMM = 0,
    OO_BUFFER_KIND_WRAPPED = 1
} oo_buffer_kind_t;

typedef enum {
    OO_TUNING_BEST_PERFORMANCE = 0,
    OO_TUNING_BEST_EFFICIENCY = 1
} oo_tuning_mode_t;

/* Utility */
size_t oo_dtype_size(
    oo_dtype_t dtype);

int oo_allreduce_supported(
    oo_dtype_t dtype,
    oo_reduce_op_t op);

/* Group */
oo_status_t oo_group_create(
    const int* devices,
    int num_devices,
    oo_group_t** out_group);

oo_status_t oo_group_create_ipc(
    const int* devices,
    int num_devices,
    int local_rank,
    const char* broker_key,
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
/* IPC / synchronization helpers */
oo_status_t oo_group_sync(
    oo_group_t* group);

/*
 * For multiprocess IPC groups:
 * - exports this rank's local wrapped/allocation buffer through the group's broker
 * - imports the peer rank's buffer
 * - returns an oo_buffer_t for the peer buffer
 *
 * This is intended for externally allocated CUDA buffers, e.g. PyTorch tensors
 * wrapped with oo_buffer_wrap().
 */
oo_status_t oo_buffer_exchange_ipc_peer(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t** out_peer);

/*
 * Run allreduce on a contiguous slice of already-registered local/peer buffers.
 * element_offset and count are in elements, not bytes.
 */
oo_status_t oo_allreduce_offset(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* peer,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    cudaStream_t stream);

oo_status_t oo_allreduce(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* peer,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    cudaStream_t stream);

oo_status_t oo_allreduce_offset_tuned(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* peer,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    oo_tuning_mode_t tuning_mode,
    cudaStream_t stream);

oo_status_t oo_allreduce_tuned(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* peer,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    oo_tuning_mode_t tuning_mode,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
