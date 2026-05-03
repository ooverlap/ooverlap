#include "comm/ooverlap_comm_internal.h"

#include "comm/tma_two_gpu_peer_allreduce_sm90.h"
#include "ooverlap/system/logging.h"
#include "comm/tuning/tuning_policy.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

constexpr size_t kOoReadySignalBytes = sizeof(int);

oo_status_t report_cuda_error(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        OOVERLAP_LOG_ERROR(
            "%s failed: %s\n",
            what,
            cudaGetErrorString(err));
        return OO_ERROR_CUDA;
    }
    return OO_SUCCESS;
}

oo_status_t report_exception(const char* where, const std::exception& e) {
    OOVERLAP_LOG_ERROR("%s threw: %s\n", where, e.what());
    return OO_ERROR_INTERNAL;
}

oo_status_t report_unknown_exception(const char* where) {
    OOVERLAP_LOG_ERROR("%s threw unknown exception\n", where);
    return OO_ERROR_INTERNAL;
}

oo_status_t cuda_status_to_oo(cudaError_t err) {
    return (err == cudaSuccess) ? OO_SUCCESS : OO_ERROR_CUDA;
}

bool valid_cuda_device(int device) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return false;
    }
    return device >= 0 && device < count;
}

bool checked_mul_size(size_t a, size_t b, size_t* out) {
    if (out == nullptr) {
        return false;
    }
    if (a != 0 && b > static_cast<size_t>(-1) / a) {
        return false;
    }
    *out = a * b;
    return true;
}

bool reduce_op_supported_for_dtype(
    oo_dtype_t dtype,
    oo_reduce_op_t op) {
    if (op == OO_REDUCE_ADD) {
        return dtype == OO_DTYPE_FLOAT16 ||
               dtype == OO_DTYPE_BFLOAT16 ||
               dtype == OO_DTYPE_FLOAT32;
    }

    if (op == OO_REDUCE_MIN || op == OO_REDUCE_MAX) {
        return dtype == OO_DTYPE_FLOAT16 ||
               dtype == OO_DTYPE_BFLOAT16;
    }

    return false;
}

bool same_group(const oo_group_t* a, const oo_group_t* b) {
    return a != nullptr && b != nullptr && a == b;
}

int find_rank_for_device(const oo_group_t* group, int device) {
    if (group == nullptr) {
        return -1;
    }

    for (int r = 0; r < group->num_devices; ++r) {
        if (group->devices[r] == device) {
            return r;
        }
    }

    return -1;
}

std::vector<int> group_devices_vector(const oo_group_t* group) {
    std::vector<int> out;
    if (group == nullptr) {
        return out;
    }

    out.reserve(static_cast<size_t>(group->num_devices));
    for (int i = 0; i < group->num_devices; ++i) {
        out.push_back(group->devices[i]);
    }

    return out;
}

void clear_ready_signal_slot(oo_ready_signal& slot) {
    slot.ptr = nullptr;
    slot.bytes = 0;
    slot.mapped_bytes = 0;
    slot.owner_rank = -1;
    slot.owner_device = -1;
    slot.kind = oo_ready_signal_kind::empty;
    slot.owned_legacy_ptr = nullptr;
}

void free_ready_signal_slot(oo_ready_signal& slot) {
    switch (slot.kind) {
        case oo_ready_signal_kind::owned_vmm:
            ooverlap::system::free_peer_visible_buffer(slot.owned_vmm);
            break;

        case oo_ready_signal_kind::owned_legacy:
            if (slot.owned_legacy_ptr != nullptr) {
                if (slot.owner_device >= 0) {
                    cudaSetDevice(slot.owner_device);
                }
                cudaFree(slot.owned_legacy_ptr);
                slot.owned_legacy_ptr = nullptr;
            }
            break;

        case oo_ready_signal_kind::imported_legacy:
            if (slot.ptr != nullptr) {
                cudaIpcCloseMemHandle(slot.ptr);
            }
            break;

        case oo_ready_signal_kind::imported_vmm:
            slot.imported.reset();
            break;

        case oo_ready_signal_kind::empty:
        default:
            break;
    }

    clear_ready_signal_slot(slot);
}

void mirror_ready_signal_for_compat(oo_group_t* group, int rank) {
    if (group == nullptr || rank < 0 || rank >= group->num_devices) {
        return;
    }

    oo_ready_signal& slot = group->ready_signal_slots[rank];
    group->ready_signals[rank].ptr = slot.ptr;
    group->ready_signals[rank].mapped_size = slot.mapped_bytes;
    group->ready_signals[rank].requested_size = slot.bytes;
    group->ready_signals[rank].owner_device = slot.owner_device;
}

void clear_ready_signal_compat_mirror(oo_group_t* group, int rank) {
    if (group == nullptr || rank < 0 || rank >= kOoMaxLocalDevices) {
        return;
    }

    group->ready_signals[rank].ptr = nullptr;
    group->ready_signals[rank].mapped_size = 0;
    group->ready_signals[rank].requested_size = 0;
    group->ready_signals[rank].owner_device = -1;
}

void free_group_ready_signals_same_process(oo_group_t* group) {
    if (group == nullptr) {
        return;
    }

    for (int r = 0; r < group->num_devices; ++r) {
        free_ready_signal_slot(group->ready_signal_slots[r]);
        clear_ready_signal_compat_mirror(group, r);
    }
}

void free_group_ready_signals_ipc(oo_group_t* group) {
    if (group == nullptr) {
        return;
    }

    try {
        if (group->broker) {
            group->broker->sync();
        }
    } catch (...) {
    }

    for (int r = 0; r < group->num_devices; ++r) {
        oo_ready_signal& slot = group->ready_signal_slots[r];
        if (slot.kind == oo_ready_signal_kind::imported_legacy ||
            slot.kind == oo_ready_signal_kind::imported_vmm) {
            free_ready_signal_slot(slot);
            clear_ready_signal_compat_mirror(group, r);
        }
    }

    try {
        if (group->broker) {
            group->broker->sync();
        }
    } catch (...) {
    }

    const int local_rank = group->local_rank;
    if (local_rank >= 0 && local_rank < group->num_devices) {
        oo_ready_signal& local_slot = group->ready_signal_slots[local_rank];
        if (local_slot.kind == oo_ready_signal_kind::owned_legacy ||
            local_slot.kind == oo_ready_signal_kind::owned_vmm) {
            free_ready_signal_slot(local_slot);
            clear_ready_signal_compat_mirror(group, local_rank);
        }
    }

    try {
        if (group->broker) {
            group->broker->sync();
        }
    } catch (...) {
    }
}

void free_group_ready_signals(oo_group_t* group) {
    if (group == nullptr) {
        return;
    }

    if (group->bootstrap_kind == oo_group_bootstrap_kind::multiprocess_ipc) {
        free_group_ready_signals_ipc(group);
    } else {
        free_group_ready_signals_same_process(group);
    }
}

oo_status_t init_group_ready_signals_same_process(oo_group_t* group) {
    if (group == nullptr || group->num_devices <= 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    try {
        std::vector<int> access_devices = group_devices_vector(group);

        for (int rank = 0; rank < group->num_devices; ++rank) {
            oo_ready_signal& slot = group->ready_signal_slots[rank];

            slot.owned_vmm =
                ooverlap::system::alloc_peer_visible_buffer(
                    kOoReadySignalBytes,
                    group->devices[rank],
                    access_devices);

            slot.ptr = slot.owned_vmm.ptr;
            slot.bytes = kOoReadySignalBytes;
            slot.mapped_bytes = slot.owned_vmm.mapped_size;
            slot.owner_rank = rank;
            slot.owner_device = group->devices[rank];
            slot.kind = oo_ready_signal_kind::owned_vmm;

            mirror_ready_signal_for_compat(group, rank);

            cudaError_t set_err = cudaSetDevice(group->devices[rank]);
            if (set_err != cudaSuccess) {
                free_group_ready_signals_same_process(group);
                return OO_ERROR_CUDA;
            }

            cudaError_t memset_err = cudaMemset(
                slot.ptr,
                0,
                slot.mapped_bytes);
            if (memset_err != cudaSuccess) {
                free_group_ready_signals_same_process(group);
                return OO_ERROR_CUDA;
            }
        }
    } catch (const std::bad_alloc&) {
        free_group_ready_signals_same_process(group);
        return OO_ERROR_INTERNAL;
    } catch (const std::exception&) {
        free_group_ready_signals_same_process(group);
        return OO_ERROR_CUDA;
    } catch (...) {
        free_group_ready_signals_same_process(group);
        return OO_ERROR_INTERNAL;
    }

    return OO_SUCCESS;
}

oo_status_t init_group_ready_signals_ipc(oo_group_t* group) {
    if (group == nullptr ||
        group->num_devices <= 0 ||
        group->local_rank < 0 ||
        group->local_rank >= group->num_devices ||
        !group->broker) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    const int local_rank = group->local_rank;
    const int signal_owner_rank = 0;
    const int signal_owner_device = group->devices[signal_owner_rank];

    const size_t signal_block_bytes =
        static_cast<size_t>(group->num_devices) * kOoReadySignalBytes;

    struct ready_signal_block_desc {
        cudaIpcMemHandle_t handle{};
        std::uint64_t bytes = 0;
        int owner_rank = -1;
        int owner_device = -1;
    };

    auto cleanup_ready_signals_local_only = [&]() {
        for (int r = 0; r < group->num_devices; ++r) {
            oo_ready_signal& slot = group->ready_signal_slots[r];

            if (slot.kind == oo_ready_signal_kind::owned_legacy) {
                if (slot.owned_legacy_ptr != nullptr) {
                    cudaSetDevice(slot.owner_device);
                    cudaFree(slot.owned_legacy_ptr);
                }
            } else if (slot.kind == oo_ready_signal_kind::imported_legacy) {
                if (slot.ptr != nullptr) {
                    cudaIpcCloseMemHandle(slot.ptr);
                }
            } else if (slot.kind == oo_ready_signal_kind::owned_vmm ||
                       slot.kind == oo_ready_signal_kind::imported_vmm) {
                free_ready_signal_slot(slot);
            }

            clear_ready_signal_slot(slot);
            clear_ready_signal_compat_mirror(group, r);
        }
    };

    try {
        void* signal_base = nullptr;
        ready_signal_block_desc local_desc{};

        if (local_rank == signal_owner_rank) {
            cudaError_t err = cudaSetDevice(signal_owner_device);
            if (err != cudaSuccess) {
                return report_cuda_error(err, "cudaSetDevice(signal_owner_device)");
            }

            err = cudaMalloc(&signal_base, signal_block_bytes);
            if (err != cudaSuccess) {
                return report_cuda_error(err, "cudaMalloc(legacy ready-signal block)");
            }

            err = cudaMemset(signal_base, 0, signal_block_bytes);
            if (err != cudaSuccess) {
                cudaFree(signal_base);
                return report_cuda_error(err, "cudaMemset(legacy ready-signal block)");
            }

            local_desc.bytes = static_cast<std::uint64_t>(signal_block_bytes);
            local_desc.owner_rank = signal_owner_rank;
            local_desc.owner_device = signal_owner_device;

            err = cudaIpcGetMemHandle(&local_desc.handle, signal_base);
            if (err != cudaSuccess) {
                cudaFree(signal_base);
                return report_cuda_error(err, "cudaIpcGetMemHandle(ready-signal block)");
            }

            oo_ready_signal& owner_slot = group->ready_signal_slots[signal_owner_rank];
            owner_slot.ptr = signal_base;
            owner_slot.bytes = signal_block_bytes;
            owner_slot.mapped_bytes = signal_block_bytes;
            owner_slot.owner_rank = signal_owner_rank;
            owner_slot.owner_device = signal_owner_device;
            owner_slot.kind = oo_ready_signal_kind::owned_legacy;
            owner_slot.owned_legacy_ptr = signal_base;
        }

        static_assert(
            sizeof(ready_signal_block_desc) <=
                ooverlap::system::broker_detail::VAULT_SIZE_PER_RANK,
            "ready_signal_block_desc does not fit in Broker exchange vault");

        std::vector<ready_signal_block_desc> all_desc(
            static_cast<size_t>(group->num_devices));

        OOVERLAP_LOG_DEBUG(
            "ready signal before broker exchange_data rank=%d broker.get()=%p group=%p sizeof(Broker)=%zu\n",
            local_rank,
            static_cast<void*>(group->broker.get()),
            static_cast<void*>(group),
            sizeof(ooverlap::system::Broker));

        group->broker->exchange_data(
            all_desc.data(),
            &local_desc,
            sizeof(local_desc));

        const ready_signal_block_desc signal_desc = all_desc[signal_owner_rank];

        if (signal_desc.bytes != signal_block_bytes ||
            signal_desc.owner_rank != signal_owner_rank ||
            signal_desc.owner_device != signal_owner_device) {
            OOVERLAP_LOG_ERROR(
                "bad legacy ready-signal descriptor: bytes=%llu owner_rank=%d owner_device=%d expected_bytes=%llu expected_owner_rank=%d expected_owner_device=%d\n",
                static_cast<unsigned long long>(signal_desc.bytes),
                signal_desc.owner_rank,
                signal_desc.owner_device,
                static_cast<unsigned long long>(signal_block_bytes),
                signal_owner_rank,
                signal_owner_device);

            cleanup_ready_signals_local_only();
            return OO_ERROR_INTERNAL;
        }

        if (local_rank != signal_owner_rank) {
            cudaError_t err = cudaSetDevice(group->devices[local_rank]);
            if (err != cudaSuccess) {
                cleanup_ready_signals_local_only();
                return report_cuda_error(err, "cudaSetDevice(before ready-signal import)");
            }

            err = cudaIpcOpenMemHandle(
                &signal_base,
                signal_desc.handle,
                cudaIpcMemLazyEnablePeerAccess);

            if (err != cudaSuccess) {
                OOVERLAP_LOG_ERROR(
                    "cudaIpcOpenMemHandle(ready-signal block) failed local_rank=%d local_device=%d owner_device=%d: %s\n",
                    local_rank,
                    group->devices[local_rank],
                    signal_owner_device,
                    cudaGetErrorString(err));

                cleanup_ready_signals_local_only();
                return OO_ERROR_CUDA;
            }

            oo_ready_signal& imported_owner_slot =
                group->ready_signal_slots[signal_owner_rank];

            imported_owner_slot.ptr = signal_base;
            imported_owner_slot.bytes = signal_block_bytes;
            imported_owner_slot.mapped_bytes = signal_block_bytes;
            imported_owner_slot.owner_rank = signal_owner_rank;
            imported_owner_slot.owner_device = signal_owner_device;
            imported_owner_slot.kind = oo_ready_signal_kind::imported_legacy;
        }

        if (signal_base == nullptr) {
            cleanup_ready_signals_local_only();
            return OO_ERROR_INTERNAL;
        }

        for (int rank = 0; rank < group->num_devices; ++rank) {
            oo_ready_signal& slot = group->ready_signal_slots[rank];

            void* slot_ptr = reinterpret_cast<void*>(
                reinterpret_cast<std::uint8_t*>(signal_base) +
                static_cast<size_t>(rank) * kOoReadySignalBytes);

            if (rank == signal_owner_rank) {
                slot.ptr = slot_ptr;
                slot.bytes = kOoReadySignalBytes;
                slot.mapped_bytes = kOoReadySignalBytes;
                slot.owner_rank = rank;
                slot.owner_device = group->devices[rank];

                if (local_rank == signal_owner_rank) {
                    slot.kind = oo_ready_signal_kind::owned_legacy;
                    slot.owned_legacy_ptr = signal_base;
                } else {
                    slot.kind = oo_ready_signal_kind::imported_legacy;
                }
            } else {
                slot.ptr = slot_ptr;
                slot.bytes = kOoReadySignalBytes;
                slot.mapped_bytes = kOoReadySignalBytes;
                slot.owner_rank = rank;
                slot.owner_device = group->devices[rank];
                slot.kind = oo_ready_signal_kind::empty;
            }

            mirror_ready_signal_for_compat(group, rank);
        }

        group->broker->sync();

        return OO_SUCCESS;

    } catch (const std::bad_alloc&) {
        cleanup_ready_signals_local_only();
        return OO_ERROR_INTERNAL;
    } catch (const std::exception& e) {
        cleanup_ready_signals_local_only();
        return report_exception("init_group_ready_signals_ipc", e);
    } catch (...) {
        cleanup_ready_signals_local_only();
        return report_unknown_exception("init_group_ready_signals_ipc");
    }
}

oo_status_t init_group_ready_signals(oo_group_t* group) {
    if (group == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (group->bootstrap_kind == oo_group_bootstrap_kind::multiprocess_ipc) {
        return init_group_ready_signals_ipc(group);
    }

    return init_group_ready_signals_same_process(group);
}

bool buffer_is_valid_for_allreduce(
    const oo_buffer_t* buffer,
    size_t required_bytes) {
    if (buffer == nullptr) {
        return false;
    }
    if (buffer->ptr == nullptr) {
        return false;
    }
    if (buffer->bytes < required_bytes) {
        return false;
    }
    if (buffer->mapped_bytes < required_bytes) {
        return false;
    }
    if (buffer->owner_device < 0) {
        return false;
    }
    if (buffer->system_kind == ooverlap::system::peer_buffer_kind::empty) {
        return false;
    }

    return true;
}

bool buffer_is_imported(const oo_buffer_t* buffer) {
    return buffer != nullptr &&
           (buffer->system_kind == ooverlap::system::peer_buffer_kind::imported_legacy ||
            buffer->system_kind == ooverlap::system::peer_buffer_kind::imported_vmm);
}

void initialize_buffer_from_view(
    oo_buffer_t* buffer,
    oo_node_t* node,
    const ooverlap::system::peer_buffer_view& view,
    oo_buffer_kind_t public_kind) {
    buffer->ptr = view.ptr;
    buffer->bytes = view.bytes;
    buffer->mapped_bytes = view.mapped_size;
    buffer->kind = public_kind;
    buffer->group = node->group;
    buffer->owner_device = view.owner_device;
    buffer->owner_rank = find_rank_for_device(node->group, view.owner_device);
    buffer->system_kind = view.kind;
}

} // namespace

oo_status_t oo_buffer_export_legacy_descriptor(
    oo_buffer_t* buffer,
    ooverlap::system::legacy_peer_buffer_descriptor* out_desc) {
    if (buffer == nullptr || out_desc == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    if (buffer->ptr == nullptr || buffer->bytes == 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    if (buffer->owner_device < 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (buffer->system_kind == ooverlap::system::peer_buffer_kind::owned_vmm ||
        buffer->kind == OO_BUFFER_KIND_VMM) {
        return OO_ERROR_UNSUPPORTED;
    }

    try {
        *out_desc = ooverlap::system::export_legacy_peer_buffer(
            buffer->ptr,
            buffer->bytes,
            buffer->owner_device,
            buffer->mapped_bytes != 0 ? buffer->mapped_bytes : buffer->bytes);
    } catch (const std::bad_alloc&) {
        return OO_ERROR_INTERNAL;
    } catch (const std::exception&) {
        return OO_ERROR_CUDA;
    } catch (...) {
        return OO_ERROR_INTERNAL;
    }

    return OO_SUCCESS;
}

oo_status_t oo_buffer_import_legacy_descriptor(
    oo_node_t* node,
    const ooverlap::system::legacy_peer_buffer_descriptor& desc,
    oo_buffer_t** out_buffer) {
    if (out_buffer == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    *out_buffer = nullptr;

    if (node == nullptr || node->group == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    if (desc.bytes == 0 || desc.mapped_size == 0 || desc.owner_device < 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    const int owner_rank = find_rank_for_device(node->group, desc.owner_device);
    if (owner_rank < 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    oo_buffer_t* buffer = new (std::nothrow) oo_buffer_t;
    if (buffer == nullptr) {
        return OO_ERROR_INTERNAL;
    }

    try {
        std::vector<int> access_devices = group_devices_vector(node->group);

        cudaError_t set_err = cudaSetDevice(node->device);
        if (set_err != cudaSuccess) {
            delete buffer;
            return OO_ERROR_CUDA;
        }

        ooverlap::system::imported_peer_buffer imported =
            ooverlap::system::import_legacy_peer_buffer(
                desc,
                access_devices);

        ooverlap::system::peer_buffer_view view = imported.view();
        initialize_buffer_from_view(buffer, node, view, OO_BUFFER_KIND_WRAPPED);

        buffer->owner_rank = owner_rank;
        buffer->imported = std::move(imported);

    } catch (const std::bad_alloc&) {
        delete buffer;
        return OO_ERROR_INTERNAL;
    } catch (const std::exception&) {
        delete buffer;
        return OO_ERROR_CUDA;
    } catch (...) {
        delete buffer;
        return OO_ERROR_INTERNAL;
    }

    *out_buffer = buffer;
    return OO_SUCCESS;
}

oo_status_t oo_buffer_adopt_imported_peer_buffer(
    oo_node_t* node,
    ooverlap::system::imported_peer_buffer&& imported,
    oo_buffer_t** out_buffer) {
    if (out_buffer == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    *out_buffer = nullptr;

    if (node == nullptr || node->group == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    if (!imported.valid()) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    const int owner_rank = find_rank_for_device(node->group, imported.owner_device);
    if (owner_rank < 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    oo_buffer_t* buffer = new (std::nothrow) oo_buffer_t;
    if (buffer == nullptr) {
        return OO_ERROR_INTERNAL;
    }

    ooverlap::system::peer_buffer_view view = imported.view();
    initialize_buffer_from_view(buffer, node, view, OO_BUFFER_KIND_WRAPPED);
    buffer->owner_rank = owner_rank;
    buffer->imported = std::move(imported);

    *out_buffer = buffer;
    return OO_SUCCESS;
}

extern "C" {

size_t oo_dtype_size(
    oo_dtype_t dtype) {
    switch (dtype) {
        case OO_DTYPE_FLOAT16:
            return sizeof(half);
        case OO_DTYPE_BFLOAT16:
            return sizeof(__nv_bfloat16);
        case OO_DTYPE_FLOAT32:
            return sizeof(float);
        default:
            return 0;
    }
}

int oo_allreduce_supported(
    oo_dtype_t dtype,
    oo_reduce_op_t op) {
    return reduce_op_supported_for_dtype(dtype, op) ? 1 : 0;
}

oo_status_t oo_group_create(
    const int* devices,
    int num_devices,
    oo_group_t** out_group) {
    if (out_group == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    *out_group = nullptr;

    if (devices == nullptr || num_devices <= 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (num_devices != 2) {
        return OO_ERROR_UNSUPPORTED;
    }

    if (num_devices > kOoMaxLocalDevices) {
        return OO_ERROR_UNSUPPORTED;
    }

    for (int i = 0; i < num_devices; ++i) {
        if (!valid_cuda_device(devices[i])) {
            return OO_ERROR_INVALID_DEVICE;
        }
        for (int j = 0; j < i; ++j) {
            if (devices[i] == devices[j]) {
                return OO_ERROR_INVALID_ARGUMENT;
            }
        }
    }

    oo_group_t* group = new (std::nothrow) oo_group_t;
    if (group == nullptr) {
        return OO_ERROR_INTERNAL;
    }

    group->num_devices = num_devices;
    group->bootstrap_kind = oo_group_bootstrap_kind::same_process;
    group->local_rank = -1;
    group->local_world_size = num_devices;

    for (int i = 0; i < num_devices; ++i) {
        group->devices[i] = devices[i];
    }

    oo_status_t sig_status = init_group_ready_signals(group);
    if (sig_status != OO_SUCCESS) {
        delete group;
        return sig_status;
    }

    *out_group = group;
    return OO_SUCCESS;
}

oo_status_t oo_group_create_ipc(
    const int* devices,
    int num_devices,
    int local_rank,
    const char* broker_key,
    oo_group_t** out_group) {
    if (out_group == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    *out_group = nullptr;

    if (devices == nullptr || broker_key == nullptr || num_devices <= 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (num_devices != 2) {
        return OO_ERROR_UNSUPPORTED;
    }

    if (num_devices > kOoMaxLocalDevices) {
        return OO_ERROR_UNSUPPORTED;
    }

    if (local_rank < 0 || local_rank >= num_devices) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    for (int i = 0; i < num_devices; ++i) {
        if (!valid_cuda_device(devices[i])) {
            return OO_ERROR_INVALID_DEVICE;
        }
        for (int j = 0; j < i; ++j) {
            if (devices[i] == devices[j]) {
                return OO_ERROR_INVALID_ARGUMENT;
            }
        }
    }

    oo_group_t* group = new (std::nothrow) oo_group_t;
    if (group == nullptr) {
        return OO_ERROR_INTERNAL;
    }

    group->num_devices = num_devices;
    group->bootstrap_kind = oo_group_bootstrap_kind::multiprocess_ipc;
    group->local_rank = local_rank;
    group->local_world_size = num_devices;

    for (int i = 0; i < num_devices; ++i) {
        group->devices[i] = devices[i];
    }

    try {
        group->broker = std::make_unique<ooverlap::system::Broker>(
            local_rank,
            num_devices,
            broker_key);
    } catch (const std::bad_alloc&) {
        delete group;
        return OO_ERROR_INTERNAL;
    } catch (const std::exception& e) {
        OOVERLAP_LOG_ERROR("oo_group_create_ipc broker init threw: %s\n", e.what());
        delete group;
        return OO_ERROR_INTERNAL;
    } catch (...) {
        delete group;
        return OO_ERROR_INTERNAL;
    }

    oo_status_t sig_status = init_group_ready_signals(group);
    if (sig_status != OO_SUCCESS) {
        delete group;
        return sig_status;
    }

    *out_group = group;
    return OO_SUCCESS;
}

void oo_group_destroy(
    oo_group_t* group) {
    if (group == nullptr) {
        return;
    }

    free_group_ready_signals(group);
    delete group;
}

int oo_group_size(
    const oo_group_t* group) {
    return (group != nullptr) ? group->num_devices : 0;
}

int oo_group_device(
    const oo_group_t* group,
    int rank) {
    if (group == nullptr || rank < 0 || rank >= group->num_devices) {
        return -1;
    }
    return group->devices[rank];
}

oo_status_t oo_node_create(
    oo_group_t* group,
    int rank,
    oo_node_t** out_node) {
    if (out_node == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    *out_node = nullptr;

    if (group == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    if (rank < 0 || rank >= group->num_devices) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (group->bootstrap_kind == oo_group_bootstrap_kind::multiprocess_ipc &&
        rank != group->local_rank) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    oo_node_t* node = new (std::nothrow) oo_node_t;
    if (node == nullptr) {
        return OO_ERROR_INTERNAL;
    }

    node->group = group;
    node->rank = rank;
    node->device = group->devices[rank];
    node->collective_epoch = 0;

    *out_node = node;
    return OO_SUCCESS;
}

void oo_node_destroy(
    oo_node_t* node) {
    delete node;
}

oo_group_t* oo_node_group(
    const oo_node_t* node) {
    return (node != nullptr) ? node->group : nullptr;
}

int oo_node_rank(
    const oo_node_t* node) {
    return (node != nullptr) ? node->rank : -1;
}

int oo_node_device(
    const oo_node_t* node) {
    return (node != nullptr) ? node->device : -1;
}

oo_status_t oo_buffer_alloc(
    oo_node_t* node,
    size_t bytes,
    oo_buffer_t** out_buffer) {
    if (out_buffer == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    *out_buffer = nullptr;

    if (node == nullptr || node->group == nullptr || bytes == 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    oo_buffer_t* buffer = new (std::nothrow) oo_buffer_t;
    if (buffer == nullptr) {
        return OO_ERROR_INTERNAL;
    }

    try {
        std::vector<int> access_devices = group_devices_vector(node->group);

        buffer->mapped = ooverlap::system::alloc_peer_visible_buffer(
            bytes,
            node->device,
            access_devices);

        ooverlap::system::peer_buffer_view view =
            ooverlap::system::make_owned_peer_buffer_view(
                buffer->mapped,
                bytes);

        initialize_buffer_from_view(buffer, node, view, OO_BUFFER_KIND_VMM);
        buffer->owner_rank = node->rank;
    } catch (const std::bad_alloc&) {
        delete buffer;
        return OO_ERROR_INTERNAL;
    } catch (const std::exception& e) {
        OOVERLAP_LOG_ERROR("oo_buffer_alloc threw: %s\n", e.what());
        delete buffer;
        return OO_ERROR_CUDA;
    } catch (...) {
        delete buffer;
        return OO_ERROR_INTERNAL;
    }

    *out_buffer = buffer;
    return OO_SUCCESS;
}

oo_status_t oo_buffer_wrap(
    oo_node_t* node,
    void* ptr,
    size_t bytes,
    oo_buffer_t** out_buffer) {
    if (out_buffer == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    *out_buffer = nullptr;

    if (node == nullptr || node->group == nullptr ||
        ptr == nullptr || bytes == 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    oo_buffer_t* buffer = new (std::nothrow) oo_buffer_t;
    if (buffer == nullptr) {
        return OO_ERROR_INTERNAL;
    }

    ooverlap::system::peer_buffer_view view =
        ooverlap::system::make_wrapped_peer_buffer_view(
            ptr,
            bytes,
            node->device);

    initialize_buffer_from_view(buffer, node, view, OO_BUFFER_KIND_WRAPPED);
    buffer->owner_rank = node->rank;

    *out_buffer = buffer;
    return OO_SUCCESS;
}

void oo_buffer_destroy(
    oo_buffer_t* buffer) {
    if (buffer == nullptr) {
        return;
    }

    if (buffer->system_kind == ooverlap::system::peer_buffer_kind::owned_vmm ||
        buffer->kind == OO_BUFFER_KIND_VMM) {
        ooverlap::system::free_peer_visible_buffer(buffer->mapped);
    } else if (
        buffer->system_kind == ooverlap::system::peer_buffer_kind::imported_legacy ||
        buffer->system_kind == ooverlap::system::peer_buffer_kind::imported_vmm) {
        buffer->imported.reset();
    }

    delete buffer;
}

void* oo_buffer_ptr(
    const oo_buffer_t* buffer) {
    return (buffer != nullptr) ? buffer->ptr : nullptr;
}

size_t oo_buffer_bytes(
    const oo_buffer_t* buffer) {
    return (buffer != nullptr) ? buffer->bytes : 0;
}

size_t oo_buffer_mapped_bytes(
    const oo_buffer_t* buffer) {
    return (buffer != nullptr) ? buffer->mapped_bytes : 0;
}

oo_buffer_kind_t oo_buffer_kind(
    const oo_buffer_t* buffer) {
    return (buffer != nullptr) ? buffer->kind : OO_BUFFER_KIND_WRAPPED;
}

oo_status_t oo_group_sync(
    oo_group_t* group) {
    if (group == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (!group->broker) {
        return OO_SUCCESS;
    }

    try {
        group->broker->sync();
        return OO_SUCCESS;
    } catch (const std::bad_alloc&) {
        return OO_ERROR_INTERNAL;
    } catch (const std::exception& e) {
        OOVERLAP_LOG_ERROR("oo_group_sync threw: %s\n", e.what());
        return OO_ERROR_INTERNAL;
    } catch (...) {
        return OO_ERROR_INTERNAL;
    }
}

oo_status_t oo_buffer_exchange_ipc_peer(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t** out_peer) {
    if (out_peer == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    *out_peer = nullptr;

    if (node == nullptr || node->group == nullptr || local == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    oo_group_t* group = node->group;

    if (group->bootstrap_kind != oo_group_bootstrap_kind::multiprocess_ipc) {
        return OO_ERROR_UNSUPPORTED;
    }

    if (!group->broker) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (group->num_devices != 2) {
        return OO_ERROR_UNSUPPORTED;
    }

    if (node->rank < 0 || node->rank >= group->num_devices) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    ooverlap::system::legacy_peer_buffer_descriptor local_desc{};

    oo_status_t st = oo_buffer_export_legacy_descriptor(
        local,
        &local_desc);

    if (st != OO_SUCCESS) {
        return st;
    }

    try {
        std::vector<ooverlap::system::legacy_peer_buffer_descriptor> all_desc(
            static_cast<size_t>(group->num_devices));

        group->broker->exchange_data(
            all_desc.data(),
            &local_desc,
            sizeof(local_desc));

        const int peer_rank = node->rank ^ 1;

        st = oo_buffer_import_legacy_descriptor(
            node,
            all_desc[peer_rank],
            out_peer);

        if (st != OO_SUCCESS) {
            return st;
        }

        group->broker->sync();

        return OO_SUCCESS;

    } catch (const std::bad_alloc&) {
        return OO_ERROR_INTERNAL;
    } catch (const std::exception& e) {
        OOVERLAP_LOG_ERROR("oo_buffer_exchange_ipc_peer threw: %s\n", e.what());
        return OO_ERROR_INTERNAL;
    } catch (...) {
        return OO_ERROR_INTERNAL;
    }
}

oo_status_t oo_allreduce_offset_impl(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* peer,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    oo_tuning_mode_t tuning_mode,
    cudaStream_t stream) {
    if (node == nullptr || node->group == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (local == nullptr || peer == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (count == 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    oo_group_t* group = node->group;

    if (group->num_devices != 2) {
        return OO_ERROR_UNSUPPORTED;
    }

    const int rank = node->rank;

    if (rank != 0 && rank != 1) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    const int peer_rank = 1 - rank;

    if (!same_group(local->group, group) ||
        !same_group(peer->group, group)) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (local->owner_rank != rank) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (peer->owner_rank != peer_rank) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (!reduce_op_supported_for_dtype(dtype, op)) {
        return OO_ERROR_UNSUPPORTED;
    }

    const size_t dtype_bytes = oo_dtype_size(dtype);

    if (dtype_bytes == 0) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    size_t end_element = 0;
    if (element_offset > static_cast<size_t>(-1) - count) {
        return OO_ERROR_INVALID_ARGUMENT;
    }
    end_element = element_offset + count;

    size_t required_bytes = 0;
    if (!checked_mul_size(end_element, dtype_bytes, &required_bytes)) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    size_t byte_offset = 0;
    if (!checked_mul_size(element_offset, dtype_bytes, &byte_offset)) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    const size_t transfer_bytes = count * dtype_bytes;

    if (!buffer_is_valid_for_allreduce(local, required_bytes)) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (!buffer_is_valid_for_allreduce(peer, required_bytes)) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    /*
     * Multiprocess IPC correctness guard:
     *
     * In IPC mode, the peer buffer must actually be an imported peer mapping.
     * Otherwise a user could accidentally pass a local/wrapped pointer for the
     * peer side and bypass the provenance guarantee.
     */
    if (group->bootstrap_kind == oo_group_bootstrap_kind::multiprocess_ipc &&
        !buffer_is_imported(peer)) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    if (group->ready_signal_slots[rank].ptr == nullptr ||
        group->ready_signal_slots[peer_rank].ptr == nullptr) {
        return OO_ERROR_INVALID_ARGUMENT;
    }

    void* local_ptr =
        reinterpret_cast<void*>(
            reinterpret_cast<unsigned char*>(local->ptr) + byte_offset);

    void* peer_ptr =
        reinterpret_cast<void*>(
            reinterpret_cast<unsigned char*>(peer->ptr) + byte_offset);

    int* local_ready_signal =
        reinterpret_cast<int*>(group->ready_signal_slots[rank].ptr);

    const int* peer_ready_signal =
        reinterpret_cast<const int*>(group->ready_signal_slots[peer_rank].ptr);

    const int dev0 = group->devices[0];
    const int dev1 = group->devices[1];

    const int collective_epoch = ++node->collective_epoch;

    const ooverlap::comm::TuningPreference preference =
        ooverlap::comm::tuning_preference_from_public(tuning_mode);

    ooverlap::comm::LaunchConfig launch_config =
        ooverlap::comm::select_launch_config_for_allreduce(
            transfer_bytes,
            preference);

    cudaError_t err =
        ooverlap::enqueue_tma_two_gpu_peer_allreduce_rank_sm90(
            local_ptr,
            local_ptr,
            peer_ptr,
            count,
            dtype,
            op,
            rank,
            dev0,
            dev1,
            stream,
            local_ready_signal,
            peer_ready_signal,
            collective_epoch,
            launch_config);

    return report_cuda_error(
        err,
        "enqueue_tma_two_gpu_peer_allreduce_rank_sm90");
}

oo_status_t oo_allreduce_offset_tuned(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* peer,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    oo_tuning_mode_t tuning_mode,
    cudaStream_t stream) {
    try {
        return oo_allreduce_offset_impl(
            node,
            local,
            peer,
            element_offset,
            count,
            dtype,
            op,
            tuning_mode,
            stream);
    } catch (const std::bad_alloc&) {
        return OO_ERROR_INTERNAL;
    } catch (const std::exception& e) {
        return report_exception("oo_allreduce_offset_tuned", e);
    } catch (...) {
        return report_unknown_exception("oo_allreduce_offset_tuned");
    }
}

oo_status_t oo_allreduce_offset(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* peer,
    size_t element_offset,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    cudaStream_t stream) {
    return oo_allreduce_offset_tuned(
        node,
        local,
        peer,
        element_offset,
        count,
        dtype,
        op,
        OO_TUNING_BEST_PERFORMANCE,
        stream);
}

oo_status_t oo_allreduce_tuned(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* peer,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    oo_tuning_mode_t tuning_mode,
    cudaStream_t stream) {
    return oo_allreduce_offset_tuned(
        node,
        local,
        peer,
        0,
        count,
        dtype,
        op,
        tuning_mode,
        stream);
}

oo_status_t oo_allreduce(
    oo_node_t* node,
    oo_buffer_t* local,
    oo_buffer_t* peer,
    size_t count,
    oo_dtype_t dtype,
    oo_reduce_op_t op,
    cudaStream_t stream) {
    return oo_allreduce_tuned(
        node,
        local,
        peer,
        count,
        dtype,
        op,
        OO_TUNING_BEST_PERFORMANCE,
        stream);
}

} // extern "C"
