#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vmm.cuh"
#include "ipc.cuh"

namespace ooverlap {
namespace system {

/*
 * Same-process / same-address-space peer-visible VMM allocation.
 *
 * This is the original raw-branch abstraction, kept source-compatible:
 *   - ptr
 *   - mapped_size
 *
 * New metadata fields are appended after the original two fields, so normal
 * uses like buf.ptr and buf.mapped_size keep working.
 */
struct mapped_peer_buffer {
    void* ptr = nullptr;
    size_t mapped_size = 0;

    // New metadata. requested_size is the user-requested size before VMM rounding.
    size_t requested_size = 0;
    int owner_device = -1;
};

/*
 * Keep the existing allocator behavior:
 *   vmm::vm_alloc_map_set_access(...)
 *
 * This returns a local mapping to VMM memory owned by owner_device and accessible
 * by access_devices. This is useful in same-process tests and benchmarks.
 */
inline mapped_peer_buffer alloc_peer_visible_buffer(
    size_t bytes,
    int owner_device,
    const std::vector<int>& access_devices) {
    if (bytes == 0) {
        throw std::invalid_argument("alloc_peer_visible_buffer: bytes must be > 0");
    }
    if (owner_device < 0) {
        throw std::invalid_argument("alloc_peer_visible_buffer: owner_device must be >= 0");
    }
    if (access_devices.empty()) {
        throw std::invalid_argument("alloc_peer_visible_buffer: access_devices must not be empty");
    }

    mapped_peer_buffer out{};
    out.requested_size = bytes;
    out.owner_device = owner_device;

    vmm::vm_alloc_map_set_access(
        &out.ptr,
        &out.mapped_size,
        bytes,
        owner_device,
        access_devices);

    return out;
}

inline void free_peer_visible_buffer(mapped_peer_buffer& buf) {
    if (buf.ptr != nullptr && buf.mapped_size != 0) {
        vmm::vm_unmap(buf.ptr, buf.mapped_size);
    }

    buf.ptr = nullptr;
    buf.mapped_size = 0;
    buf.requested_size = 0;
    buf.owner_device = -1;
}

/*
 * Buffer provenance/kind.
 *
 * The important part for multiprocess support is that an imported IPC mapping
 * is not treated as an anonymous wrapped pointer. The comm layer can use this
 * kind to validate that peer buffers are actually imported peer mappings.
 */
enum class peer_buffer_kind {
    empty = 0,

    // Local same-process VMM allocation from alloc_peer_visible_buffer.
    owned_vmm = 1,

    // Pointer owned by some external allocator, e.g. torch/cudaMalloc.
    // This does NOT imply peer visibility or IPC import.
    wrapped = 2,

    // Pointer obtained by cudaIpcOpenMemHandle.
    imported_legacy = 3,

    // Pointer obtained by cuMemImportFromShareableHandle + VMM map.
    imported_vmm = 4
};

inline const char* peer_buffer_kind_name(peer_buffer_kind kind) {
    switch (kind) {
        case peer_buffer_kind::empty:
            return "empty";
        case peer_buffer_kind::owned_vmm:
            return "owned_vmm";
        case peer_buffer_kind::wrapped:
            return "wrapped";
        case peer_buffer_kind::imported_legacy:
            return "imported_legacy";
        case peer_buffer_kind::imported_vmm:
            return "imported_vmm";
        default:
            return "unknown";
    }
}

/*
 * Non-owning typed view used by comm code.
 *
 * This is deliberately small and cheap to pass around. It does not free memory.
 */
struct peer_buffer_view {
    void* ptr = nullptr;

    // Requested/user-visible byte size.
    size_t bytes = 0;

    // Actual mapped size. For legacy/wrapped this is usually bytes.
    // For VMM it may be rounded up to allocation granularity.
    size_t mapped_size = 0;

    int owner_device = -1;
    peer_buffer_kind kind = peer_buffer_kind::empty;
};

inline bool is_imported_peer_buffer(const peer_buffer_view& view) {
    return view.kind == peer_buffer_kind::imported_legacy ||
           view.kind == peer_buffer_kind::imported_vmm;
}

inline bool is_valid_peer_buffer_view(const peer_buffer_view& view) {
    return view.ptr != nullptr &&
           view.bytes != 0 &&
           view.mapped_size != 0 &&
           view.owner_device >= 0 &&
           view.kind != peer_buffer_kind::empty;
}

inline peer_buffer_view make_owned_peer_buffer_view(
    const mapped_peer_buffer& buf,
    size_t bytes = 0) {
    if (buf.ptr == nullptr || buf.mapped_size == 0) {
        throw std::invalid_argument("make_owned_peer_buffer_view: buffer is empty");
    }

    const size_t visible_bytes =
        bytes != 0 ? bytes :
        buf.requested_size != 0 ? buf.requested_size :
        buf.mapped_size;

    if (visible_bytes > buf.mapped_size) {
        throw std::invalid_argument("make_owned_peer_buffer_view: bytes exceed mapped_size");
    }

    peer_buffer_view out{};
    out.ptr = buf.ptr;
    out.bytes = visible_bytes;
    out.mapped_size = buf.mapped_size;
    out.owner_device = buf.owner_device;
    out.kind = peer_buffer_kind::owned_vmm;
    return out;
}

inline peer_buffer_view make_wrapped_peer_buffer_view(
    void* ptr,
    size_t bytes,
    int owner_device) {
    if (ptr == nullptr) {
        throw std::invalid_argument("make_wrapped_peer_buffer_view: ptr is null");
    }
    if (bytes == 0) {
        throw std::invalid_argument("make_wrapped_peer_buffer_view: bytes must be > 0");
    }
    if (owner_device < 0) {
        throw std::invalid_argument("make_wrapped_peer_buffer_view: owner_device must be >= 0");
    }

    peer_buffer_view out{};
    out.ptr = ptr;
    out.bytes = bytes;
    out.mapped_size = bytes;
    out.owner_device = owner_device;
    out.kind = peer_buffer_kind::wrapped;
    return out;
}

/*
 * Return a sub-view of a larger peer buffer.
 *
 * This is the helper you want for FlashOverlap segments:
 *
 *   full local/peer buffer registered once
 *   segment = slice_peer_buffer_view(full, acc_addr * sizeof(half), seg_bytes)
 */
inline peer_buffer_view slice_peer_buffer_view(
    const peer_buffer_view& base,
    size_t offset_bytes,
    size_t bytes) {
    if (!is_valid_peer_buffer_view(base)) {
        throw std::invalid_argument("slice_peer_buffer_view: base view is invalid");
    }
    if (bytes == 0) {
        throw std::invalid_argument("slice_peer_buffer_view: bytes must be > 0");
    }
    if (offset_bytes > base.bytes || bytes > base.bytes - offset_bytes) {
        throw std::out_of_range("slice_peer_buffer_view: slice exceeds base visible size");
    }

    peer_buffer_view out = base;
    out.ptr = reinterpret_cast<void*>(
        reinterpret_cast<std::uint8_t*>(base.ptr) + offset_bytes);
    out.bytes = bytes;

    /*
     * The slice is only a view. mapped_size here is the visible slice size, not
     * the full VMM reservation size. Freeing must happen through the owner
     * object, not through this view.
     */
    out.mapped_size = bytes;
    return out;
}

template <typename T>
inline T* peer_buffer_ptr(peer_buffer_view view, size_t byte_offset = 0) {
    if (!is_valid_peer_buffer_view(view)) {
        throw std::invalid_argument("peer_buffer_ptr: view is invalid");
    }
    if (byte_offset > view.bytes) {
        throw std::out_of_range("peer_buffer_ptr: byte_offset exceeds view size");
    }

    return reinterpret_cast<T*>(
        reinterpret_cast<std::uint8_t*>(view.ptr) + byte_offset);
}

/*
 * Legacy CUDA IPC descriptor.
 *
 * This is POD-like and can be exchanged through Broker::exchange_data or
 * Broker::exchange_pod.
 *
 * Use this for cudaMalloc / PyTorch CUDA storage pointers.
 */
struct legacy_peer_buffer_descriptor {
    ipc::legacy_handle handle{};
    std::uint64_t bytes = 0;
    std::uint64_t mapped_size = 0;
    int owner_device = -1;
};

inline legacy_peer_buffer_descriptor export_legacy_peer_buffer(
    void* ptr,
    size_t bytes,
    int owner_device,
    size_t mapped_size = 0) {
    if (ptr == nullptr) {
        throw std::invalid_argument("export_legacy_peer_buffer: ptr is null");
    }
    if (bytes == 0) {
        throw std::invalid_argument("export_legacy_peer_buffer: bytes must be > 0");
    }
    if (owner_device < 0) {
        throw std::invalid_argument("export_legacy_peer_buffer: owner_device must be >= 0");
    }

    legacy_peer_buffer_descriptor out{};
    ipc::export_handle(&out.handle, ptr);
    out.bytes = static_cast<std::uint64_t>(bytes);
    out.mapped_size = static_cast<std::uint64_t>(mapped_size != 0 ? mapped_size : bytes);
    out.owner_device = owner_device;
    return out;
}

/*
 * VMM descriptor.
 *
 * The FD itself is intentionally not stored here. POSIX FDs must be exchanged
 * with Broker::exchange_fds or Broker::broadcast_fd, not memcpy'd through shared
 * memory.
 */
struct vmm_peer_buffer_descriptor {
    std::uint64_t bytes = 0;
    std::uint64_t mapped_size = 0;
    int owner_device = -1;
};

inline vmm_peer_buffer_descriptor make_vmm_peer_buffer_descriptor(
    const mapped_peer_buffer& buf,
    size_t bytes = 0) {
    if (buf.ptr == nullptr || buf.mapped_size == 0) {
        throw std::invalid_argument("make_vmm_peer_buffer_descriptor: buffer is empty");
    }
    if (buf.owner_device < 0) {
        throw std::invalid_argument("make_vmm_peer_buffer_descriptor: owner_device is invalid");
    }

    const size_t visible_bytes =
        bytes != 0 ? bytes :
        buf.requested_size != 0 ? buf.requested_size :
        buf.mapped_size;

    if (visible_bytes > buf.mapped_size) {
        throw std::invalid_argument("make_vmm_peer_buffer_descriptor: bytes exceed mapped_size");
    }

    vmm_peer_buffer_descriptor out{};
    out.bytes = static_cast<std::uint64_t>(visible_bytes);
    out.mapped_size = static_cast<std::uint64_t>(buf.mapped_size);
    out.owner_device = buf.owner_device;
    return out;
}

inline ipc::vmm_handle export_vmm_peer_buffer_fd(void* ptr) {
    if (ptr == nullptr) {
        throw std::invalid_argument("export_vmm_peer_buffer_fd: ptr is null");
    }

    ipc::vmm_handle out{};
    ipc::export_handle(&out, ptr);
    return out;
}

inline ipc::vmm_handle export_vmm_peer_buffer_fd(const mapped_peer_buffer& buf) {
    if (buf.ptr == nullptr || buf.mapped_size == 0) {
        throw std::invalid_argument("export_vmm_peer_buffer_fd: buffer is empty");
    }

    return export_vmm_peer_buffer_fd(buf.ptr);
}

/*
 * RAII owner for imported mappings.
 *
 * This is the key change versus pretending imported peer memory is just wrapped:
 * the mapping knows how it was imported and how it must be closed.
 */
struct imported_peer_buffer {
    void* ptr = nullptr;

    // Requested/user-visible byte size.
    size_t bytes = 0;

    // Actual import/map size.
    size_t mapped_size = 0;

    int owner_device = -1;
    peer_buffer_kind kind = peer_buffer_kind::empty;

    imported_peer_buffer() = default;

    imported_peer_buffer(const imported_peer_buffer&) = delete;
    imported_peer_buffer& operator=(const imported_peer_buffer&) = delete;

    imported_peer_buffer(imported_peer_buffer&& other) noexcept
        : ptr(other.ptr),
          bytes(other.bytes),
          mapped_size(other.mapped_size),
          owner_device(other.owner_device),
          kind(other.kind) {
        other.ptr = nullptr;
        other.bytes = 0;
        other.mapped_size = 0;
        other.owner_device = -1;
        other.kind = peer_buffer_kind::empty;
    }

    imported_peer_buffer& operator=(imported_peer_buffer&& other) noexcept {
        if (this != &other) {
            reset();

            ptr = other.ptr;
            bytes = other.bytes;
            mapped_size = other.mapped_size;
            owner_device = other.owner_device;
            kind = other.kind;

            other.ptr = nullptr;
            other.bytes = 0;
            other.mapped_size = 0;
            other.owner_device = -1;
            other.kind = peer_buffer_kind::empty;
        }

        return *this;
    }

    ~imported_peer_buffer() {
        reset();
    }

    bool valid() const {
        return ptr != nullptr &&
               bytes != 0 &&
               mapped_size != 0 &&
               owner_device >= 0 &&
               is_imported_peer_buffer(view());
    }

    peer_buffer_view view() const {
        peer_buffer_view out{};
        out.ptr = ptr;
        out.bytes = bytes;
        out.mapped_size = mapped_size;
        out.owner_device = owner_device;
        out.kind = kind;
        return out;
    }

    void reset() {
        if (ptr != nullptr) {
            if (kind == peer_buffer_kind::imported_legacy) {
                ipc::free_handle<ipc::flavor::legacy>(ptr, mapped_size);
            } else if (kind == peer_buffer_kind::imported_vmm) {
                ipc::free_handle<ipc::flavor::vmm>(ptr, mapped_size);
            }
        }

        ptr = nullptr;
        bytes = 0;
        mapped_size = 0;
        owner_device = -1;
        kind = peer_buffer_kind::empty;
    }
};

inline imported_peer_buffer import_legacy_peer_buffer(
    legacy_peer_buffer_descriptor desc,
    const std::vector<int>& accessible_device_ids) {
    if (desc.bytes == 0 || desc.mapped_size == 0) {
        throw std::invalid_argument("import_legacy_peer_buffer: descriptor size is zero");
    }
    if (desc.owner_device < 0) {
        throw std::invalid_argument("import_legacy_peer_buffer: owner_device is invalid");
    }

    imported_peer_buffer out{};
    ipc::import_handle(
        &out.ptr,
        desc.handle,
        static_cast<size_t>(desc.mapped_size),
        accessible_device_ids);

    out.bytes = static_cast<size_t>(desc.bytes);
    out.mapped_size = static_cast<size_t>(desc.mapped_size);
    out.owner_device = desc.owner_device;
    out.kind = peer_buffer_kind::imported_legacy;
    return out;
}

inline imported_peer_buffer import_legacy_peer_buffer(
    legacy_peer_buffer_descriptor desc,
    int num_devices) {
    if (desc.bytes == 0 || desc.mapped_size == 0) {
        throw std::invalid_argument("import_legacy_peer_buffer: descriptor size is zero");
    }
    if (desc.owner_device < 0) {
        throw std::invalid_argument("import_legacy_peer_buffer: owner_device is invalid");
    }

    imported_peer_buffer out{};
    ipc::import_handle(
        &out.ptr,
        desc.handle,
        static_cast<size_t>(desc.mapped_size),
        num_devices);

    out.bytes = static_cast<size_t>(desc.bytes);
    out.mapped_size = static_cast<size_t>(desc.mapped_size);
    out.owner_device = desc.owner_device;
    out.kind = peer_buffer_kind::imported_legacy;
    return out;
}

inline imported_peer_buffer import_vmm_peer_buffer(
    ipc::vmm_handle& fd_handle,
    const vmm_peer_buffer_descriptor& desc,
    const std::vector<int>& accessible_device_ids) {
    if (desc.bytes == 0 || desc.mapped_size == 0) {
        throw std::invalid_argument("import_vmm_peer_buffer: descriptor size is zero");
    }
    if (desc.owner_device < 0) {
        throw std::invalid_argument("import_vmm_peer_buffer: owner_device is invalid");
    }
    if (fd_handle.value < 0) {
        throw std::invalid_argument("import_vmm_peer_buffer: fd handle is invalid");
    }

    imported_peer_buffer out{};
    ipc::import_handle(
        &out.ptr,
        fd_handle,
        static_cast<size_t>(desc.mapped_size),
        accessible_device_ids);

    out.bytes = static_cast<size_t>(desc.bytes);
    out.mapped_size = static_cast<size_t>(desc.mapped_size);
    out.owner_device = desc.owner_device;
    out.kind = peer_buffer_kind::imported_vmm;
    return out;
}

inline imported_peer_buffer import_vmm_peer_buffer(
    int fd,
    const vmm_peer_buffer_descriptor& desc,
    const std::vector<int>& accessible_device_ids) {
    ipc::vmm_handle h{};
    h.value = fd;
    return import_vmm_peer_buffer(h, desc, accessible_device_ids);
}

inline imported_peer_buffer import_vmm_peer_buffer(
    ipc::vmm_handle& fd_handle,
    const vmm_peer_buffer_descriptor& desc,
    int num_devices) {
    if (desc.bytes == 0 || desc.mapped_size == 0) {
        throw std::invalid_argument("import_vmm_peer_buffer: descriptor size is zero");
    }
    if (desc.owner_device < 0) {
        throw std::invalid_argument("import_vmm_peer_buffer: owner_device is invalid");
    }
    if (fd_handle.value < 0) {
        throw std::invalid_argument("import_vmm_peer_buffer: fd handle is invalid");
    }

    imported_peer_buffer out{};
    ipc::import_handle(
        &out.ptr,
        fd_handle,
        static_cast<size_t>(desc.mapped_size),
        num_devices);

    out.bytes = static_cast<size_t>(desc.bytes);
    out.mapped_size = static_cast<size_t>(desc.mapped_size);
    out.owner_device = desc.owner_device;
    out.kind = peer_buffer_kind::imported_vmm;
    return out;
}

inline imported_peer_buffer import_vmm_peer_buffer(
    int fd,
    const vmm_peer_buffer_descriptor& desc,
    int num_devices) {
    ipc::vmm_handle h{};
    h.value = fd;
    return import_vmm_peer_buffer(h, desc, num_devices);
}

/*
 * Convenience for comm validation.
 *
 * Use this in oo_allreduce validation later:
 *   - local can be owned_vmm, wrapped, or maybe imported self-buffer
 *   - peer should be imported_legacy/imported_vmm in multiprocess mode
 */
inline void require_peer_imported_buffer(const peer_buffer_view& view) {
    if (!is_valid_peer_buffer_view(view)) {
        throw std::invalid_argument("require_peer_imported_buffer: invalid view");
    }
    if (!is_imported_peer_buffer(view)) {
        throw std::invalid_argument(
            "require_peer_imported_buffer: expected imported_legacy or imported_vmm buffer");
    }
}

inline void require_accessible_buffer(const peer_buffer_view& view) {
    if (!is_valid_peer_buffer_view(view)) {
        throw std::invalid_argument("require_accessible_buffer: invalid view");
    }
}

/*
 * Compatibility helper for old code that only wants ptr/mapped_size.
 */
inline mapped_peer_buffer as_mapped_peer_buffer_unsafe(const peer_buffer_view& view) {
    if (!is_valid_peer_buffer_view(view)) {
        throw std::invalid_argument("as_mapped_peer_buffer_unsafe: invalid view");
    }

    mapped_peer_buffer out{};
    out.ptr = view.ptr;
    out.mapped_size = view.mapped_size;
    out.requested_size = view.bytes;
    out.owner_device = view.owner_device;
    return out;
}

} // namespace system
} // namespace ooverlap
