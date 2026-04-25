#pragma once

/*
 * ooverlap/system/broker.cuh
 *
 * Small node-local process broker adapted from ThunderKittens' KittensBroker.
 *
 * Purpose:
 *   - one-node, one-process-per-GPU bootstrap
 *   - barrier/sync between local ranks
 *   - exchange small fixed-size blobs, e.g. cudaIpcMemHandle_t + metadata
 *   - exchange or broadcast POSIX file descriptors for CUDA VMM handles
 *
 * Important:
 *   - This is not a multi-node transport.
 *   - The broker key must be unique per local process group/job.
 *   - Do not call this in the hot path; use it only during init/register/destroy.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#error "ooverlap::system::Broker is not supported on Windows"
#endif

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <unistd.h>

namespace ooverlap {
namespace system {

namespace broker_detail {

static constexpr int MAX_LOCAL_WORLD_SIZE = 128;

/*
 * Keep this larger than sizeof(cudaIpcMemHandle_t). 64 bytes is enough for a
 * raw cudaIpcMemHandle_t, but 256 lets you exchange small descriptors like:
 *
 *   struct {
 *     cudaIpcMemHandle_t handle;
 *     uint64_t bytes;
 *     int owner_device;
 *   };
 */
static constexpr int VAULT_SIZE_PER_RANK = 256;

struct Vault {
    static constexpr int INIT_CODE = 0x4f4f4950; // "OOIP"

    int init;
    int barrier;
    int sense;

    uint8_t data[MAX_LOCAL_WORLD_SIZE * VAULT_SIZE_PER_RANK];
};

static constexpr size_t SHM_SIZE = ((sizeof(Vault) + 4095) / 4096) * 4096;

inline std::string sanitize_key_component(const char* key) {
    if (key == nullptr || key[0] == '\0') {
        throw std::runtime_error("Broker: key must be non-empty");
    }

    std::string out;
    out.reserve(std::strlen(key));

    for (const char* p = key; *p; ++p) {
        char c = *p;
        const bool ok =
            (c >= 'a' && c <= 'z') ||
            (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') ||
            c == '_' || c == '-' || c == '.';

        out.push_back(ok ? c : '_');
    }

    if (out.empty()) {
        throw std::runtime_error("Broker: sanitized key is empty");
    }

    return out;
}

inline std::string make_shm_key(const char* key) {
    std::string s = "/ooverlap_broker_";
    s += sanitize_key_component(key);

    /*
     * POSIX shm names are implementation-defined, but on Linux they should be
     * short and start with exactly one slash. Keep margin under common limits.
     */
    if (s.size() >= 240) {
        throw std::runtime_error("Broker: shm key is too long");
    }
    return s;
}

inline std::string make_socket_prefix(const char* key) {
    std::string s = "/tmp/ooverlap_broker_";
    s += sanitize_key_component(key);
    s += ".sock.";

    /*
     * sockaddr_un::sun_path is usually 108 bytes. We also append rank digits.
     */
    if (s.size() >= 90) {
        throw std::runtime_error("Broker: socket key is too long");
    }
    return s;
}

inline std::string make_socket_path(const std::string& prefix, int local_rank) {
    std::string s = prefix + std::to_string(local_rank);
    if (s.size() >= sizeof(sockaddr_un::sun_path)) {
        throw std::runtime_error("Broker: socket path is too long");
    }
    return s;
}

inline void wait_until_initialized(volatile Vault* vault, const char* where) {
    if (vault == nullptr) {
        throw std::runtime_error("Broker: vault pointer is null");
    }

    /*
     * Rank startup can race under Python multiprocessing spawn. Rank 1 can map
     * the shm before rank 0 has published INIT_CODE. Treat that as a waitable
     * state, not a fatal state.
     */
    int spins = 0;
    while (vault->init != Vault::INIT_CODE) {
        if ((++spins % 1000000) == 0) {
            std::fprintf(
                stderr,
                "[ooverlap][broker] waiting for vault init in %s, current init=0x%x expected=0x%x\n",
                where ? where : "unknown",
                vault->init,
                Vault::INIT_CODE);
            std::fflush(stderr);
        }
        usleep(1);
    }

    __sync_synchronize();
}

inline void init_sync(int local_rank, volatile Vault* vault) {
    if (vault == nullptr) {
        throw std::runtime_error("Broker: vault pointer is null");
    }

    if (local_rank == 0) {
        vault->barrier = 0;
        vault->sense = 0;
        __sync_synchronize();
        vault->init = Vault::INIT_CODE;
    } else {
        wait_until_initialized(vault, "init_sync");
    }

    __sync_synchronize();
}

inline void sync(int local_world_size, volatile Vault* vault) {
    if (local_world_size <= 0) {
        throw std::runtime_error("Broker: invalid local_world_size");
    }

    wait_until_initialized(vault, "sync");

    int arrived = __sync_add_and_fetch(&vault->barrier, 1);
    if (arrived == local_world_size) {
        vault->sense = 1;
    }

    while (!vault->sense) {
        usleep(1);
    }

    __sync_synchronize();

    arrived = __sync_add_and_fetch(&vault->barrier, -1);
    if (arrived == 0) {
        vault->sense = 0;
    }

    while (vault->sense) {
        usleep(1);
    }
}

inline void* create_shm(const char* key, size_t size) {
    /*
     * Unlink stale shm from crashed previous jobs using the same key.
     * Safe only if caller gives a unique key per live process group.
     */
    shm_unlink(key);

    int shm_fd = shm_open(key, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (shm_fd < 0) {
        throw std::runtime_error("Broker: failed to create shared memory");
    }

    if (ftruncate(shm_fd, static_cast<off_t>(size)) != 0) {
        shm_unlink(key);
        close(shm_fd);
        throw std::runtime_error("Broker: failed to resize shared memory");
    }

    void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    close(shm_fd);

    if (addr == MAP_FAILED) {
        shm_unlink(key);
        throw std::runtime_error("Broker: failed to map shared memory");
    }

    return addr;
}

inline void* open_shm(const char* key, size_t size) {
    int shm_fd = -1;

    while (true) {
        shm_fd = shm_open(key, O_RDWR | O_CLOEXEC, 0);
        if (shm_fd >= 0) {
            break;
        }
        if (errno != ENOENT) {
            throw std::runtime_error("Broker: failed to open shared memory");
        }
        usleep(1);
    }

    struct stat shm_st {};
    do {
        if (fstat(shm_fd, &shm_st) != 0) {
            close(shm_fd);
            throw std::runtime_error("Broker: failed to stat shared memory");
        }
        usleep(1);
    } while (static_cast<size_t>(shm_st.st_size) < size);

    void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    close(shm_fd);

    if (addr == MAP_FAILED) {
        throw std::runtime_error("Broker: failed to map shared memory");
    }

    return addr;
}

inline void unlink_shm(const char* key) {
    shm_unlink(key);
}

inline void unmap_shm(void* addr, size_t size) {
    if (addr != nullptr) {
        munmap(addr, size);
    }
}

inline int create_socket(const std::string& socket_prefix, int local_rank) {
    int sock_fd = socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
    if (sock_fd < 0) {
        throw std::runtime_error("Broker: socket creation failed");
    }

    const std::string socket_path = make_socket_path(socket_prefix, local_rank);
    unlink(socket_path.c_str());

    sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(sock_fd, reinterpret_cast<sockaddr*>(&addr), SUN_LEN(&addr)) < 0) {
        close(sock_fd);
        throw std::runtime_error("Broker: failed to bind socket");
    }

    return sock_fd;
}

inline void unlink_socket(const std::string& socket_prefix, int local_rank) {
    if (local_rank < 0) {
        return;
    }
    const std::string socket_path = make_socket_path(socket_prefix, local_rank);
    unlink(socket_path.c_str());
}

inline void close_socket(int sock_fd) {
    if (sock_fd >= 0) {
        close(sock_fd);
    }
}

inline void send_fd(
    int sock_fd,
    int data_fd,
    const std::string& dst_socket_prefix,
    int dst_local_rank,
    int src_local_rank
) {
    if (sock_fd < 0) {
        throw std::runtime_error("Broker: invalid socket fd");
    }
    if (data_fd < 0) {
        throw std::runtime_error("Broker: invalid data fd");
    }

    const std::string dst_path = make_socket_path(dst_socket_prefix, dst_local_rank);

    sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, dst_path.c_str(), sizeof(addr.sun_path) - 1);

    char control[CMSG_SPACE(sizeof(int))] {};
    msghdr msg {};
    msg.msg_name = &addr;
    msg.msg_namelen = sizeof(addr);
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);

    cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    std::memcpy(CMSG_DATA(cmsg), &data_fd, sizeof(data_fd));

    int payload = src_local_rank;
    iovec iov {};
    iov.iov_base = &payload;
    iov.iov_len = sizeof(payload);
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;

    while (true) {
        ssize_t sent = sendmsg(sock_fd, &msg, 0);
        if (sent >= static_cast<ssize_t>(sizeof(payload))) {
            break;
        }
        if (errno == EINTR) {
            continue;
        }
        throw std::runtime_error("Broker: failed to send fd");
    }
}

inline void recv_fd(int sock_fd, int* out_fd, int* out_src_local_rank) {
    if (sock_fd < 0) {
        throw std::runtime_error("Broker: invalid socket fd");
    }
    if (out_fd == nullptr || out_src_local_rank == nullptr) {
        throw std::runtime_error("Broker: null recv_fd output");
    }

    char control[CMSG_SPACE(sizeof(int))] {};
    msghdr msg {};
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);

    int payload = -1;
    iovec iov {};
    iov.iov_base = &payload;
    iov.iov_len = sizeof(payload);
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;

    while (true) {
        ssize_t received = recvmsg(sock_fd, &msg, 0);
        if (received >= static_cast<ssize_t>(sizeof(payload))) {
            break;
        }
        if (received < 0 && errno == EINTR) {
            msg.msg_controllen = sizeof(control);
            msg.msg_iovlen = 1;
            continue;
        }
        throw std::runtime_error("Broker: failed to receive fd");
    }

    if (msg.msg_flags & MSG_CTRUNC) {
        throw std::runtime_error("Broker: fd control data truncated");
    }

    cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    if (cmsg == nullptr ||
        cmsg->cmsg_len != CMSG_LEN(sizeof(int)) ||
        cmsg->cmsg_level != SOL_SOCKET ||
        cmsg->cmsg_type != SCM_RIGHTS) {
        throw std::runtime_error("Broker: invalid fd control message");
    }

    std::memcpy(out_fd, CMSG_DATA(cmsg), sizeof(*out_fd));
    *out_src_local_rank = payload;
}

} // namespace broker_detail

class Broker {
public:
    Broker(int local_rank, int local_world_size, const char* key)
        : local_rank_(local_rank),
          local_world_size_(local_world_size),
          shm_key_(broker_detail::make_shm_key(key)),
          socket_prefix_(broker_detail::make_socket_prefix(key)),
          shm_raw_(nullptr),
          shm_(nullptr),
          sock_(-1) {
        if (local_rank_ < 0) {
            throw std::runtime_error("Broker: local_rank must be non-negative");
        }
        if (local_world_size_ <= 0) {
            throw std::runtime_error("Broker: local_world_size must be positive");
        }
        if (local_rank_ >= local_world_size_) {
            throw std::runtime_error("Broker: local_rank >= local_world_size");
        }
        if (local_world_size_ > broker_detail::MAX_LOCAL_WORLD_SIZE) {
            throw std::runtime_error("Broker: local_world_size exceeds MAX_LOCAL_WORLD_SIZE");
        }

        if (local_rank_ == 0) {
            shm_raw_ = broker_detail::create_shm(shm_key_.c_str(), broker_detail::SHM_SIZE);
            shm_ = reinterpret_cast<volatile broker_detail::Vault*>(shm_raw_);
            std::memset(shm_raw_, 0, broker_detail::SHM_SIZE);
        } else {
            shm_raw_ = broker_detail::open_shm(shm_key_.c_str(), broker_detail::SHM_SIZE);
            shm_ = reinterpret_cast<volatile broker_detail::Vault*>(shm_raw_);
        }

        broker_detail::init_sync(local_rank_, shm_);
        broker_detail::sync(local_world_size_, shm_);

        sock_ = broker_detail::create_socket(socket_prefix_, local_rank_);
        broker_detail::sync(local_world_size_, shm_);
    }

    Broker(const Broker&) = delete;
    Broker& operator=(const Broker&) = delete;

    /*
     * Do not allow moving Broker objects. std::unique_ptr<Broker> can move the
     * pointer safely; moving the object itself can leave a moved-from Broker
     * whose shm_ is null and later used during cleanup/error paths.
     */
    Broker(Broker&&) = delete;
    Broker& operator=(Broker&&) = delete;

    ~Broker() {
        destroy();
    }

    int local_rank() const {
        return local_rank_;
    }

    int local_world_size() const {
        return local_world_size_;
    }

    void sync(int num_ranks = -1) {
        ensure_vault_mapped("Broker::sync");

        if (num_ranks == -1) {
            num_ranks = local_world_size_;
        }
        if (num_ranks <= 0 || num_ranks > local_world_size_) {
            throw std::runtime_error("Broker: invalid num_ranks");
        }

        broker_detail::sync(num_ranks, shm_);
    }

    /*
     * All-gather a fixed-size blob from each local rank.
     *
     * dst must have local_world_size * size bytes.
     */
    void exchange_data(void* dst, const void* src, size_t size) {
        ensure_vault_mapped("Broker::exchange_data");

        if (dst == nullptr || src == nullptr) {
            throw std::runtime_error("Broker: exchange_data got null pointer");
        }
        if (size == 0 || size > broker_detail::VAULT_SIZE_PER_RANK) {
            throw std::runtime_error("Broker: invalid exchange_data size");
        }

        uint8_t* dst_bytes = reinterpret_cast<uint8_t*>(dst);
        const uint8_t* src_bytes = reinterpret_cast<const uint8_t*>(src);

        sync();

        std::memcpy(
            const_cast<uint8_t*>(shm_->data) +
                local_rank_ * broker_detail::VAULT_SIZE_PER_RANK,
            src_bytes,
            size);

        sync();

        for (int r = 0; r < local_world_size_; ++r) {
            std::memcpy(
                dst_bytes + r * size,
                const_cast<uint8_t*>(shm_->data) +
                    r * broker_detail::VAULT_SIZE_PER_RANK,
                size);
        }

        sync();
    }

    template <typename T>
    void exchange_pod(T* dst, const T& src) {
        static_assert(std::is_trivially_copyable<T>::value,
                      "Broker::exchange_pod requires trivially copyable T");
        exchange_data(dst, &src, sizeof(T));
    }

    /*
     * Exchange one FD from each rank. The input fd is consumed/closed by this
     * function when it is no longer needed.
     *
     * dst_fds must have local_world_size entries. dst_fds[local_rank] is set
     * to -1 because you do not import your own fd.
     */
    void exchange_fds(int* dst_fds, int src_fd) {
        ensure_vault_mapped("Broker::exchange_fds");

        if (dst_fds == nullptr) {
            throw std::runtime_error("Broker: exchange_fds dst is null");
        }
        if (src_fd < 0) {
            throw std::runtime_error("Broker: exchange_fds source fd is invalid");
        }

        for (int r = 0; r < local_world_size_; ++r) {
            dst_fds[r] = -1;
        }

        sync();

        if (local_rank_ == 0) {
            std::vector<int> gathered(local_world_size_, -1);
            gathered[0] = src_fd;

            for (int i = 0; i < local_world_size_ - 1; ++i) {
                int received_fd = -1;
                int src_rank = -1;
                broker_detail::recv_fd(sock_, &received_fd, &src_rank);

                if (src_rank <= 0 || src_rank >= local_world_size_) {
                    if (received_fd >= 0) {
                        close(received_fd);
                    }
                    throw std::runtime_error("Broker: invalid received source rank");
                }

                gathered[src_rank] = received_fd;
            }

            /*
             * Send every rank all other ranks' FDs.
             */
            for (int dst_rank = 1; dst_rank < local_world_size_; ++dst_rank) {
                for (int src_rank = 0; src_rank < local_world_size_; ++src_rank) {
                    if (dst_rank == src_rank) {
                        continue;
                    }
                    broker_detail::send_fd(
                        sock_,
                        gathered[src_rank],
                        socket_prefix_,
                        dst_rank,
                        src_rank);
                }
            }

            /*
             * Rank 0 keeps imported peer FDs, but not its own.
             */
            for (int src_rank = 1; src_rank < local_world_size_; ++src_rank) {
                dst_fds[src_rank] = gathered[src_rank];
            }

            close(gathered[0]);
            gathered[0] = -1;
        } else {
            broker_detail::send_fd(sock_, src_fd, socket_prefix_, 0, local_rank_);
            close(src_fd);

            for (int i = 0; i < local_world_size_ - 1; ++i) {
                int received_fd = -1;
                int src_rank = -1;
                broker_detail::recv_fd(sock_, &received_fd, &src_rank);

                if (src_rank < 0 || src_rank >= local_world_size_ ||
                    src_rank == local_rank_) {
                    if (received_fd >= 0) {
                        close(received_fd);
                    }
                    throw std::runtime_error("Broker: invalid received source rank");
                }

                dst_fds[src_rank] = received_fd;
            }
        }

        dst_fds[local_rank_] = -1;
        sync();
    }

    /*
     * Broadcast one FD from src_rank to all other local ranks. The input fd is
     * consumed/closed by this function on src_rank.
     */
    void broadcast_fd(int* dst_fd, int src_fd, int src_rank) {
        ensure_vault_mapped("Broker::broadcast_fd");

        if (src_rank < 0 || src_rank >= local_world_size_) {
            throw std::runtime_error("Broker: invalid broadcast source rank");
        }

        sync();

        if (local_rank_ == src_rank) {
            if (src_fd < 0) {
                throw std::runtime_error("Broker: invalid broadcast source fd");
            }

            for (int dst_rank = 0; dst_rank < local_world_size_; ++dst_rank) {
                if (dst_rank == src_rank) {
                    continue;
                }
                broker_detail::send_fd(
                    sock_,
                    src_fd,
                    socket_prefix_,
                    dst_rank,
                    src_rank);
            }

            close(src_fd);
        } else {
            if (dst_fd == nullptr) {
                throw std::runtime_error("Broker: broadcast dst is null");
            }

            int received_src = -1;
            broker_detail::recv_fd(sock_, dst_fd, &received_src);

            if (*dst_fd < 0 || received_src != src_rank) {
                if (*dst_fd >= 0) {
                    close(*dst_fd);
                }
                throw std::runtime_error("Broker: invalid broadcast fd receive");
            }
        }

        sync();
    }

    void destroy() {
        const int old_rank = local_rank_;

        /*
         * No barrier in destructor.
         *
         * Destructors can run during failed init/error paths where the peer rank
         * may already have exited or may be stuck in another phase. A destructor
         * barrier can turn the original error into a broker error or deadlock.
         *
         * Tests should do explicit broker syncs before normal teardown.
         */
        if (old_rank == 0 && !shm_key_.empty()) {
            broker_detail::unlink_shm(shm_key_.c_str());
        }

        if (shm_raw_ != nullptr) {
            broker_detail::unmap_shm(shm_raw_, broker_detail::SHM_SIZE);
            shm_raw_ = nullptr;
            shm_ = nullptr;
        }

        if (sock_ >= 0) {
            broker_detail::unlink_socket(socket_prefix_, old_rank);
            broker_detail::close_socket(sock_);
            sock_ = -1;
        }

        local_rank_ = -1;
        local_world_size_ = -1;
    }

private:
    void ensure_vault_mapped(const char* where) {
        if (shm_ != nullptr) {
            return;
        }

        /*
         * This should normally never happen after the constructor succeeds.
         * But if a moved-from/partially-cleaned Broker path or failed init path
         * leaves shm_ null, recover by reopening the shm by name. The shm name
         * is kept alive until rank 0 destroy(), so this is safe for tests.
         */
        std::fprintf(
            stderr,
            "[ooverlap][broker] remapping null vault in %s rank=%d world=%d key=%s\n",
            where ? where : "unknown",
            local_rank_,
            local_world_size_,
            shm_key_.c_str());
        std::fflush(stderr);

        if (local_world_size_ <= 0 || local_rank_ < 0) {
            throw std::runtime_error("Broker: cannot remap invalid broker state");
        }

        if (shm_raw_ != nullptr) {
            shm_ = reinterpret_cast<volatile broker_detail::Vault*>(shm_raw_);
            return;
        }

        shm_raw_ = broker_detail::open_shm(
            shm_key_.c_str(),
            broker_detail::SHM_SIZE);

        shm_ = reinterpret_cast<volatile broker_detail::Vault*>(shm_raw_);

        if (shm_ == nullptr) {
            throw std::runtime_error("Broker: failed to remap vault");
        }
    }

    int local_rank_;
    int local_world_size_;

    std::string shm_key_;
    std::string socket_prefix_;

    void* shm_raw_;
    volatile broker_detail::Vault* shm_;
    int sock_;
};

} // namespace system
} // namespace ooverlap
