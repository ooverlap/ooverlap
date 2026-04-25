#include "ooverlap/system/broker.cuh"
#include "ooverlap/system/logging.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <pthread.h>
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

static constexpr uint32_t INIT_CODE = 0x4f4f4950u; // "OOIP"

struct Vault {
    uint32_t init;
    int world_size;

    pthread_mutex_t mutex;
    pthread_cond_t cond;

    int barrier_count;
    int barrier_generation;

    uint8_t data[MAX_LOCAL_WORLD_SIZE * VAULT_SIZE_PER_RANK];
};

static constexpr size_t SHM_SIZE = ((sizeof(Vault) + 4095) / 4096) * 4096;

std::string sanitize_key_component(const char* key) {
    if (key == nullptr || key[0] == '\0') {
        throw std::runtime_error("Broker: key must be non-empty");
    }

    std::string out;
    out.reserve(std::strlen(key));

    for (const char* p = key; *p; ++p) {
        const char c = *p;
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

std::string make_shm_key(const char* key) {
    std::string s = "/ooverlap_broker_";
    s += sanitize_key_component(key);

    if (s.size() >= 240) {
        throw std::runtime_error("Broker: shm key is too long");
    }

    return s;
}

std::string make_socket_prefix(const char* key) {
    std::string s = "/tmp/ooverlap_broker_";
    s += sanitize_key_component(key);
    s += ".sock.";

    if (s.size() >= 90) {
        throw std::runtime_error("Broker: socket key is too long");
    }

    return s;
}

std::string make_socket_path(const std::string& prefix, int local_rank) {
    std::string s = prefix + std::to_string(local_rank);

    if (s.size() >= sizeof(sockaddr_un::sun_path)) {
        throw std::runtime_error("Broker: socket path is too long");
    }

    return s;
}

void die_pthread(int err, const char* what) {
    if (err != 0) {
        OOVERLAP_LOG_ERROR("%s failed: %s\n", what, std::strerror(err));
        throw std::runtime_error(std::string("Broker: ") + what + " failed");
    }
}

void wait_until_initialized(volatile Vault* vault, const char* where) {
    if (vault == nullptr) {
        throw std::runtime_error("Broker: vault pointer is null");
    }

    int spins = 0;
    while (vault->init != INIT_CODE) {
        if ((++spins % 1000000) == 0) {
            OOVERLAP_LOG_TRACE(
                "waiting for init in %s, init=0x%x expected=0x%x\n",
                where ? where : "unknown",
                static_cast<unsigned>(vault->init),
                static_cast<unsigned>(INIT_CODE));
        }

        usleep(1);
    }

    __sync_synchronize();
}

void init_vault_rank0(Vault* vault, int local_world_size) {
    if (vault == nullptr) {
        throw std::runtime_error("Broker: init_vault_rank0 got null vault");
    }

    std::memset(vault, 0, SHM_SIZE);

    pthread_mutexattr_t mutex_attr;
    pthread_condattr_t cond_attr;

    die_pthread(pthread_mutexattr_init(&mutex_attr), "pthread_mutexattr_init");
    die_pthread(pthread_mutexattr_setpshared(&mutex_attr, PTHREAD_PROCESS_SHARED),
                "pthread_mutexattr_setpshared");

    die_pthread(pthread_condattr_init(&cond_attr), "pthread_condattr_init");
    die_pthread(pthread_condattr_setpshared(&cond_attr, PTHREAD_PROCESS_SHARED),
                "pthread_condattr_setpshared");

    die_pthread(pthread_mutex_init(&vault->mutex, &mutex_attr),
                "pthread_mutex_init");
    die_pthread(pthread_cond_init(&vault->cond, &cond_attr),
                "pthread_cond_init");

    pthread_mutexattr_destroy(&mutex_attr);
    pthread_condattr_destroy(&cond_attr);

    vault->world_size = local_world_size;
    vault->barrier_count = 0;
    vault->barrier_generation = 0;

    __sync_synchronize();
    vault->init = INIT_CODE;
    __sync_synchronize();
}

void sync(int local_world_size, Vault* vault) {
    if (local_world_size <= 0) {
        throw std::runtime_error("Broker: invalid local_world_size");
    }

    wait_until_initialized(vault, "sync");

    int err = pthread_mutex_lock(&vault->mutex);
    die_pthread(err, "pthread_mutex_lock");

    if (vault->world_size != local_world_size) {
        pthread_mutex_unlock(&vault->mutex);
        throw std::runtime_error("Broker: world_size mismatch");
    }

    const int generation = vault->barrier_generation;

    vault->barrier_count += 1;

    if (vault->barrier_count == local_world_size) {
        vault->barrier_count = 0;
        vault->barrier_generation += 1;

        err = pthread_cond_broadcast(&vault->cond);
        if (err != 0) {
            pthread_mutex_unlock(&vault->mutex);
            die_pthread(err, "pthread_cond_broadcast");
        }
    } else {
        while (generation == vault->barrier_generation) {
            err = pthread_cond_wait(&vault->cond, &vault->mutex);
            if (err != 0) {
                pthread_mutex_unlock(&vault->mutex);
                die_pthread(err, "pthread_cond_wait");
            }
        }
    }

    err = pthread_mutex_unlock(&vault->mutex);
    die_pthread(err, "pthread_mutex_unlock");
}

void* create_shm(const char* key, size_t size) {
    shm_unlink(key);

    int shm_fd = shm_open(key, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (shm_fd < 0) {
        OOVERLAP_LOG_ERROR(
            "shm_open create failed for %s: %s\n",
            key,
            std::strerror(errno));
        throw std::runtime_error("Broker: failed to create shared memory");
    }

    if (ftruncate(shm_fd, static_cast<off_t>(size)) != 0) {
        OOVERLAP_LOG_ERROR(
            "ftruncate failed for %s: %s\n",
            key,
            std::strerror(errno));
        shm_unlink(key);
        close(shm_fd);
        throw std::runtime_error("Broker: failed to resize shared memory");
    }

    void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    close(shm_fd);

    if (addr == MAP_FAILED) {
        OOVERLAP_LOG_ERROR(
            "mmap create failed for %s: %s\n",
            key,
            std::strerror(errno));
        shm_unlink(key);
        throw std::runtime_error("Broker: failed to map shared memory");
    }

    return addr;
}

void* open_shm(const char* key, size_t size) {
    int shm_fd = -1;

    while (true) {
        shm_fd = shm_open(key, O_RDWR | O_CLOEXEC, 0);
        if (shm_fd >= 0) {
            break;
        }

        if (errno != ENOENT) {
            OOVERLAP_LOG_ERROR(
                "shm_open open failed for %s: %s\n",
                key,
                std::strerror(errno));
            throw std::runtime_error("Broker: failed to open shared memory");
        }

        usleep(1);
    }

    struct stat shm_st {};
    do {
        if (fstat(shm_fd, &shm_st) != 0) {
            OOVERLAP_LOG_ERROR(
                "fstat failed for %s: %s\n",
                key,
                std::strerror(errno));
            close(shm_fd);
            throw std::runtime_error("Broker: failed to stat shared memory");
        }

        usleep(1);
    } while (static_cast<size_t>(shm_st.st_size) < size);

    void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    close(shm_fd);

    if (addr == MAP_FAILED) {
        OOVERLAP_LOG_ERROR(
            "mmap open failed for %s: %s\n",
            key,
            std::strerror(errno));
        throw std::runtime_error("Broker: failed to map shared memory");
    }

    return addr;
}

void unlink_shm(const char* key) {
    if (key != nullptr) {
        shm_unlink(key);
    }
}

void unmap_shm(void* addr, size_t size) {
    if (addr != nullptr) {
        munmap(addr, size);
    }
}

int create_socket(const std::string& socket_prefix, int local_rank) {
    int sock_fd = socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
    if (sock_fd < 0) {
        OOVERLAP_LOG_ERROR("socket failed: %s\n", std::strerror(errno));
        throw std::runtime_error("Broker: socket creation failed");
    }

    const std::string socket_path = make_socket_path(socket_prefix, local_rank);
    unlink(socket_path.c_str());

    sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(sock_fd, reinterpret_cast<sockaddr*>(&addr), SUN_LEN(&addr)) < 0) {
        OOVERLAP_LOG_ERROR(
            "bind failed for %s: %s\n",
            socket_path.c_str(),
            std::strerror(errno));
        close(sock_fd);
        throw std::runtime_error("Broker: failed to bind socket");
    }

    return sock_fd;
}

void unlink_socket(const std::string& socket_prefix, int local_rank) {
    if (local_rank < 0) {
        return;
    }

    const std::string socket_path = make_socket_path(socket_prefix, local_rank);
    unlink(socket_path.c_str());
}

void close_socket(int sock_fd) {
    if (sock_fd >= 0) {
        close(sock_fd);
    }
}

void send_fd(
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

        OOVERLAP_LOG_ERROR(
            "sendmsg fd src_rank=%d dst_rank=%d failed: %s\n",
            src_local_rank,
            dst_local_rank,
            std::strerror(errno));

        throw std::runtime_error("Broker: failed to send fd");
    }
}

void recv_fd(int sock_fd, int* out_fd, int* out_src_local_rank) {
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

        OOVERLAP_LOG_ERROR("recvmsg fd failed: %s\n", std::strerror(errno));
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

Broker::Broker(int local_rank, int local_world_size, const char* key)
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

    OOVERLAP_LOG_DEBUG(
        "broker construct begin this=%p rank=%d world=%d shm=%s socket_prefix=%s sizeof(Broker)=%zu\n",
        static_cast<void*>(this),
        local_rank_,
        local_world_size_,
        shm_key_.c_str(),
        socket_prefix_.c_str(),
        sizeof(Broker));

    if (local_rank_ == 0) {
        shm_raw_ = broker_detail::create_shm(
            shm_key_.c_str(),
            broker_detail::SHM_SIZE);

        shm_ = reinterpret_cast<broker_detail::Vault*>(shm_raw_);

        broker_detail::init_vault_rank0(shm_, local_world_size_);
    } else {
        shm_raw_ = broker_detail::open_shm(
            shm_key_.c_str(),
            broker_detail::SHM_SIZE);

        shm_ = reinterpret_cast<broker_detail::Vault*>(shm_raw_);

        broker_detail::wait_until_initialized(
            shm_,
            "Broker ctor rank>0");
    }

    broker_detail::sync(local_world_size_, shm_);

    sock_ = broker_detail::create_socket(socket_prefix_, local_rank_);

    broker_detail::sync(local_world_size_, shm_);

    OOVERLAP_LOG_DEBUG(
        "broker construct end this=%p rank=%d shm_raw=%p shm=%p sock=%d sizeof(Broker)=%zu\n",
        static_cast<void*>(this),
        local_rank_,
        shm_raw_,
        static_cast<void*>(shm_),
        sock_,
        sizeof(Broker));
}

Broker::~Broker() {
    destroy();
}

int Broker::local_rank() const {
    return local_rank_;
}

int Broker::local_world_size() const {
    return local_world_size_;
}

void Broker::check_alive(const char* where) const {
    if (local_rank_ < 0 || local_world_size_ <= 0) {
        throw std::runtime_error(std::string("Broker dead in ") + where);
    }

    if (shm_raw_ == nullptr || shm_ == nullptr) {
        OOVERLAP_LOG_ERROR(
            "broker null shm in %s this=%p rank=%d world=%d shm_raw=%p shm=%p key=%s sizeof(Broker)=%zu\n",
            where,
            static_cast<const void*>(this),
            local_rank_,
            local_world_size_,
            shm_raw_,
            static_cast<void*>(shm_),
            shm_key_.c_str(),
            sizeof(Broker));

        throw std::runtime_error(std::string("Broker null shm in ") + where);
    }

    if (shm_->init != broker_detail::INIT_CODE) {
        OOVERLAP_LOG_ERROR(
            "broker bad magic in %s this=%p rank=%d init=0x%x expected=0x%x shm=%p sizeof(Broker)=%zu\n",
            where,
            static_cast<const void*>(this),
            local_rank_,
            static_cast<unsigned>(shm_->init),
            static_cast<unsigned>(broker_detail::INIT_CODE),
            static_cast<void*>(shm_),
            sizeof(Broker));

        throw std::runtime_error(std::string("Broker bad magic in ") + where);
    }
}

void Broker::sync(int num_ranks) {
    check_alive("Broker::sync");

    if (num_ranks == -1) {
        num_ranks = local_world_size_;
    }

    if (num_ranks <= 0 || num_ranks > local_world_size_) {
        throw std::runtime_error("Broker: invalid num_ranks");
    }

    broker_detail::sync(num_ranks, shm_);
}

void Broker::exchange_data(void* dst, const void* src, size_t size) {
    check_alive("Broker::exchange_data");

    if (dst == nullptr || src == nullptr) {
        throw std::runtime_error("Broker: exchange_data got null pointer");
    }

    if (size == 0 || size > broker_detail::VAULT_SIZE_PER_RANK) {
        throw std::runtime_error("Broker: invalid exchange_data size");
    }

    OOVERLAP_LOG_TRACE(
        "broker exchange_data begin this=%p rank=%d size=%zu shm_raw=%p shm=%p generation=%d sizeof(Broker)=%zu\n",
        static_cast<void*>(this),
        local_rank_,
        size,
        shm_raw_,
        static_cast<void*>(shm_),
        shm_ ? shm_->barrier_generation : -1,
        sizeof(Broker));

    uint8_t* dst_bytes = reinterpret_cast<uint8_t*>(dst);
    const uint8_t* src_bytes = reinterpret_cast<const uint8_t*>(src);

    sync();

    std::memcpy(
        shm_->data + local_rank_ * broker_detail::VAULT_SIZE_PER_RANK,
        src_bytes,
        size);

    sync();

    for (int r = 0; r < local_world_size_; ++r) {
        std::memcpy(
            dst_bytes + r * size,
            shm_->data + r * broker_detail::VAULT_SIZE_PER_RANK,
            size);
    }

    sync();

    OOVERLAP_LOG_TRACE(
        "broker exchange_data end this=%p rank=%d size=%zu generation=%d sizeof(Broker)=%zu\n",
        static_cast<void*>(this),
        local_rank_,
        size,
        shm_->barrier_generation,
        sizeof(Broker));
}

void Broker::exchange_fds(int* dst_fds, int src_fd) {
    check_alive("Broker::exchange_fds");

    if (dst_fds == nullptr) {
        throw std::runtime_error("Broker: exchange_fds dst is null");
    }

    if (src_fd < 0) {
        throw std::runtime_error("Broker: exchange_fds source fd is invalid");
    }

    OOVERLAP_LOG_TRACE(
        "broker exchange_fds begin this=%p rank=%d src_fd=%d sizeof(Broker)=%zu\n",
        static_cast<void*>(this),
        local_rank_,
        src_fd,
        sizeof(Broker));

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

            OOVERLAP_LOG_TRACE("broker rank0 recv_fd wait\n");

            broker_detail::recv_fd(sock_, &received_fd, &src_rank);

            OOVERLAP_LOG_TRACE(
                "broker rank0 recv_fd got src_rank=%d fd=%d\n",
                src_rank,
                received_fd);

            if (src_rank <= 0 || src_rank >= local_world_size_) {
                if (received_fd >= 0) {
                    close(received_fd);
                }

                throw std::runtime_error("Broker: invalid received source rank");
            }

            gathered[src_rank] = received_fd;
        }

        for (int dst_rank = 1; dst_rank < local_world_size_; ++dst_rank) {
            for (int src_rank = 0; src_rank < local_world_size_; ++src_rank) {
                if (dst_rank == src_rank) {
                    continue;
                }

                OOVERLAP_LOG_TRACE(
                    "broker rank0 send fd from src_rank=%d to dst_rank=%d fd=%d\n",
                    src_rank,
                    dst_rank,
                    gathered[src_rank]);

                broker_detail::send_fd(
                    sock_,
                    gathered[src_rank],
                    socket_prefix_,
                    dst_rank,
                    src_rank);
            }
        }

        for (int src_rank = 1; src_rank < local_world_size_; ++src_rank) {
            dst_fds[src_rank] = gathered[src_rank];
        }

        close(gathered[0]);
        gathered[0] = -1;
    } else {
        OOVERLAP_LOG_TRACE(
            "broker rank%d send fd to rank0 fd=%d\n",
            local_rank_,
            src_fd);

        broker_detail::send_fd(
            sock_,
            src_fd,
            socket_prefix_,
            0,
            local_rank_);

        close(src_fd);

        for (int i = 0; i < local_world_size_ - 1; ++i) {
            int received_fd = -1;
            int src_rank = -1;

            OOVERLAP_LOG_TRACE(
                "broker rank%d recv_fd wait\n",
                local_rank_);

            broker_detail::recv_fd(sock_, &received_fd, &src_rank);

            OOVERLAP_LOG_TRACE(
                "broker rank%d recv_fd got src_rank=%d fd=%d\n",
                local_rank_,
                src_rank,
                received_fd);

            if (src_rank < 0 ||
                src_rank >= local_world_size_ ||
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

    OOVERLAP_LOG_TRACE(
        "broker exchange_fds end this=%p rank=%d sizeof(Broker)=%zu\n",
        static_cast<void*>(this),
        local_rank_,
        sizeof(Broker));
}

void Broker::broadcast_fd(int* dst_fd, int src_fd, int src_rank) {
    check_alive("Broker::broadcast_fd");

    if (src_rank < 0 || src_rank >= local_world_size_) {
        throw std::runtime_error("Broker: invalid broadcast source rank");
    }

    OOVERLAP_LOG_TRACE(
        "broker broadcast_fd begin this=%p rank=%d src_rank=%d src_fd=%d sizeof(Broker)=%zu\n",
        static_cast<void*>(this),
        local_rank_,
        src_rank,
        src_fd,
        sizeof(Broker));

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

    OOVERLAP_LOG_TRACE(
        "broker broadcast_fd end this=%p rank=%d sizeof(Broker)=%zu\n",
        static_cast<void*>(this),
        local_rank_,
        sizeof(Broker));
}

void Broker::destroy() {
    const int old_rank = local_rank_;

    if (old_rank >= 0) {
        OOVERLAP_LOG_DEBUG(
            "broker destroy begin this=%p rank=%d shm=%p sock=%d sizeof(Broker)=%zu\n",
            static_cast<void*>(this),
            old_rank,
            static_cast<void*>(shm_),
            sock_,
            sizeof(Broker));
    }

    if (old_rank == 0 && !shm_key_.empty()) {
        broker_detail::unlink_shm(shm_key_.c_str());
    }

    if (shm_raw_ != nullptr) {
        broker_detail::unmap_shm(
            shm_raw_,
            broker_detail::SHM_SIZE);

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

    if (old_rank >= 0) {
        OOVERLAP_LOG_DEBUG(
            "broker destroy end this=%p rank=%d sizeof(Broker)=%zu\n",
            static_cast<void*>(this),
            old_rank,
            sizeof(Broker));
    }
}

} // namespace system
} // namespace ooverlap
