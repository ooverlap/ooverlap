#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

namespace ooverlap {
namespace system {

namespace broker_detail {

static constexpr int MAX_LOCAL_WORLD_SIZE = 128;
static constexpr int VAULT_SIZE_PER_RANK = 256;

struct Vault;

} // namespace broker_detail

class Broker {
public:
    Broker(int local_rank, int local_world_size, const char* key);

    Broker(const Broker&) = delete;
    Broker& operator=(const Broker&) = delete;

    Broker(Broker&&) = delete;
    Broker& operator=(Broker&&) = delete;

    ~Broker();

    int local_rank() const;
    int local_world_size() const;

    void sync(int num_ranks = -1);

    void exchange_data(void* dst, const void* src, size_t size);

    template <typename T>
    void exchange_pod(T* dst, const T& src) {
        static_assert(std::is_trivially_copyable<T>::value,
                      "Broker::exchange_pod requires trivially copyable T");
        exchange_data(dst, &src, sizeof(T));
    }

    void exchange_fds(int* dst_fds, int src_fd);
    void broadcast_fd(int* dst_fd, int src_fd, int src_rank);

    void destroy();

private:
    void check_alive(const char* where) const;

    int local_rank_;
    int local_world_size_;

    std::string shm_key_;
    std::string socket_prefix_;

    void* shm_raw_;
    broker_detail::Vault* shm_;
    int sock_;
};

} // namespace system
} // namespace ooverlap
