#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cerrno>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <rclcpp/rclcpp.hpp>
#include <unitree_api/msg/request.hpp>

namespace {

constexpr int32_t kStopMove = 1003;
constexpr int32_t kMove = 1008;
constexpr auto kCommandPeriod = std::chrono::milliseconds(20);
constexpr auto kCommandTimeout = std::chrono::milliseconds(1000);

struct Motion {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

class BciUnitreeBridge final : public rclcpp::Node {
 public:
  BciUnitreeBridge()
      : Node("bci_unitree_bridge"),
        publisher_(create_publisher<unitree_api::msg::Request>(
            "/api/sport/request", rclcpp::QoS(1).reliable())),
        running_(true),
        last_update_(std::chrono::steady_clock::now()) {
    socket_path_ = declare_parameter<std::string>(
        "socket_path", "/tmp/bci_unitree.sock");
    setup_socket();
    timer_ = create_wall_timer(kCommandPeriod, [this]() { publish_current(); });
    socket_thread_ = std::thread(&BciUnitreeBridge::socket_loop, this);
    RCLCPP_INFO(get_logger(), "Listening for bci_result_v1 on %s",
                socket_path_.c_str());
  }

  ~BciUnitreeBridge() override {
    running_.store(false);
    if (server_fd_ >= 0) {
      shutdown(server_fd_, SHUT_RDWR);
      close(server_fd_);
      server_fd_ = -1;
    }
    if (socket_thread_.joinable()) socket_thread_.join();
    unlink(socket_path_.c_str());
    publish_stop();
  }

 private:
  static bool extract_string(const std::string &line, const std::string &key,
                             std::string &value) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
    std::smatch match;
    if (!std::regex_search(line, match, pattern)) return false;
    value = match[1].str();
    return true;
  }

  static bool extract_bool(const std::string &line, const std::string &key,
                           bool &value) {
    const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*(true|false)");
    std::smatch match;
    if (!std::regex_search(line, match, pattern)) return false;
    value = match[1].str() == "true";
    return true;
  }

  static bool extract_number(const std::string &line, const std::string &key,
                             double &value) {
    const std::regex pattern(
        "\\\"" + key + "\\\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)");
    std::smatch match;
    if (!std::regex_search(line, match, pattern)) return false;
    try {
      value = std::stod(match[1].str());
      return std::isfinite(value);
    } catch (const std::exception &) {
      return false;
    }
  }

  void setup_socket() {
    unlink(socket_path_.c_str());
    server_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd_ < 0) throw std::runtime_error("cannot create Unix Socket");

    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    if (socket_path_.size() >= sizeof(address.sun_path)) {
      throw std::runtime_error("Unix Socket path is too long");
    }
    std::strncpy(address.sun_path, socket_path_.c_str(),
                 sizeof(address.sun_path) - 1);
    if (bind(server_fd_, reinterpret_cast<sockaddr *>(&address),
             sizeof(address)) < 0 || listen(server_fd_, 1) < 0) {
      close(server_fd_);
      server_fd_ = -1;
      throw std::runtime_error("cannot bind/listen on Unix Socket");
    }
  }

  void socket_loop() {
    while (running_.load()) {
      const int client = accept(server_fd_, nullptr, nullptr);
      if (client < 0) {
        if (running_.load()) RCLCPP_WARN(get_logger(), "Unix Socket accept failed");
        continue;
      }
      timeval timeout{0, 500000};
      setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
      std::string pending;
      char buffer[4096];
      while (running_.load()) {
        const ssize_t count = recv(client, buffer, sizeof(buffer), 0);
        if (count == 0) break;
        if (count < 0) {
          if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) continue;
          break;
        }
        pending.append(buffer, static_cast<size_t>(count));
        size_t newline = 0;
        while ((newline = pending.find('\n')) != std::string::npos) {
          const std::string line = pending.substr(0, newline);
          pending.erase(0, newline + 1);
          handle_result(line);
        }
      }
      close(client);
      set_motion(Motion{});
    }
  }

  void handle_result(const std::string &line) {
    std::string protocol;
    std::string label;
    bool is_accepted = false;
    if (!extract_string(line, "protocol", protocol) ||
        protocol != "bci_result_v1" || !extract_string(line, "label", label) ||
        !extract_bool(line, "accepted", is_accepted)) {
      return;
    }

    Motion motion{};
    const bool has_motion = extract_number(line, "x", motion.x) &&
                            extract_number(line, "y", motion.y) &&
                            extract_number(line, "z", motion.z);
    if (!is_accepted || !has_motion) motion = Motion{};
    set_motion(motion);
    RCLCPP_INFO(get_logger(), "BCI label=%s accepted=%s -> x=%.3f y=%.3f z=%.3f",
                label.c_str(), is_accepted ? "true" : "false", motion.x,
                motion.y, motion.z);
  }

  void set_motion(const Motion &motion) {
    std::lock_guard<std::mutex> lock(mutex_);
    motion_ = motion;
    last_update_ = std::chrono::steady_clock::now();
  }

  void publish_current() {
    Motion motion;
    auto last_update = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      motion = motion_;
      last_update = last_update_;
    }
    if (std::chrono::steady_clock::now() - last_update > kCommandTimeout ||
        (std::abs(motion.x) < 1e-6 && std::abs(motion.y) < 1e-6 &&
         std::abs(motion.z) < 1e-6)) {
      publish_stop();
      return;
    }
    unitree_api::msg::Request request;
    request.header.identity.api_id = kMove;
    request.header.policy.noreply = true;
    request.parameter = "{\"x\":" + std::to_string(motion.x) +
                        ",\"y\":" + std::to_string(motion.y) +
                        ",\"z\":" + std::to_string(motion.z) + "}";
    publisher_->publish(request);
  }

  void publish_stop() {
    unitree_api::msg::Request request;
    request.header.identity.api_id = kStopMove;
    request.header.policy.noreply = true;
    publisher_->publish(request);
  }

  rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::atomic<bool> running_;
  std::mutex mutex_;
  Motion motion_;
  std::chrono::steady_clock::time_point last_update_;
  std::string socket_path_;
  int server_fd_{-1};
  std::thread socket_thread_;
};

}  // namespace

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<BciUnitreeBridge>();
    rclcpp::spin(node);
  } catch (const std::exception &error) {
    fprintf(stderr, "bci_unitree_bridge: %s\n", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}

