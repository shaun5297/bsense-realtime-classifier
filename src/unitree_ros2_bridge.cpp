#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <rclcpp/rclcpp.hpp>
#include <unitree_api/msg/request.hpp>
#include <unitree_go/msg/sport_mode_state.hpp>

namespace {

constexpr int32_t kStopMove = 1003;
constexpr int32_t kMove = 1008;
constexpr auto kCommandPeriod = std::chrono::milliseconds(20);
constexpr const char *kSchema = "bsense.unitree.command.v1";

struct Motion {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

bool is_zero(const Motion &motion) {
  return std::abs(motion.x) < 1e-6 && std::abs(motion.y) < 1e-6 &&
         std::abs(motion.z) < 1e-6;
}

std::string json_escape(const std::string &value) {
  std::ostringstream output;
  for (const char character : value) {
    switch (character) {
      case '\\':
        output << "\\\\";
        break;
      case '"':
        output << "\\\"";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        output << character;
        break;
    }
  }
  return output.str();
}

class BciUnitreeBridge final : public rclcpp::Node {
 public:
  BciUnitreeBridge()
      : Node("bsense_unitree_bridge"),
        publisher_(create_publisher<unitree_api::msg::Request>(
            "/api/sport/request", rclcpp::QoS(10).reliable())),
        running_(true),
        last_state_(std::chrono::steady_clock::time_point::min()),
        motion_deadline_(std::chrono::steady_clock::time_point::min()) {
    socket_path_ =
        declare_parameter<std::string>("socket_path", "/tmp/bsense_unitree.sock");
    sport_state_topic_ =
        declare_parameter<std::string>("sport_state_topic", "sportmodestate");
    obstacle_stop_distance_m_ =
        declare_parameter<double>("obstacle_stop_distance_m", 0.45);
    max_linear_mps_ = declare_parameter<double>("max_linear_mps", 0.5);
    max_angular_rps_ = declare_parameter<double>("max_angular_rps", 1.0);
    max_motion_duration_s_ =
        declare_parameter<double>("max_motion_duration_s", 2.0);
    state_timeout_ms_ = declare_parameter<int>("state_timeout_ms", 1000);
    require_robot_state_ =
        declare_parameter<bool>("require_robot_state", true);
    require_obstacle_clear_ =
        declare_parameter<bool>("require_obstacle_clear", true);
    validate_parameters();

    state_subscription_ =
        create_subscription<unitree_go::msg::SportModeState>(
            sport_state_topic_, rclcpp::SensorDataQoS(),
            [this](const unitree_go::msg::SportModeState::SharedPtr message) {
              update_robot_state(*message);
            });
    setup_socket();
    timer_ = create_wall_timer(kCommandPeriod, [this]() { publish_current(); });
    socket_thread_ = std::thread(&BciUnitreeBridge::socket_loop, this);
    RCLCPP_INFO(
        get_logger(),
        "Listening on %s; state topic=%s; obstacle stop=%.2f m",
        socket_path_.c_str(), sport_state_topic_.c_str(),
        obstacle_stop_distance_m_);
  }

  ~BciUnitreeBridge() override {
    running_.store(false);
    if (server_fd_ >= 0) {
      shutdown(server_fd_, SHUT_RDWR);
      close(server_fd_);
      server_fd_ = -1;
    }
    if (socket_thread_.joinable()) {
      socket_thread_.join();
    }
    unlink(socket_path_.c_str());
    publish_stop();
  }

 private:
  static bool extract_string(const std::string &line, const std::string &key,
                             std::string &value) {
    const std::regex pattern("\\\"" + key +
                             "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    std::smatch match;
    if (!std::regex_search(line, match, pattern)) {
      return false;
    }
    value = match[1].str();
    return true;
  }

  static bool extract_bool(const std::string &line, const std::string &key,
                           bool &value) {
    const std::regex pattern("\\\"" + key +
                             "\\\"\\s*:\\s*(true|false)");
    std::smatch match;
    if (!std::regex_search(line, match, pattern)) {
      return false;
    }
    value = match[1].str() == "true";
    return true;
  }

  static bool extract_number(const std::string &line, const std::string &key,
                             double &value) {
    const std::regex pattern(
        "\\\"" + key +
        "\\\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)");
    std::smatch match;
    if (!std::regex_search(line, match, pattern)) {
      return false;
    }
    try {
      value = std::stod(match[1].str());
      return std::isfinite(value);
    } catch (const std::exception &) {
      return false;
    }
  }

  void validate_parameters() const {
    if (socket_path_.empty()) {
      throw std::runtime_error("socket_path cannot be empty");
    }
    if (obstacle_stop_distance_m_ <= 0.0 || max_linear_mps_ <= 0.0 ||
        max_angular_rps_ <= 0.0 || max_motion_duration_s_ <= 0.0 ||
        state_timeout_ms_ <= 0) {
      throw std::runtime_error("safety parameters must be positive");
    }
  }

  void setup_socket() {
    unlink(socket_path_.c_str());
    server_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
      throw std::runtime_error("cannot create Unix Socket");
    }
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    if (socket_path_.size() >= sizeof(address.sun_path)) {
      throw std::runtime_error("Unix Socket path is too long");
    }
    std::strncpy(address.sun_path, socket_path_.c_str(),
                 sizeof(address.sun_path) - 1);
    if (bind(server_fd_, reinterpret_cast<sockaddr *>(&address),
             sizeof(address)) < 0 ||
        listen(server_fd_, 1) < 0) {
      close(server_fd_);
      server_fd_ = -1;
      throw std::runtime_error("cannot bind/listen on Unix Socket");
    }
    if (chmod(socket_path_.c_str(), S_IRUSR | S_IWUSR) != 0) {
      throw std::runtime_error("cannot restrict Unix Socket permissions");
    }
  }

  void update_robot_state(const unitree_go::msg::SportModeState &message) {
    bool known = false;
    bool detected = false;
    double minimum = std::numeric_limits<double>::infinity();
    for (const float raw_distance : message.range_obstacle) {
      const double distance = static_cast<double>(raw_distance);
      if (!std::isfinite(distance) || distance <= 0.01) {
        continue;
      }
      known = true;
      minimum = std::min(minimum, distance);
      detected = detected || distance <= obstacle_stop_distance_m_;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    state_received_ = true;
    obstacle_known_ = known;
    obstacle_detected_ = detected;
    minimum_obstacle_m_ = known ? minimum : 0.0;
    last_state_ = std::chrono::steady_clock::now();
  }

  bool state_fresh_locked(
      const std::chrono::steady_clock::time_point now) const {
    if (!state_received_) {
      return false;
    }
    return now - last_state_ <= std::chrono::milliseconds(state_timeout_ms_);
  }

  bool robot_connected_locked(
      const std::chrono::steady_clock::time_point now) const {
    return !require_robot_state_ || state_fresh_locked(now);
  }

  bool obstacle_clear_locked(
      const std::chrono::steady_clock::time_point now) const {
    if (!require_obstacle_clear_) {
      return true;
    }
    return state_fresh_locked(now) && obstacle_known_ &&
           !obstacle_detected_;
  }

  std::string status_json(const bool ok,
                          const std::string &detail_override = "") const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto now = std::chrono::steady_clock::now();
    const bool connected = robot_connected_locked(now);
    std::string detail = detail_override;
    if (detail.empty()) {
      if (!connected) {
        detail = "robot state unavailable or stale";
      } else if (require_obstacle_clear_ &&
                 (!obstacle_known_ || !state_fresh_locked(now))) {
        detail = "obstacle state unavailable";
      } else if (obstacle_detected_) {
        detail = "obstacle detected";
      } else if (emergency_latched_) {
        detail = "emergency stop latched";
      } else {
        detail = "ready";
      }
    }
    std::ostringstream output;
    output << std::boolalpha << std::setprecision(6)
           << "{\"ok\":" << ok << ",\"connected\":" << connected
           << ",\"obstacle_detected\":";
    if (!obstacle_known_ || !state_fresh_locked(now)) {
      output << "null";
    } else {
      output << obstacle_detected_;
    }
    output << ",\"emergency_stopped\":" << emergency_latched_
           << ",\"armed\":" << !emergency_latched_
           << ",\"minimum_obstacle_m\":";
    if (obstacle_known_ && state_fresh_locked(now)) {
      output << minimum_obstacle_m_;
    } else {
      output << "null";
    }
    output << ",\"detail\":\"" << json_escape(detail) << "\"}";
    return output.str();
  }

  std::string handle_line(const std::string &line) {
    std::string schema;
    std::string kind;
    if (!extract_string(line, "schema", schema) || schema != kSchema ||
        !extract_string(line, "kind", kind)) {
      return status_json(false, "invalid schema or kind");
    }
    if (kind == "status") {
      return status_json(true);
    }
    if (kind == "arm") {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto now = std::chrono::steady_clock::now();
        if (!robot_connected_locked(now)) {
          return status_json_unlocked(false, "robot state unavailable or stale",
                                      now);
        }
        if (!obstacle_clear_locked(now)) {
          return status_json_unlocked(false, "obstacle is not confirmed clear",
                                      now);
        }
        emergency_latched_ = false;
        motion_ = Motion{};
        stop_published_ = false;
      }
      return status_json(true, "armed");
    }
    if (kind != "command") {
      return status_json(false, "unsupported request kind");
    }

    std::string command;
    bool emergency = false;
    if (!extract_string(line, "command", command) ||
        !extract_bool(line, "emergency", emergency)) {
      return status_json(false, "invalid command fields");
    }
    if (command == "stop" || emergency) {
      std::lock_guard<std::mutex> lock(mutex_);
      motion_ = Motion{};
      emergency_latched_ = emergency_latched_ || emergency;
      stop_published_ = false;
      return status_json_unlocked(true,
                                  emergency ? "emergency stop latched" : "stopped",
                                  std::chrono::steady_clock::now());
    }
    if (command == "idle") {
      std::lock_guard<std::mutex> lock(mutex_);
      motion_ = Motion{};
      stop_published_ = false;
      return status_json_unlocked(true, "idle",
                                  std::chrono::steady_clock::now());
    }

    double linear_x = 0.0;
    double angular_z = 0.0;
    double duration_s = 0.0;
    if (!extract_number(line, "linear_x", linear_x) ||
        !extract_number(line, "angular_z", angular_z) ||
        !extract_number(line, "duration_s", duration_s)) {
      return status_json(false, "invalid motion fields");
    }
    Motion requested{};
    if (command == "forward") {
      requested.x = std::min(std::abs(linear_x), max_linear_mps_);
    } else if (command == "backward") {
      requested.x = -std::min(std::abs(linear_x), max_linear_mps_);
    } else if (command == "left") {
      requested.z = std::min(std::abs(angular_z), max_angular_rps_);
    } else if (command == "right") {
      requested.z = -std::min(std::abs(angular_z), max_angular_rps_);
    } else {
      return status_json(false, "unsupported robot command");
    }
    if (is_zero(requested)) {
      return status_json(false, "motion magnitude is zero");
    }
    const double bounded_duration =
        std::clamp(duration_s, 0.1, max_motion_duration_s_);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto now = std::chrono::steady_clock::now();
      if (emergency_latched_) {
        return status_json_unlocked(false, "emergency stop is latched", now);
      }
      if (!robot_connected_locked(now)) {
        return status_json_unlocked(false, "robot state unavailable or stale",
                                    now);
      }
      if (!obstacle_clear_locked(now)) {
        motion_ = Motion{};
        stop_published_ = false;
        return status_json_unlocked(false, "obstacle is not confirmed clear",
                                    now);
      }
      motion_ = requested;
      motion_deadline_ =
          now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(bounded_duration));
      stop_published_ = false;
    }
    return status_json(true, "motion accepted");
  }

  std::string status_json_unlocked(
      const bool ok, const std::string &detail,
      const std::chrono::steady_clock::time_point now) const {
    const bool connected = robot_connected_locked(now);
    std::ostringstream output;
    output << std::boolalpha << std::setprecision(6)
           << "{\"ok\":" << ok << ",\"connected\":" << connected
           << ",\"obstacle_detected\":";
    if (!obstacle_known_ || !state_fresh_locked(now)) {
      output << "null";
    } else {
      output << obstacle_detected_;
    }
    output << ",\"emergency_stopped\":" << emergency_latched_
           << ",\"armed\":" << !emergency_latched_
           << ",\"minimum_obstacle_m\":";
    if (obstacle_known_ && state_fresh_locked(now)) {
      output << minimum_obstacle_m_;
    } else {
      output << "null";
    }
    output << ",\"detail\":\"" << json_escape(detail) << "\"}";
    return output.str();
  }

  static bool send_all(const int client, const std::string &response) {
    size_t sent = 0;
    while (sent < response.size()) {
      const ssize_t count =
          send(client, response.data() + sent, response.size() - sent,
               MSG_NOSIGNAL);
      if (count < 0 && errno == EINTR) {
        continue;
      }
      if (count <= 0) {
        return false;
      }
      sent += static_cast<size_t>(count);
    }
    return true;
  }

  void socket_loop() {
    while (running_.load()) {
      const int client = accept(server_fd_, nullptr, nullptr);
      if (client < 0) {
        if (running_.load()) {
          RCLCPP_WARN(get_logger(), "Unix Socket accept failed");
        }
        continue;
      }
      timeval timeout{0, 500000};
      setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
      std::string pending;
      char buffer[4096];
      bool response_failed = false;
      while (running_.load()) {
        const ssize_t count = recv(client, buffer, sizeof(buffer), 0);
        if (count == 0) {
          break;
        }
        if (count < 0) {
          if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
            continue;
          }
          break;
        }
        pending.append(buffer, static_cast<size_t>(count));
        if (pending.size() > 65536) {
          break;
        }
        size_t newline = 0;
        while ((newline = pending.find('\n')) != std::string::npos) {
          const std::string line = pending.substr(0, newline);
          pending.erase(0, newline + 1);
          const std::string response = handle_line(line) + "\n";
          if (!send_all(client, response)) {
            response_failed = true;
            break;
          }
        }
        if (response_failed) {
          break;
        }
      }
      close(client);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        motion_ = Motion{};
        stop_published_ = false;
      }
    }
  }

  void publish_current() {
    Motion motion;
    bool publish_stop_request = false;
    bool publish_move_request = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto now = std::chrono::steady_clock::now();
      const bool must_stop =
          emergency_latched_ || now >= motion_deadline_ ||
          !robot_connected_locked(now) || !obstacle_clear_locked(now) ||
          is_zero(motion_);
      if (must_stop) {
        motion_ = Motion{};
        if (!stop_published_) {
          stop_published_ = true;
          publish_stop_request = true;
        }
      } else {
        motion = motion_;
        stop_published_ = false;
        publish_move_request = true;
      }
    }
    if (publish_stop_request) {
      publish_stop();
    } else if (publish_move_request) {
      publish_move(motion);
    }
  }

  void publish_move(const Motion &motion) {
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
  rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr
      state_subscription_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::atomic<bool> running_;
  mutable std::mutex mutex_;
  Motion motion_;
  std::chrono::steady_clock::time_point last_state_;
  std::chrono::steady_clock::time_point motion_deadline_;
  std::string socket_path_;
  std::string sport_state_topic_;
  double obstacle_stop_distance_m_{0.45};
  double max_linear_mps_{0.5};
  double max_angular_rps_{1.0};
  double max_motion_duration_s_{2.0};
  int state_timeout_ms_{1000};
  bool require_robot_state_{true};
  bool require_obstacle_clear_{true};
  bool state_received_{false};
  bool obstacle_known_{false};
  bool obstacle_detected_{false};
  bool emergency_latched_{true};
  bool stop_published_{false};
  double minimum_obstacle_m_{0.0};
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
    fprintf(stderr, "bsense_unitree_bridge: %s\n", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
