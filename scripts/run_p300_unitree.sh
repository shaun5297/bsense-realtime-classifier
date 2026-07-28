#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_SETUP="${BCI_ROS_SETUP:-/opt/ros/humble/setup.bash}"
UNITREE_SETUP="${BCI_UNITREE_SETUP:-${HOME}/unitree_ros2/setup.sh}"
PROJECT_SETUP="${PROJECT_ROOT}/install/setup.bash"
SOCKET_PATH="${BCI_UNITREE_SOCKET:-/tmp/bsense_unitree.sock}"
MODEL_PATH="${BCI_P300_MODEL:-${PROJECT_ROOT}/models/m7_p300/model.joblib}"
BRIDGE_CONFIG="${BCI_BRIDGE_CONFIG:-${PROJECT_ROOT}/config/unitree_bridge_ros2.json}"
SPORT_STATE_TOPIC="${BCI_SPORT_STATE_TOPIC:-/lf/sportmodestate}"
OBSTACLE_DISTANCE="${BCI_OBSTACLE_DISTANCE_M:-0.45}"
ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/bsense_ros_logs}"

export AMENT_TRACE_SETUP_FILES=""
source "${ROS_SETUP}"
source "${UNITREE_SETUP}"
source "${PROJECT_SETUP}"

export ROS_LOG_DIR
mkdir -p "${ROS_LOG_DIR}"

if [[ -n "${BCI_CYCLONEDDS_INTERFACE:-}" ]]; then
  export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"${BCI_CYCLONEDDS_INTERFACE}\" priority=\"default\" multicast=\"default\" /></Interfaces></General></Domain></CycloneDDS>"
fi

BRIDGE_PID=""
cleanup() {
  if [[ -n "${BRIDGE_PID}" ]]; then
    kill -INT "${BRIDGE_PID}" 2>/dev/null || true
    wait "${BRIDGE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ros2 run bsense_realtime_classifier bsense_unitree_bridge --ros-args \
  -p "socket_path:=${SOCKET_PATH}" \
  -p "sport_state_topic:=${SPORT_STATE_TOPIC}" \
  -p "obstacle_stop_distance_m:=${OBSTACLE_DISTANCE}" \
  -p "require_obstacle_clear:=${BCI_REQUIRE_OBSTACLE_CLEAR:-false}" &
BRIDGE_PID=$!

for _attempt in $(seq 1 50); do
  if [[ -S "${SOCKET_PATH}" ]]; then
    break
  fi
  if ! kill -0 "${BRIDGE_PID}" 2>/dev/null; then
    echo "ERROR: ROS 2 Bridge 启动失败，请检查 ${ROS_LOG_DIR}。" >&2
    exit 1
  fi
  sleep 0.1
done
if [[ ! -S "${SOCKET_PATH}" ]]; then
  echo "ERROR: 等待 Unix Socket 超时：${SOCKET_PATH}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
APP_ARGS=(
  --bridge-config "${BRIDGE_CONFIG}"
  --socket-path "${SOCKET_PATH}"
)
if [[ -f "${MODEL_PATH}" ]]; then
  APP_ARGS+=(--model "${MODEL_PATH}")
else
  echo "WARN: 尚未找到 M7 模型，将以桥接联调模式启动：${MODEL_PATH}" >&2
fi
python3 -m bsense_classifier.robot_app "${APP_ARGS[@]}" "$@"
