#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_SETUP="${BCI_ROS_SETUP:-/opt/ros/humble/setup.bash}"
UNITREE_SETUP="${BCI_UNITREE_SETUP:-${HOME}/unitree_ros2/setup.sh}"
PROJECT_SETUP="${PROJECT_ROOT}/install/setup.bash"
SOCKET_PATH="${BCI_UNITREE_SOCKET:-/tmp/bsense_unitree.sock}"
MODEL_PATH="${BCI_P300_MODEL:-${PROJECT_ROOT}/models/m7_p300/model.joblib}"
SPORT_STATE_TOPIC="${BCI_SPORT_STATE_TOPIC:-/lf/sportmodestate}"
OBSTACLE_DISTANCE="${BCI_OBSTACLE_DISTANCE_M:-0.45}"
REQUIRE_OBSTACLE_CLEAR="${BCI_REQUIRE_OBSTACLE_CLEAR:-true}"
ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/bsense_ros_logs}"

case "${REQUIRE_OBSTACLE_CLEAR}" in
  true)
    DEFAULT_BRIDGE_CONFIG="${PROJECT_ROOT}/config/unitree_bridge_ros2.json"
    ;;
  false)
    DEFAULT_BRIDGE_CONFIG="${PROJECT_ROOT}/config/unitree_bridge_ros2_no_obstacle_sensor.json"
    echo "WARN: 避障联锁已关闭；仅允许在无避障传感器的受控联调环境使用，禁止用于比赛。" >&2
    ;;
  *)
    echo "ERROR: BCI_REQUIRE_OBSTACLE_CLEAR 只能设置为 true 或 false。" >&2
    exit 2
    ;;
esac
BRIDGE_CONFIG="${BCI_BRIDGE_CONFIG:-${DEFAULT_BRIDGE_CONFIG}}"

export AMENT_TRACE_SETUP_FILES=""
set +u
source "${ROS_SETUP}"
source "${UNITREE_SETUP}"
source "${PROJECT_SETUP}"
set -u

export ROS_LOG_DIR
mkdir -p "${ROS_LOG_DIR}"

CONFIG_REQUIRE_OBSTACLE_CLEAR="$(
  python3 -c \
    'import json, sys; print(str(json.load(open(sys.argv[1], encoding="utf-8")).get("require_obstacle_clear", True)).lower())' \
    "${BRIDGE_CONFIG}"
)"
if [[ "${CONFIG_REQUIRE_OBSTACLE_CLEAR}" != "${REQUIRE_OBSTACLE_CLEAR}" ]]; then
  echo "ERROR: ${BRIDGE_CONFIG} 的 require_obstacle_clear=${CONFIG_REQUIRE_OBSTACLE_CLEAR}，与 BCI_REQUIRE_OBSTACLE_CLEAR=${REQUIRE_OBSTACLE_CLEAR} 不一致。" >&2
  exit 2
fi

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
  -p "require_obstacle_clear:=${REQUIRE_OBSTACLE_CLEAR}" &
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
