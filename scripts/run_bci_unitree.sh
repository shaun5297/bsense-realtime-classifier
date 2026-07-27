#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="/home/dd/bsense-realtime-classifier"
PY_RUNTIME="/tmp/bsense_runtime"
PY_LSL="/tmp/bsense_pylsl"
LSL_LIB="/tmp/liblsl/liblsl-1.17.7-jammy_arm64/lib/liblsl.so.1.17.7"
SOCKET_PATH="/tmp/bci_unitree.sock"

export ROS_LOG_DIR="/tmp/bsense_ros_logs"
mkdir -p "${ROS_LOG_DIR}"

source /opt/ros/humble/setup.bash
source /home/dd/unitree_ros2/setup.sh
source "${PROJECT_ROOT}/install/setup.bash"

set -u

if [[ -n "${BCI_CYCLONEDDS_INTERFACE:-}" ]]; then
  export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"${BCI_CYCLONEDDS_INTERFACE}\" priority=\"default\" multicast=\"default\" /></Interfaces></General></Domain></CycloneDDS>"
fi

BRIDGE_PID=""
cleanup() {
  if [[ -n "${BRIDGE_PID}" ]]; then
    kill "${BRIDGE_PID}" 2>/dev/null || true
    wait "${BRIDGE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ros2 run bsense_realtime_classifier unitree_ros2_bridge &
BRIDGE_PID=$!
sleep 0.5
if ! kill -0 "${BRIDGE_PID}" 2>/dev/null; then
  echo "ERROR: unitree_ros2_bridge failed to start; check ROS_LOG_DIR and BCI_CYCLONEDDS_INTERFACE." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
env \
  PYTHONPATH="${PY_RUNTIME}:${PY_LSL}:${PROJECT_ROOT}/src" \
  PYLSL_LIB="${LSL_LIB}" \
  python3 -m bsense_classifier.cli \
    --model "${PROJECT_ROOT}/models/m1_mi/model.joblib" \
    --stream-type EEG \
    --threshold 0.30 \
    --smoothing 1 \
    --confirmation 1 \
    --no-marker \
    --transport unitree_ros2 \
    --socket-path "${SOCKET_PATH}"
