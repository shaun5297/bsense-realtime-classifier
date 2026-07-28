#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${BCI_MACOS_VENV:-${PROJECT_ROOT}/.venv-macos-arm64}"
PYTHON="${VENV_PATH}/bin/python"
MODEL_PATH="${BCI_P300_MODEL:-${PROJECT_ROOT}/models/m7_p300/model.joblib}"
CONTROL_MODE="${BCI_CONTROL_MODE:-stdout}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: 此启动器只支持 Apple Silicon macOS（arm64）。" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: 尚未安装 macOS 环境，请先运行 ./scripts/setup_macos_arm64.sh。" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "ERROR: 找不到 P300 模型：${MODEL_PATH}" >&2
  exit 1
fi

case "${CONTROL_MODE}" in
  stdout)
    BRIDGE_CONFIG="${PROJECT_ROOT}/config/unitree_bridge.json"
    echo "安全联调模式：机器狗指令只输出到终端，不会发送到真机。"
    ;;
  http)
    : "${BCI_UNITREE_HOST:?ERROR: HTTP 真机模式必须设置 Ubuntu 主机地址 BCI_UNITREE_HOST。}"
    : "${BCI_BRIDGE_TOKEN:?ERROR: HTTP 真机模式必须设置 BCI_BRIDGE_TOKEN。}"
    if [[ ${#BCI_BRIDGE_TOKEN} -lt 16 ]]; then
      echo "ERROR: BCI_BRIDGE_TOKEN 至少需要 16 个字符。" >&2
      exit 1
    fi
    BRIDGE_CONFIG="${PROJECT_ROOT}/config/unitree_bridge_macos_http.json"
    echo "HTTP 真机模式：Ubuntu=${BCI_UNITREE_HOST}；请确认场地安全和急停可用。"
    ;;
  *)
    echo "ERROR: BCI_CONTROL_MODE 只能是 stdout 或 http。" >&2
    exit 1
    ;;
esac

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON}" -m bsense_classifier.robot_app \
  --model "${MODEL_PATH}" \
  --stream-name "${BCI_EEG_STREAM_NAME:-}" \
  --bridge-config "${BRIDGE_CONFIG}" \
  "$@"
