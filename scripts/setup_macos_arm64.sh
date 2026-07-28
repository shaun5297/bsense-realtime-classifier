#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: 此脚本只支持 Apple Silicon macOS（arm64）。" >&2
  exit 1
fi

PYTHON_BIN="${BCI_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  for candidate in python3.12 python3.11 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "ERROR: 未找到 Python 3.11/3.12。可先运行 brew install python@3.12 python-tk@3.12。" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import platform
import sys

if not ((3, 11) <= sys.version_info[:2] < (3, 14)):
    raise SystemExit("ERROR: 需要 Python 3.11–3.13。")
if platform.machine() != "arm64":
    raise SystemExit("ERROR: Python 不是原生 arm64，请避免在 Rosetta 下安装。")
try:
    import tkinter
except ImportError as exc:
    raise SystemExit(
        "ERROR: Python 缺少 Tk。Homebrew 用户可安装 python-tk@3.12。"
    ) from exc
PY

VENV_PATH="${BCI_MACOS_VENV:-${PROJECT_ROOT}/.venv-macos-arm64}"
"${PYTHON_BIN}" -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if [[ "${BCI_INSTALL_EEGNET:-0}" == "1" ]]; then
  python -m pip install -e "${PROJECT_ROOT}[xdf,deep]"
else
  python -m pip install -e "${PROJECT_ROOT}[xdf]"
fi

python - <<'PY'
import platform
import tkinter
import pylsl

print(f"Python architecture: {platform.machine()}")
print(f"pylsl: {pylsl.__version__}")
print(f"Tk: {tkinter.TkVersion}")
PY

echo ""
echo "macOS ARM64 环境已就绪：${VENV_PATH}"
echo "安全联调：./scripts/run_dog_controller_macos.sh"
echo "若使用 EEGNet，重新运行前设置 BCI_INSTALL_EEGNET=1。"
