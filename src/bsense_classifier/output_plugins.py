"""Output plugins for machine-readable inference results."""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from typing import Any


DEFAULT_MOTIONS: dict[str, dict[str, float]] = {
    "idle": {"x": 0.0, "y": 0.0, "z": 0.0},
    "left_mi": {"x": 0.0, "y": 0.0, "z": 0.25},
    "right_mi": {"x": 0.0, "y": 0.0, "z": -0.25},
    "forward": {"x": 0.15, "y": 0.0, "z": 0.0},
    "backward": {"x": -0.15, "y": 0.0, "z": 0.0},
    "strafe_left": {"x": 0.0, "y": 0.15, "z": 0.0},
    "strafe_right": {"x": 0.0, "y": -0.15, "z": 0.0},
    "turn_left": {"x": 0.0, "y": 0.0, "z": 0.25},
    "turn_right": {"x": 0.0, "y": 0.0, "z": -0.25},
}


class UnitreeOutput:
    """Send bci_result_v1 records to the local ROS 2 adapter over Unix Socket."""

    def __init__(self, socket_path: str, connect_timeout: float = 10.0) -> None:
        self.socket_path = Path(socket_path)
        deadline = time.monotonic() + connect_timeout
        self._socket: socket.socket | None = None
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(str(self.socket_path))
                self._socket = sock
                return
            except OSError as exc:
                last_error = exc
                try:
                    sock.close()
                except UnboundLocalError:
                    pass
                time.sleep(0.1)
        raise ConnectionError(
            f"无法连接 ROS 2 Bridge Unix Socket: {self.socket_path}"
        ) from last_error

    def send(self, record: dict[str, Any]) -> None:
        label = str(record.get("label", "unknown"))
        accepted = bool(record.get("accepted", False))
        motion = DEFAULT_MOTIONS.get(label, DEFAULT_MOTIONS["idle"])
        message = {
            "protocol": "bci_result_v1",
            "label": label,
            "accepted": accepted,
            "confidence": record.get("confidence"),
            "timestamp": record.get("lsl_timestamp"),
            "motion": motion if accepted else DEFAULT_MOTIONS["idle"],
        }
        assert self._socket is not None
        self._socket.sendall((json.dumps(message, ensure_ascii=False) + "\n").encode())

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
