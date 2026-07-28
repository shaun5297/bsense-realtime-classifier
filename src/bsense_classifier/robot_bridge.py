"""Safe, transport-agnostic command delivery for a Unitree bridge."""

from __future__ import annotations

import json
import socket
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


RobotCommand = Literal["forward", "backward", "left", "right", "stop", "idle"]
MOVEMENT_COMMANDS = frozenset({"forward", "backward", "left", "right"})
COMMAND_VELOCITIES: dict[RobotCommand, tuple[float, float]] = {
    "forward": (0.35, 0.0),
    "backward": (-0.25, 0.0),
    "left": (0.0, 0.65),
    "right": (0.0, -0.65),
    "stop": (0.0, 0.0),
    "idle": (0.0, 0.0),
}


class BridgeError(RuntimeError):
    """Raised when the robot bridge cannot accept a command."""


class SafetyInterlockError(BridgeError):
    """Raised when a command is rejected by a safety interlock."""


@dataclass(frozen=True)
class RobotBridgeConfig:
    transport: Literal["stdout", "http", "udp", "unitree_ros2"] = "stdout"
    command_url: str = "http://127.0.0.1:8000/api/v1/command"
    status_url: str = "http://127.0.0.1:8000/api/v1/status"
    udp_host: str = "127.0.0.1"
    udp_port: int = 5005
    socket_path: str = "/tmp/bsense_unitree.sock"
    timeout_seconds: float = 1.0
    require_obstacle_clear: bool = True
    motion_duration_seconds: float = 0.8
    linear_speed_mps: float = 0.35
    backward_speed_mps: float = 0.25
    angular_speed_rps: float = 0.65

    @classmethod
    def load(cls, path: Path | str) -> "RobotBridgeConfig":
        source = Path(path).expanduser().resolve()
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise BridgeError(f"机器狗桥接配置不存在：{source}") from exc
        except json.JSONDecodeError as exc:
            raise BridgeError(f"机器狗桥接配置不是有效 JSON：{exc}") from exc
        if not isinstance(raw, dict):
            raise BridgeError("机器狗桥接配置必须是 JSON 对象。")
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise BridgeError(f"机器狗桥接配置包含未知字段：{', '.join(unknown)}")
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if self.transport not in {"stdout", "http", "udp", "unitree_ros2"}:
            raise BridgeError(f"不支持的桥接传输方式：{self.transport}")
        if not 1 <= int(self.udp_port) <= 65535:
            raise BridgeError("udp_port 必须在 1–65535 范围内。")
        if self.timeout_seconds <= 0:
            raise BridgeError("timeout_seconds 必须大于 0。")
        if not 0.1 <= self.motion_duration_seconds <= 5.0:
            raise BridgeError("motion_duration_seconds 必须在 0.1–5 秒范围内。")
        if self.linear_speed_mps <= 0 or self.backward_speed_mps <= 0:
            raise BridgeError("直线速度必须大于 0。")
        if self.angular_speed_rps <= 0:
            raise BridgeError("转向角速度必须大于 0。")
        if self.transport == "unitree_ros2" and not self.socket_path.strip():
            raise BridgeError("unitree_ros2 传输必须提供 socket_path。")


@dataclass(frozen=True)
class BridgeStatus:
    connected: bool
    obstacle_detected: bool | None
    emergency_stopped: bool
    detail: str
    raw: dict[str, Any] | None = None

    @property
    def obstacle_clear(self) -> bool:
        return self.obstacle_detected is False


class RobotBridgeClient:
    """Send a stable JSON command contract over stdout, HTTP, or UDP."""

    def __init__(self, config: RobotBridgeConfig) -> None:
        config.validate()
        self.config = config
        self._unix_socket: socket.socket | None = None
        self._unix_buffer = bytearray()
        self._unix_lock = threading.RLock()

    def _velocity(self, command: RobotCommand) -> tuple[float, float]:
        if command == "forward":
            return self.config.linear_speed_mps, 0.0
        if command == "backward":
            return -self.config.backward_speed_mps, 0.0
        if command == "left":
            return 0.0, self.config.angular_speed_rps
        if command == "right":
            return 0.0, -self.config.angular_speed_rps
        return COMMAND_VELOCITIES[command]

    def build_payload(
        self,
        command: RobotCommand,
        *,
        confidence: float | None = None,
        source: str = "bsense_p300",
        emergency: bool = False,
    ) -> dict[str, Any]:
        linear_x, angular_z = self._velocity(command)
        return {
            "schema": "bsense.unitree.command.v1",
            "kind": "command",
            "command": command,
            "linear_x": linear_x,
            "angular_z": angular_z,
            "duration_s": (
                self.config.motion_duration_seconds
                if command in MOVEMENT_COMMANDS
                else 0.0
            ),
            "source": source,
            "confidence": confidence,
            "emergency": emergency,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

    def send(
        self,
        command: RobotCommand,
        *,
        confidence: float | None = None,
        source: str = "bsense_p300",
        emergency: bool = False,
    ) -> dict[str, Any]:
        payload = self.build_payload(
            command,
            confidence=confidence,
            source=source,
            emergency=emergency,
        )
        return self._deliver(payload)

    def _deliver(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.config.transport == "stdout":
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            return {"ok": True, "transport": "stdout"}
        if self.config.transport == "unitree_ros2":
            return self._unix_exchange(payload)
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if self.config.transport == "udp":
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.settimeout(self.config.timeout_seconds)
                    sock.sendto(
                        encoded,
                        (self.config.udp_host, int(self.config.udp_port)),
                    )
            except OSError as exc:
                raise BridgeError(f"UDP 指令发送失败：{exc}") from exc
            return {"ok": True, "transport": "udp"}
        request = Request(
            self.config.command_url,
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                body = response.read().decode("utf-8", errors="replace")
                if not body:
                    return {"ok": True, "status": response.status}
                parsed = json.loads(body)
                if isinstance(parsed, dict):
                    if parsed.get("ok") is False:
                        detail = parsed.get("detail", parsed.get("message", "未知原因"))
                        raise BridgeError(f"桥接端拒绝指令：{detail}")
                    return parsed
                return {"ok": True, "data": parsed}
        except BridgeError:
            raise
        except (HTTPError, URLError, OSError, json.JSONDecodeError) as exc:
            raise BridgeError(f"HTTP 指令发送失败：{exc}") from exc

    def _close_unix(self) -> None:
        if self._unix_socket is not None:
            try:
                self._unix_socket.close()
            except OSError:
                pass
        self._unix_socket = None
        self._unix_buffer.clear()

    def close(self) -> None:
        with self._unix_lock:
            self._close_unix()

    def _connect_unix(self) -> socket.socket:
        if self._unix_socket is not None:
            return self._unix_socket
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.config.timeout_seconds)
        try:
            sock.connect(self.config.socket_path)
        except OSError:
            sock.close()
            raise
        self._unix_socket = sock
        return sock

    def _read_unix_line(self, sock: socket.socket) -> bytes:
        while b"\n" not in self._unix_buffer:
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("ROS 2 Bridge 已关闭 Unix Socket。")
            self._unix_buffer.extend(chunk)
            if len(self._unix_buffer) > 65536:
                raise ConnectionError("ROS 2 Bridge 响应超过 64 KiB。")
        line, _, remaining = self._unix_buffer.partition(b"\n")
        self._unix_buffer = bytearray(remaining)
        return bytes(line)

    def _unix_exchange(self, payload: dict[str, Any]) -> dict[str, Any]:
        encoded = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        last_error: Exception | None = None
        with self._unix_lock:
            for _attempt in range(2):
                try:
                    sock = self._connect_unix()
                    sock.sendall(encoded)
                    response = json.loads(self._read_unix_line(sock).decode("utf-8"))
                    if not isinstance(response, dict):
                        raise BridgeError("ROS 2 Bridge 响应必须是 JSON 对象。")
                    if response.get("ok") is False:
                        detail = response.get(
                            "detail",
                            response.get("message", "未知原因"),
                        )
                        raise SafetyInterlockError(f"ROS 2 Bridge 拒绝请求：{detail}")
                    return response
                except SafetyInterlockError:
                    raise
                except (
                    OSError,
                    ConnectionError,
                    json.JSONDecodeError,
                    UnicodeDecodeError,
                    BridgeError,
                ) as exc:
                    last_error = exc
                    self._close_unix()
            raise BridgeError(f"ROS 2 Bridge 通信失败：{last_error}") from last_error

    def status(self) -> BridgeStatus:
        if self.config.transport == "stdout":
            return BridgeStatus(True, False, False, "stdout 联调模式")
        if self.config.transport == "udp":
            return BridgeStatus(
                True,
                None,
                False,
                "UDP 为单向传输，无法读取避障状态",
            )
        if self.config.transport == "unitree_ros2":
            try:
                raw = self._unix_exchange(
                    {
                        "schema": "bsense.unitree.command.v1",
                        "kind": "status",
                    }
                )
            except BridgeError as exc:
                return BridgeStatus(False, None, True, str(exc))
            return self._parse_status(raw)
        request = Request(self.config.status_url, method="GET")
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, OSError, json.JSONDecodeError) as exc:
            return BridgeStatus(False, None, True, f"状态读取失败：{exc}")
        if not isinstance(raw, dict):
            return BridgeStatus(False, None, True, "状态响应必须是 JSON 对象")
        return self._parse_status(raw)

    @staticmethod
    def _parse_status(raw: dict[str, Any]) -> BridgeStatus:
        obstacle = raw.get("obstacle_detected")
        if obstacle is None and "obstacle_clear" in raw:
            obstacle_clear = raw["obstacle_clear"]
            obstacle = not obstacle_clear if isinstance(obstacle_clear, bool) else None
        obstacle_value = obstacle if isinstance(obstacle, bool) else None
        emergency = bool(
            raw.get("emergency_stopped", raw.get("emergency_stop", False))
        )
        connected = bool(raw.get("connected", raw.get("ok", True)))
        detail = str(raw.get("detail", raw.get("message", "桥接状态正常")))
        return BridgeStatus(connected, obstacle_value, emergency, detail, raw)

    def clear_emergency(self) -> BridgeStatus:
        if self.config.transport == "stdout":
            return self.status()
        if self.config.transport == "udp":
            raise BridgeError("UDP 传输无法确认或解除急停。")
        if self.config.transport == "unitree_ros2":
            raw = self._unix_exchange(
                {
                    "schema": "bsense.unitree.command.v1",
                    "kind": "arm",
                }
            )
            return self._parse_status(raw)
        payload = self.build_payload("idle", source="bsense_arm_reset")
        payload["reset_emergency"] = True
        self._deliver(payload)
        return self.status()


class SafeRobotController:
    """Fail-closed arming, obstacle interlock, and automatic motion expiry."""

    def __init__(self, bridge: RobotBridgeClient) -> None:
        self.bridge = bridge
        self._lock = threading.RLock()
        self._armed = False
        self._motion_timer: threading.Timer | None = None

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._armed

    def arm(self) -> BridgeStatus:
        status = self.bridge.status()
        if not status.connected:
            raise SafetyInterlockError(status.detail)
        if status.emergency_stopped:
            status = self.bridge.clear_emergency()
        if status.emergency_stopped:
            raise SafetyInterlockError("桥接端仍处于急停状态，不能解锁。")
        if self.bridge.config.require_obstacle_clear and not status.obstacle_clear:
            raise SafetyInterlockError("未确认避障区域安全，不能解锁。")
        self.bridge.send("stop", source="bsense_arm_check")
        with self._lock:
            self._armed = True
        return status

    def disarm(self, *, emergency: bool = False) -> dict[str, Any]:
        with self._lock:
            self._armed = False
            self._cancel_motion_timer()
        return self.bridge.send(
            "stop",
            source="bsense_emergency_stop" if emergency else "bsense_disarm",
            emergency=emergency,
        )

    def emergency_stop(self) -> dict[str, Any]:
        return self.disarm(emergency=True)

    def execute(
        self,
        command: RobotCommand,
        *,
        confidence: float | None = None,
        source: str = "bsense_p300",
    ) -> dict[str, Any]:
        if command == "stop":
            return self.emergency_stop()
        if command == "idle":
            with self._lock:
                self._cancel_motion_timer()
            return self.bridge.send("idle", confidence=confidence, source=source)
        with self._lock:
            if not self._armed:
                raise SafetyInterlockError("控制器未解锁，移动指令已拦截。")
        status = self.bridge.status()
        if not status.connected:
            self.emergency_stop()
            raise SafetyInterlockError(status.detail)
        if status.emergency_stopped:
            with self._lock:
                self._armed = False
            raise SafetyInterlockError("桥接端处于急停状态，移动指令已拦截。")
        if self.bridge.config.require_obstacle_clear and not status.obstacle_clear:
            self.bridge.send("stop", source="bsense_obstacle_interlock")
            raise SafetyInterlockError(
                "检测到障碍或避障状态未知，移动指令已拦截。"
            )
        response = self.bridge.send(
            command,
            confidence=confidence,
            source=source,
        )
        with self._lock:
            self._cancel_motion_timer()
            self._motion_timer = threading.Timer(
                self.bridge.config.motion_duration_seconds,
                self._expire_motion,
            )
            self._motion_timer.daemon = True
            self._motion_timer.start()
        return response

    def _cancel_motion_timer(self) -> None:
        if self._motion_timer is not None:
            self._motion_timer.cancel()
            self._motion_timer = None

    def _expire_motion(self) -> None:
        try:
            self.bridge.send("stop", source="bsense_motion_watchdog")
        finally:
            with self._lock:
                self._motion_timer = None

    def close(self) -> None:
        with self._lock:
            self._cancel_motion_timer()
        self.bridge.close()
