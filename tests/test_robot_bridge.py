from __future__ import annotations

import json
from pathlib import Path
import socket
import threading

import pytest

from bsense_classifier.robot_bridge import (
    BridgeStatus,
    RobotBridgeClient,
    RobotBridgeConfig,
    SafeRobotController,
    SafetyInterlockError,
)


def test_http_payload_uses_stable_command_contract() -> None:
    client = RobotBridgeClient(
        RobotBridgeConfig(
            transport="stdout",
            linear_speed_mps=0.4,
            motion_duration_seconds=0.7,
        )
    )
    payload = client.build_payload("forward", confidence=0.81)
    assert payload["schema"] == "bsense.unitree.command.v1"
    assert payload["command"] == "forward"
    assert payload["linear_x"] == 0.4
    assert payload["angular_z"] == 0.0
    assert payload["duration_s"] == 0.7
    assert payload["confidence"] == 0.81


def test_unitree_profiles_keep_obstacle_interlock_explicit() -> None:
    project_root = Path(__file__).resolve().parents[1]
    competition = RobotBridgeConfig.load(
        project_root / "config" / "unitree_bridge_ros2.json"
    )
    no_sensor_debug = RobotBridgeConfig.load(
        project_root
        / "config"
        / "unitree_bridge_ros2_no_obstacle_sensor.json"
    )

    assert competition.transport == "unitree_ros2"
    assert competition.require_obstacle_clear is True
    assert no_sensor_debug.transport == "unitree_ros2"
    assert no_sensor_debug.require_obstacle_clear is False


def test_http_token_is_read_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("BSENSE_TEST_TOKEN", "secret-token")
    client = RobotBridgeClient(
        RobotBridgeConfig(
            transport="http",
            auth_token_env="BSENSE_TEST_TOKEN",
        )
    )
    assert client._http_headers(content_type=True) == {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Bearer secret-token",
    }


class _FakeBridge:
    def __init__(self, status: BridgeStatus) -> None:
        self.config = RobotBridgeConfig(
            transport="stdout",
            require_obstacle_clear=True,
            motion_duration_seconds=1.0,
        )
        self.current_status = status
        self.commands: list[tuple[str, dict]] = []

    def status(self) -> BridgeStatus:
        return self.current_status

    def send(self, command: str, **kwargs: object) -> dict:
        self.commands.append((command, kwargs))
        return {"ok": True}

    def clear_emergency(self) -> BridgeStatus:
        self.current_status = BridgeStatus(True, False, False, "cleared")
        return self.current_status

    def close(self) -> None:
        return None


def test_controller_requires_explicit_arm_and_clear_obstacle() -> None:
    bridge = _FakeBridge(BridgeStatus(True, False, False, "ok"))
    controller = SafeRobotController(bridge)
    with pytest.raises(SafetyInterlockError):
        controller.execute("forward")
    controller.arm()
    controller.execute("forward", confidence=0.8)
    controller.disarm()
    assert [command for command, _ in bridge.commands] == [
        "stop",
        "forward",
        "stop",
    ]


def test_controller_fails_closed_when_obstacle_status_is_unknown() -> None:
    bridge = _FakeBridge(BridgeStatus(True, None, False, "unknown"))
    controller = SafeRobotController(bridge)
    with pytest.raises(SafetyInterlockError, match="避障"):
        controller.arm()
    assert controller.armed is False


def test_emergency_stop_is_allowed_while_disarmed() -> None:
    bridge = _FakeBridge(BridgeStatus(True, False, False, "ok"))
    controller = SafeRobotController(bridge)
    controller.emergency_stop()
    assert bridge.commands[-1][0] == "stop"
    assert bridge.commands[-1][1]["emergency"] is True


def test_arm_explicitly_clears_bridge_emergency_latch() -> None:
    bridge = _FakeBridge(BridgeStatus(True, False, True, "latched"))
    controller = SafeRobotController(bridge)
    controller.arm()
    assert controller.armed is True
    assert bridge.current_status.emergency_stopped is False


@pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"),
    reason="Unix domain sockets are unavailable on this Python build",
)
def test_unitree_ros2_transport_is_bidirectional(tmp_path: Path) -> None:
    socket_path = tmp_path / "unitree.sock"
    ready = threading.Event()
    received: list[dict] = []

    def serve() -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(socket_path))
            server.listen(1)
            ready.set()
            client, _address = server.accept()
            with client:
                pending = bytearray()
                while len(received) < 3:
                    chunk = client.recv(4096)
                    if not chunk:
                        return
                    pending.extend(chunk)
                    while b"\n" in pending:
                        line, _, remainder = pending.partition(b"\n")
                        pending = bytearray(remainder)
                        message = json.loads(line)
                        received.append(message)
                        emergency = message["kind"] == "status"
                        response = {
                            "ok": True,
                            "connected": True,
                            "obstacle_detected": False,
                            "emergency_stopped": emergency,
                            "detail": "ok",
                        }
                        client.sendall(
                            (json.dumps(response) + "\n").encode("utf-8")
                        )

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    assert ready.wait(1.0)
    client = RobotBridgeClient(
        RobotBridgeConfig(
            transport="unitree_ros2",
            socket_path=str(socket_path),
            timeout_seconds=1.0,
        )
    )
    assert client.status().emergency_stopped is True
    assert client.clear_emergency().emergency_stopped is False
    client.send("forward", confidence=0.8)
    client.close()
    thread.join(1.0)
    assert [message["kind"] for message in received] == [
        "status",
        "arm",
        "command",
    ]
