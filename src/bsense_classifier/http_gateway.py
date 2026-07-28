"""Authenticated HTTP gateway for the local Unitree ROS 2 Unix bridge."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import socket
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


SCHEMA = "bsense.unitree.command.v1"
COMMANDS = frozenset({"forward", "backward", "left", "right", "stop", "idle"})
MAX_BODY_BYTES = 65536


class GatewayError(RuntimeError):
    """Raised when an HTTP request cannot be relayed safely."""


def translate_command_payload(payload: Any) -> dict[str, Any]:
    """Validate the public HTTP contract and map reset to the ROS arm request."""

    if not isinstance(payload, dict):
        raise GatewayError("请求体必须是 JSON 对象。")
    if payload.get("schema") != SCHEMA or payload.get("kind") != "command":
        raise GatewayError("schema 或 kind 无效。")
    if payload.get("reset_emergency") is True:
        return {"schema": SCHEMA, "kind": "arm"}
    command = payload.get("command")
    if command not in COMMANDS:
        raise GatewayError("不支持的机器狗指令。")
    return payload


class UnixBridgeRelay:
    def __init__(self, socket_path: str, timeout_seconds: float = 1.0) -> None:
        self.socket_path = socket_path
        self.timeout_seconds = timeout_seconds

    def exchange(self, payload: dict[str, Any]) -> dict[str, Any]:
        encoded = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(self.timeout_seconds)
                client.connect(self.socket_path)
                client.sendall(encoded)
                response = bytearray()
                while b"\n" not in response:
                    chunk = client.recv(4096)
                    if not chunk:
                        raise GatewayError("ROS 2 Bridge 提前关闭连接。")
                    response.extend(chunk)
                    if len(response) > MAX_BODY_BYTES:
                        raise GatewayError("ROS 2 Bridge 响应过大。")
        except OSError as exc:
            raise GatewayError(f"ROS 2 Bridge 不可用：{exc}") from exc
        line = bytes(response).split(b"\n", 1)[0]
        try:
            parsed = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GatewayError("ROS 2 Bridge 返回了无效 JSON。") from exc
        if not isinstance(parsed, dict):
            raise GatewayError("ROS 2 Bridge 响应必须是 JSON 对象。")
        return parsed


class GatewayHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        relay: UnixBridgeRelay,
        token: str,
    ) -> None:
        super().__init__(server_address, GatewayRequestHandler)
        self.relay = relay
        self.token = token


class GatewayRequestHandler(BaseHTTPRequestHandler):
    server: GatewayHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        print(
            f"{self.address_string()} - {format % args}",
            flush=True,
        )

    def _authorized(self) -> bool:
        expected = f"Bearer {self.server.token}"
        supplied = self.headers.get("Authorization", "")
        return hmac.compare_digest(supplied, expected)

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _require_authorization(self) -> bool:
        if self._authorized():
            return True
        self._write_json(
            HTTPStatus.UNAUTHORIZED,
            {"ok": False, "detail": "unauthorized"},
        )
        return False

    def do_GET(self) -> None:
        if self.path != "/api/v1/status":
            self._write_json(
                HTTPStatus.NOT_FOUND,
                {"ok": False, "detail": "not found"},
            )
            return
        if not self._require_authorization():
            return
        try:
            response = self.server.relay.exchange(
                {"schema": SCHEMA, "kind": "status"}
            )
        except GatewayError as exc:
            self._write_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"ok": False, "connected": False, "detail": str(exc)},
            )
            return
        self._write_json(HTTPStatus.OK, response)

    def do_POST(self) -> None:
        if self.path != "/api/v1/command":
            self._write_json(
                HTTPStatus.NOT_FOUND,
                {"ok": False, "detail": "not found"},
            )
            return
        if not self._require_authorization():
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if not 0 < length <= MAX_BODY_BYTES:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "detail": "invalid content length"},
            )
            return
        try:
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
            relay_payload = translate_command_payload(payload)
            response = self.server.relay.exchange(relay_payload)
        except (UnicodeDecodeError, json.JSONDecodeError, GatewayError) as exc:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "detail": str(exc)},
            )
            return
        self._write_json(HTTPStatus.OK, response)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--socket-path",
        default="/tmp/bsense_unitree.sock",
    )
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument(
        "--token-env",
        default="BCI_BRIDGE_TOKEN",
        help="包含网关访问令牌的环境变量名。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    token = os.environ.get(args.token_env, "").strip()
    if len(token) < 16:
        raise SystemExit(
            f"{args.token_env} 必须设置为至少 16 个字符的随机令牌。"
        )
    socket_path = str(Path(args.socket_path).expanduser())
    relay = UnixBridgeRelay(socket_path, timeout_seconds=args.timeout)
    server = GatewayHTTPServer((args.host, args.port), relay, token)
    print(
        f"BSense Unitree HTTP gateway listening on "
        f"http://{args.host}:{args.port}; socket={socket_path}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            relay.exchange(
                {
                    "schema": SCHEMA,
                    "kind": "command",
                    "command": "stop",
                    "linear_x": 0.0,
                    "angular_z": 0.0,
                    "duration_s": 0.0,
                    "source": "bsense_http_gateway_shutdown",
                    "confidence": None,
                    "emergency": True,
                }
            )
        except GatewayError:
            pass
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
