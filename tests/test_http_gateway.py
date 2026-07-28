from __future__ import annotations

import json
import threading
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from bsense_classifier.http_gateway import (
    GatewayHTTPServer,
    SCHEMA,
    translate_command_payload,
)


class _FakeRelay:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def exchange(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(payload)
        return {
            "ok": True,
            "connected": True,
            "obstacle_detected": False,
            "emergency_stopped": False,
            "detail": "ok",
        }


def test_gateway_maps_http_emergency_reset_to_arm() -> None:
    payload = {
        "schema": SCHEMA,
        "kind": "command",
        "command": "idle",
        "reset_emergency": True,
    }
    assert translate_command_payload(payload) == {
        "schema": SCHEMA,
        "kind": "arm",
    }


def test_gateway_requires_token_and_relays_status() -> None:
    relay = _FakeRelay()
    token = "test-token-at-least-16"
    server = GatewayHTTPServer(("127.0.0.1", 0), relay, token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = f"http://127.0.0.1:{server.server_port}/api/v1/status"
    try:
        with pytest.raises(HTTPError) as unauthorized:
            urlopen(address, timeout=1.0)
        assert unauthorized.value.code == 401

        request = Request(
            address,
            headers={"Authorization": f"Bearer {token}"},
            method="GET",
        )
        with urlopen(request, timeout=1.0) as response:
            body = json.loads(response.read().decode("utf-8"))
        assert body["connected"] is True
        assert relay.requests == [{"schema": SCHEMA, "kind": "status"}]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(1.0)
