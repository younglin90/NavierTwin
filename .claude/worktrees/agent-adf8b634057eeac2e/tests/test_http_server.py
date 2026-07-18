"""Round 167 — 경량 HTTP 추론 서버."""

from __future__ import annotations

import json
import urllib.request


class TestServer:
    def test_roundtrip(self) -> None:
        from naviertwin.core.serving.http_server import InferenceServer

        def handler(data: dict):
            x = data.get("x", 0)
            return {"double": 2 * x}

        with InferenceServer(handler) as srv:
            url = f"http://{srv.host}:{srv.port}/"
            req = urllib.request.Request(
                url, data=json.dumps({"x": 21}).encode(), method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                body = json.loads(resp.read().decode())
            assert body["ok"] is True
            assert body["result"]["double"] == 42

    def test_error(self) -> None:
        from naviertwin.core.serving.http_server import InferenceServer

        def handler(data: dict):  # noqa: ARG001
            raise ValueError("bad")

        with InferenceServer(handler) as srv:
            url = f"http://{srv.host}:{srv.port}/"
            req = urllib.request.Request(
                url, data=b"{}", method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    body = json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                body = json.loads(e.read().decode())
            assert body["ok"] is False
            assert "bad" in body["error"]
