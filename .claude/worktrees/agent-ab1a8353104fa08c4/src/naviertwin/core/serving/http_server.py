"""경량 HTTP 추론 서버 — stdlib BaseHTTPRequestHandler.

POST / with JSON body → 핸들러 함수 호출 → JSON 응답.

Examples:
    >>> # 테스트에서 start_server → requests → stop 사용
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable


def _make_handler(fn: Callable[[dict], Any]):
    class _H(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                data = json.loads(body) if body else {}
                result = fn(data)
                resp = json.dumps({"ok": True, "result": result}).encode("utf-8")
                self.send_response(200)
            except Exception as e:  # noqa: BLE001
                resp = json.dumps({"ok": False, "error": str(e)}).encode("utf-8")
                self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, *a):  # noqa: ANN001, N802
            pass
    return _H


class InferenceServer:
    def __init__(
        self, handler: Callable[[dict], Any], host: str = "127.0.0.1", port: int = 0,
    ) -> None:
        self.handler = handler
        self._server = HTTPServer((host, port), _make_handler(handler))
        self.host, self.port = self._server.server_address
        self._thread: threading.Thread | None = None

    def start(self) -> "InferenceServer":
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread:
            self._thread.join(timeout=1.0)

    def __enter__(self) -> "InferenceServer":
        return self.start()

    def __exit__(self, *exc) -> None:  # noqa: ANN001
        self.stop()


__all__ = ["InferenceServer"]
