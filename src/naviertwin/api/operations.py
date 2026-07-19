"""Dependency-light API security, rate limiting, and observability primitives."""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class APISettings:
    """Operational API settings loaded from environment variables."""

    api_keys: tuple[str, ...] = ()
    api_key_hashes: tuple[str, ...] = ()
    rate_limit_requests: int = 0
    rate_limit_window_seconds: float = 60.0
    rate_limit_store_path: str | None = None
    max_request_bytes: int = 64 * 1024 * 1024
    expose_metrics: bool = True

    def __post_init__(self) -> None:
        if self.rate_limit_requests < 0:
            raise ValueError("rate_limit_requests must be non-negative")
        if self.rate_limit_window_seconds <= 0:
            raise ValueError("rate_limit_window_seconds must be positive")
        if self.max_request_bytes < 1:
            raise ValueError("max_request_bytes must be positive")
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value.lower())
            for value in self.api_key_hashes
        ):
            raise ValueError("api_key_hashes must contain SHA-256 hex digests")

    @classmethod
    def from_env(cls) -> "APISettings":
        keys = tuple(
            value.strip()
            for value in os.getenv("NAVIERTWIN_API_KEYS", "").split(",")
            if value.strip()
        )
        key_hashes = tuple(
            value.strip().lower()
            for value in os.getenv("NAVIERTWIN_API_KEY_HASHES", "").split(",")
            if value.strip()
        )
        if any(len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in key_hashes):
            raise ValueError("NAVIERTWIN_API_KEY_HASHES must contain SHA-256 hex digests")
        return cls(
            api_keys=keys,
            api_key_hashes=key_hashes,
            rate_limit_requests=max(
                0, int(os.getenv("NAVIERTWIN_RATE_LIMIT_REQUESTS", "0"))
            ),
            rate_limit_window_seconds=max(
                1.0, float(os.getenv("NAVIERTWIN_RATE_LIMIT_WINDOW_SECONDS", "60"))
            ),
            rate_limit_store_path=os.getenv("NAVIERTWIN_RATE_LIMIT_STORE") or None,
            max_request_bytes=max(
                1024, int(os.getenv("NAVIERTWIN_MAX_REQUEST_BYTES", str(64 * 1024 * 1024)))
            ),
            expose_metrics=os.getenv("NAVIERTWIN_EXPOSE_METRICS", "1").lower()
            not in {"0", "false", "no"},
        )


class SlidingWindowRateLimiter:
    """Thread-safe in-process sliding-window limiter keyed by client identity."""

    def __init__(self, limit: int, window_seconds: float, clock: Any = time.monotonic) -> None:
        self.limit = max(0, int(limit))
        self.window_seconds = max(0.001, float(window_seconds))
        self._clock = clock
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, identity: str) -> tuple[bool, float]:
        if self.limit == 0:
            return True, 0.0
        now = float(self._clock())
        cutoff = now - self.window_seconds
        with self._lock:
            events = self._events[identity]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                return False, max(0.001, self.window_seconds - (now - events[0]))
            events.append(now)
            return True, 0.0


class SQLiteWindowRateLimiter:
    """Process-shared fixed-window limiter backed by transactional SQLite."""

    def __init__(
        self,
        path: str,
        limit: int,
        window_seconds: float,
        clock: Any = time.time,
    ) -> None:
        self.path = str(path)
        self.limit = max(0, int(limit))
        self.window_seconds = max(1.0, float(window_seconds))
        self._clock = clock
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=10.0) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS rate_limit_buckets ("
                "identity TEXT NOT NULL, bucket INTEGER NOT NULL, count INTEGER NOT NULL, "
                "PRIMARY KEY(identity, bucket))"
            )

    def allow(self, identity: str) -> tuple[bool, float]:
        if self.limit == 0:
            return True, 0.0
        now = float(self._clock())
        bucket = int(now // self.window_seconds)
        bucket_end = (bucket + 1) * self.window_seconds
        with sqlite3.connect(self.path, timeout=10.0, isolation_level=None) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM rate_limit_buckets WHERE bucket < ?", (bucket - 1,)
            )
            row = connection.execute(
                "SELECT count FROM rate_limit_buckets WHERE identity = ? AND bucket = ?",
                (identity, bucket),
            ).fetchone()
            count = int(row[0]) if row else 0
            if count >= self.limit:
                connection.execute("COMMIT")
                return False, max(0.001, bucket_end - now)
            connection.execute(
                "INSERT INTO rate_limit_buckets(identity, bucket, count) VALUES (?, ?, 1) "
                "ON CONFLICT(identity, bucket) DO UPDATE SET count = count + 1",
                (identity, bucket),
            )
            connection.execute("COMMIT")
        return True, 0.0


class APIMetrics:
    """Bounded route-template counters and latency aggregates."""

    def __init__(self) -> None:
        self.started_at = time.time()
        self._lock = threading.Lock()
        self._in_flight = 0
        self._requests = 0
        self._duration_seconds = 0.0
        self._status: dict[str, int] = defaultdict(int)
        self._routes: dict[tuple[str, str], dict[str, float]] = defaultdict(
            lambda: {"requests": 0, "duration_seconds": 0.0, "errors": 0}
        )

    def begin(self) -> None:
        with self._lock:
            self._in_flight += 1

    def finish(self, method: str, route: str, status: int, duration: float) -> None:
        status_class = f"{int(status) // 100}xx"
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._requests += 1
            self._duration_seconds += float(duration)
            self._status[status_class] += 1
            item = self._routes[(method, route)]
            item["requests"] += 1
            item["duration_seconds"] += float(duration)
            if status >= 500:
                item["errors"] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "uptime_seconds": max(0.0, time.time() - self.started_at),
                "in_flight": self._in_flight,
                "requests": self._requests,
                "duration_seconds": self._duration_seconds,
                "status": dict(self._status),
                "routes": {
                    f"{method} {route}": dict(values)
                    for (method, route), values in sorted(self._routes.items())
                },
            }

    def prometheus(self) -> str:
        snapshot = self.snapshot()
        lines = [
            "# HELP naviertwin_api_requests_total HTTP requests.",
            "# TYPE naviertwin_api_requests_total counter",
            f"naviertwin_api_requests_total {snapshot['requests']}",
            "# HELP naviertwin_api_in_flight Current HTTP requests.",
            "# TYPE naviertwin_api_in_flight gauge",
            f"naviertwin_api_in_flight {snapshot['in_flight']}",
            "# HELP naviertwin_api_request_duration_seconds_total Request time.",
            "# TYPE naviertwin_api_request_duration_seconds_total counter",
            f"naviertwin_api_request_duration_seconds_total {snapshot['duration_seconds']}",
        ]
        for status_class, count in sorted(snapshot["status"].items()):
            lines.append(
                f'naviertwin_api_responses_total{{status_class="{status_class}"}} {count}'
            )
        return "\n".join(lines) + "\n"


def _extract_api_key(headers: Any) -> str:
    direct = str(headers.get("x-api-key", "")).strip()
    if direct:
        return direct
    authorization = str(headers.get("authorization", ""))
    scheme, _, value = authorization.partition(" ")
    return value.strip() if scheme.lower() == "bearer" else ""


def api_key_sha256(api_key: str) -> str:
    """Return deployment-safe SHA-256 representation for one API key."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _valid_api_key(
    candidate: str,
    configured: tuple[str, ...],
    configured_hashes: tuple[str, ...],
) -> bool:
    if not candidate:
        return False
    plain_match = any(
        secrets.compare_digest(candidate, expected) for expected in configured
    )
    digest = api_key_sha256(candidate)
    hash_match = any(
        secrets.compare_digest(digest, expected) for expected in configured_hashes
    )
    return plain_match or hash_match


def install_operations(app: Any, settings: APISettings) -> APIMetrics:
    """Install middleware and return app-local metrics storage."""
    from fastapi.responses import JSONResponse

    metrics = APIMetrics()
    limiter: SlidingWindowRateLimiter | SQLiteWindowRateLimiter
    if settings.rate_limit_store_path:
        limiter = SQLiteWindowRateLimiter(
            settings.rate_limit_store_path,
            settings.rate_limit_requests,
            settings.rate_limit_window_seconds,
        )
    else:
        limiter = SlidingWindowRateLimiter(
            settings.rate_limit_requests, settings.rate_limit_window_seconds
        )
    public_paths = {
        "/health",
        "/ready",
        "/api/v1/health",
        "/api/v1/ready",
        "/docs",
        "/openapi.json",
        "/redoc",
    }

    @app.middleware("http")
    async def operations_middleware(request: Any, call_next: Any) -> Any:
        started = time.perf_counter()
        metrics.begin()
        status = 500
        route = "unmatched"
        request_id = str(request.headers.get("x-request-id", "")).strip()
        if (
            not request_id
            or len(request_id) > 128
            or not all(character.isalnum() or character in "-_.:" for character in request_id)
        ):
            request_id = uuid.uuid4().hex
        try:
            api_key = _extract_api_key(request.headers)
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    too_large = int(content_length) > settings.max_request_bytes
                except ValueError:
                    too_large = True
                if too_large:
                    status = 413
                    response = JSONResponse(
                        {"detail": "request body too large"}, status_code=status
                    )
                    return response
            authentication_enabled = bool(settings.api_keys or settings.api_key_hashes)
            if authentication_enabled and request.url.path not in public_paths:
                if not _valid_api_key(
                    api_key, settings.api_keys, settings.api_key_hashes
                ):
                    status = 401
                    response = JSONResponse(
                        {"detail": "valid API key required"}, status_code=status
                    )
                    response.headers["WWW-Authenticate"] = "Bearer"
                    return response
            client_host = request.client.host if request.client else "unknown"
            identity = api_key_sha256(api_key if api_key else client_host)
            allowed, retry_after = limiter.allow(identity)
            if not allowed:
                status = 429
                response = JSONResponse({"detail": "rate limit exceeded"}, status_code=status)
                response.headers["Retry-After"] = str(max(1, int(retry_after + 0.999)))
                return response
            response = await call_next(request)
            status = int(response.status_code)
            route_object = request.scope.get("route")
            route = str(getattr(route_object, "path", "unmatched"))
            return response
        finally:
            duration = time.perf_counter() - started
            metrics.finish(request.method, route, status, duration)
            if "response" in locals():
                response.headers["X-Request-ID"] = request_id
                response.headers["X-API-Version"] = "1"
                response.headers["Server-Timing"] = f"app;dur={duration * 1000.0:.3f}"
                response.headers["X-Content-Type-Options"] = "nosniff"
                response.headers["X-Frame-Options"] = "DENY"
                response.headers["Referrer-Policy"] = "no-referrer"
                response.headers["Permissions-Policy"] = (
                    "camera=(), microphone=(), geolocation=()"
                )
                if request.url.scheme == "https":
                    response.headers["Strict-Transport-Security"] = (
                        "max-age=31536000; includeSubDomains"
                    )
                if route.startswith("/") and not route.startswith("/api/v1"):
                    response.headers["Link"] = (
                        f'</api/v1{route}>; rel="successor-version"'
                    )

    return metrics


__all__ = [
    "APIMetrics",
    "APISettings",
    "SQLiteWindowRateLimiter",
    "SlidingWindowRateLimiter",
    "api_key_sha256",
    "install_operations",
]
