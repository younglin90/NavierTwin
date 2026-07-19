"""API authentication, rate limiting, versioning, and metrics tests."""

from __future__ import annotations

import inspect
import threading

from fastapi.testclient import TestClient

from naviertwin.api import operations as operations_module
from naviertwin.api.operations import (
    APISettings,
    SlidingWindowRateLimiter,
    SQLiteWindowRateLimiter,
    _valid_api_key,
    api_key_sha256,
)
from naviertwin.api.server import create_app


def test_sliding_window_limiter_expires_old_requests() -> None:
    now = [10.0]
    limiter = SlidingWindowRateLimiter(2, 5.0, clock=lambda: now[0])
    assert limiter.allow("client")[0]
    assert limiter.allow("client")[0]
    allowed, retry_after = limiter.allow("client")
    assert not allowed
    assert retry_after == 5.0
    now[0] = 15.1
    assert limiter.allow("client")[0]


def test_api_key_version_headers_and_metrics() -> None:
    client = TestClient(create_app(APISettings(api_keys=("secret",))))

    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json()["api_version"] == "1"
    assert health.headers["x-api-version"] == "1"
    assert health.headers["x-request-id"]
    assert health.headers["x-content-type-options"] == "nosniff"
    assert health.headers["x-frame-options"] == "DENY"

    denied = client.post("/analytic/couette", json={})
    assert denied.status_code == 401
    assert denied.headers["www-authenticate"] == "Bearer"

    accepted = client.post(
        "/analytic/couette", json={"n_points": 4}, headers={"X-API-Key": "secret"}
    )
    assert accepted.status_code == 200
    assert len(accepted.json()["velocity"]) == 4

    metrics = client.get("/api/v1/metrics", headers={"Authorization": "Bearer secret"})
    assert metrics.status_code == 200
    payload = metrics.json()
    assert payload["requests"] >= 3
    assert payload["status"]["2xx"] >= 2

    prometheus = client.get(
        "/api/v1/metrics/prometheus", headers={"X-API-Key": "secret"}
    )
    assert prometheus.status_code == 200
    assert "naviertwin_api_requests_total" in prometheus.text


def test_rate_limit_returns_retry_after() -> None:
    settings = APISettings(
        api_keys=("key",), rate_limit_requests=1, rate_limit_window_seconds=60.0
    )
    client = TestClient(create_app(settings))
    headers = {"X-API-Key": "key"}

    assert client.post("/analytic/couette", json={}, headers=headers).status_code == 200
    limited = client.post("/analytic/couette", json={}, headers=headers)
    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) >= 1


def test_metrics_can_be_disabled() -> None:
    client = TestClient(create_app(APISettings(expose_metrics=False)))
    assert client.get("/api/v1/metrics").status_code == 404


def test_hashed_api_key_and_openapi_security() -> None:
    settings = APISettings(api_key_hashes=(api_key_sha256("secret"),))
    client = TestClient(create_app(settings))

    assert client.get("/ready").status_code == 200
    assert client.get("/api/v1/ready").status_code == 200
    denied = client.post("/api/v1/analytic/couette", json={})
    assert denied.status_code == 401
    accepted = client.post(
        "/api/v1/analytic/couette", json={}, headers={"X-API-Key": "secret"}
    )
    assert accepted.status_code == 200
    legacy = client.post(
        "/analytic/couette", json={}, headers={"X-API-Key": "secret"}
    )
    assert legacy.headers["link"] == (
        '</api/v1/analytic/couette>; rel="successor-version"'
    )

    schema = client.get("/openapi.json").json()
    schemes = schema["components"]["securitySchemes"]
    assert set(schemes) >= {"ApiKeyAuth", "BearerAuth"}
    assert schema["paths"]["/ready"]["get"]["security"] == []
    assert schema["paths"]["/api/v1/analytic/couette"]["post"]["security"]


def test_every_legacy_business_route_has_v1_alias() -> None:
    app = create_app()
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    excluded = {"/docs", "/docs/oauth2-redirect", "/openapi.json", "/redoc"}
    legacy = {path for path in paths if not path.startswith("/api/") and path not in excluded}
    assert {f"/api/v1{path}" for path in legacy}.issubset(paths)


def test_sqlite_rate_limit_is_shared_between_workers(tmp_path) -> None:
    now = [120.0]
    path = tmp_path / "limits.sqlite3"
    first = SQLiteWindowRateLimiter(str(path), 2, 60.0, clock=lambda: now[0])
    second = SQLiteWindowRateLimiter(str(path), 2, 60.0, clock=lambda: now[0])

    assert first.allow("identity")[0]
    assert second.allow("identity")[0]
    allowed, retry_after = first.allow("identity")
    assert not allowed
    assert retry_after == 60.0
    now[0] = 180.0
    assert second.allow("identity")[0]


def test_request_size_limit_and_https_hsts() -> None:
    client = TestClient(
        create_app(APISettings(max_request_bytes=8)), base_url="https://testserver"
    )
    too_large = client.post("/analytic/couette", content=b"012345678")
    assert too_large.status_code == 413
    assert "max-age=31536000" in too_large.headers["strict-transport-security"]


def test_settings_from_environment_supports_hashes_and_shared_store(
    monkeypatch, tmp_path
) -> None:
    store = tmp_path / "rate.sqlite3"
    monkeypatch.setenv("NAVIERTWIN_API_KEY_HASHES", api_key_sha256("key"))
    monkeypatch.setenv("NAVIERTWIN_RATE_LIMIT_REQUESTS", "25")
    monkeypatch.setenv("NAVIERTWIN_RATE_LIMIT_WINDOW_SECONDS", "30")
    monkeypatch.setenv("NAVIERTWIN_RATE_LIMIT_STORE", str(store))
    monkeypatch.setenv("NAVIERTWIN_MAX_REQUEST_BYTES", "4096")

    settings = APISettings.from_env()
    assert settings.api_key_hashes == (api_key_sha256("key"),)
    assert settings.rate_limit_requests == 25
    assert settings.rate_limit_window_seconds == 30.0
    assert settings.rate_limit_store_path == str(store)
    assert settings.max_request_bytes == 4096


def test_invalid_hashed_key_configuration_is_rejected(monkeypatch) -> None:
    import pytest

    monkeypatch.setenv("NAVIERTWIN_API_KEY_HASHES", "not-a-sha256")
    with pytest.raises(ValueError, match="SHA-256"):
        APISettings.from_env()


def test_api_key_comparison_uses_constant_time_digest() -> None:
    """Pin the timing-attack mitigation: key comparison must go through
    ``secrets.compare_digest`` (equivalent to ``hmac.compare_digest``), never a
    plain ``==`` on attacker-controlled input. A regression here (e.g. someone
    "simplifying" the comparison back to ``==``) must fail this test."""
    source = inspect.getsource(operations_module._valid_api_key)
    assert "secrets.compare_digest" in source
    # No direct equality comparison of the candidate/digest against configured
    # secrets anywhere in the function body.
    assert "candidate ==" not in source
    assert "== candidate" not in source
    assert "digest ==" not in source
    assert "== digest" not in source


def test_valid_api_key_checks_both_plain_and_hashed_forms() -> None:
    assert _valid_api_key("plain-secret", ("plain-secret",), ()) is True
    assert _valid_api_key("wrong", ("plain-secret",), ()) is False
    assert _valid_api_key("", ("plain-secret",), ()) is False
    hashed = api_key_sha256("hashed-secret")
    assert _valid_api_key("hashed-secret", (), (hashed,)) is True
    assert _valid_api_key("hashed-secret", (), ("0" * 64,)) is False


def test_sliding_window_limiter_is_thread_safe_under_concurrency() -> None:
    """Concurrent requests must never let more than `limit` through — a naive
    read-modify-write without the lock would leak extra admissions."""
    limit = 25
    limiter = SlidingWindowRateLimiter(limit, 60.0)
    allowed_count = 0
    count_lock = threading.Lock()

    def worker() -> None:
        nonlocal allowed_count
        ok, _ = limiter.allow("shared-client")
        if ok:
            with count_lock:
                allowed_count += 1

    threads = [threading.Thread(target=worker) for _ in range(limit * 4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert allowed_count == limit


def _sqlite_limiter_worker_process(
    db_path: str, limit: int, calls: int, queue: "object"
) -> None:
    """Module-level (picklable) worker used by
    ``test_sqlite_limiter_is_safe_across_real_processes`` to simulate a real
    ``--workers N`` uvicorn worker process."""
    worker_limiter = SQLiteWindowRateLimiter(db_path, limit, 60.0)
    admitted = 0
    for _ in range(calls):
        ok, _retry = worker_limiter.allow("shared-client")
        if ok:
            admitted += 1
    queue.put(admitted)


def test_sqlite_limiter_is_safe_across_real_processes(tmp_path) -> None:
    """Simulate real ``--workers N`` behaviour: separate OS processes sharing
    one SQLite-backed limiter file must still enforce the exact limit."""
    import multiprocessing as mp

    limit = 20
    path = str(tmp_path / "mp_limits.sqlite3")
    # pre-create schema before forking workers
    SQLiteWindowRateLimiter(path, limit, 60.0)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    n_procs = 8
    calls_per_proc = 10
    procs = [
        ctx.Process(
            target=_sqlite_limiter_worker_process,
            args=(path, limit, calls_per_proc, queue),
        )
        for _ in range(n_procs)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=30)

    total_admitted = sum(queue.get(timeout=5) for _ in procs)
    assert total_admitted == limit


def test_content_security_policy_present_and_scoped() -> None:
    client = TestClient(create_app(APISettings()))

    api_response = client.get("/health")
    assert (
        api_response.headers["content-security-policy"]
        == "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"
    )

    docs_response = client.get("/docs")
    docs_csp = docs_response.headers["content-security-policy"]
    assert "cdn.jsdelivr.net" in docs_csp
    assert "frame-ancestors 'none'" in docs_csp

    redoc_response = client.get("/redoc")
    assert "cdn.jsdelivr.net" in redoc_response.headers["content-security-policy"]


def test_prometheus_metrics_follow_exposition_format() -> None:
    client = TestClient(create_app(APISettings()))
    client.get("/health")

    response = client.get("/api/v1/metrics/prometheus")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    lines = body.splitlines()
    assert any(line.startswith("# HELP naviertwin_api_requests_total") for line in lines)
    assert any(line.startswith("# TYPE naviertwin_api_requests_total counter") for line in lines)
    assert any(
        line.startswith("naviertwin_api_requests_total ") and not line.startswith("#")
        for line in lines
    )


def test_tls_server_actually_serves_https(tmp_path) -> None:
    """End-to-end proof that --ssl-certfile/--ssl-keyfile produce a working
    HTTPS listener, not just CLI flags that get silently ignored."""
    import shutil
    import socket
    import ssl as ssl_module
    import subprocess
    import sys
    import time
    import urllib.request

    if shutil.which("openssl") is None:
        import pytest

        pytest.skip("openssl not available to generate a test certificate")

    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(key_path), "-out", str(cert_path),
            "-days", "1", "-subj", "/CN=localhost",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "naviertwin.main", "server",
            "--host", "127.0.0.1", "--port", str(port),
            "--ssl-certfile", str(cert_path), "--ssl-keyfile", str(key_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        ctx = ssl_module.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl_module.CERT_NONE

        deadline = time.time() + 20.0
        last_error = None
        body = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"https://127.0.0.1:{port}/health", timeout=1.0, context=ctx
                ) as resp:
                    body = resp.read()
                    break
            except Exception as exc:  # noqa: BLE001 — server may not be up yet
                last_error = exc
                time.sleep(0.3)

        assert body is not None, f"TLS server never came up: {last_error}"
        assert b'"status":"ok"' in body
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
