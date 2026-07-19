"""API authentication, rate limiting, versioning, and metrics tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from naviertwin.api.operations import (
    APISettings,
    SlidingWindowRateLimiter,
    SQLiteWindowRateLimiter,
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
