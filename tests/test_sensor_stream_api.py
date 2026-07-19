"""Timestamp-aware sensor stream API tests."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="FastAPI is required for REST API tests")


def _routes():
    from naviertwin.api import create_app

    return {
        route.path: route.endpoint
        for route in create_app().routes
        if hasattr(route, "path") and hasattr(route, "endpoint")
    }


def test_sensor_alignment_assimilates_complete_observation_once() -> None:
    from naviertwin.api import (
        TwinStreamAlignReq,
        TwinStreamInitReq,
        TwinStreamSensorReq,
    )

    routes = _routes()
    routes["/twin/stream/init"](
        TwinStreamInitReq(
            session_id="sensor-case",
            state_dim=2,
            n_ensemble=20,
            initial_mean=[0.0, 0.0],
            initial_std=0.1,
            seed=3,
            assimilation_interval=0.5,
        )
    )
    for sensor_id, value in (("pressure", 1.0), ("velocity", -0.5)):
        payload = routes["/twin/stream/sensor"](
            TwinStreamSensorReq(
                session_id="sensor-case",
                sensor_id=sensor_id,
                timestamp=10.0,
                values=[value],
            )
        )
        assert payload["accepted"]

    request = TwinStreamAlignReq(
        session_id="sensor-case",
        sensor_ids=["pressure", "velocity"],
        timestamp=10.0,
    )
    first = routes["/twin/stream/align"](request)
    duplicate = routes["/twin/stream/align"](request)
    assert first["alignment"]["complete"]
    assert first["assimilated"]
    assert first["observation_count"] == 1
    assert not duplicate["assimilated"]
    assert duplicate["observation_count"] == 1


def test_incomplete_alignment_does_not_update_twin() -> None:
    from naviertwin.api import TwinStreamAlignReq, TwinStreamInitReq

    routes = _routes()
    routes["/twin/stream/init"](
        TwinStreamInitReq(session_id="missing-sensor", state_dim=1, n_ensemble=10)
    )
    payload = routes["/twin/stream/align"](
        TwinStreamAlignReq(
            session_id="missing-sensor",
            sensor_ids=["pressure"],
            timestamp=1.0,
        )
    )
    assert not payload["alignment"]["complete"]
    assert payload["alignment"]["missing_sensor_ids"] == ["pressure"]
    assert not payload["assimilated"]
    assert payload["observation_count"] == 0


def test_aligned_dimension_mismatch_is_rejected() -> None:
    from fastapi import HTTPException

    from naviertwin.api import (
        TwinStreamAlignReq,
        TwinStreamInitReq,
        TwinStreamSensorReq,
    )

    routes = _routes()
    routes["/twin/stream/init"](
        TwinStreamInitReq(session_id="bad-dim", state_dim=2, n_ensemble=10)
    )
    routes["/twin/stream/sensor"](
        TwinStreamSensorReq(
            session_id="bad-dim", sensor_id="p", timestamp=1.0, values=[1.0]
        )
    )
    with pytest.raises(HTTPException, match="observation size"):
        routes["/twin/stream/align"](
            TwinStreamAlignReq(
                session_id="bad-dim", sensor_ids=["p"], timestamp=1.0
            )
        )


def test_sensor_session_persists_recovers_reports_metrics_and_closes(tmp_path) -> None:
    from fastapi import HTTPException

    from naviertwin.api import (
        TwinStreamCloseReq,
        TwinStreamInitReq,
        TwinStreamSensorReq,
    )

    routes = _routes()
    store_path = tmp_path / "sensor-session.sqlite3"
    init = TwinStreamInitReq(
        session_id="durable",
        state_dim=1,
        n_ensemble=10,
        sensor_store_path=str(store_path),
    )
    routes["/twin/stream/init"](init)
    routes["/twin/stream/sensor"](
        TwinStreamSensorReq(
            session_id="durable",
            sensor_id="p",
            timestamp=1.0,
            values=[2.0],
            sequence=1,
        )
    )
    first_metrics = routes["/twin/stream/metrics"](session_id="durable")
    assert first_metrics["metrics"]["accepted_samples"] == 1
    assert first_metrics["buffered_sensor_ids"] == ["p"]

    replaced = routes["/twin/stream/init"](init)
    assert replaced["replaced"]
    recovered_metrics = routes["/twin/stream/metrics"](session_id="durable")
    assert recovered_metrics["metrics"]["recovered_samples"] == 1
    closed = routes["/twin/stream/close"](
        TwinStreamCloseReq(session_id="durable")
    )
    assert closed["event"] == "close"
    with pytest.raises(HTTPException) as exc:
        routes["/twin/stream/metrics"](session_id="durable")
    assert exc.value.status_code == 404
