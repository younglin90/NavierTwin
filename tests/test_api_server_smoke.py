"""Customer-facing FastAPI endpoint smoke tests."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="FastAPI is required for REST API smoke tests")


def test_advertised_rest_endpoints_return_json() -> None:
    import numpy as np

    from naviertwin.api import CouetteReq, LBMReq, PODReq, create_app

    app = create_app()
    route_map = {
        route.path: route.endpoint
        for route in app.routes
        if hasattr(route, "path") and hasattr(route, "endpoint")
    }
    assert "/twin/build" in route_map
    assert "/twin/predict" in route_map
    assert "/twin/benchmark" in route_map

    assert route_map["/health"]() == {"status": "ok", "service": "naviertwin"}

    couette_payload = route_map["/analytic/couette"](
        CouetteReq(U_top=1.0, H=1.0, n_points=5)
    )
    assert len(couette_payload["coords"]) == 5
    assert couette_payload["velocity"][-1] == pytest.approx(1.0)

    snapshots = np.eye(4).tolist()
    reduce_payload = route_map["/reduce"](
        PODReq(snapshots=snapshots, n_modes=2, reducer_kind="pod")
    )
    assert reduce_payload["reducer_kind"] == "pod"
    assert reduce_payload["n_modes"] == 2
    assert len(reduce_payload["singular_values"]) >= 2

    lbm_payload = route_map["/simulate/lbm_cavity"](
        LBMReq(nx=6, ny=6, tau=0.8, u_top=0.05, n_steps=2, record_every=1)
    )
    assert lbm_payload["n_snapshots"] == 2
    assert lbm_payload["shape"] == [2, 6, 6, 3]
    assert abs(lbm_payload["ux_max"]) < 1.0


def _write_snapshot_series(tmp_path, *, steps: int = 8, width: int = 6) -> list:
    paths = []
    for step in range(steps):
        path = tmp_path / f"snapshot_{step:03d}.csv"
        rows = ["x,U"]
        for index in range(width):
            rows.append(f"{index},{step * 0.2 + index * 0.01}")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def test_twin_build_endpoint_creates_predictable_artifacts(tmp_path) -> None:
    from naviertwin.api import TwinBuildReq, TwinPredictReq, create_app

    paths = _write_snapshot_series(tmp_path)
    outdir = tmp_path / "twin"
    app = create_app()
    route_map = {
        route.path: route.endpoint
        for route in app.routes
        if hasattr(route, "path") and hasattr(route, "endpoint")
    }

    payload = route_map["/twin/build"](
        TwinBuildReq(
            csv_snapshots=",".join(str(path) for path in paths),
            field_column="U",
            outdir=str(outdir),
            n_modes=2,
            surrogate="rbf",
            validation_count=2,
        )
    )

    assert payload["status"] == "ok"
    assert payload["field"] == "U"
    assert payload["training"]["n_snapshots"] == len(paths)
    assert payload["training"]["validation_count"] == 2
    assert payload["training"]["parameter_contract"]["names"] == ["normalized_index"]
    assert (outdir / "engine.pkl").exists()
    assert (outdir / "manifest.json").exists()
    assert (outdir / "metrics.json").exists()

    predict_payload = route_map["/twin/predict"](
        TwinPredictReq(artifacts_dir=str(outdir), params=[0.25])
    )
    assert predict_payload["status"] == "ok"
    assert predict_payload["artifacts_dir"] == str(outdir)
    assert predict_payload["prediction_shape"] == [6]


def test_twin_build_endpoint_reports_invalid_source_contract(tmp_path) -> None:
    from fastapi import HTTPException

    from naviertwin.api import TwinBuildReq, create_app

    app = create_app()
    route_map = {
        route.path: route.endpoint
        for route in app.routes
        if hasattr(route, "path") and hasattr(route, "endpoint")
    }

    with pytest.raises(HTTPException) as exc:
        route_map["/twin/build"](
            TwinBuildReq(
                input_path=str(tmp_path / "case.vtu"),
                csv_snapshots=str(tmp_path / "snapshot.csv"),
                outdir=str(tmp_path / "twin"),
            )
        )

    assert exc.value.status_code == 400
    assert "exactly one of input_path or csv_snapshots" in str(exc.value.detail)


def test_twin_predict_endpoint_serves_saved_engine(tmp_path) -> None:
    import numpy as np

    from naviertwin.api import TwinPredictReq, create_app
    from naviertwin.core.digital_twin.twin_engine import TwinEngine

    snapshots = np.vstack(
        [
            np.linspace(0.0, 1.0, 8),
            np.linspace(1.0, 2.0, 8),
            np.linspace(2.0, 3.0, 8),
            np.linspace(3.0, 4.0, 8),
        ]
    )
    params = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
    engine = TwinEngine(reducer_type="pod", surrogate_type="rbf", n_modes=2)
    engine.fit(snapshots, params)
    engine_path = tmp_path / "engine.pkl"
    engine.save(engine_path)

    app = create_app()
    route_map = {
        route.path: route.endpoint
        for route in app.routes
        if hasattr(route, "path") and hasattr(route, "endpoint")
    }
    payload = route_map["/twin/predict"](
        TwinPredictReq(engine_path=str(engine_path), params=[0.5])
    )

    assert payload["status"] == "ok"
    assert payload["engine"] == str(engine_path)
    assert payload["input_shape"] == [1]
    assert payload["prediction_shape"] == [4]
    assert len(payload["preview"]) == 4
    assert len(payload["prediction"]) == 4


def test_twin_predict_endpoint_reports_bad_parameter_shape(tmp_path) -> None:
    import numpy as np
    from fastapi import HTTPException

    from naviertwin.api import TwinPredictReq, create_app
    from naviertwin.core.digital_twin.twin_engine import TwinEngine

    snapshots = np.vstack([np.linspace(0.0, 1.0, 6), np.linspace(1.0, 2.0, 6)])
    params = np.linspace(0.0, 1.0, 6).reshape(-1, 1)
    engine = TwinEngine(reducer_type="pod", surrogate_type="rbf", n_modes=1)
    engine.fit(snapshots, params)
    engine_path = tmp_path / "engine.pkl"
    engine.save(engine_path)

    app = create_app()
    route_map = {
        route.path: route.endpoint
        for route in app.routes
        if hasattr(route, "path") and hasattr(route, "endpoint")
    }

    with pytest.raises(HTTPException) as exc:
        route_map["/twin/predict"](
            TwinPredictReq(engine_path=str(engine_path), params=[[[0.5]]])
        )

    assert exc.value.status_code == 400
    assert "params must be 1D or 2D" in str(exc.value.detail)


def test_twin_predict_endpoint_reports_corrupt_engine(tmp_path) -> None:
    from fastapi import HTTPException

    from naviertwin.api import TwinPredictReq, create_app

    engine_path = tmp_path / "engine.pkl"
    engine_path.write_bytes(b"not a pickle")
    app = create_app()
    route_map = {
        route.path: route.endpoint
        for route in app.routes
        if hasattr(route, "path") and hasattr(route, "endpoint")
    }

    with pytest.raises(HTTPException) as exc:
        route_map["/twin/predict"](
            TwinPredictReq(engine_path=str(engine_path), params=[0.5])
        )

    assert exc.value.status_code == 400


def test_twin_benchmark_endpoint_reports_latency_and_acceptance(tmp_path) -> None:
    import numpy as np

    from naviertwin.api import TwinBenchmarkReq, create_app
    from naviertwin.core.digital_twin.twin_engine import TwinEngine

    snapshots = np.vstack(
        [
            np.linspace(0.0, 1.0, 8),
            np.linspace(1.0, 2.0, 8),
            np.linspace(2.0, 3.0, 8),
            np.linspace(3.0, 4.0, 8),
        ]
    )
    params = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
    engine = TwinEngine(reducer_type="pod", surrogate_type="rbf", n_modes=2)
    engine.fit(snapshots, params)
    engine_path = tmp_path / "engine.pkl"
    engine.save(engine_path)

    app = create_app()
    route_map = {
        route.path: route.endpoint
        for route in app.routes
        if hasattr(route, "path") and hasattr(route, "endpoint")
    }
    payload = route_map["/twin/benchmark"](
        TwinBenchmarkReq(
            engine_path=str(engine_path),
            params=[0.5],
            warmup=0,
            repeat=3,
            max_p95_ms=100000.0,
            min_throughput_hz=0.0001,
        )
    )

    assert payload["status"] == "ok"
    assert payload["engine"] == str(engine_path)
    assert payload["repeat"] == 3
    assert len(payload["samples_ms"]) == 3
    assert payload["latency_ms"]["p95"] >= payload["latency_ms"]["min"]
    assert payload["throughput_hz"] is not None
    assert payload["acceptance"]["configured"] is True
    assert payload["acceptance"]["passed"] is True


def test_twin_benchmark_endpoint_reports_slo_failure(tmp_path) -> None:
    import numpy as np

    from naviertwin.api import TwinBenchmarkReq, create_app
    from naviertwin.core.digital_twin.twin_engine import TwinEngine

    snapshots = np.vstack([np.linspace(0.0, 1.0, 6), np.linspace(1.0, 2.0, 6)])
    params = np.linspace(0.0, 1.0, 6).reshape(-1, 1)
    engine = TwinEngine(reducer_type="pod", surrogate_type="rbf", n_modes=1)
    engine.fit(snapshots, params)
    engine_path = tmp_path / "engine.pkl"
    engine.save(engine_path)

    app = create_app()
    route_map = {
        route.path: route.endpoint
        for route in app.routes
        if hasattr(route, "path") and hasattr(route, "endpoint")
    }
    payload = route_map["/twin/benchmark"](
        TwinBenchmarkReq(
            engine_path=str(engine_path),
            params=[0.5],
            warmup=0,
            repeat=1,
            max_mean_ms=0.0,
        )
    )

    assert payload["status"] == "failed"
    assert payload["acceptance"]["configured"] is True
    assert payload["acceptance"]["passed"] is False
    assert payload["acceptance"]["checks"][0]["metric"] == "latency_ms.mean"
