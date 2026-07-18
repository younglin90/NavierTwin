"""Customer-facing FastAPI endpoint smoke tests."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="FastAPI is required for REST API smoke tests")


def test_advertised_rest_endpoints_return_json() -> None:
    import numpy as np

    from naviertwin.api import CouetteReq, LBMReq, PODReq, PreflightReq, create_app

    app = create_app()
    route_map = {
        route.path: route.endpoint
        for route in app.routes
        if hasattr(route, "path") and hasattr(route, "endpoint")
    }
    assert "/preflight" in route_map
    assert "/twin/build" in route_map
    assert "/twin/predict" in route_map
    assert "/twin/benchmark" in route_map
    assert "/twin/package" in route_map
    assert "/twin/package/inspect" in route_map
    assert "/twin/package/verify" in route_map
    assert "/twin/package/accept" in route_map
    assert "/twin/stream/init" in route_map
    assert "/twin/stream/step" in route_map
    assert "/twin/stream/observe" in route_map
    assert "/twin/stream/observe-batch" in route_map
    assert "/twin/stream/observe-line" in route_map
    assert "/twin/stream/state" in route_map

    assert route_map["/health"]() == {"status": "ok", "service": "naviertwin"}
    doctor_payload = route_map["/doctor"]()
    assert doctor_payload["status"] in {"ok", "warn", "error"}
    assert {"timestamp", "version", "environment", "checks", "warnings", "errors"} <= set(
        doctor_payload
    )
    assert any(check["name"] == "python_version" for check in doctor_payload["checks"])
    optional_doctor_payload = route_map["/doctor"](include_optional=True)
    assert any(
        check["name"] == "optional_dependencies"
        for check in optional_doctor_payload["checks"]
    )

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

    preflight_payload = route_map["/preflight"](
        PreflightReq(path="tests/fixtures/tiny_square.su2")
    )
    assert preflight_payload["status"] == "ok"
    assert preflight_payload["readiness_score"] == 100
    assert preflight_payload["summary"]["n_points"] == 4


def test_preflight_endpoint_reports_missing_input(tmp_path) -> None:
    from naviertwin.api import PreflightReq, create_app

    app = create_app()
    route_map = {
        route.path: route.endpoint
        for route in app.routes
        if hasattr(route, "path") and hasattr(route, "endpoint")
    }

    payload = route_map["/preflight"](PreflightReq(path=str(tmp_path / "missing.su2")))

    assert payload["status"] == "error"
    assert payload["readiness_score"] == 0
    assert payload["errors"] == ["path_exists"]
    assert payload["checks"][0]["code"] == "INPUT_PATH_MISSING"


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


def _build_packaged_twin(tmp_path):
    from naviertwin.api import TwinBuildReq, TwinPackageCreateReq, create_app

    paths = _write_snapshot_series(tmp_path)
    outdir = tmp_path / "twin"
    package_path = tmp_path / "twin.zip"
    app = create_app()
    route_map = {
        route.path: route.endpoint
        for route in app.routes
        if hasattr(route, "path") and hasattr(route, "endpoint")
    }
    build_payload = route_map["/twin/build"](
        TwinBuildReq(
            csv_snapshots=",".join(str(path) for path in paths),
            field_column="U",
            outdir=str(outdir),
            n_modes=2,
            surrogate="rbf",
            validation_count=2,
        )
    )
    assert build_payload["status"] == "ok"
    package_payload = route_map["/twin/package"](
        TwinPackageCreateReq(
            artifacts_dir=str(outdir),
            output=str(package_path),
            no_latency_slo=True,
        )
    )
    assert package_payload["status"] == "ok"
    assert package_payload["output"] == str(package_path)
    assert package_payload["latency_slo"] is None
    return app, route_map, package_path


def test_twin_package_endpoint_creates_delivery_zip(tmp_path) -> None:
    import zipfile

    from naviertwin.api import TwinBuildReq, TwinPackageCreateReq, create_app

    paths = _write_snapshot_series(tmp_path)
    outdir = tmp_path / "twin"
    package_path = tmp_path / "delivery.zip"
    app = create_app()
    route_map = {
        route.path: route.endpoint
        for route in app.routes
        if hasattr(route, "path") and hasattr(route, "endpoint")
    }
    build_payload = route_map["/twin/build"](
        TwinBuildReq(
            csv_snapshots=",".join(str(path) for path in paths),
            field_column="U",
            outdir=str(outdir),
            n_modes=2,
            surrogate="rbf",
            validation_count=2,
        )
    )
    assert build_payload["status"] == "ok"

    payload = route_map["/twin/package"](
        TwinPackageCreateReq(
            artifacts_dir=str(outdir),
            output=str(package_path),
            max_p95_ms=123.0,
            min_throughput_hz=4.5,
        )
    )

    assert payload["status"] == "ok"
    assert payload["output"] == str(package_path)
    assert payload["latency_slo"]["thresholds"]["max_p95_ms"] == 123.0
    assert payload["latency_slo"]["thresholds"]["min_throughput_hz"] == 4.5
    assert {"README.txt", "delivery.json", "sample_params.csv"} <= set(
        payload["generated_entries"]
    )
    assert {"engine.pkl", "manifest.json"} <= set(payload["files"])
    assert package_path.exists()
    with zipfile.ZipFile(package_path) as archive:
        assert {"MANIFEST.json", "delivery.json", "README.txt", "engine.pkl", "manifest.json"} <= set(
            archive.namelist()
        )


def test_twin_package_endpoint_reports_invalid_artifacts_dir(tmp_path) -> None:
    from fastapi import HTTPException

    from naviertwin.api import TwinPackageCreateReq, create_app

    app = create_app()
    route_map = {
        route.path: route.endpoint
        for route in app.routes
        if hasattr(route, "path") and hasattr(route, "endpoint")
    }

    with pytest.raises(HTTPException) as exc:
        route_map["/twin/package"](
            TwinPackageCreateReq(
                artifacts_dir=str(tmp_path / "missing"),
                output=str(tmp_path / "missing.zip"),
            )
        )

    assert exc.value.status_code == 400
    assert "artifacts-dir not found" in str(exc.value.detail)


def test_twin_package_inspect_endpoint_reports_delivery_metadata(tmp_path) -> None:
    from naviertwin.api import TwinPackageInspectReq

    _app, route_map, package_path = _build_packaged_twin(tmp_path)
    payload = route_map["/twin/package/inspect"](
        TwinPackageInspectReq(package=str(package_path))
    )

    assert payload["status"] == "ok"
    assert payload["delivery_metadata_present"] is True
    assert payload["format"] == "NavierTwin delivery package"
    assert payload["schema"] == "naviertwin-delivery-v1"
    assert payload["verification"]["status"] == "ok"
    assert payload["manifest_entry_count"] >= 7
    assert payload["parameter_contract"]["names"] == ["normalized_index"]
    assert {"README.txt", "delivery.json", "sample_params.csv"} <= set(
        payload["generated_entries"]
    )


def test_twin_package_verify_endpoint_checks_and_extracts_zip(tmp_path) -> None:
    from naviertwin.api import TwinPackageVerifyReq

    _app, route_map, package_path = _build_packaged_twin(tmp_path)
    extract_to = tmp_path / "verified"
    payload = route_map["/twin/package/verify"](
        TwinPackageVerifyReq(package=str(package_path), extract_to=str(extract_to))
    )

    assert payload["status"] == "ok"
    assert payload["extracted_to"] == str(extract_to)
    assert payload["manifest_entry_count"] >= 7
    assert all(check["passed"] for check in payload["checks"])
    assert (extract_to / "engine.pkl").exists()
    assert (extract_to / "manifest.json").exists()


def test_twin_package_cli_text_mode_uses_payload_output(tmp_path, capsys) -> None:
    from naviertwin.api import TwinBuildReq, create_app
    from naviertwin.main import _run_package_twin

    paths = _write_snapshot_series(tmp_path)
    outdir = tmp_path / "twin"
    package_path = tmp_path / "cli-package.zip"
    app = create_app()
    route_map = {
        route.path: route.endpoint
        for route in app.routes
        if hasattr(route, "path") and hasattr(route, "endpoint")
    }
    route_map["/twin/build"](
        TwinBuildReq(
            csv_snapshots=",".join(str(path) for path in paths),
            field_column="U",
            outdir=str(outdir),
            n_modes=2,
            surrogate="rbf",
            validation_count=2,
        )
    )

    code = _run_package_twin(
        artifacts_dir=str(outdir),
        output=str(package_path),
        include_validation=None,
        no_latency_slo=True,
        as_json=False,
    )

    output = capsys.readouterr().out
    assert code == 0
    assert f"output={package_path}" in output


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


def test_twin_package_accept_endpoint_runs_delivery_gate(tmp_path) -> None:
    from naviertwin.api import TwinPackageAcceptReq

    _app, route_map, package_path = _build_packaged_twin(tmp_path)
    payload = route_map["/twin/package/accept"](
        TwinPackageAcceptReq(
            package=str(package_path),
            extract_to=str(tmp_path / "accepted"),
            warmup=0,
            repeat=2,
            max_p95_ms=100000.0,
            min_throughput_hz=0.0001,
        )
    )

    assert payload["status"] == "ok"
    assert payload["temporary_extraction"] is False
    assert payload["verification"]["status"] == "ok"
    assert payload["inspection"]["status"] == "ok"
    assert payload["acceptance"]["package"] is True
    assert payload["acceptance"]["prediction"] is True
    assert payload["acceptance"]["benchmark"] is True
    assert payload["acceptance"]["passed"] is True
    assert payload["prediction"]["prediction_shape"] == [6, 1]
    assert payload["benchmark"]["repeat"] == 2
    assert len(payload["benchmark"]["samples_ms"]) == 2
    assert (tmp_path / "accepted" / "engine.pkl").exists()


def test_twin_package_accept_endpoint_reports_slo_failure(tmp_path) -> None:
    from naviertwin.api import TwinPackageAcceptReq

    _app, route_map, package_path = _build_packaged_twin(tmp_path)
    payload = route_map["/twin/package/accept"](
        TwinPackageAcceptReq(
            package=str(package_path),
            extract_to=str(tmp_path / "accepted-fail"),
            warmup=0,
            repeat=1,
            max_mean_ms=0.0,
        )
    )

    assert payload["status"] == "failed"
    assert payload["acceptance"]["package"] is True
    assert payload["acceptance"]["prediction"] is True
    assert payload["acceptance"]["benchmark"] is False
    assert payload["benchmark"]["acceptance"]["checks"][0]["metric"] == "latency_ms.mean"


def test_twin_package_accept_endpoint_reports_missing_package(tmp_path) -> None:
    from fastapi import HTTPException

    from naviertwin.api import TwinPackageAcceptReq, create_app

    app = create_app()
    route_map = {
        route.path: route.endpoint
        for route in app.routes
        if hasattr(route, "path") and hasattr(route, "endpoint")
    }

    with pytest.raises(HTTPException) as exc:
        route_map["/twin/package/accept"](
            TwinPackageAcceptReq(package=str(tmp_path / "missing.zip"))
        )

    assert exc.value.status_code == 400
    assert "package not found" in str(exc.value.detail)


def test_twin_stream_endpoints_run_state_assimilation() -> None:
    import numpy as np

    from naviertwin.api import (
        TwinStreamInitReq,
        TwinStreamObserveBatchReq,
        TwinStreamObserveLineReq,
        TwinStreamObserveReq,
        TwinStreamStepReq,
        create_app,
    )

    app = create_app()
    route_map = {
        route.path: route.endpoint
        for route in app.routes
        if hasattr(route, "path") and hasattr(route, "endpoint")
    }
    assert "/twin/stream/init" in route_map
    assert "/twin/stream/step" in route_map
    assert "/twin/stream/observe" in route_map
    assert "/twin/stream/observe-batch" in route_map
    assert "/twin/stream/observe-line" in route_map
    assert "/twin/stream/state" in route_map

    init_payload = route_map["/twin/stream/init"](
        TwinStreamInitReq(
            session_id="case-001",
            state_dim=2,
            n_ensemble=30,
            transition=[[0.95, 0.0], [0.0, 0.9]],
            observation_matrix=[[1.0, 0.0], [0.0, 1.0]],
            observation_covariance=(0.01 * np.eye(2)).tolist(),
            initial_mean=[1.0, -0.5],
            initial_std=0.02,
            history_size=6,
            seed=7,
        )
    )

    assert init_payload["status"] == "ok"
    assert init_payload["event"] == "init"
    assert init_payload["session_id"] == "case-001"
    assert init_payload["state_dim"] == 2
    assert init_payload["n_ensemble"] == 30
    assert init_payload["history_length"] == 1
    assert len(init_payload["estimate"]) == 2
    assert len(init_payload["uncertainty"]) == 2

    step_payload = route_map["/twin/stream/step"](
        TwinStreamStepReq(session_id="case-001", steps=2)
    )
    assert step_payload["event"] == "step"
    assert step_payload["step_count"] == 2
    assert step_payload["history_length"] == 3

    observe_payload = route_map["/twin/stream/observe"](
        TwinStreamObserveReq(
            session_id="case-001",
            observation=[0.25, -0.1],
            advance=True,
        )
    )
    assert observe_payload["event"] == "observe"
    assert observe_payload["step_count"] == 3
    assert observe_payload["observation_count"] == 1
    assert observe_payload["history_length"] <= 6
    assert len(observe_payload["history_tail"]) <= 5
    assert max(observe_payload["uncertainty"]) < 0.2

    state_payload = route_map["/twin/stream/state"](session_id="case-001")
    assert state_payload["event"] == "state"
    assert state_payload["estimate"] == observe_payload["estimate"]
    assert state_payload["uncertainty"] == observe_payload["uncertainty"]

    batch_payload = route_map["/twin/stream/observe-batch"](
        TwinStreamObserveBatchReq(
            session_id="case-001",
            observations=[[0.22, -0.08], [0.2, -0.05]],
            advance=True,
        )
    )
    assert batch_payload["event"] == "observe-batch"
    assert batch_payload["processed_observations"] == 2
    assert batch_payload["observation_count"] == 3
    assert batch_payload["step_count"] == 5

    line_payload = route_map["/twin/stream/observe-line"](
        TwinStreamObserveLineReq(
            session_id="case-001",
            line="123.0,0.18,-0.04",
            value_columns=[1, 2],
            advance=False,
        )
    )
    assert line_payload["event"] == "observe-line"
    assert line_payload["parsed_observation"] == [0.18, -0.04]
    assert line_payload["observation_count"] == 4
    assert line_payload["step_count"] == 5


def test_twin_stream_endpoints_report_invalid_requests() -> None:
    from fastapi import HTTPException

    from naviertwin.api import (
        TwinStreamInitReq,
        TwinStreamObserveBatchReq,
        TwinStreamObserveLineReq,
        TwinStreamStepReq,
        create_app,
    )

    app = create_app()
    route_map = {
        route.path: route.endpoint
        for route in app.routes
        if hasattr(route, "path") and hasattr(route, "endpoint")
    }

    with pytest.raises(HTTPException) as exc:
        route_map["/twin/stream/init"](
            TwinStreamInitReq(
                state_dim=2,
                n_ensemble=10,
                transition=[[1.0, 0.0, 0.0]],
            )
        )
    assert exc.value.status_code == 400
    assert "transition shape" in str(exc.value.detail)

    with pytest.raises(HTTPException) as exc:
        route_map["/twin/stream/step"](
            TwinStreamStepReq(session_id="missing", steps=1)
        )
    assert exc.value.status_code == 404
    assert "stream session not found" in str(exc.value.detail)

    route_map["/twin/stream/init"](
        TwinStreamInitReq(
            session_id="invalid-inputs",
            state_dim=2,
            n_ensemble=10,
            initial_std=0.01,
        )
    )

    with pytest.raises(HTTPException) as exc:
        route_map["/twin/stream/observe-batch"](
            TwinStreamObserveBatchReq(session_id="invalid-inputs", observations=[])
        )
    assert exc.value.status_code == 400
    assert "observations must not be empty" in str(exc.value.detail)

    with pytest.raises(HTTPException) as exc:
        route_map["/twin/stream/observe-line"](
            TwinStreamObserveLineReq(
                session_id="invalid-inputs",
                line="time,not-a-number",
                value_columns=[1],
            )
        )
    assert exc.value.status_code == 400
    assert "non-numeric observation token" in str(exc.value.detail)


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
    assert payload["prediction_returned"] is True
    assert payload["prediction_size"] == 4
    assert payload["prediction_bytes"] > 0
    assert payload["latency_ms"] >= 0.0
    assert payload["output_path"] is None
    assert len(payload["preview"]) == 4
    assert len(payload["prediction"]) == 4

    output_path = tmp_path / "prediction.csv"
    file_payload = route_map["/twin/predict"](
        TwinPredictReq(
            engine_path=str(engine_path),
            params=[[0.5], [0.75]],
            return_prediction=False,
            output_path=str(output_path),
            output_format="csv",
        )
    )

    assert file_payload["status"] == "ok"
    assert file_payload["prediction_shape"] == [4, 2]
    assert file_payload["prediction_returned"] is False
    assert "prediction" not in file_payload
    assert file_payload["output_path"] == str(output_path)
    assert file_payload["output_format"] == "csv"
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").splitlines()[0] == "sample_0,sample_1"

    npy_path = tmp_path / "prediction.npy"
    npy_payload = route_map["/twin/predict"](
        TwinPredictReq(
            engine_path=str(engine_path),
            params=[0.5],
            return_prediction=False,
            output_path=str(npy_path),
            output_format="npy",
        )
    )

    assert npy_payload["status"] == "ok"
    assert npy_payload["output_path"] == str(npy_path)
    assert npy_payload["output_format"] == "npy"
    assert "prediction" not in npy_payload
    assert np.load(npy_path).shape == (4,)


def test_twin_predict_endpoint_reports_bad_output_format(tmp_path) -> None:
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
            TwinPredictReq(
                engine_path=str(engine_path),
                params=[0.5],
                output_path=str(tmp_path / "prediction.bin"),
                output_format="vtk",
            )
        )

    assert exc.value.status_code == 400
    assert "unsupported output_format" in str(exc.value.detail)


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
