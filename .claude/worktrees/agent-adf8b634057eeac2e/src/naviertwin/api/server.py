"""FastAPI REST 서버 — NavierTwin 기능을 HTTP 엔드포인트로 노출.

엔드포인트:
    - GET  /health                        : 헬스 체크
    - GET  /doctor                        : 설치/런타임 환경 진단
    - POST /reduce                         : reducer 수행, 모드/에너지 반환
    - POST /reduce/pod                     : POD 전용(하위 호환)
    - POST /preflight                      : CFD 입력 readiness 점검
    - POST /twin/build                     : CFD/CSV dataset → TwinEngine 산출물 생성
    - POST /twin/predict                   : 저장/배포된 TwinEngine 예측/파일 출력
    - POST /twin/benchmark                 : TwinEngine 예측 latency/SLO 측정
    - POST /twin/package                   : TwinEngine 산출물 → 전달 ZIP 패키징
    - POST /twin/package/inspect           : 전달 ZIP 구성/메타데이터 조회
    - POST /twin/package/verify            : 전달 ZIP 무결성 검증/선택 추출
    - POST /twin/package/accept            : 전달 ZIP 검증/예측/latency 수락 검사
    - POST /twin/stream/init               : StreamingDigitalTwin 세션 초기화
    - POST /twin/stream/step               : StreamingDigitalTwin forecast 전파
    - POST /twin/stream/observe            : StreamingDigitalTwin 관측 동화
    - POST /twin/stream/observe-batch      : StreamingDigitalTwin 관측 배치 동화
    - POST /twin/stream/observe-line       : CSV/log 라인 관측 동화
    - GET  /twin/stream/state              : StreamingDigitalTwin 현재 상태 조회
    - POST /analytic/couette              : Couette 해석해 샘플
    - POST /analytic/poiseuille_2d        : Poiseuille 2D 해석해 샘플
    - POST /optimize/bayesian             : BO 최소화 (간단 quadratic)

Usage:
    uvicorn naviertwin.api.server:app --host 0.0.0.0 --port 8000
"""

from typing import Any, List, Optional  # noqa: UP035 — pydantic v1 호환

from naviertwin import __version__

_HAS_FASTAPI = True
try:
    import fastapi
    from fastapi import Body, FastAPI
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    _HAS_FASTAPI = False
    fastapi = None
    Body = None  # type: ignore[assignment]
    FastAPI = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[misc,assignment]


if _HAS_FASTAPI:

    class CouetteReq(BaseModel):
        U_top: float = 1.0
        H: float = 1.0
        n_points: int = 50

    class PoiseuilleReq(BaseModel):
        dpdx: float = -1.0
        mu: float = 1.0
        H: float = 1.0
        n_points: int = 50

    class PODReq(BaseModel):
        snapshots: List[List[float]]  # noqa: UP006
        n_modes: int = 5
        reducer_kind: str = "pod"

    class BayesianOptReq(BaseModel):
        bounds: List[List[float]]  # noqa: UP006
        n_initial: int = 5
        max_iter: int = 10
        problem: str = "quadratic"

    class PreflightReq(BaseModel):
        path: str

    class TwinBuildReq(BaseModel):
        input_path: Optional[str] = None
        csv_snapshots: Optional[str] = None
        field: Optional[str] = None
        field_column: Optional[str] = None
        params: Optional[str] = None
        param_columns: Optional[str] = None
        outdir: str
        reducer: str = "pod"
        n_modes: int = 3
        surrogate: str = "rbf"
        validation_count: int = 3

    class TwinPredictReq(BaseModel):
        engine_path: Optional[str] = None
        artifacts_dir: Optional[str] = None
        params: Any
        preview_limit: int = 8
        return_prediction: bool = True
        output_path: Optional[str] = None
        output_format: str = "csv"

    class TwinStreamInitReq(BaseModel):
        session_id: Optional[str] = None
        state_dim: int
        n_ensemble: int = 40
        transition: Optional[List[List[float]]] = None  # noqa: UP006
        observation_matrix: Optional[List[List[float]]] = None  # noqa: UP006
        observation_covariance: Optional[List[List[float]]] = None  # noqa: UP006
        process_noise: float = 0.01
        history_size: int = 100
        seed: Optional[int] = 0
        initial_mean: Optional[List[float]] = None  # noqa: UP006
        initial_std: float = 1.0
        initial_ensemble: Optional[List[List[float]]] = None  # noqa: UP006

    class TwinStreamStepReq(BaseModel):
        session_id: str
        steps: int = 1

    class TwinStreamObserveReq(BaseModel):
        session_id: str
        observation: List[float]  # noqa: UP006
        advance: bool = True

    class TwinStreamObserveBatchReq(BaseModel):
        session_id: str
        observations: List[List[float]]  # noqa: UP006
        advance: bool = True

    class TwinStreamObserveLineReq(BaseModel):
        session_id: str
        line: str
        delimiter: str = ","
        value_columns: Optional[List[int]] = None  # noqa: UP006
        advance: bool = True

    class TwinBenchmarkReq(BaseModel):
        engine_path: Optional[str] = None
        artifacts_dir: Optional[str] = None
        params: Any
        warmup: int = 2
        repeat: int = 20
        max_mean_ms: Optional[float] = None
        max_p50_ms: Optional[float] = None
        max_p95_ms: Optional[float] = None
        max_p99_ms: Optional[float] = None
        min_throughput_hz: Optional[float] = None

    class TwinPackageAcceptReq(BaseModel):
        package: str
        extract_to: Optional[str] = None
        prediction_output: Optional[str] = None
        warmup: int = 2
        repeat: int = 20
        max_mean_ms: Optional[float] = None
        max_p50_ms: Optional[float] = None
        max_p95_ms: Optional[float] = None
        max_p99_ms: Optional[float] = None
        min_throughput_hz: Optional[float] = None
        skip_benchmark: bool = False

    class TwinPackageCreateReq(BaseModel):
        artifacts_dir: str
        output: str
        include_validation: Optional[str] = None
        max_mean_ms: Optional[float] = None
        max_p50_ms: Optional[float] = None
        max_p95_ms: Optional[float] = 100.0
        max_p99_ms: Optional[float] = None
        min_throughput_hz: Optional[float] = 10.0
        no_latency_slo: bool = False

    class TwinPackageInspectReq(BaseModel):
        package: str

    class TwinPackageVerifyReq(BaseModel):
        package: str
        extract_to: Optional[str] = None

    class LBMReq(BaseModel):
        nx: int = 32
        ny: int = 32
        tau: float = 0.8
        u_top: float = 0.05
        n_steps: int = 200
        record_every: int = 200


def create_app() -> Any:
    """FastAPI app 팩토리."""
    if not _HAS_FASTAPI:
        raise RuntimeError(
            "fastapi 설치 필요: pip install fastapi uvicorn"
        )
    import numpy as np

    app = FastAPI(title="NavierTwin API", version=__version__)
    stream_sessions: dict[str, dict[str, Any]] = {}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "naviertwin"}

    @app.get("/doctor")
    def doctor(include_optional: bool = False) -> dict[str, Any]:
        from naviertwin.utils.doctor import build_doctor_report

        return build_doctor_report(include_optional=include_optional)

    @app.post("/analytic/couette")
    def couette(req: CouetteReq = Body(...)) -> dict[str, list]:
        from naviertwin.core.validation.analytic_solutions import couette_flow

        y = np.linspace(0.0, req.H, req.n_points)
        sol = couette_flow(U_top=req.U_top, H=req.H, y=y)
        return {"coords": y.tolist(), "velocity": sol.velocity.tolist()}

    @app.post("/analytic/poiseuille_2d")
    def poiseuille(req: PoiseuilleReq = Body(...)) -> dict[str, list]:
        from naviertwin.core.validation.analytic_solutions import poiseuille_flow_2d

        y = np.linspace(0.0, req.H, req.n_points)
        sol = poiseuille_flow_2d(dpdx=req.dpdx, mu=req.mu, H=req.H, y=y)
        return {"coords": y.tolist(), "velocity": sol.velocity.tolist()}

    def _run_reducer(req: PODReq) -> dict[str, Any]:
        X = np.asarray(req.snapshots, dtype=np.float64)
        if req.reducer_kind == "pod":
            from naviertwin.core.dimensionality_reduction.linear.pod import SnapshotPOD

            reducer = SnapshotPOD(n_modes=req.n_modes)
        elif req.reducer_kind == "incremental_pod":
            from naviertwin.core.dimensionality_reduction.linear.incremental_pod import (
                IncrementalPOD,
            )

            reducer = IncrementalPOD(n_modes=req.n_modes)
        elif req.reducer_kind == "mrpod":
            from naviertwin.core.dimensionality_reduction.linear.mrpod import MRPOD

            reducer = MRPOD(n_scales=3, n_modes_per_scale=req.n_modes)
        else:
            raise fastapi.HTTPException(
                status_code=400,
                detail=f"unknown reducer_kind: {req.reducer_kind}",
            )

        reducer.fit(X)
        singular_values = getattr(reducer, "singular_values_", getattr(reducer, "singular_values", None))
        if singular_values is None:
            singular_values = []
        energy = getattr(reducer, "energy_ratio_", None)
        if energy is None:
            energy = getattr(reducer, "energy_ratio", [])
        n_components = int(getattr(reducer, "n_components", req.n_modes))
        return {
            "reducer_kind": req.reducer_kind,
            "n_modes": n_components,
            "singular_values": np.asarray(singular_values, dtype=np.float64).tolist(),
            "cumulative_energy": np.asarray(energy, dtype=np.float64).tolist(),
        }

    @app.post("/reduce")
    def reduce(req: PODReq = Body(...)) -> dict[str, Any]:
        return _run_reducer(req)

    @app.post("/reduce/pod")
    def pod(req: PODReq = Body(...)) -> dict[str, Any]:
        # 하위 호환: 기존 요청은 reducer_kind 없이 /reduce/pod를 호출한다.
        req.reducer_kind = "pod"
        return _run_reducer(req)

    @app.post("/preflight")
    def preflight(req: PreflightReq = Body(...)) -> dict[str, Any]:
        from naviertwin.core.validation.dataset_preflight import (
            build_dataset_preflight_report,
        )

        try:
            if not req.path.strip():
                raise ValueError("path is required")
            return build_dataset_preflight_report(req.path)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/twin/build")
    def twin_build(req: TwinBuildReq = Body(...)) -> dict[str, Any]:
        from naviertwin.main import _build_twin_payload

        try:
            sources = [bool(req.input_path), bool(req.csv_snapshots)]
            if sum(sources) != 1:
                raise ValueError("exactly one of input_path or csv_snapshots is required")
            if not req.outdir.strip():
                raise ValueError("outdir is required")
            if req.reducer not in {"pod", "incremental_pod", "mrpod", "ae"}:
                raise ValueError(f"unsupported reducer: {req.reducer}")
            if req.surrogate not in {"kriging", "rbf"}:
                raise ValueError(f"unsupported surrogate: {req.surrogate}")
            payload = _build_twin_payload(
                input_path=req.input_path,
                csv_snapshots=req.csv_snapshots,
                field=req.field,
                field_column=req.field_column,
                params=req.params,
                param_columns=req.param_columns,
                outdir=req.outdir,
                reducer=req.reducer,
                n_modes=req.n_modes,
                surrogate=req.surrogate,
                validation_count=req.validation_count,
            )
        except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc

        return payload

    def _write_prediction_output(
        prediction: np.ndarray,
        *,
        output_path: Optional[str],
        output_format: str,
    ) -> tuple[str | None, str | None]:
        from pathlib import Path

        normalized_format = output_format.lower()
        if normalized_format not in {"csv", "npy"}:
            raise ValueError(
                f"unsupported output_format: {output_format}. supported: csv, npy"
            )
        if output_path is None:
            return None, None

        target = Path(output_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        if normalized_format == "csv":
            matrix = prediction.reshape(-1, 1) if prediction.ndim == 1 else prediction
            if matrix.ndim != 2:
                raise ValueError(
                    f"csv output requires 1D or 2D prediction, got shape {prediction.shape}"
                )
            header = ",".join(map(_format_sample_header, range(matrix.shape[1])))
            np.savetxt(target, matrix, delimiter=",", header=header, comments="")
        else:
            with target.open("wb") as handle:
                np.save(handle, prediction)
        return str(target), normalized_format

    def _format_sample_header(index: int) -> str:
        return f"sample_{index}"

    @app.post("/twin/predict")
    def twin_predict(req: TwinPredictReq = Body(...)) -> dict[str, Any]:
        import pickle
        from time import perf_counter

        from naviertwin.core.digital_twin.twin_engine import TwinEngine
        from naviertwin.main import (
            _check_twin_parameter_contract,
            _load_twin_parameter_contract,
            _resolve_twin_engine_path,
        )

        try:
            engine_file = _resolve_twin_engine_path(
                engine_path=req.engine_path,
                artifacts_dir=req.artifacts_dir,
            )
            engine = TwinEngine.load(engine_file)
            params = np.asarray(req.params, dtype=np.float64)
            if params.ndim not in {1, 2}:
                raise ValueError(f"params must be 1D or 2D, got shape {params.shape}")
            parameter_contract = _load_twin_parameter_contract(engine_file)
            parameter_check = _check_twin_parameter_contract(params, parameter_contract)
            started_at = perf_counter()
            prediction = np.asarray(engine.predict(params), dtype=np.float64)
            latency_ms = (perf_counter() - started_at) * 1000.0
            output_path, output_format = _write_prediction_output(
                prediction,
                output_path=req.output_path,
                output_format=req.output_format,
            )
        except (
            AttributeError,
            EOFError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            pickle.PickleError,
        ) as exc:
            raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc

        preview = prediction.reshape(-1)[: max(0, req.preview_limit)]
        payload = {
            "status": "ok",
            "engine": str(engine_file),
            "artifacts_dir": req.artifacts_dir,
            "input_shape": list(params.shape),
            "prediction_shape": list(prediction.shape),
            "prediction_size": int(prediction.size),
            "prediction_bytes": int(prediction.nbytes),
            "prediction_returned": bool(req.return_prediction),
            "latency_ms": float(latency_ms),
            "output_path": output_path,
            "output_format": output_format,
            "parameter_contract": parameter_contract,
            "parameter_check": parameter_check,
            "preview": list(map(float, preview)),
        }
        if req.return_prediction:
            payload["prediction"] = prediction.tolist()
        return payload

    def _stream_matrix(
        raw: Any,
        default: np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        matrix = np.asarray(default if raw is None else raw, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(f"{name} must be 2D, got shape {matrix.shape}")
        return matrix

    def _stream_state_payload(
        session_id: str,
        record: dict[str, Any],
        *,
        event: str,
    ) -> dict[str, Any]:
        twin = record["twin"]
        estimate = np.asarray(twin.estimate(), dtype=np.float64)
        uncertainty = np.asarray(twin.uncertainty(), dtype=np.float64)
        history_tail = []
        history_items = list(twin.history)[-5:]
        history_index = 0
        while history_index < len(history_items):
            item = history_items[history_index]
            history_tail.append(np.asarray(item, dtype=np.float64).tolist())
            history_index += 1
        return {
            "status": "ok",
            "event": event,
            "session_id": session_id,
            "state_dim": int(twin.state_dim),
            "n_ensemble": int(twin.n_ensemble),
            "step_count": int(record["step_count"]),
            "observation_count": int(record["observation_count"]),
            "history_length": int(len(twin.history)),
            "history_limit": int(twin.history.maxlen or 0),
            "estimate": estimate.tolist(),
            "uncertainty": uncertainty.tolist(),
            "history_tail": history_tail,
        }

    def _get_stream_record(session_id: str) -> dict[str, Any]:
        try:
            return stream_sessions[session_id]
        except KeyError as exc:
            raise fastapi.HTTPException(
                status_code=404,
                detail=f"stream session not found: {session_id}",
            ) from exc

    def _apply_stream_observation(
        record: dict[str, Any],
        observation: Any,
        *,
        advance: bool,
    ) -> None:
        if advance:
            record["twin"].step()
            record["step_count"] += 1
        record["twin"].assimilate(np.asarray(observation, dtype=np.float64))
        record["observation_count"] += 1

    def _parse_stream_observation_line(req: TwinStreamObserveLineReq) -> list[float]:
        tokens = list(map(str.strip, req.line.strip().split(req.delimiter)))
        if not tokens or not any(tokens):
            raise ValueError("line must contain observation values")
        if req.value_columns is not None:
            try:
                selected_tokens = []
                column_index = 0
                while column_index < len(req.value_columns):
                    selected_tokens.append(tokens[req.value_columns[column_index]])
                    column_index += 1
                tokens = selected_tokens
            except IndexError as exc:
                raise ValueError("value_columns index out of range") from exc
        values: list[float] = []
        token_index = 0
        while token_index < len(tokens):
            token = tokens[token_index]
            if not token:
                token_index += 1
                continue
            try:
                values.append(float(token))
            except ValueError as exc:
                raise ValueError(f"non-numeric observation token: {token}") from exc
            token_index += 1
        if not values:
            raise ValueError("line must contain numeric observation values")
        return values

    @app.post("/twin/stream/init")
    def twin_stream_init(req: TwinStreamInitReq = Body(...)) -> dict[str, Any]:
        from uuid import uuid4

        from naviertwin.core.digital_twin.streaming_twin import StreamingDigitalTwin

        try:
            if req.state_dim < 1:
                raise ValueError("state_dim must be >= 1")
            if req.n_ensemble < 2:
                raise ValueError("n_ensemble must be >= 2")
            if req.history_size < 1:
                raise ValueError("history_size must be >= 1")
            if req.initial_std < 0.0:
                raise ValueError("initial_std must be >= 0")

            state_dim = int(req.state_dim)
            transition = _stream_matrix(
                req.transition,
                np.eye(state_dim, dtype=np.float64),
                name="transition",
            )
            if transition.shape != (state_dim, state_dim):
                raise ValueError(
                    "transition shape must be "
                    f"({state_dim}, {state_dim}), got {transition.shape}"
                )
            observation_matrix = _stream_matrix(
                req.observation_matrix,
                np.eye(state_dim, dtype=np.float64),
                name="observation_matrix",
            )
            if observation_matrix.shape[1] != state_dim:
                raise ValueError(
                    "observation_matrix columns must equal state_dim: "
                    f"{observation_matrix.shape[1]} != {state_dim}"
                )
            obs_dim = int(observation_matrix.shape[0])
            observation_covariance = _stream_matrix(
                req.observation_covariance,
                0.01 * np.eye(obs_dim, dtype=np.float64),
                name="observation_covariance",
            )
            if observation_covariance.shape != (obs_dim, obs_dim):
                raise ValueError(
                    "observation_covariance shape must be "
                    f"({obs_dim}, {obs_dim}), got {observation_covariance.shape}"
                )

            rng = np.random.default_rng(req.seed)
            if req.initial_ensemble is not None:
                initial_ensemble = np.asarray(req.initial_ensemble, dtype=np.float64)
            else:
                initial_mean = (
                    np.zeros(state_dim, dtype=np.float64)
                    if req.initial_mean is None
                    else np.asarray(req.initial_mean, dtype=np.float64)
                )
                if initial_mean.shape != (state_dim,):
                    raise ValueError(
                        f"initial_mean shape must be ({state_dim},), got {initial_mean.shape}"
                    )
                initial_ensemble = initial_mean + req.initial_std * rng.standard_normal(
                    (req.n_ensemble, state_dim)
                )

            twin = StreamingDigitalTwin(
                state_dim=state_dim,
                n_ensemble=req.n_ensemble,
                model_fn=lambda x, matrix=transition: matrix @ x,
                H=observation_matrix,
                R=observation_covariance,
                process_noise=req.process_noise,
                history_size=req.history_size,
                rng=rng,
            )
            twin.initialize(initial_ensemble)
            session_id = (req.session_id or uuid4().hex).strip()
            if not session_id:
                raise ValueError("session_id must not be empty")
            replaced = session_id in stream_sessions
            stream_sessions[session_id] = {
                "twin": twin,
                "step_count": 0,
                "observation_count": 0,
            }
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc

        payload = _stream_state_payload(session_id, stream_sessions[session_id], event="init")
        payload["replaced"] = replaced
        payload["observation_dim"] = obs_dim
        return payload

    @app.post("/twin/stream/step")
    def twin_stream_step(req: TwinStreamStepReq = Body(...)) -> dict[str, Any]:
        if req.steps < 1:
            raise fastapi.HTTPException(status_code=400, detail="steps must be >= 1")
        record = _get_stream_record(req.session_id)
        try:
            step_index = 0
            while step_index < req.steps:
                record["twin"].step()
                step_index += 1
            record["step_count"] += req.steps
        except (RuntimeError, TypeError, ValueError) as exc:
            raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc
        return _stream_state_payload(req.session_id, record, event="step")

    @app.post("/twin/stream/observe")
    def twin_stream_observe(req: TwinStreamObserveReq = Body(...)) -> dict[str, Any]:
        record = _get_stream_record(req.session_id)
        try:
            _apply_stream_observation(record, req.observation, advance=req.advance)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc
        return _stream_state_payload(req.session_id, record, event="observe")

    @app.post("/twin/stream/observe-batch")
    def twin_stream_observe_batch(
        req: TwinStreamObserveBatchReq = Body(...),
    ) -> dict[str, Any]:
        record = _get_stream_record(req.session_id)
        try:
            if not req.observations:
                raise ValueError("observations must not be empty")
            observation_index = 0
            while observation_index < len(req.observations):
                observation = req.observations[observation_index]
                _apply_stream_observation(record, observation, advance=req.advance)
                observation_index += 1
        except (RuntimeError, TypeError, ValueError) as exc:
            raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc
        payload = _stream_state_payload(req.session_id, record, event="observe-batch")
        payload["processed_observations"] = len(req.observations)
        return payload

    @app.post("/twin/stream/observe-line")
    def twin_stream_observe_line(
        req: TwinStreamObserveLineReq = Body(...),
    ) -> dict[str, Any]:
        record = _get_stream_record(req.session_id)
        try:
            observation = _parse_stream_observation_line(req)
            _apply_stream_observation(record, observation, advance=req.advance)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc
        payload = _stream_state_payload(req.session_id, record, event="observe-line")
        payload["parsed_observation"] = observation
        return payload

    @app.get("/twin/stream/state")
    def twin_stream_state(session_id: str) -> dict[str, Any]:
        record = _get_stream_record(session_id)
        return _stream_state_payload(session_id, record, event="state")

    @app.post("/twin/benchmark")
    def twin_benchmark(req: TwinBenchmarkReq = Body(...)) -> dict[str, Any]:
        import pickle

        from naviertwin.main import _benchmark_twin_payload

        try:
            params = np.asarray(req.params, dtype=np.float64)
            payload = _benchmark_twin_payload(
                engine_path=req.engine_path,
                artifacts_dir=req.artifacts_dir,
                params_array=params,
                warmup=req.warmup,
                repeat=req.repeat,
                max_mean_ms=req.max_mean_ms,
                max_p50_ms=req.max_p50_ms,
                max_p95_ms=req.max_p95_ms,
                max_p99_ms=req.max_p99_ms,
                min_throughput_hz=req.min_throughput_hz,
            )
        except (
            AttributeError,
            EOFError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            pickle.PickleError,
        ) as exc:
            raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc

        return payload

    @app.post("/twin/package")
    def twin_package(req: TwinPackageCreateReq = Body(...)) -> dict[str, Any]:
        from naviertwin.main import _package_twin_payload

        try:
            if not req.artifacts_dir.strip():
                raise ValueError("artifacts_dir is required")
            if not req.output.strip():
                raise ValueError("output is required")
            payload = _package_twin_payload(
                artifacts_dir=req.artifacts_dir,
                output=req.output,
                include_validation=req.include_validation,
                max_mean_ms=req.max_mean_ms,
                max_p50_ms=req.max_p50_ms,
                max_p95_ms=req.max_p95_ms,
                max_p99_ms=req.max_p99_ms,
                min_throughput_hz=req.min_throughput_hz,
                no_latency_slo=req.no_latency_slo,
            )
        except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc

        return payload

    @app.post("/twin/package/inspect")
    def twin_package_inspect(req: TwinPackageInspectReq = Body(...)) -> dict[str, Any]:
        import zipfile
        from pathlib import Path

        from naviertwin.main import _inspect_twin_package_archive

        try:
            payload = _inspect_twin_package_archive(Path(req.package).expanduser())
        except (
            ImportError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            zipfile.BadZipFile,
        ) as exc:
            raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc

        return payload

    @app.post("/twin/package/verify")
    def twin_package_verify(req: TwinPackageVerifyReq = Body(...)) -> dict[str, Any]:
        import zipfile
        from pathlib import Path

        from naviertwin.main import _verify_twin_package_archive

        try:
            payload = _verify_twin_package_archive(
                Path(req.package).expanduser(),
                extract_to=Path(req.extract_to).expanduser() if req.extract_to else None,
            )
        except (
            ImportError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            zipfile.BadZipFile,
        ) as exc:
            raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc

        return payload

    @app.post("/twin/package/accept")
    def twin_package_accept(req: TwinPackageAcceptReq = Body(...)) -> dict[str, Any]:
        import tempfile
        import zipfile
        from pathlib import Path

        from naviertwin.main import _accept_twin_package_archive

        try:
            package = Path(req.package).expanduser()
            prediction_output = (
                Path(req.prediction_output).expanduser()
                if req.prediction_output
                else None
            )
            if req.extract_to:
                payload = _accept_twin_package_archive(
                    package,
                    extract_to=Path(req.extract_to).expanduser(),
                    temporary_extraction=False,
                    prediction_output=prediction_output,
                    warmup=req.warmup,
                    repeat=req.repeat,
                    max_mean_ms=req.max_mean_ms,
                    max_p50_ms=req.max_p50_ms,
                    max_p95_ms=req.max_p95_ms,
                    max_p99_ms=req.max_p99_ms,
                    min_throughput_hz=req.min_throughput_hz,
                    skip_benchmark=req.skip_benchmark,
                )
            else:
                with tempfile.TemporaryDirectory(prefix="naviertwin-api-accept-") as tmp_raw:
                    payload = _accept_twin_package_archive(
                        package,
                        extract_to=Path(tmp_raw) / "twin",
                        temporary_extraction=True,
                        prediction_output=prediction_output,
                        warmup=req.warmup,
                        repeat=req.repeat,
                        max_mean_ms=req.max_mean_ms,
                        max_p50_ms=req.max_p50_ms,
                        max_p95_ms=req.max_p95_ms,
                        max_p99_ms=req.max_p99_ms,
                        min_throughput_hz=req.min_throughput_hz,
                        skip_benchmark=req.skip_benchmark,
                    )
        except (
            ImportError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            zipfile.BadZipFile,
        ) as exc:
            raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc

        return payload

    @app.post("/simulate/lbm_cavity")
    def lbm_cavity(req: LBMReq = Body(...)) -> dict[str, Any]:
        from naviertwin.core.solver_interfaces.lbm_d2q9 import LBMD2Q9

        lbm = LBMD2Q9(nx=req.nx, ny=req.ny, tau=req.tau, u_top=req.u_top)
        snaps = lbm.run(n_steps=req.n_steps, record_every=req.record_every)
        last = snaps[-1]
        return {
            "n_snapshots": int(snaps.shape[0]),
            "shape": list(snaps.shape),
            "ux_max": float(last[..., 1].max()),
            "ux_min": float(last[..., 1].min()),
            "uy_max": float(last[..., 2].max()),
            "rho_mean": float(last[..., 0].mean()),
        }

    @app.post("/optimize/bayesian")
    def bayesian(req: BayesianOptReq = Body(...)) -> dict[str, Any]:
        from naviertwin.core.optimization.bayesian_opt import BayesianOptimizer

        bounds = np.array(req.bounds, dtype=np.float64)
        if req.problem == "quadratic":
            def obj(x: np.ndarray) -> float:
                return float(np.sum(x ** 2))
        else:
            def obj(x: np.ndarray) -> float:
                return float(np.sum(np.sin(x)))

        opt = BayesianOptimizer(
            bounds=bounds, n_initial=req.n_initial, max_iter=req.max_iter, seed=0
        )
        x_best, f_best = opt.minimize(obj)
        return {"x_best": x_best.tolist(), "f_best": f_best}

    return app


# 모듈 레벨 app
try:
    app: Optional[Any] = create_app() if _HAS_FASTAPI else None
except RuntimeError:
    app = None


__all__ = [
    "BayesianOptReq",
    "CouetteReq",
    "LBMReq",
    "PODReq",
    "PoiseuilleReq",
    "PreflightReq",
    "TwinBuildReq",
    "TwinBenchmarkReq",
    "TwinPackageAcceptReq",
    "TwinPackageCreateReq",
    "TwinPackageInspectReq",
    "TwinPackageVerifyReq",
    "TwinPredictReq",
    "TwinStreamObserveBatchReq",
    "TwinStreamObserveLineReq",
    "TwinStreamInitReq",
    "TwinStreamObserveReq",
    "TwinStreamStepReq",
    "app",
    "create_app",
]
