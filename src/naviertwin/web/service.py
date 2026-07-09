"""웹 워크플로우 오케스트레이션 (Qt/GL 비의존).

trame 앱과 독립적으로 테스트 가능한 순수 서비스 계층. Import → Analyze →
Reduce → Twin MVP 워크플로우를 ``core`` 모듈 위에 얇게 조립한다.

함수는 모두 ``CFDDataset`` 또는 학습된 모델 객체를 받고/돌려주며, trame
state 나 PyVista 렌더 컨텍스트(GL)에 의존하지 않는다. 따라서 headless CI
에서도 그대로 단위 테스트할 수 있다.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from naviertwin.utils.logger import get_logger

if TYPE_CHECKING:
    from naviertwin.core.cfd_reader.base import CFDDataset

logger = get_logger(__name__)

# Analyze 패널과 동일한 와류 식별 기법 키.
VORTEX_METHODS = ("q_criterion", "lambda2")
RESULT_FIELD = {"q_criterion": "Q-criterion", "lambda2": "lambda2"}

# 파생(분석/모드/예측) field 판정 — app/service 단일 출처(SSOT).
DERIVED_PREFIXES = ("pod_mode", "twin_pred")
DERIVED_EXTRA = ("vorticity",)


def is_derived_field(name: str) -> bool:
    """분석/POD모드/예측으로 생성된 파생 field 인지 판정한다 (원본 물리량 제외용)."""
    if name in RESULT_FIELD.values() or name in DERIVED_EXTRA:
        return True
    return name.startswith(DERIVED_PREFIXES)


def _warn_overwrite(mesh: Any, name: str) -> None:
    """비파생 동명 사용자 field 를 분석/예측 결과로 덮어쓸 때 경고한다."""
    has_field = name in getattr(mesh, "point_data", {}) or name in getattr(mesh, "cell_data", {})
    if has_field and not is_derived_field(name):
        logger.warning("기존 사용자 field '%s' 를 분석/예측 결과로 덮어씁니다.", name)


# ──────────────────────────────────────────────────────────────────────
# Import
# ──────────────────────────────────────────────────────────────────────


def load_dataset(path: str | Path) -> CFDDataset:
    """파일/디렉토리 경로에서 CFD 데이터셋을 로드한다 (ReaderFactory)."""
    from naviertwin.core.cfd_reader import ReaderFactory

    target = Path(path).expanduser()
    if not target.exists():
        raise FileNotFoundError(f"경로를 찾을 수 없습니다: {target}")
    return ReaderFactory().create_and_read(target)


def load_project(path: str | Path) -> tuple[CFDDataset, Any | None]:
    """``.ntwin`` 프로젝트를 로드한다 (+ 같은 폴더의 ``.engine.pkl`` 트윈 복원).

    ``ReaderFactory`` 는 ``.ntwin`` 을 처리하지 못하므로 전용 VTKHDF 리더를 쓴다.
    ``<stem>.engine.pkl`` 사이드카가 있으면 ``TwinEngine`` 도 복원한다(실패 시 None).

    Returns:
        ``(dataset, engine|None)``.

    Raises:
        FileNotFoundError: 경로가 없는 경우.
        ValueError: ``.ntwin`` 파일이 아닌 경우.
    """
    from naviertwin.core.export.ntwin_format import load_dataset as _load_ntwin

    target = Path(path).expanduser()
    if not target.exists():
        raise FileNotFoundError(f"경로를 찾을 수 없습니다: {target}")
    if target.suffix.lower() != ".ntwin":
        raise ValueError(f".ntwin 프로젝트 파일이 아닙니다: {target}")

    dataset = _load_ntwin(target)
    engine: Any | None = None
    engine_path = target.with_suffix(".engine.pkl")
    if engine_path.exists():
        from naviertwin.core.digital_twin.twin_engine import TwinEngine

        try:
            engine = TwinEngine.load(engine_path)
        except Exception as exc:  # noqa: BLE001 — 엔진은 선택, dataset 은 살린다
            logger.warning("엔진 사이드카 로드 실패 (%s): %s", engine_path, exc)
            engine = None
    return dataset, engine


def make_demo_dataset(
    nx: int = 48,
    ny: int = 48,
    n_steps: int = 12,
    nu: float = 0.05,
    oscillation_hz: float = 1.5,
    oscillation_amp: float = 0.25,
) -> CFDDataset:
    """Taylor–Green 감쇠 와류 합성 데이터셋을 생성한다.

    파일 없이 전체 파이프라인(Analyze/Reduce/Twin)을 즉시 체험할 수 있는
    데모용 데이터셋이다. 2D 정형 격자 위에 시간에 따라 감쇠하는 속도장 ``U``
    와 압력장 ``p`` 를 시계열로 담는다.

    Args:
        nx: x 방향 격자점 수.
        ny: y 방향 격자점 수.
        n_steps: 타임스텝 수.
        nu: 동점성계수 (감쇠율 제어).

    Returns:
        시계열 ``U``/``p`` 를 metadata 에 담은 ``CFDDataset``.
    """
    import pyvista as pv

    from naviertwin.core.cfd_reader.base import CFDDataset

    nx = max(4, int(nx))
    ny = max(4, int(ny))
    n_steps = max(1, int(n_steps))

    two_pi = 2.0 * np.pi
    xs = np.linspace(0.0, two_pi, nx)
    ys = np.linspace(0.0, two_pi, ny)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")  # (ny, nx)
    x_flat = grid_x.ravel(order="C")  # ImageData 점 순서: x fastest
    y_flat = grid_y.ravel(order="C")
    n_points = x_flat.size

    dx = two_pi / (nx - 1) if nx > 1 else 1.0
    dy = two_pi / (ny - 1) if ny > 1 else 1.0
    image = pv.ImageData(
        dimensions=(nx, ny, 1),
        spacing=(float(dx), float(dy), 1.0),
        origin=(0.0, 0.0, 0.0),
    )

    times = np.linspace(0.0, 2.0, n_steps)
    velocity = np.zeros((n_steps, n_points, 3), dtype=np.float64)
    pressure = np.zeros((n_steps, n_points), dtype=np.float64)
    step = 0
    while step < n_steps:
        t = float(times[step])
        # 단조 감쇠 포락선에 약한 시간 진동을 더해 FFT/PSD 가 검출할 주파수를 만든다.
        # (공간 와류 구조는 유지된다.)
        envelope = np.exp(-2.0 * nu * t) * (
            1.0 + oscillation_amp * np.sin(2.0 * np.pi * oscillation_hz * t)
        )
        decay = envelope
        u = np.cos(x_flat) * np.sin(y_flat) * decay
        v = -np.sin(x_flat) * np.cos(y_flat) * decay
        velocity[step, :, 0] = u
        velocity[step, :, 1] = v
        pressure[step] = -0.25 * (np.cos(2.0 * x_flat) + np.cos(2.0 * y_flat)) * (decay ** 2)
        step += 1

    image.point_data["U"] = velocity[0]
    image.point_data["p"] = pressure[0]

    return CFDDataset(
        mesh=image,
        time_steps=list(map(float, times)),
        field_names=["U", "p"],
        metadata={
            "source": "demo_taylor_green",
            "description": "Taylor–Green decaying vortex (synthetic demo)",
            "time_series_fields": {"U": velocity, "p": pressure},
            "time_series_locations": {"U": "point", "p": "point"},
        },
    )


def dataset_info(dataset: CFDDataset) -> dict[str, Any]:
    """GUI 표시용 데이터셋 요약 정보를 만든다."""
    return {
        "points": int(getattr(dataset, "n_points", 0)),
        "cells": int(getattr(dataset, "n_cells", 0)),
        "time_steps": int(getattr(dataset, "n_time_steps", 0)),
        "fields": list(getattr(dataset, "field_names", []) or []),
        "source": str((getattr(dataset, "metadata", {}) or {}).get("source", "")),
    }


def sync_timestep_to_mesh(dataset: CFDDataset, timestep: int) -> None:
    """시계열 데이터셋의 현재 timestep 기본 field 를 mesh.point_data 에 반영한다.

    Analyze(Q-criterion/λ₂)는 ``mesh.point_data['U']`` 를 직접 읽으므로, 표시
    중인 timestep 과 분석 대상이 일치하도록 동기화한다. 시계열이 없는(실파일)
    데이터셋은 변경하지 않는다.
    """
    meta = getattr(dataset, "metadata", {}) or {}
    series = meta.get("time_series_fields", {})
    if not isinstance(series, dict) or not series:
        return
    locations = meta.get("time_series_locations", {})
    n_steps = max(1, int(getattr(dataset, "n_time_steps", 1)))
    step = min(max(0, int(timestep)), n_steps - 1)
    for name, arr in series.items():
        values = np.asarray(arr)
        if values.ndim >= 1 and values.shape[0] == n_steps:
            snapshot = values[step]
        else:
            snapshot = values
        location = locations.get(name, "point") if isinstance(locations, dict) else "point"
        target = dataset.mesh.cell_data if location == "cell" else dataset.mesh.point_data
        target[name] = np.asarray(snapshot)


# ──────────────────────────────────────────────────────────────────────
# Analyze — 와류 식별
# ──────────────────────────────────────────────────────────────────────


def run_vortex_analysis(
    dataset: CFDDataset,
    method: str,
    *,
    velocity_name: str = "U",
    timestep: int = 0,
) -> str:
    """Q-criterion 또는 λ₂ 와류 식별을 실행하고 결과 field 이름을 반환한다.

    결과 스칼라는 ``dataset.mesh.point_data`` 에 추가되고 ``field_names`` 에도
    등록되어 3D 뷰어에서 즉시 선택할 수 있다.

    Args:
        dataset: 속도장을 포함한 데이터셋.
        method: ``"q_criterion"`` 또는 ``"lambda2"``.
        velocity_name: 속도 벡터 field 이름.
        timestep: 분석에 사용할 timestep (시계열 데이터셋).

    Returns:
        메쉬에 추가된 결과 스칼라 field 이름.

    Raises:
        ValueError: 지원하지 않는 method 인 경우.
        KeyError: 속도 field 가 메쉬에 없는 경우.
    """
    if method not in VORTEX_METHODS:
        raise ValueError(f"지원하지 않는 분석 기법: {method}")

    from naviertwin.core.flow_analysis.vortex.q_criterion import (
        compute_lambda2,
        compute_q_criterion,
    )

    sync_timestep_to_mesh(dataset, timestep)
    mesh = dataset.mesh

    if method == "q_criterion":
        result_mesh = compute_q_criterion(mesh, velocity_name)
        field_name = "Q-criterion"
        values = result_mesh.point_data.get(field_name)
    else:
        result_mesh = compute_lambda2(mesh, velocity_name)
        field_name = "lambda2"
        values = result_mesh.point_data.get(field_name)

    if values is not None:
        _warn_overwrite(mesh, field_name)
        mesh.point_data[field_name] = np.asarray(values)
        if field_name not in dataset.field_names:
            dataset.field_names.append(field_name)
    return field_name


def estimate_dt(dataset: CFDDataset) -> float:
    """타임스텝 값에서 샘플링 간격 dt 를 추정한다 (없으면 1.0)."""
    times = np.asarray(getattr(dataset, "time_steps", []), dtype=np.float64)
    if times.size < 2:
        return 1.0
    diffs = np.diff(times)
    median = float(np.median(diffs))
    return median if median > 0 else 1.0


def compute_fft_psd(
    dataset: CFDDataset,
    field: str,
    *,
    dt: float | None = None,
    point_index: int | None = None,
    n_peaks: int = 3,
) -> dict[str, Any]:
    """필드 시계열의 FFT 진폭 스펙트럼 + Welch PSD + 지배 주파수를 계산한다.

    ``point_index`` 가 None 이면 공간 평균 신호를, 지정하면 해당 인덱스의 시간
    이력을 신호로 사용한다.

    Returns:
        ``freqs``/``amplitudes`` (FFT), ``psd_freqs``/``psd`` (Welch),
        ``dominant`` (지배 주파수 dict 리스트), ``dt``, ``probe``, ``signal``,
        ``times`` 를 담은 dict (배열은 list).

    Raises:
        ValueError: 타임스텝이 2개 미만이거나 시계열 복원이 불가능한 경우.
    """
    from naviertwin.core.flow_analysis.statistics.fft_psd import (
        compute_fft,
        compute_psd,
        find_dominant_frequencies,
    )

    n_steps = int(getattr(dataset, "n_time_steps", 0))
    if n_steps < 2:
        raise ValueError("FFT/PSD 에는 2개 이상의 타임스텝이 필요합니다.")
    snapshots = dataset.extract_field_snapshots(field)
    if snapshots.shape[1] != n_steps:
        raise ValueError("현재 데이터 구조에서는 시계열을 복원할 수 없습니다.")

    if point_index is None:
        signal = snapshots.mean(axis=0).astype(np.float64)
        probe = "spatial_mean"
    else:
        idx = max(0, min(int(point_index), snapshots.shape[0] - 1))
        signal = snapshots[idx].astype(np.float64)
        probe = f"point[{idx}]"

    sample_dt = float(dt) if dt and dt > 0 else estimate_dt(dataset)
    freqs, amps = compute_fft(signal, sample_dt)
    psd_freqs, psd = compute_psd(signal, sample_dt)
    dominant = find_dominant_frequencies(freqs, amps, n_peaks=n_peaks)
    times = np.asarray(getattr(dataset, "time_steps", []), dtype=np.float64)
    return {
        "field": field,
        "dt": sample_dt,
        "probe": probe,
        "signal": signal.tolist(),
        "times": times.tolist(),
        "freqs": np.asarray(freqs, dtype=np.float64).tolist(),
        "amplitudes": np.asarray(amps, dtype=np.float64).tolist(),
        "psd_freqs": np.asarray(psd_freqs, dtype=np.float64).tolist(),
        "psd": np.asarray(psd, dtype=np.float64).tolist(),
        "dominant": dominant,
    }


# ──────────────────────────────────────────────────────────────────────
# Reduce — POD
# ──────────────────────────────────────────────────────────────────────


def run_pod(dataset: CFDDataset, field: str, n_modes: int) -> dict[str, Any]:
    """필드 스냅샷에 대해 POD 차원 축소를 수행한다.

    Returns:
        ``reducer`` (학습된 SnapshotPOD), ``singular_values``,
        ``cumulative_energy``, ``n_modes``, ``n_snapshots`` 를 담은 dict.
    """
    from naviertwin.core.dimensionality_reduction.linear.pod import SnapshotPOD

    snapshots = dataset.extract_field_snapshots(field)
    n_snapshots = int(snapshots.shape[1])
    n_modes = max(1, min(int(n_modes), n_snapshots, int(snapshots.shape[0])))

    reducer = SnapshotPOD(n_modes=n_modes)
    reducer.fit(snapshots)
    reducer.training_metadata = {"field_name": field, "n_modes": n_modes}

    singular = np.asarray(getattr(reducer, "singular_values_", []), dtype=np.float64)
    energy = np.asarray(getattr(reducer, "energy_ratio", []), dtype=np.float64)
    return {
        "reducer": reducer,
        "field": field,
        "n_modes": int(getattr(reducer, "n_components", n_modes)),
        "n_snapshots": n_snapshots,
        "singular_values": singular.tolist(),
        "cumulative_energy": energy.tolist(),
    }


def pod_mode_field(reducer: Any, mode_index: int = 0) -> np.ndarray:
    """학습된 POD reducer 에서 지정 모드 형상을 1D 배열로 추출한다."""
    modes = np.asarray(getattr(reducer, "modes_", None))
    if modes is None or modes.ndim != 2:
        raise ValueError("reducer 에 학습된 POD 모드가 없습니다.")
    idx = max(0, min(int(mode_index), modes.shape[1] - 1))
    return modes[:, idx]


# ──────────────────────────────────────────────────────────────────────
# Twin — 파라미터 → 필드 예측
# ──────────────────────────────────────────────────────────────────────


REDUCERS = ("pod", "randomized_pod")
SURROGATES = ("rbf", "kriging")


def build_twin(
    dataset: CFDDataset,
    field: str,
    n_modes: int,
    *,
    reducer: str = "pod",
    surrogate: str = "rbf",
) -> dict[str, Any]:
    """시계열 데이터셋에서 (시간 → 필드) 디지털 트윈을 학습한다.

    timestep 시간값을 파라미터로, 각 timestep 의 필드 스냅샷을 출력으로 하는
    (reducer + surrogate) 파이프라인(TwinEngine)을 구성한다. 학습 후 임의의 시간
    파라미터에서 필드를 실시간 재구성·예측할 수 있다.

    Args:
        dataset: 시계열 데이터셋.
        field: 학습 대상 물리량 field.
        n_modes: POD 모드 수.
        reducer: 차원 축소기 ``"pod"`` | ``"randomized_pod"``.
        surrogate: 서로게이트 ``"rbf"`` | ``"kriging"``.

    Returns:
        ``engine`` (TwinEngine), ``param_min``/``param_max`` (예측 슬라이더
        범위), ``params`` (학습 파라미터), ``n_modes``, ``reducer``,
        ``surrogate`` 를 담은 dict.

    Raises:
        ValueError: 타임스텝이 2개 미만이거나 미지원 reducer/surrogate 인 경우.
    """
    from naviertwin.core.digital_twin.twin_engine import TwinEngine

    n_steps = int(getattr(dataset, "n_time_steps", 0))
    if n_steps < 2:
        raise ValueError(
            "디지털 트윈 학습에는 2개 이상의 타임스텝이 필요합니다. "
            f"현재: {n_steps}"
        )

    snapshots = dataset.extract_field_snapshots(field)
    if snapshots.shape[1] != n_steps:
        # 단일 스냅샷으로 폴백된 경우 등 — 시계열 파라미터화 불가.
        raise ValueError(
            "필드 스냅샷 수와 타임스텝 수가 일치하지 않아 트윈을 학습할 수 없습니다."
        )

    if reducer not in REDUCERS:
        raise ValueError(f"지원하지 않는 reducer: {reducer}")
    if surrogate not in SURROGATES:
        raise ValueError(f"지원하지 않는 surrogate: {surrogate}")

    times = np.asarray(getattr(dataset, "time_steps", []), dtype=np.float64)
    params = times.reshape(-1, 1)
    n_modes = max(1, min(int(n_modes), n_steps, int(snapshots.shape[0])))

    engine = TwinEngine(reducer_type=reducer, surrogate_type=surrogate, n_modes=n_modes)
    engine.fit(snapshots, params)
    engine.training_metadata = {
        "field_name": field,
        "n_modes": n_modes,
        "reducer": reducer,
        "surrogate": surrogate,
        # 저장/복원 시 예측 슬라이더가 학습 범위를 벗어나지 않도록 기록
        # (TwinEngine 자체는 학습 파라미터 범위를 저장하지 않는다).
        "param_min": float(times.min()),
        "param_max": float(times.max()),
    }

    return {
        "engine": engine,
        "field": field,
        "n_modes": n_modes,
        "reducer": reducer,
        "surrogate": surrogate,
        "params": params.reshape(-1).tolist(),
        "param_min": float(times.min()),
        "param_max": float(times.max()),
    }


def predict_twin(engine: Any, param_value: float) -> np.ndarray:
    """학습된 TwinEngine 으로 단일 파라미터에서 필드를 예측한다."""
    params = np.asarray([float(param_value)], dtype=np.float64)
    prediction = np.asarray(engine.predict(params), dtype=np.float64).reshape(-1)
    return prediction


def attach_prediction(
    dataset: CFDDataset,
    prediction: np.ndarray,
    field_name: str = "twin_prediction",
) -> str:
    """예측 필드를 현재 메쉬에 붙이고 field 이름을 반환한다 (3D 표시용)."""
    mesh = dataset.mesh
    values = np.asarray(prediction, dtype=np.float64).reshape(-1)
    n_points = int(getattr(dataset, "n_points", 0))
    n_cells = int(getattr(dataset, "n_cells", 0))
    _warn_overwrite(mesh, field_name)
    if values.size == n_points:
        mesh.point_data[field_name] = values
    elif values.size == n_cells:
        mesh.cell_data[field_name] = values
    else:
        raise ValueError(
            f"예측 크기({values.size})가 메쉬 point({n_points})/cell({n_cells})와 "
            "맞지 않아 표시할 수 없습니다."
        )
    if field_name not in dataset.field_names:
        dataset.field_names.append(field_name)
    return field_name


# ──────────────────────────────────────────────────────────────────────
# Compare — reducer×surrogate 조합 벤치마크
# ──────────────────────────────────────────────────────────────────────


def compare_models(
    dataset: CFDDataset,
    field: str,
    n_modes: int,
    *,
    combos: list[tuple[str, str]] | None = None,
    repeat: int = 10,
    progress_cb: Any = None,
) -> dict[str, Any]:
    """여러 reducer×surrogate 조합을 학습/평가해 정확도·지연시간 순위표를 만든다.

    각 조합으로 (시간→필드) TwinEngine 을 학습하고, 학습 파라미터에서의 재구성
    오차(RMSE / R² / 상대 L2)와 단일 예측 지연시간(ms)을 측정한다.

    Args:
        dataset: 시계열 데이터셋.
        field: 평가 대상 물리량 field.
        n_modes: POD 모드 수.
        combos: 평가할 (reducer, surrogate) 목록. None 이면 전체 조합.
        repeat: 지연시간 측정 반복 횟수.

    Returns:
        ``rows`` (조합별 결과 dict 리스트, RMSE 오름차순 정렬), ``best``
        (최우수 조합 요약), ``field``, ``n_modes`` 를 담은 dict.

    Raises:
        ValueError: 타임스텝이 2개 미만인 경우.
    """
    from time import perf_counter

    from naviertwin.core.validation.metrics import r2_score, relative_l2_error, rmse

    n_steps = int(getattr(dataset, "n_time_steps", 0))
    if n_steps < 2:
        raise ValueError("모델 비교에는 2개 이상의 타임스텝이 필요합니다.")

    truth = dataset.extract_field_snapshots(field)
    times = np.asarray(getattr(dataset, "time_steps", []), dtype=np.float64)
    params = times.reshape(-1, 1)
    single = params[len(params) // 2 : len(params) // 2 + 1]
    if combos is None:
        combos = [(reducer, surrogate) for reducer in REDUCERS for surrogate in SURROGATES]

    rows: list[dict[str, Any]] = []
    n_combos = len(combos)
    for combo_index, (reducer, surrogate) in enumerate(combos):
        if progress_cb is not None:
            progress_cb(combo_index, n_combos, f"{reducer}+{surrogate}")
        try:
            result = build_twin(dataset, field, n_modes, reducer=reducer, surrogate=surrogate)
            engine = result["engine"]
            prediction = np.asarray(engine.predict(params), dtype=np.float64)
            engine.predict(single)  # warmup
            started = perf_counter()
            iteration = 0
            while iteration < max(1, repeat):
                engine.predict(single)
                iteration += 1
            latency_ms = (perf_counter() - started) / max(1, repeat) * 1000.0
            rows.append(
                {
                    "combo": f"{reducer}+{surrogate}",
                    "reducer": reducer,
                    "surrogate": surrogate,
                    "n_modes": int(result["n_modes"]),
                    "rmse": float(rmse(truth, prediction)),
                    "r2": float(r2_score(truth, prediction)),
                    "rel_l2": float(relative_l2_error(truth, prediction)),
                    "latency_ms": float(latency_ms),
                    "status": "ok",
                }
            )
        except Exception as exc:  # noqa: BLE001 — 한 조합 실패가 전체를 막지 않도록
            rows.append(
                {
                    "combo": f"{reducer}+{surrogate}",
                    "reducer": reducer,
                    "surrogate": surrogate,
                    "n_modes": int(n_modes),
                    "rmse": float("inf"),
                    "r2": float("nan"),
                    "rel_l2": float("inf"),
                    "latency_ms": float("nan"),
                    "status": f"error: {exc}",
                }
            )

    if progress_cb is not None:
        progress_cb(n_combos, n_combos, "완료")
    rows.sort(key=lambda row: row["rmse"])
    best = next((row for row in rows if row["status"] == "ok"), None)
    return {"rows": rows, "best": best, "field": field, "n_modes": int(n_modes)}


def attach_pod_mode(
    dataset: CFDDataset,
    reducer: Any,
    mode_index: int = 0,
    field_name: str = "pod_mode",
) -> str:
    """학습된 POD 모드 형상을 메쉬에 붙이고 3D 표시용 field 이름을 반환한다."""
    mode = pod_mode_field(reducer, mode_index)
    label = f"{field_name}_{int(mode_index)}"
    return attach_prediction(dataset, mode, field_name=label)


# ──────────────────────────────────────────────────────────────────────
# Export — 산출물 저장 (Qt/GL 비의존)
# ──────────────────────────────────────────────────────────────────────


def _ensure_parent(path: Path) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _as_unstructured(dataset: CFDDataset) -> CFDDataset:
    """메쉬를 UnstructuredGrid 로 cast 한 데이터셋을 반환한다 (.ntwin 호환)."""
    import pyvista as pv

    from naviertwin.core.cfd_reader.base import CFDDataset

    mesh = dataset.mesh
    if isinstance(mesh, pv.UnstructuredGrid):
        return dataset
    try:
        cast = mesh.cast_to_unstructured_grid()
    except Exception:  # noqa: BLE001 — cast 불가 시 원본 유지
        return dataset
    return CFDDataset(
        mesh=cast,
        time_steps=list(getattr(dataset, "time_steps", []) or []),
        field_names=list(getattr(dataset, "field_names", []) or []),
        metadata=dict(getattr(dataset, "metadata", {}) or {}),
    )


def snapshot_dataset(dataset: CFDDataset, timestep: int = 0) -> CFDDataset:
    """현재 timestep 을 단일 스냅샷으로 materialize 한 새 데이터셋을 만든다.

    시계열을 metadata 에 담는 데모/스트리밍 데이터셋을 ``.ntwin``/VTK/CSV 로
    내보낼 때, 메쉬에 현재 timestep field 를 모두 부착하고 ``time_steps`` 를 1개로
    맞춘 자기완결적(round-trip 가능) 데이터셋을 반환한다. 원본은 변경하지 않는다.
    """
    from naviertwin.core.cfd_reader.base import CFDDataset

    mesh = dataset.mesh.copy(deep=True)
    meta = dict(getattr(dataset, "metadata", {}) or {})
    series = meta.pop("time_series_fields", {})
    locations = meta.pop("time_series_locations", {})

    n_steps = max(1, int(getattr(dataset, "n_time_steps", 1)))
    step = min(max(0, int(timestep)), n_steps - 1)
    if isinstance(series, dict):
        for name, arr in series.items():
            values = np.asarray(arr)
            current = values[step] if values.ndim >= 1 and values.shape[0] == n_steps else values
            location = locations.get(name, "point") if isinstance(locations, dict) else "point"
            target = mesh.cell_data if location == "cell" else mesh.point_data
            target[name] = np.asarray(current)

    times = list(getattr(dataset, "time_steps", []) or [0.0])
    time_value = float(times[step]) if times else 0.0
    return CFDDataset(
        mesh=mesh,
        time_steps=[time_value],
        field_names=list(getattr(dataset, "field_names", []) or []),
        metadata=meta,
    )


def export_vtk(dataset: CFDDataset, path: str | Path) -> str:
    """현재 메쉬를 VTK 포맷으로 저장한다.

    메쉬 타입이 요청 확장자를 지원하지 않으면(예: ImageData 에 ``.vtu``),
    모든 메쉬 타입이 지원하는 legacy ``.vtk`` 로 폴백한다.
    """
    target = _ensure_parent(path)
    try:
        dataset.mesh.save(str(target))
        return str(target)
    except ValueError:
        fallback = target.with_suffix(".vtk")
        dataset.mesh.save(str(fallback))
        return str(fallback)


def export_field_csv(dataset: CFDDataset, path: str | Path) -> str:
    """좌표 + point_data field 들을 CSV 로 저장한다."""
    target = _ensure_parent(path)
    mesh = dataset.mesh
    pts = np.asarray(mesh.points, dtype=np.float64)
    header = "x,y,z"
    columns: list[np.ndarray] = [pts]
    for name in list(getattr(dataset, "field_names", []) or []):
        if name not in mesh.point_data:
            continue
        arr = np.asarray(mesh.point_data[name])
        if arr.ndim == 1:
            header += f",{name}"
            columns.append(arr.reshape(-1, 1))
        else:
            comp = 0
            while comp < arr.shape[1]:
                header += f",{name}_{comp}"
                comp += 1
            columns.append(arr.reshape(len(pts), -1))
    data = np.hstack([c.reshape(len(pts), -1) for c in columns])
    np.savetxt(str(target), data, delimiter=",", header=header, comments="")
    return str(target)


def save_project(
    dataset: CFDDataset,
    path: str | Path,
    *,
    engine: Any = None,
    compression: str | None = "gzip",
) -> dict[str, str]:
    """데이터셋을 ``.ntwin`` (VTKHDF 기반)으로 저장하고, 엔진이 있으면 동봉한다.

    Returns:
        저장된 경로들 (``project``, 선택적으로 ``engine``).
    """
    from naviertwin.core.export.ntwin_format import save_dataset

    target = _ensure_parent(path)
    if target.suffix.lower() != ".ntwin":
        target = target.with_suffix(".ntwin")
    # .ntwin(VTKHDF) 은 UnstructuredGrid 토폴로지를 기대한다. ImageData/
    # StructuredGrid 등은 cast 해야 round-trip(재로드)이 보장된다.
    save_dataset(_as_unstructured(dataset), target, compression=compression)
    paths = {"project": str(target)}
    if engine is not None and hasattr(engine, "save"):
        engine_path = target.with_suffix(".engine.pkl")
        engine.save(engine_path)
        paths["engine"] = str(engine_path)
    return paths


def save_engine(engine: Any, path: str | Path) -> str:
    """학습된 TwinEngine 을 pickle 로 저장한다."""
    if engine is None or not hasattr(engine, "save"):
        raise ValueError("저장할 TwinEngine 이 없습니다. 먼저 Model 단계에서 학습하세요.")
    target = _ensure_parent(path)
    if target.suffix.lower() != ".pkl":
        target = target.with_suffix(".pkl")
    engine.save(target)
    return str(target)


def export_report(
    dataset: CFDDataset,
    path: str | Path,
    *,
    project: str = "NavierTwin Web",
    summary: str = "",
    model_info: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    figures: list[dict[str, str]] | None = None,
) -> str:
    """데이터셋/모델 요약을 HTML 보고서로 저장한다 (Jinja2)."""
    from naviertwin.core.report.generator import ReportGenerator

    target = _ensure_parent(path)
    if target.suffix.lower() not in {".html", ".htm"}:
        target = target.with_suffix(".html")
    info = dataset_info(dataset)
    base_model_info = {
        "points": info["points"],
        "cells": info["cells"],
        "time_steps": info["time_steps"],
        "fields": ", ".join(info["fields"]) or "-",
        "source": info["source"] or "-",
    }
    if model_info:
        base_model_info.update({k: str(v) for k, v in model_info.items()})
    data: dict[str, Any] = {
        "project": project,
        "summary": summary or "NavierTwin 웹 워크플로우 산출물 요약 보고서.",
        "model_info": base_model_info,
        "metrics": metrics or {},
        "figures": figures or [],
    }
    out = ReportGenerator().render_html(data, target)
    return str(out)


__all__ = [
    "DERIVED_EXTRA",
    "DERIVED_PREFIXES",
    "REDUCERS",
    "RESULT_FIELD",
    "SURROGATES",
    "VORTEX_METHODS",
    "attach_pod_mode",
    "attach_prediction",
    "build_twin",
    "compare_models",
    "compute_fft_psd",
    "dataset_info",
    "estimate_dt",
    "export_field_csv",
    "export_report",
    "export_vtk",
    "is_derived_field",
    "load_dataset",
    "load_project",
    "make_demo_dataset",
    "pod_mode_field",
    "predict_twin",
    "run_pod",
    "run_vortex_analysis",
    "save_engine",
    "save_project",
    "snapshot_dataset",
    "sync_timestep_to_mesh",
]
