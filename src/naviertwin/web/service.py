"""웹 워크플로우 오케스트레이션 (Qt/GL 비의존).

trame 앱과 독립적으로 테스트 가능한 순수 서비스 계층. Import → Analyze →
Reduce → Twin MVP 워크플로우를 ``core`` 모듈 위에 얇게 조립한다.

함수는 모두 ``CFDDataset`` 또는 학습된 모델 객체를 받고/돌려주며, trame
state 나 PyVista 렌더 컨텍스트(GL)에 의존하지 않는다. 따라서 headless CI
에서도 그대로 단위 테스트할 수 있다.
"""

from __future__ import annotations

from collections.abc import Sequence
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
DERIVED_PREFIXES = ("pod_mode", "twin_")  # twin_prediction + 다중출력 twin_<field>

# sdf = 재샘플이 만든 형상 부호거리 — 뷰어로 보되 학습 대상(출력)은 아니다.
DERIVED_EXTRA = ("vorticity", "sdf")


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


def list_directory(path: str | Path | None = None) -> dict[str, Any]:
    """파일 브라우저용 — 디렉토리를 나열한다 (하위 디렉토리 + 로드 가능 파일).

    로드 가능 파일 = ReaderFactory 등록 확장자 + OpenFOAM(.foam) + ``.ntwin``.
    숨김 항목은 제외. 경로가 없거나 없으면 홈으로 폴백.

    Returns:
        ``cwd``, ``parent``(없으면 None), ``entries``(name/path/is_dir 정렬 목록),
        ``home`` 을 담은 dict.
    """
    from naviertwin.core.cfd_reader import ReaderFactory

    exts = set(ReaderFactory.registered_extensions()) | {".foam", ".openfoam", ".ntwin"}
    home = Path.home()
    base = Path(path).expanduser() if path else home
    if base.is_file():
        base = base.parent
    if not base.is_dir():
        base = home

    entries: list[dict[str, Any]] = []
    try:
        children = sorted(base.iterdir(), key=lambda q: (not q.is_dir(), q.name.lower()))
    except (PermissionError, OSError):
        children = []
    for child in children:
        if child.name.startswith("."):
            continue
        is_dir = child.is_dir()
        if not is_dir and child.suffix.lower() not in exts:
            continue
        entries.append({"name": child.name, "path": str(child), "is_dir": is_dir})

    parent = str(base.parent) if base.parent != base else None
    return {"cwd": str(base), "parent": parent, "entries": entries, "home": str(home)}


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


def meshes_are_identical(datasets: Sequence[CFDDataset]) -> bool:
    """케이스들이 같은 메쉬(좌표/토폴로지)를 공유하는지 판정한다."""
    cases = list(datasets)
    if len(cases) < 2:
        return True
    ref = np.asarray(cases[0].mesh.points, dtype=np.float64)
    for case in cases[1:]:
        points = np.asarray(case.mesh.points, dtype=np.float64)
        if points.shape != ref.shape or not np.allclose(points, ref):
            return False
    return True


def resample_cases_to_common_grid(
    datasets: Sequence[CFDDataset],
    *,
    resolution: int = 32,
    fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    """형상이 다른 케이스들을 공통 배경 격자로 재샘플한다 (형상 가변 지원).

    케이스마다 메쉬가 다르면(형상 파라미터가 바뀌는 외부유동 등) 스냅샷 행렬도
    신경장 학습도 불가능하다 — 모든 케이스를 **같은 균일 격자**에 올려 문제를
    유형 B(파라미터 스윕)로 되돌린다. 이는 형상 가변 ROM 의 표준 전처리다.

    각 격자점에 부호거리 ``sdf`` 를 함께 계산한다 (+ = 유체, − = 고체/도메인
    밖). VTK 표면 추출 대신 ``vtkValidPointMask`` 의 거리변환(EDT)으로 구하므로
    2D/3D·열린/닫힌 형상 어디서나 동작한다. 고체 내부 격자점의 물리량은 0 으로
    채워진다(``sample`` 기본) — sdf 가 그 영역을 식별해준다.

    Args:
        datasets: 케이스 목록 (메쉬가 서로 달라도 됨).
        resolution: 최장 축 방향 격자 분할 수.
        fields: 옮길 field 목록. None 이면 첫 케이스의 field 전체.

    Returns:
        ``datasets``(공통 격자 위 케이스 목록), ``grid_summary``,
        ``fields``, ``resolution`` 을 담은 dict.

    Raises:
        ValueError: 케이스가 없거나 격자를 만들 수 없는 경우.
    """
    from scipy.ndimage import distance_transform_edt

    from naviertwin.core.cfd_reader.base import CFDDataset as _CFDDataset

    cases = list(datasets)
    if not cases:
        raise ValueError("재샘플할 케이스가 없습니다.")

    # 모든 케이스를 덮는 합집합 바운딩 박스 위에 등방 격자를 만든다.
    box = np.asarray([case.mesh.bounds for case in cases], dtype=np.float64)
    union = [
        box[:, 0].min(), box[:, 1].max(),
        box[:, 2].min(), box[:, 3].max(),
        box[:, 4].min(), box[:, 5].max(),
    ]
    grid = _uniform_grid_over(union, resolution)
    dims = list(grid.dimensions)
    spacing = list(grid.spacing)

    names = (
        [str(f) for f in fields]
        if fields
        else [n for n in (getattr(cases[0], "field_names", []) or [])]
    )

    # EDT 는 격자 인덱스 공간에서 계산 → 축별 물리 간격으로 환산.
    # 배열은 VTK point 순서(z, y, x)로 reshape 하므로 spacing 도 뒤집는다.
    # 두께 0 축(2D)은 거리가 항상 0 이라 값이 무의미하지만 길이는 맞춰야 한다.
    edt_sampling = list(reversed(spacing))

    out: list[CFDDataset] = []
    for case in cases:
        sampled = grid.sample(case.mesh)
        mask = np.asarray(
            sampled.point_data.get("vtkValidPointMask", np.ones(grid.n_points)),
            dtype=np.int8,
        )
        shaped = mask.reshape(tuple(reversed(dims)))  # VTK point order = z,y,x
        # 부호거리: 유체 안쪽은 +(고체까지 거리), 고체 안쪽은 −(유체까지 거리).
        d_fluid = distance_transform_edt(shaped, sampling=edt_sampling)
        d_solid = distance_transform_edt(1 - shaped, sampling=edt_sampling)
        sdf = (np.asarray(d_fluid) - np.asarray(d_solid)).reshape(-1)

        mesh = grid.copy(deep=True)
        for name in names:
            if name in sampled.point_data:
                mesh.point_data[name] = np.asarray(sampled.point_data[name])
        mesh.point_data["sdf"] = sdf.astype(np.float64)
        out.append(
            _CFDDataset(
                mesh=mesh,
                time_steps=[float((getattr(case, "time_steps", None) or [0.0])[-1])],
                field_names=[*names, "sdf"],
                metadata={
                    "source": "resampled_common_grid",
                    "resample_resolution": int(resolution),
                },
            )
        )

    summary = (
        f"공통 격자 {'×'.join(str(d) for d in dims)} ({grid.n_points:,}점)"
        f" · 재샘플 {len(cases)}케이스 · sdf 포함"
    )
    return {
        "datasets": out,
        "grid_summary": summary,
        "fields": [*names, "sdf"],
        "resolution": int(resolution),
    }


def _uniform_grid_over(bounds: Sequence[float], resolution: int) -> Any:
    """바운딩 박스를 덮는 등방 균일 격자를 만든다 (두께 0 축은 1칸 → 2D 자동)."""
    import pyvista as pv

    box = np.asarray(bounds, dtype=np.float64)
    lo = box[[0, 2, 4]]
    hi = box[[1, 3, 5]]
    span = hi - lo
    pad = np.where(span > 0, span * 0.02, 0.0)
    lo, hi = lo - pad, hi + pad
    span = hi - lo
    longest = float(span.max())
    if longest <= 0:
        raise ValueError("바운딩 박스가 비어 있어 격자를 만들 수 없습니다.")
    step = longest / max(2, int(resolution))
    dims = [max(1, int(round(s / step)) + 1) if s > 0 else 1 for s in span]
    spacing = [step if s > 0 else 1.0 for s in span]
    return pv.ImageData(
        dimensions=tuple(dims), spacing=tuple(spacing), origin=tuple(lo)
    )


def coarsen_dataset(
    dataset: CFDDataset,
    *,
    resolution: int = 48,
    fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    """대용량 메쉬를 성긴 균일 격자로 재샘플해 메모리·학습 비용을 줄인다.

    왜 필요한가: :meth:`CFDDataset.extract_field_snapshots` 는 (n_features,
    n_steps) **전체 행렬을 메모리에 올린다** — 1천만 점 × 100 스텝이면 float64
    로 8GB 다. Physics AI 의 ``max_train_points`` 는 이 행렬을 만든 **뒤**
    부분추출하므로 피크 메모리를 줄이지 못한다. 학습 전에 해상도를 낮추는 것이
    유일하게 효과적인 대응이다.

    셀 몇 개를 하나로 합치는 게 아니라 **성긴 격자점에서 원본을 보간 샘플**한다
    (VTK ``sample``) — 결과는 같은 물리 도메인의 저해상도 표현이며 시계열도
    보존한다. 손실 압축이므로 급격한 구배/불연속은 뭉개진다.

    Args:
        dataset: 원본 데이터셋 (시계열 포함 가능).
        resolution: 최장 축 방향 격자 분할 수. 작을수록 가볍고 거칠다.
        fields: 옮길 field. None 이면 파생이 아닌 원본 물리량 전체.

    Returns:
        ``dataset``(성긴 격자 데이터셋), ``summary``, ``points_before``/
        ``points_after``, ``ratio``(축소 배율) 를 담은 dict.

    Raises:
        ValueError: 옮길 field 가 없거나 격자를 만들 수 없는 경우.
    """
    from naviertwin.core.cfd_reader.base import CFDDataset as _CFDDataset

    names = (
        [str(f) for f in fields]
        if fields
        else [
            n
            for n in (getattr(dataset, "field_names", []) or [])
            if not is_derived_field(n)
        ]
    )
    if not names:
        raise ValueError("성기게 만들 원본 물리량 field 가 없습니다.")

    grid = _uniform_grid_over(dataset.mesh.bounds, resolution)
    points_before = int(getattr(dataset, "n_points", 0))
    n_steps = max(1, int(getattr(dataset, "n_time_steps", 1)))
    meta = getattr(dataset, "metadata", {}) or {}
    series_in = meta.get("time_series_fields")
    locations = meta.get("time_series_locations") or {}

    # 원본 메쉬를 복사해 타임스텝별 값을 얹어가며 샘플한다 (원본 불변).
    source = dataset.mesh.copy(deep=True)
    series_out: dict[str, list[Any]] = {name: [] for name in names}
    for step in range(n_steps):
        if isinstance(series_in, dict):
            for name in names:
                raw = series_in.get(name)
                if raw is None:
                    continue
                values = np.asarray(raw)
                current = values[step] if values.shape[0] == n_steps else values
                location = str(locations.get(name, "point"))
                target = source.cell_data if location == "cell" else source.point_data
                target[name] = np.asarray(current)
        sampled = grid.sample(source)
        for name in names:
            if name in sampled.point_data:
                series_out[name].append(np.asarray(sampled.point_data[name]))

    stacked = {
        name: np.stack(values, axis=0) for name, values in series_out.items() if values
    }
    if not stacked:
        raise ValueError("성긴 격자로 옮겨진 field 가 없습니다 (도메인 불일치?).")

    mesh = grid.copy(deep=True)
    for name, values in stacked.items():
        mesh.point_data[name] = np.asarray(values[0])

    times = [float(t) for t in (getattr(dataset, "time_steps", None) or [0.0])]
    out = _CFDDataset(
        mesh=mesh,
        time_steps=times[:n_steps] if len(times) >= n_steps else times,
        field_names=list(stacked),
        metadata={
            "source": f"{meta.get('source', 'unknown')}+coarsened",
            "coarsen_resolution": int(resolution),
            "coarsen_points_before": points_before,
            "time_series_fields": stacked,
            "time_series_locations": {name: "point" for name in stacked},
        },
    )
    points_after = int(out.n_points)
    ratio = (points_before / points_after) if points_after else float("nan")
    dims = mesh.dimensions
    summary = (
        f"{points_before:,}점 → {points_after:,}점 "
        f"({'×'.join(str(d) for d in dims)} 격자, {ratio:.1f}× 축소)"
    )
    logger.info("데이터셋 해상도 낮춤: %s", summary)
    return {
        "dataset": out,
        "summary": summary,
        "points_before": points_before,
        "points_after": points_after,
        "ratio": float(ratio),
    }


def load_case_set(
    directory: str | Path,
    *,
    params_path: str | Path | None = None,
    param_columns: Sequence[str] | None = None,
    resample: bool | str = "auto",
    resolution: int = 32,
) -> dict[str, Any]:
    """폴더의 CFD 파일들을 "정상 파라미터 스윕" 케이스 세트로 로드한다.

    파일 1개 = 운전조건 1개(케이스). 같은 폴더(또는 ``params_path``)의 CSV 가
    케이스별 입력 파라미터 표를 준다 — 행 = 케이스, 열 = 파라미터이며 **행
    순서는 파일명 정렬 순서와 일치**해야 한다. CSV 가 없으면 케이스 인덱스를
    유일한 입력으로 폴백한다 (실제 운전조건 학습에는 CSV 필요).

    각 케이스는 마지막 타임스텝(정상 해석의 수렴 해)만 단일 스냅샷으로
    materialize 한다 — 케이스당 정확히 1 스냅샷이라 파라미터 표 행 수와 맞는다.

    Args:
        directory: 케이스 파일들이 있는 폴더.
        params_path: 파라미터 CSV 경로 (None 이면 폴더 안에서 자동 탐색).
        param_columns: 쓸 CSV 열 (None 이면 숫자 열 전부).
        resample: 형상 가변 지원 — ``"auto"``(기본)면 케이스 메쉬가 서로 다를
            때만 공통 격자로 재샘플, ``True`` 면 항상, ``False`` 면 안 함.
        resolution: 재샘플 격자의 최장 축 분할 수.

    Returns:
        ``datasets``(단일 스냅샷 CFDDataset 목록), ``params`` (N, k),
        ``param_names``, ``case_names``, ``params_source``, ``resampled``,
        ``grid_summary`` 를 담은 dict.

    Raises:
        FileNotFoundError: 폴더가 없는 경우.
        ValueError: 로드 가능한 케이스가 2개 미만이거나 CSV 행 수가 안 맞거나,
            메쉬가 다른데 ``resample=False`` 인 경우.
    """
    from naviertwin.core.cfd_reader import ReaderFactory
    from naviertwin.core.physnemo.cfd_field_model import load_parameter_table

    base = Path(directory).expanduser()
    if not base.is_dir():
        raise FileNotFoundError(f"케이스 폴더를 찾을 수 없습니다: {base}")

    exts = set(ReaderFactory.registered_extensions())
    files = sorted(
        (
            p
            for p in base.iterdir()
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in exts
        ),
        key=lambda p: p.name.lower(),
    )
    if len(files) < 2:
        raise ValueError(
            f"케이스 세트에는 CFD 파일이 2개 이상 필요합니다 (찾음: {len(files)}개). "
            "파일 하나가 운전조건 하나여야 합니다."
        )

    datasets: list[CFDDataset] = []
    for target in files:
        loaded = load_dataset(target)
        n_steps = max(1, int(getattr(loaded, "n_time_steps", 1)))
        # 정상 해석 = 마지막 스텝이 수렴 해.
        datasets.append(snapshot_dataset(loaded, n_steps - 1))

    csv: Path | None = Path(params_path).expanduser() if params_path else None
    if csv is None:
        candidates = sorted(p for p in base.glob("*.csv") if not p.name.startswith("."))
        csv = candidates[0] if len(candidates) == 1 else None

    if csv is not None:
        values, names = load_parameter_table(
            csv, columns=param_columns, expected_rows=len(datasets)
        )
        source = f"CSV: {csv.name}"
    else:
        values = np.arange(len(datasets), dtype=np.float64).reshape(-1, 1)
        names = ["case_index"]
        source = "case_index (파라미터 CSV 없음)"
        logger.warning("파라미터 CSV 가 없어 case index 를 입력으로 사용합니다: %s", base)

    # 형상 가변 지원: 메쉬가 서로 다르면 공통 격자로 재샘플해야 스냅샷 행렬/
    # 신경장 학습이 성립한다 (근거: model-taxonomy-plan.md §15, M4a).
    identical = meshes_are_identical(datasets)
    grid_summary = ""
    resampled = False
    if resample is True or (resample == "auto" and not identical):
        result = resample_cases_to_common_grid(datasets, resolution=resolution)
        datasets = result["datasets"]
        grid_summary = str(result["grid_summary"])
        resampled = True
        logger.info("케이스 메쉬가 달라 공통 격자로 재샘플: %s", grid_summary)
    elif not identical:
        raise ValueError(
            "케이스마다 메쉬가 다릅니다 — 형상 가변 케이스는 공통 격자 재샘플이 "
            "필요합니다 (resample='auto' 또는 True)."
        )

    return {
        "datasets": datasets,
        "params": np.asarray(values, dtype=np.float64),
        "param_names": list(names),
        "case_names": [f.name for f in files],
        "params_source": source,
        "resampled": resampled,
        "grid_summary": grid_summary,
    }


def build_twin_from_cases(
    datasets: Sequence[CFDDataset],
    field: str,
    n_modes: int,
    params: np.ndarray,
    *,
    param_names: Sequence[str] | None = None,
    reducer: str = "pod",
    surrogate: str = "rbf",
) -> dict[str, Any]:
    """케이스 세트에서 (운전조건 → 필드) ROM 트윈을 학습한다 (정상 파라미터 스윕).

    :func:`build_twin` 의 시계열 버전과 달리 입력이 시간이 아니라 k 차원 운전
    조건 벡터다 — 케이스마다 스냅샷 1장, 파라미터 표 행 1개.

    Raises:
        ValueError: 케이스가 2개 미만, 파라미터 행 수 불일치, 메쉬 크기 불일치,
            또는 미지원 reducer/surrogate 인 경우.
    """
    from naviertwin.core.digital_twin.twin_engine import TwinEngine

    cases = list(datasets)
    if len(cases) < 2:
        raise ValueError(f"트윈 학습에는 케이스가 2개 이상 필요합니다. 현재: {len(cases)}")
    if reducer not in REDUCERS:
        raise ValueError(f"지원하지 않는 reducer: {reducer}")
    if surrogate not in SURROGATES:
        raise ValueError(f"지원하지 않는 surrogate: {surrogate}")

    params_arr = np.asarray(params, dtype=np.float64)
    if params_arr.ndim == 1:
        params_arr = params_arr.reshape(-1, 1)
    if params_arr.shape[0] != len(cases):
        raise ValueError(
            f"파라미터 행 수({params_arr.shape[0]})와 케이스 수({len(cases)})가 다릅니다."
        )

    columns = [
        np.asarray(case.extract_field_snapshots(field), dtype=np.float64)[:, -1]
        for case in cases
    ]
    n_features = int(columns[0].shape[0])
    if any(int(column.shape[0]) != n_features for column in columns):
        raise ValueError(
            "케이스마다 메쉬 크기가 달라 ROM 학습이 불가능합니다 (동일 메쉬 필요). "
            "형상이 다른 케이스는 기하 인지 연산자(GNN/GINO)가 필요합니다."
        )
    snapshots = np.stack(columns, axis=1)  # (n_features, n_cases)

    names = (
        list(param_names)
        if param_names
        else [f"param_{i}" for i in range(params_arr.shape[1])]
    )
    n_modes = max(1, min(int(n_modes), len(cases), n_features))
    engine = TwinEngine(reducer_type=reducer, surrogate_type=surrogate, n_modes=n_modes)
    engine.fit(snapshots, params_arr)

    mins = [float(v) for v in params_arr.min(axis=0)]
    maxs = [float(v) for v in params_arr.max(axis=0)]
    engine.training_metadata = {
        "field_name": field,
        "n_modes": n_modes,
        "reducer": reducer,
        "surrogate": surrogate,
        "problem_type": "steady_sweep",
        "param_names": names,
        "param_mins": mins,
        "param_maxs": maxs,
    }
    return {
        "engine": engine,
        "field": field,
        "n_modes": n_modes,
        "reducer": reducer,
        "surrogate": surrogate,
        "n_cases": len(cases),
        "param_names": names,
        "param_mins": mins,
        "param_maxs": maxs,
    }


def build_physics_ai_twin_from_cases(
    datasets: Sequence[CFDDataset],
    field: str | Sequence[str],
    params: np.ndarray,
    *,
    param_names: Sequence[str] | None = None,
    hidden: int = 32,
    max_epochs: int = 150,
    max_train_points: int = 20_000,
) -> dict[str, Any]:
    """케이스 세트에서 (좌표 + 운전조건 → 필드) Physics AI 트윈을 학습한다.

    :func:`build_physics_ai_twin` 의 다중 케이스 버전 — 입력이 (x, y, z, μ₁..μₖ).
    다중 출력 필드도 그대로 지원한다.

    Raises:
        ValueError: 케이스 2개 미만, 필드 미선택, 또는 파라미터 행 수 불일치.
    """
    from naviertwin.core.digital_twin.physics_ai_engine import PhysicsAITwinEngine
    from naviertwin.core.physnemo.cfd_field_model import PhysicsNeMoCFDFieldModel

    fields = [field] if isinstance(field, str) else [str(f) for f in field if str(f).strip()]
    if not fields:
        raise ValueError("학습할 출력 필드를 최소 1개 선택하세요.")
    cases = list(datasets)
    if len(cases) < 2:
        raise ValueError(f"트윈 학습에는 케이스가 2개 이상 필요합니다. 현재: {len(cases)}")

    params_arr = np.asarray(params, dtype=np.float64)
    if params_arr.ndim == 1:
        params_arr = params_arr.reshape(-1, 1)
    if params_arr.shape[0] != len(cases):
        raise ValueError(
            f"파라미터 행 수({params_arr.shape[0]})와 케이스 수({len(cases)})가 다릅니다."
        )

    model = PhysicsNeMoCFDFieldModel.from_datasets(
        cases,
        field_names=fields,
        params=params_arr,
        parameter_names=list(param_names) if param_names else None,
        hidden=hidden,
        max_epochs=max_epochs,
        max_train_points=max_train_points,
    )
    engine = PhysicsAITwinEngine.from_fitted_model(model, model_type="physicsnemo_cfd_field")
    meta = model.training_metadata
    names = list(meta.get("parameter_names", []) or [])
    mins = [float(v) for v in meta["parameter_min"]]
    maxs = [float(v) for v in meta["parameter_max"]]
    engine.training_metadata.update(
        {
            "field_name": ",".join(fields),
            "field_names": list(fields),
            "reducer": "direct_physics_ai",
            "surrogate": "physicsnemo_cfd_field",
            "problem_type": "steady_sweep",
            "param_names": names,
            "param_mins": mins,
            "param_maxs": maxs,
        }
    )
    return {
        "engine": engine,
        "field": ",".join(fields),
        "fields": list(fields),
        "n_cases": len(cases),
        "param_names": names,
        "param_mins": mins,
        "param_maxs": maxs,
        "validation_metrics": dict(meta.get("validation_metrics", {})),
        "per_field_metrics": dict(meta.get("per_field_metrics", {})),
        "train_losses": list(model.train_losses_),
    }


def make_demo_dataset(
    nx: int = 48,
    ny: int = 48,
    n_steps: int = 12,
    kind: str = "advecting",
    nu: float = 0.05,
    oscillation_hz: float = 1.5,
    oscillation_amp: float = 0.25,
    drift_cycles_x: float = 1.5,
    drift_cycles_y: float = 0.5,
) -> CFDDataset:
    """합성 시계열 데모 데이터셋을 생성한다 (파이프라인 즉시 체험용).

    ``kind`` 로 난이도가 다른 두 종류를 만든다:

    - ``"advecting"``: 이류 Taylor–Green 와류 — 부드럽고(연속) 공간 주기적.
      진폭이 시간 진동(FFT 검출용)하고 셀이 대각선으로 흐른다. POD/FNO 가
      쉽게 학습(빠른 특이값 감쇠). 단위 테스트가 쓰는 기본값.
    - ``"filament"``: **소용돌이(차등 회전) 유동이 날카로운 불연속 스칼라
      덩어리(슬롯 원반·사각형·바)를 나선 필라멘트로 감아올린다.** 경계가
      뚜렷한 불연속 + 국소적(비주기) + 필라멘트화로 POD 특이값이 느리게
      감쇠 → 학습이 어렵다. 데모 버튼(:meth:`load_demo`)이 쓰는 값.

    Args:
        nx/ny: 격자점 수.
        n_steps: 타임스텝 수.
        kind: ``"advecting"`` | ``"filament"``.
        nu: 감쇠율(advecting).
        oscillation_hz/oscillation_amp: 진폭 진동(advecting, FFT 검출용).
        drift_cycles_x/drift_cycles_y: 이류 주기 수(advecting).

    Returns:
        시계열 ``U``/``p`` 를 metadata 에 담은 ``CFDDataset``.

    Raises:
        ValueError: 지원하지 않는 ``kind``.
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

    if kind not in {"advecting", "filament"}:
        raise ValueError(f"지원하지 않는 demo kind: {kind}. 지원: ['advecting', 'filament']")

    times = np.linspace(0.0, 2.0, n_steps)
    t_span = float(times[-1]) if times[-1] > 0 else 1.0
    velocity = np.zeros((n_steps, n_points, 3), dtype=np.float64)
    pressure = np.zeros((n_steps, n_points), dtype=np.float64)

    if kind == "advecting":
        step = 0
        while step < n_steps:
            t = float(times[step])
            # 단조 감쇠 포락선 + 약한 시간 진동(FFT/PSD 가 검출할 주파수).
            envelope = np.exp(-2.0 * nu * t) * (
                1.0 + oscillation_amp * np.sin(2.0 * np.pi * oscillation_hz * t)
            )
            # 이류: 정규화 시간 s 만큼 공간 위상을 이동 → 셀이 대각선으로 흐른다.
            s = t / t_span
            px = x_flat - two_pi * drift_cycles_x * s
            py = y_flat - two_pi * drift_cycles_y * s
            velocity[step, :, 0] = np.cos(px) * np.sin(py) * envelope
            velocity[step, :, 1] = -np.sin(px) * np.cos(py) * envelope
            pressure[step] = -0.25 * (np.cos(2.0 * px) + np.cos(2.0 * py)) * (envelope ** 2)
            step += 1
        source = "demo_taylor_green"
        description = "Advecting Taylor–Green vortex (smooth, periodic)"
    else:  # "filament" — 소용돌이 유동이 불연속 스칼라를 나선으로 감아올림
        # 좌표를 [0,1] 로 정규화 (도메인은 경계가 있는 비주기 사각형).
        xn = x_flat / two_pi
        yn = y_flat / two_pi
        dx0 = xn - 0.5
        dy0 = yn - 0.5
        r = np.sqrt(dx0 * dx0 + dy0 * dy0)
        phi = np.arctan2(dy0, dx0)
        # 차등 회전: 전체 구간 동안 감는 각(중심 빠름 → 나선). 반경 불변.
        winding = 5.5 * np.exp(-r / 0.30)
        base_ang = winding / t_span
        # 비정상 유동: 각속도가 시간에 따라 진화 → U 의 크기뿐 아니라 공간 패턴도 변한다.
        #   방위류(순수 회전)라 유체 입자의 반경은 보존되고 각만 이동 → 역추적이 정확.
        #   순간 배율   R(s,r) = 0.35 + 1.3 s + 0.8 s·G(r)
        #   누적 배율   W(s,r) = ∫R ds = 0.35 s + 0.65 s² + 0.4 s²·G(r)   (R 의 정확한 적분)
        #   G(r) = exp(-((r-0.25)/0.12)²): r≈0.25 에 시간이 지날수록 커지는 '빠른 회전 링'.
        g_ring = np.exp(-((r - 0.25) / 0.12) ** 2)
        step = 0
        while step < n_steps:
            s = float(times[step]) / t_span
            rate = 0.35 + 1.3 * s + 0.8 * s * g_ring
            wound = 0.35 * s + 0.65 * s * s + 0.4 * s * s * g_ring
            ang_speed = base_ang * rate  # 시간·반경 가변 각속도
            phi0 = phi - winding * wound  # 역추적 각 (경계 날카로움 보존)
            x0 = 0.5 + r * np.cos(phi0)
            y0 = 0.5 + r * np.sin(phi0)
            pressure[step] = _demo_sharp_blobs(x0, y0)
            velocity[step, :, 0] = -ang_speed * dy0
            velocity[step, :, 1] = ang_speed * dx0
            step += 1
        source = "demo_swirl_filament"
        description = "Swirling advection of sharp scalar blobs (discontinuous, non-periodic)"

    image.point_data["U"] = velocity[0]
    image.point_data["p"] = pressure[0]

    return CFDDataset(
        mesh=image,
        time_steps=list(map(float, times)),
        field_names=["U", "p"],
        metadata={
            "source": source,
            "description": description,
            "time_series_fields": {"U": velocity, "p": pressure},
            "time_series_locations": {"U": "point", "p": "point"},
        },
    )


def _demo_sharp_blobs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """[0,1]² 위 날카로운 불연속 스칼라 초기장 (슬롯 원반 + 사각형 + 바)."""
    val = np.zeros_like(x)
    # Zalesak 슬롯 원반 (비볼록 → 어려움)
    disk = np.sqrt((x - 0.34) ** 2 + (y - 0.64) ** 2) < 0.17
    slot = (np.abs(x - 0.34) < 0.035) & (y < 0.66)
    val[disk & ~slot] = 1.0
    # 사각형 패치
    square = (x > 0.56) & (x < 0.80) & (y > 0.20) & (y < 0.44)
    val[square] = 0.65
    # 얇은 고대비 바
    bar = (x > 0.18) & (x < 0.72) & (np.abs(y - 0.28) < 0.018)
    val[bar] = -0.55
    return val


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


def predict_twin(engine: Any, param_value: float | Sequence[float]) -> np.ndarray:
    """학습된 엔진으로 파라미터 지점에서 필드를 예측한다.

    ``param_value`` 는 시계열 트윈이면 단일 시간값, 케이스 세트(파라미터 스윕)
    트윈이면 k 차원 운전조건 벡터다.
    """
    params = np.asarray(param_value, dtype=np.float64).reshape(-1)
    prediction = np.asarray(engine.predict(params), dtype=np.float64).reshape(-1)
    return prediction


def recommend_method(dataset: CFDDataset) -> dict[str, str]:
    """로드된 데이터 특성으로 ④Model 학습 방식을 추천한다.

    문헌 통례 기반 휴리스틱:
      - 비정형/스냅샷 적음 → ROM(축소+보간)이 표준.
      - 균일 격자 + 스냅샷 많음 → 신경 연산자(FNO)도 후보.
      - 타임스텝 1개 → 학습 불가(예측 전용).

    Returns:
        ``method`` ("rom" | "operator" | "none") 과 한국어 ``reason``.
    """
    n_steps = int(getattr(dataset, "n_time_steps", 0))
    n_points = int(getattr(dataset, "n_points", 0))
    if n_steps < 2:
        return {
            "method": "none",
            "reason": "타임스텝이 1개뿐이라 (시간→필드) 학습이 불가능합니다. "
            "저장된 트윈이 있다면 ⑤Twin 예측만 가능합니다.",
        }
    try:
        import pyvista as pv

        uniform = isinstance(dataset.mesh, pv.ImageData)
    except Exception:  # noqa: BLE001
        uniform = False
    prefix = f"현재 데이터: {n_steps} 타임스텝 · {n_points:,} 포인트"
    if uniform and n_steps >= 100:
        return {
            "method": "operator",
            "reason": f"{prefix} · 균일 격자 — 스냅샷이 많아 신경 연산자(FNO)도 "
            "고려할 만합니다. ROM 은 여전히 안전한 기본값입니다.",
        }
    if n_steps < 50:
        return {
            "method": "rom",
            "reason": f"{prefix} — 스냅샷이 적어 축소+보간(ROM)이 가장 안정적입니다. "
            "신경 연산자 학습에는 샘플이 부족합니다.",
        }
    return {
        "method": "rom",
        "reason": f"{prefix} — 단일 시계열은 ROM 권장. 물리 제약이 필요하거나 "
        "데이터가 희소하면 Physics AI 가 대안입니다.",
    }


def build_physics_ai_twin(
    dataset: CFDDataset,
    field: str | Sequence[str],
    *,
    input_fields: Sequence[str] | None = None,
    hidden: int = 32,
    max_epochs: int = 150,
    max_train_points: int = 20_000,
) -> dict[str, Any]:
    """NVIDIA PhysicsNeMo 스타일 직접 필드 예측 모델을 학습한다.

    POD reducer 없이 (좌표 + 시간) → 필드값을 곧장 매핑하는 PyTorch MLP
    (:class:`PhysicsNeMoCFDFieldModel`)를 학습해 :class:`PhysicsAITwinEngine`
    으로 감싼다. 이 엔진은 ``predict()`` 만 노출하는 ``TwinEngine`` 과 동일한
    덕타이핑 계약을 따르므로 :func:`predict_twin` / :func:`attach_prediction`
    / :func:`save_engine` / ``app._restore_engine`` 을 그대로 재사용한다 —
    ``physicsnemo`` 패키지 설치 여부와 무관하게(순수 torch 로) 학습된다.

    Args:
        dataset: 시계열 CFD 데이터셋.
        field: 학습 대상 물리량 field — 문자열 하나 또는 목록(다중 출력).
            다중 출력이면 한 신경망이 모든 필드를 동시에 학습한다 (예측은
            :func:`split_multi_prediction` 으로 필드별로 분해).
        input_fields: **입력**으로 쓸 다른 field 목록 (예: U → p). 주면 모델이
            (좌표 + 입력장 + 시간) → 출력 의 field-to-field 연산자가 된다.
            ③Twin 예측은 학습 입력장을 시간 보간해 자동으로 채운다.
        hidden: 은닉층 폭.
        max_epochs: 학습 epoch 수.
        max_train_points: (좌표×스냅샷) 학습 샘플 상한 — 초과 시 무작위 부분추출.

    Returns:
        ``engine`` (PhysicsAITwinEngine), ``fields`` (학습된 필드 목록),
        ``param_min``/``param_max``, 검증 지표(``validation_metrics``), 손실
        곡선(``train_losses``) 등을 담은 dict.

    Raises:
        ValueError: 타임스텝이 2개 미만이거나 필드 목록이 빈 경우.
    """
    from naviertwin.core.digital_twin.physics_ai_engine import PhysicsAITwinEngine
    from naviertwin.core.physnemo.cfd_field_model import PhysicsNeMoCFDFieldModel

    fields = [field] if isinstance(field, str) else [str(f) for f in field if str(f).strip()]
    if not fields:
        raise ValueError("학습할 출력 필드를 최소 1개 선택하세요.")

    n_steps = int(getattr(dataset, "n_time_steps", 0))
    if n_steps < 2:
        raise ValueError(
            f"Physics AI 학습에는 2개 이상의 타임스텝이 필요합니다. 현재: {n_steps}"
        )

    inputs = [str(f) for f in (input_fields or []) if str(f).strip()]
    model = PhysicsNeMoCFDFieldModel.from_datasets(
        [dataset],
        field_names=fields,
        input_field_names=inputs or None,
        hidden=hidden,
        max_epochs=max_epochs,
        max_train_points=max_train_points,
    )
    engine = PhysicsAITwinEngine.from_fitted_model(model, model_type="physicsnemo_cfd_field")
    meta = model.training_metadata
    param_min = float(meta["parameter_min"][0])
    param_max = float(meta["parameter_max"][0])
    engine.training_metadata.update(
        {
            "field_name": ",".join(fields),
            "field_names": list(fields),
            "input_field_names": list(inputs),
            "reducer": "direct_physics_ai",
            "surrogate": "physicsnemo_cfd_field",
            "param_min": param_min,
            "param_max": param_max,
        }
    )
    return {
        "engine": engine,
        "field": ",".join(fields),
        "fields": list(fields),
        "input_fields": list(inputs),
        "output_fields": list(model.output_fields),
        "param_min": param_min,
        "param_max": param_max,
        "validation_metrics": dict(meta.get("validation_metrics", {})),
        "per_field_metrics": dict(meta.get("per_field_metrics", {})),
        "n_locations": int(meta.get("n_locations", 0)),
        "n_snapshots": int(meta.get("n_snapshots", 0)),
        "train_losses": list(model.train_losses_),
    }


# 검증된 PyDMD 변형만 노출한다 — fbdmd 는 이상적인 데이터에서도 발산하고
# hodmd 는 지연 임베딩으로 reconstruct 와 차원이 안 맞는다 (core dmd.py Note).
DMD_METHODS = ("dmd", "spdmd")


def build_dmd_twin(
    dataset: CFDDataset,
    field: str,
    *,
    method: str = "dmd",
    n_modes: int | None = None,
    forecast_factor: float = 1.5,
) -> dict[str, Any]:
    """시계열에서 DMD 동역학 트윈을 학습한다 (계열 Ⓓ — 시간 외삽 가능).

    PyDMD(:class:`DMDAnalyzer`) 로 상태의 **시간 전이 규칙**을 학습한다. 다른
    계열(POD+보간, 신경장)은 학습 파라미터 범위 안 내삽이 전제인 반면, DMD 는
    모드별 고유값(성장률·주파수)으로 **학습 구간 너머를 외삽**할 수 있다.

    적용 한계가 뚜렷하다 — 유동이 "공간모드 × 고유 주파수"의 저랭크 선형
    동역학으로 근사될 때만 맞는다. 강한 이류/불연속(예: 이 앱의 필라멘트 데모)
    이나 공간 구조가 사실상 하나뿐인 경우엔 크게 빗나간다. 그래서 학습 구간
    **재구성 상대오차(``reconstruction_error``)를 함께 돌려주니 UI 에 반드시
    노출할 것** — 사용자가 적합 여부를 눈으로 판단할 유일한 근거다.

    Args:
        dataset: 시계열 데이터셋 (타임스텝 2개 이상).
        field: 학습 대상 물리량.
        method: PyDMD 변형 — ``dmd``(기본) | ``spdmd``(희소). 둘 다 검증됨.
        n_modes: 모드 수. None(권장)이면 PyDMD 가 자동 결정 — 실수 진동은
            켤레쌍 때문에 물리 모드당 랭크 2 가 필요해 수동 지정 시 과소적합
            되기 쉽다.
        forecast_factor: 예측 슬라이더 상한 배수 (1.5 = 학습 구간의 1.5배까지
            외삽 허용).

    Returns:
        ``engine`` (DMDTwinEngine), ``param_min``/``param_max`` (학습 범위),
        ``forecast_max`` (외삽 포함 상한), ``reconstruction_error`` (학습 구간
        상대오차 — 적합도), ``frequencies``/``growth_rates`` (모드 진단),
        ``n_modes``, ``method`` 를 담은 dict.

    Raises:
        ValueError: 타임스텝 2개 미만 또는 미지원 method.
        ImportError: pydmd 미설치.
    """
    from naviertwin.core.digital_twin.dmd_engine import DMDTwinEngine
    from naviertwin.core.flow_analysis.modal.dmd import DMDAnalyzer

    if method not in DMD_METHODS:
        raise ValueError(f"지원하지 않는 DMD method: {method}")
    n_steps = int(getattr(dataset, "n_time_steps", 0))
    if n_steps < 2:
        raise ValueError(
            f"DMD 학습에는 2개 이상의 타임스텝이 필요합니다. 현재: {n_steps}"
        )

    snapshots = np.asarray(dataset.extract_field_snapshots(field), dtype=np.float64)
    times = np.asarray(getattr(dataset, "time_steps", []), dtype=np.float64)
    dt = estimate_dt(dataset)

    analyzer = DMDAnalyzer(method=method, dt=dt, n_modes=n_modes)
    analyzer.fit(snapshots)

    # 적합도 = 학습 구간 재구성 상대오차. DMD 는 데이터가 맞지 않으면 조용히
    # 크게 빗나가므로(경고 없이) 이 값을 반드시 사용자에게 보여줘야 한다.
    rebuilt = np.asarray(analyzer.reconstruct(times), dtype=np.float64)
    denom = float(np.linalg.norm(snapshots))
    reconstruction_error = (
        float(np.linalg.norm(rebuilt - snapshots) / denom) if denom > 0 else float("nan")
    )

    param_min, param_max = float(times.min()), float(times.max())
    span = max(param_max - param_min, 1e-9)
    forecast_max = param_min + span * float(forecast_factor)
    engine = DMDTwinEngine(
        analyzer,
        method=method,
        metadata={
            "field_name": field,
            "reducer": "dmd_dynamics",
            "surrogate": method,
            "problem_type": "dynamics_forecast",
            "param_min": param_min,
            "param_max": param_max,
            "forecast_max": forecast_max,
            "dt": dt,
            "reconstruction_error": reconstruction_error,
        },
    )
    frequencies = np.asarray(analyzer.frequencies, dtype=np.float64)
    growth = np.asarray(analyzer.growth_rates, dtype=np.float64)
    return {
        "engine": engine,
        "field": field,
        "method": method,
        "n_modes": int(np.asarray(analyzer.modes).shape[1]),
        "dt": dt,
        "param_min": param_min,
        "param_max": param_max,
        "forecast_max": forecast_max,
        "reconstruction_error": reconstruction_error,
        "frequencies": frequencies.tolist(),
        "growth_rates": growth.tolist(),
    }


def predict_to_mesh(
    engine: Any,
    param_value: float | Sequence[float],
    target: CFDDataset,
) -> tuple[CFDDataset, list[str]]:
    """대상 메쉬 좌표에서 예측해 필드를 붙인 새 데이터셋을 반환한다 (출력 격자 자유).

    Physics AI(신경장)는 (좌표, 파라미터) → 물리량이라 학습 메쉬에 묶이지
    않는다 — 더 촘촘한 격자, 다른 형식의 메쉬, 프로브 점 어디서든 평가된다.
    ROM 은 모드가 학습 메쉬 벡터라 이 경로를 쓸 수 없다.

    Returns:
        ``(예측 필드가 붙은 새 CFDDataset, 붙은 field 이름 목록)``.

    Raises:
        RuntimeError: Physics AI 트윈이 아닌 경우 (ROM 등).
    """
    from naviertwin.core.cfd_reader.base import CFDDataset as _CFDDataset

    model = getattr(engine, "model", None)
    if model is None or not hasattr(model, "predict_at"):
        raise RuntimeError(
            "다른 격자 예측은 Physics AI(직접 회귀) 트윈만 지원합니다 — "
            "ROM 은 POD 모드가 학습 메쉬에 묶여 있습니다."
        )

    mesh = target.mesh.copy(deep=True)
    coords = np.asarray(mesh.points, dtype=np.float64)[:, :3]
    params = np.asarray(param_value, dtype=np.float64).reshape(-1)
    values = np.asarray(model.predict_at(coords, params), dtype=np.float64).reshape(-1)

    specs = list(getattr(model, "output_fields", []) or [])
    labels = (
        [str(spec["display_name"]) for spec in specs]
        if specs
        else [str(name) for name in model.field_names]
    )
    blocks = values.reshape(len(labels), coords.shape[0])

    attached: list[str] = []
    for label, block in zip(labels, blocks):
        name = f"twin_{label}"
        mesh.point_data[name] = np.asarray(block, dtype=np.float64)
        attached.append(name)
    dataset = _CFDDataset(
        mesh=mesh,
        time_steps=[0.0],
        field_names=list(attached),
        metadata={"source": "twin_prediction_on_mesh"},
    )
    return dataset, attached


def split_multi_prediction(
    engine: Any,
    prediction: np.ndarray,
) -> list[tuple[str, np.ndarray]] | None:
    """다중 출력 Physics AI 예측 벡터를 (표시명, 필드값) 목록으로 분해한다.

    :class:`PhysicsNeMoCFDFieldModel` 은 다중 출력을 field-major 로 이어붙인
    벡터를 반환한다 — ``output_fields`` spec(start/end)으로 자른다.
    단일 출력 엔진(고전 TwinEngine 포함)은 None 을 반환해 기존 단일 필드
    attach 경로를 그대로 쓰게 한다.
    """
    specs = getattr(getattr(engine, "model", None), "output_fields", None) or []
    if len(specs) <= 1:
        return None
    values = np.asarray(prediction, dtype=np.float64).reshape(-1)
    parts: list[tuple[str, np.ndarray]] = []
    for spec in specs:
        segment = values[int(spec["start"]) : int(spec["end"])]
        parts.append((str(spec["display_name"]), segment))
    return parts


def export_physicsnemo_module(engine: Any, path: str | Path) -> str:
    """학습된 Physics AI 엔진의 PyTorch 모델을 PhysicsNeMo Module 로 감싸 저장한다.

    실제 ``physicsnemo`` 패키지(``pip install nvidia-physicsnemo``)가 필요하다
    — 학습 자체는 순수 torch 로 이뤄지므로 :func:`build_physics_ai_twin` 은
    이 패키지 없이도 동작하지만, 표준 PhysicsNeMo 체크포인트로 내보내려면
    필요하다.

    Raises:
        RuntimeError: 감쌀 PyTorch 모델이 없거나 ``physicsnemo`` 미설치인 경우.
    """
    from naviertwin.core.physnemo.physicsnemo_model import (
        physicsnemo_available,
        save_checkpoint,
        wrap_as_physicsnemo_module,
    )

    torch_model = getattr(getattr(engine, "model", None), "_model", None)
    if torch_model is None:
        raise RuntimeError("PhysicsNeMo Module로 저장할 학습된 Physics AI 모델이 없습니다.")
    if not physicsnemo_available():
        raise RuntimeError("physicsnemo 미설치: pip install nvidia-physicsnemo")
    wrapped = wrap_as_physicsnemo_module(
        torch_model, name=f"naviertwin_{getattr(engine, 'model_type', 'physics_ai')}"
    )
    out = save_checkpoint(wrapped, path)
    return str(out)


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
    include_physics: bool = True,
    physics_epochs: int = 120,
) -> dict[str, Any]:
    """여러 reducer×surrogate 조합(+Physics AI)을 학습/평가해 순위표를 만든다.

    각 조합으로 (시간→필드) TwinEngine 을 학습하고, 학습 파라미터에서의 재구성
    오차(RMSE / R² / 상대 L2)와 단일 예측 지연시간(ms)을 측정한다.
    ``include_physics=True`` 면 Ⓑ 직접 회귀(PhysicsNeMo)도 같은 지표로
    리더보드에 포함한다. 벡터 필드는 양쪽 모두 크기(magnitude) 스냅샷으로
    학습하므로(extract_field_snapshots 계약) 동일 조건 비교다.

    Args:
        dataset: 시계열 데이터셋.
        field: 평가 대상 물리량 field.
        n_modes: POD 모드 수.
        combos: 평가할 (reducer, surrogate) 목록. None 이면 전체 조합.
        repeat: 지연시간 측정 반복 횟수.
        include_physics: Physics AI 행 포함 여부.
        physics_epochs: 비교용 Physics AI 학습 epoch 수.

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
    # 안전장치: truth 가 point/cell 단위 스칼라 스냅샷일 때만 Physics AI 참전
    # (extract_field_snapshots 는 벡터도 크기로 펴므로 통상 항상 성립).
    n_points = int(getattr(dataset, "n_points", 0))
    n_cells = int(getattr(dataset, "n_cells", 0))
    physics_comparable = bool(include_physics) and truth.shape[0] in {n_points, n_cells}
    n_total = n_combos + (1 if physics_comparable else 0)
    for combo_index, (reducer, surrogate) in enumerate(combos):
        if progress_cb is not None:
            progress_cb(combo_index, n_total, f"{reducer}+{surrogate}")
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

    if physics_comparable:
        # Ⓑ 직접 회귀(PhysicsNeMo) 도 같은 지표로 리더보드에 참전시킨다.
        if progress_cb is not None:
            progress_cb(n_combos, n_total, "physicsnemo (직접 회귀)")
        try:
            physics = build_physics_ai_twin(
                dataset, field, max_epochs=int(physics_epochs)
            )
            engine = physics["engine"]
            prediction = np.asarray(engine.predict(params), dtype=np.float64)
            if prediction.shape != truth.shape:
                raise ValueError(
                    "벡터 필드는 Physics AI(크기 필드 학습)와 직접 비교할 수 "
                    f"없습니다: {prediction.shape} vs {truth.shape}"
                )
            engine.predict(single)  # warmup
            started = perf_counter()
            iteration = 0
            while iteration < max(1, repeat):
                engine.predict(single)
                iteration += 1
            latency_ms = (perf_counter() - started) / max(1, repeat) * 1000.0
            rows.append(
                {
                    "combo": "physicsnemo (직접 회귀)",
                    "reducer": "direct",
                    "surrogate": "physicsnemo",
                    "n_modes": 0,
                    "rmse": float(rmse(truth, prediction)),
                    "r2": float(r2_score(truth, prediction)),
                    "rel_l2": float(relative_l2_error(truth, prediction)),
                    "latency_ms": float(latency_ms),
                    "status": "ok",
                }
            )
        except Exception as exc:  # noqa: BLE001 — 실패해도 ROM 순위표는 유지
            rows.append(
                {
                    "combo": "physicsnemo (직접 회귀)",
                    "reducer": "direct",
                    "surrogate": "physicsnemo",
                    "n_modes": 0,
                    "rmse": float("inf"),
                    "r2": float("nan"),
                    "rel_l2": float("inf"),
                    "latency_ms": float("nan"),
                    "status": f"error: {exc}",
                }
            )

    if progress_cb is not None:
        progress_cb(n_total, n_total, "완료")
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
