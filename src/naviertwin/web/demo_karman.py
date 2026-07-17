"""카르만 와열 데모 데이터 — 진짜 LBM 해석 + 진짜 구멍 뚫린 격자.

기존 데모(``service.make_demo_dataset``)는 해석적 합성장을 균일 격자에 얹은
것이라 장애물이 없다. 여기서는 실제로 LBM 을 돌리고, **장애물 자리의 셀을 격자에서
제거**해 진짜 구멍을 만든다:

- SDF·마스크·0 채움 같은 "가짜 empty" 를 쓰지 않는다. 벽 안쪽에는 셀이 아예 없다.
- 벽면 노드(bounce-back 으로 u=0)는 점으로 남는다 — no-slip 경계가 데이터에 그대로
  나타난다.

결과적으로 케이스마다 셀 수가 다르므로(형상이 다르면 구멍도 다르다) **POD/ROM 은
쓸 수 없고**(길이가 다른 스냅샷 벡터는 쌓지 못한다) 좌표 기반인 Physics AI 만
학습할 수 있다. 이건 요구사항의 귀결이지 버그가 아니다 — 공통 격자로 맞추는 순간
그게 바로 가짜 empty 다.

해석은 비싸므로(400×160 기준) **기본 파라미터 결과를 저장소에 함께 커밋**한다
(``demo_data/``). 기본 데모는 그 파일에서 즉시 로드하고 절대 재계산하지 않는다 —
데모가 "숙제" 가 되지 않도록. 파라미터를 바꾼 커스텀 실행만 실제로 계산하고 결과를
사용자 캐시에 저장한다. 커밋된 데이터를 갱신하려면 :func:`regenerate_bundled_data`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.cfd_reader.base import CFDDataset
from naviertwin.core.solvers.lbm_obstacle_2d import shape_mask, solve_obstacle_flow

logger = logging.getLogger(__name__)

__all__ = [
    "KARMAN_SHAPE_CASES",
    "build_karman_case_set",
    "build_karman_unsteady",
    "cache_dir",
    "clear_cache",
    "grid_with_hole",
    "regenerate_bundled_data",
]

# 정상 형상 스윕 — (이름, 형상, 정면 높이).
#   원기둥은 직경을 훑어 **진짜 연속 스윕**(학습에 없는 중간 직경도 예측 가능)을
#   만들고, 사각/삼각은 형상 자체가 달라지는 경우를 덮는다.
KARMAN_SHAPE_CASES: tuple[tuple[str, str, int], ...] = (
    ("circle_d18", "circle", 18),
    ("circle_d24", "circle", 24),
    ("circle_d30", "circle", 30),
    ("circle_d36", "circle", 36),
    ("square_d24", "square", 24),
    ("triangle_d24", "triangle", 24),
)

# 형상 코드 — Physics AI 입력 파라미터로 쓴다. 변의 수라 "많을수록 둥글다" 는
# 순서가 있지만 연속량은 아니다 (원=64 는 사실상 범주값).
_SHAPE_SIDES = {"triangle": 3.0, "square": 4.0, "circle": 64.0}

_CACHE_VERSION = 5  # 물리/격자 규칙이 바뀌면 올린다 → 옛 캐시 자동 무효화

# 저장소에 함께 커밋되는 계산 완료 데이터. 기본 파라미터 데모는 여기서 즉시
# 로드하고 절대 재계산하지 않는다 — 데모가 "숙제" 가 되지 않도록. 커스텀
# 파라미터일 때만 계산 후 사용자 캐시에 저장한다.
_BUNDLED_DIR = Path(__file__).parent / "demo_data"


def cache_dir() -> Path:
    """데모 해석 결과 캐시 폴더 (환경변수 ``NAVIERTWIN_DEMO_CACHE`` 로 변경 가능)."""
    override = os.environ.get("NAVIERTWIN_DEMO_CACHE")
    base = Path(override) if override else Path.home() / ".naviertwin" / "demo_cache"
    base.mkdir(parents=True, exist_ok=True)
    return base


def clear_cache() -> int:
    """사용자 캐시된 해석 결과를 지운다 (저장소 번들 데이터는 건드리지 않는다)."""
    removed = 0
    for path in cache_dir().glob("karman_*.npz"):
        path.unlink()
        removed += 1
    return removed


def _cache_path(tag: str, spec: dict[str, Any]) -> Path:
    """해석 조건이 조금이라도 다르면 다른 파일이 되도록 해시로 이름을 만든다."""
    blob = json.dumps({**spec, "v": _CACHE_VERSION}, sort_keys=True)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    return cache_dir() / f"karman_{tag}_{digest}.npz"


def _spec_key(spec: dict[str, Any]) -> str:
    """스펙(+버전)을 정규화한 문자열 — 저장된 데이터가 요청과 맞는지 대조용."""
    return json.dumps({**spec, "v": _CACHE_VERSION}, sort_keys=True)


def _save_solved(path: Path, spec: dict[str, Any], arrays: dict[str, Any]) -> None:
    """계산 결과를 float32 로 저장한다 (부동 배열만). 스펙을 함께 넣어 대조한다.

    float32: 솔버가 이미 f32 로 돌고, 격자 단위 값이 O(0.1~1) 이라 정밀도가
    충분하다. 난류장이라 압축은 잘 안 되므로 정밀도로 크기를 줄인다.
    """
    blob: dict[str, Any] = {"_spec": np.frombuffer(_spec_key(spec).encode(), dtype=np.uint8)}
    for key, value in arrays.items():
        arr = np.asarray(value)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        blob[key] = arr
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **blob)


def _load_solved(path: Path, spec: dict[str, Any]) -> dict[str, Any] | None:
    """저장된 결과를 읽되, 스펙이 요청과 정확히 일치할 때만 돌려준다.

    스펙이 다르면(해상도·Re·케이스 목록 등) None — 호출자가 계산하게 한다.
    """
    if not path.exists():
        return None
    with np.load(path) as data:
        stored = bytes(data["_spec"]).decode() if "_spec" in data.files else ""
        if stored != _spec_key(spec):
            return None
        return {key: data[key] for key in data.files if key != "_spec"}


def _resolve_solved(
    tag: str, spec: dict[str, Any], use_cache: bool
) -> dict[str, Any] | None:
    """계산 결과를 찾는다: 저장소 번들 → 사용자 캐시 순. 둘 다 없으면 None."""
    bundled = _load_solved(_BUNDLED_DIR / f"karman_{tag}.npz", spec)
    if bundled is not None:
        logger.info("카르만 데모: 저장소 번들 데이터 사용 (%s)", tag)
        return bundled
    if use_cache:
        cached = _load_solved(_cache_path(tag, spec), spec)
        if cached is not None:
            logger.info("카르만 데모: 사용자 캐시 사용 (%s)", tag)
            return cached
    return None


def grid_with_hole(
    solid: NDArray[np.bool_], *, spacing: float = 1.0
) -> tuple[Any, NDArray[np.intp]]:
    """고체 자리의 셀을 **제거한** UnstructuredGrid 를 만든다.

    격자점 = LBM 격자 노드. 셀은 노드 사이에 놓이며, **네 꼭짓점이 모두 고체인 셀만**
    버린다. 그래서 벽에 걸친 셀은 남고, 그 셀이 참조하는 벽면 노드(u=0)도 점으로
    남는다 — no-slip 벽이 데이터에 실제로 존재하고, 장애물 내부는 진짜 빈 공간이 된다.

    Args:
        solid: ``(ny, nx)`` 고체 노드 마스크.
        spacing: 격자 간격.

    Returns:
        ``(mesh, keep_point_ids)`` — ``keep_point_ids`` 는 원본 노드를 평탄화한
        인덱스에서 살아남은 점들의 위치(필드 값을 골라내는 데 쓴다).

    Raises:
        ValueError: 남는 셀이 없는 경우.
    """
    import pyvista as pv

    ny, nx = solid.shape
    image = pv.ImageData(dimensions=(nx, ny, 1), spacing=(spacing, spacing, spacing))
    # 셀 (i, j) 의 네 꼭짓점: (j,i), (j,i+1), (j+1,i), (j+1,i+1)
    corner = np.stack(
        [solid[:-1, :-1], solid[:-1, 1:], solid[1:, :-1], solid[1:, 1:]], axis=0
    )
    cell_all_solid = corner.all(axis=0)  # (ny-1, nx-1)
    keep_cells = np.flatnonzero(~cell_all_solid.ravel())
    if keep_cells.size == 0:
        raise ValueError("남는 셀이 없습니다 — 장애물이 격자를 다 덮었습니다.")

    # 원본 점 인덱스를 실어 보내 재매핑을 정확히 추적한다 (extract_cells 는 점을
    # 재번호매김하므로 추측하면 안 된다).
    image.point_data["_orig_pid"] = np.arange(nx * ny, dtype=np.int64)
    mesh = image.extract_cells(keep_cells)
    keep_point_ids = np.asarray(mesh.point_data["_orig_pid"], dtype=np.intp)
    mesh.point_data.remove("_orig_pid")
    return mesh, keep_point_ids


def _fields_from_state(
    ux: NDArray[np.float64],
    uy: NDArray[np.float64],
    rho: NDArray[np.float64],
    keep: NDArray[np.intp],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """LBM 상태 → 살아남은 점들의 (U, p). p = cs²·ρ (LBM 상태방정식)."""
    vel = np.zeros((keep.size, 3), dtype=np.float64)
    vel[:, 0] = ux.ravel()[keep]
    vel[:, 1] = uy.ravel()[keep]
    pressure = rho.ravel()[keep] / 3.0  # cs² = 1/3
    return vel, pressure


def _compute_unsteady_arrays(
    solid: NDArray[np.bool_],
    u_in: float,
    reynolds: float,
    size: int,
    n_snapshots: int,
    progress_cb: Callable[[int, int], None] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """실제 LBM 을 돌려 비정상 시계열 (velocity, pressure, times) 를 만든다.

    번들 데이터 생성과 폴백 계산이 같은 물리를 쓰도록 여기 한 곳에 둔다.
    """
    # 셔딩 주기 ≈ D / (St · u), St ≈ 0.2 → 발달에 넉넉히, 기록은 딱 2 주기라
    # 저장 프레임이 그대로 반복 재생된다.
    period = size / (0.2 * u_in)
    settle = int(period * 6)
    record_span = int(period * 2)
    record_every = max(1, record_span // n_snapshots)
    total = settle + record_span
    logger.info(
        "카르만 비정상 데모 해석: Re=%.0f, %d 스텝 (저장 데이터 없음)", reynolds, total
    )
    result = solve_obstacle_flow(
        solid,
        u_in=u_in,
        reynolds=reynolds,
        max_steps=total,
        record_every=record_every,
        record_from=settle,
        perturb=2e-2,
        perturb_steps=settle // 3,
        progress_cb=progress_cb,
    )
    snaps = result["snapshots"][:n_snapshots]
    if not snaps:
        raise ValueError("스냅샷이 기록되지 않았습니다 — n_snapshots 를 확인하세요.")
    _mesh, keep = grid_with_hole(solid)
    vel = np.zeros((len(snaps), keep.size, 3), dtype=np.float64)
    pres = np.zeros((len(snaps), keep.size), dtype=np.float64)
    for i, snap in enumerate(snaps):
        vel[i], pres[i] = _fields_from_state(snap["ux"], snap["uy"], snap["rho"], keep)
    times = np.arange(len(snaps), dtype=np.float64) * record_every
    return vel, pres, times


def build_karman_unsteady(
    *,
    nx: int = 400,
    ny: int = 160,
    size: int = 24,
    u_in: float = 0.1,
    reynolds: float = 140.0,
    n_snapshots: int = 20,
    use_cache: bool = True,
    progress_cb: Callable[[int, int], None] | None = None,
) -> CFDDataset:
    """실린더 주위 카르만 와열 시계열 (비정상, 문제 유형 A).

    기본 파라미터면 저장소 번들 데이터에서 즉시 로드한다 (계산 없음). 파라미터를
    바꾸면 그때만 실제로 해석한다. 셔딩이 발달한 뒤 ~2 주기만 기록해 그대로
    반복 재생되게 한다.

    Args:
        nx/ny: 격자 크기.
        size: 실린더 직경 (격자 단위).
        u_in: 입구 속도 (격자 단위).
        reynolds: 레이놀즈 수 (셔딩하려면 ≳ 47).
        n_snapshots: 기록할 스냅샷 수 (기본 20 — ~2 주기, 부드러운 반복).
        use_cache: 사용자 캐시 사용 여부 (번들 데이터는 항상 우선).
        progress_cb: ``(step, total)`` 진행 콜백.

    Returns:
        시계열 ``U``/``p`` 를 담은 ``CFDDataset`` (격자에 실린더 구멍이 뚫려 있음).
    """
    spec = {
        "nx": nx, "ny": ny, "size": size, "u_in": u_in,
        "re": reynolds, "n": n_snapshots, "kind": "unsteady",
    }
    solid = shape_mask(nx, ny, kind="circle", size=size)

    data = _resolve_solved("unsteady", spec, use_cache)
    if data is not None:
        vel, pres, times = data["velocity"], data["pressure"], data["times"]
    else:
        vel, pres, times = _compute_unsteady_arrays(
            solid, u_in, reynolds, size, n_snapshots, progress_cb
        )
        if use_cache:
            _save_solved(
                _cache_path("unsteady", spec),
                spec,
                {"velocity": vel, "pressure": pres, "times": times},
            )

    mesh, _keep = grid_with_hole(solid)
    mesh.point_data["U"] = vel[0]
    mesh.point_data["p"] = pres[0]
    return CFDDataset(
        mesh=mesh,
        time_steps=[float(t) for t in times],
        field_names=["U", "p"],
        metadata={
            "source": "demo_karman_unsteady",
            "description": (
                f"Kármán vortex street behind a cylinder (LBM D2Q9, Re={reynolds:.0f}, "
                f"{nx}×{ny}, real hole — no cells inside the body)"
            ),
            "time_series_fields": {"U": vel, "p": pres},
            "time_series_locations": {"U": "point", "p": "point"},
            "reynolds": float(reynolds),
            "solver": "lbm_obstacle_2d",
        },
    )


def _solve_one_case(args: tuple[str, str, int, int, int, float, float]) -> dict[str, Any]:
    """정상 케이스 1개 (프로세스 풀 워커 — 최상위 함수여야 pickle 된다)."""
    name, kind, size, nx, ny, u_in, reynolds = args
    solid = shape_mask(nx, ny, kind=kind, size=size)
    # 정상 케이스만 float64 를 쓴다 — "진짜 정상해" 라는 주장이 수렴 판정에
    # 걸려 있는데, float32 는 잔차가 ~3e-5 에서 바닥을 쳐 그걸 판정할 수 없다.
    # tol=5e-6: u_in 대비 잔차라 충분히 수렴한 정상해이고, 1e-6 보다 스텝이 ~16%
    # 적다(400×160 d=36 기준 19,801 → 16,601).
    result = solve_obstacle_flow(
        solid,
        u_in=u_in,
        reynolds=reynolds,
        max_steps=40_000,
        steady_tol=5e-6,
        check_every=200,
        dtype=np.float64,
    )
    return {
        "name": name, "kind": kind, "size": size,
        "ux": result["ux"], "uy": result["uy"], "rho": result["rho"],
        "steps": result["steps"], "converged": result["converged"],
        "residual": result["residual"],
    }


def _solve_caseset(
    nx: int,
    ny: int,
    u_in: float,
    reynolds: float,
    cases: tuple[tuple[str, str, int], ...],
    max_workers: int | None,
    progress_cb: Callable[[int, int], None] | None,
) -> list[dict[str, Any]]:
    """정상 케이스들을 병렬로 푼다 (번들 생성·폴백 공용)."""
    from concurrent.futures import ProcessPoolExecutor

    logger.info(
        "카르만 정상 케이스 세트 해석: %d 케이스 × %dx%d Re=%.0f (저장 데이터 없음)",
        len(cases), nx, ny, reynolds,
    )
    jobs = [(name, kind, size, nx, ny, u_in, reynolds) for name, kind, size in cases]
    workers = max_workers if max_workers else min(len(jobs), (os.cpu_count() or 2))
    solved: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, res in enumerate(pool.map(_solve_one_case, jobs), start=1):
            solved.append(res)
            if progress_cb is not None:
                progress_cb(done, len(jobs))
    return solved


def _caseset_blob(solved: list[dict[str, Any]]) -> dict[str, Any]:
    """케이스별 결과를 npz 저장용 평탄 dict 로. 정수/불리언도 float 로 담는다."""
    blob: dict[str, Any] = {}
    for i, res in enumerate(solved):
        blob[f"ux_{i}"] = res["ux"]
        blob[f"uy_{i}"] = res["uy"]
        blob[f"rho_{i}"] = res["rho"]
        blob[f"steps_{i}"] = np.float64(res["steps"])
        blob[f"conv_{i}"] = np.float64(1.0 if res["converged"] else 0.0)
        blob[f"resid_{i}"] = np.float64(res["residual"])
    return blob


def build_karman_case_set(
    *,
    # 정상해는 비정상보다 도메인이 짧아도 된다 — Re=30 재순환 거품은 ~2D(≈48셀)라
    # 뒤로 300셀을 더 끌고 가봐야 빈 공간을 계산할 뿐이다. **셀 크기는 그대로**라
    # 형상 해상도(화질)는 400 일 때와 같고, 스텝당 비용과 수렴까지의 스텝 수
    # (유동 통과 시간 = nx/u)만 줄어든다. 와열을 보여줘야 하는 비정상 데모는
    # 하류가 길어야 하므로 400 을 유지한다.
    nx: int = 240,
    ny: int = 160,
    u_in: float = 0.1,
    reynolds: float = 30.0,
    cases: tuple[tuple[str, str, int], ...] = KARMAN_SHAPE_CASES,
    use_cache: bool = True,
    max_workers: int | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """형상별 정상 케이스 세트 (문제 유형 D — 형상 가변).

    Re 는 기본 30 — 실린더 주위 유동은 Re ≲ 47 에서만 **진짜 정상해**가 존재하고,
    그 위에서는 반드시 셔딩해 정상해 자체가 없다.

    각 케이스는 자기 격자(자기 구멍)를 가진다 — 공통 격자로 재샘플하지 않는다.

    Args:
        nx/ny: 격자 크기.
        u_in: 입구 속도.
        reynolds: 레이놀즈 수 (정상해가 존재하려면 ≲ 47).
        cases: ``(이름, 형상, 정면 높이)`` 목록.
        use_cache: 캐시 사용 여부.
        max_workers: 병렬 프로세스 수 (기본 = 케이스 수).
        progress_cb: ``(완료 케이스 수, 전체)`` 진행 콜백.

    Returns:
        :func:`naviertwin.web.service.load_case_set` 과 같은 형태의 dict.
    """

    spec = {
        "nx": nx, "ny": ny, "u_in": u_in, "re": reynolds,
        "cases": [list(c) for c in cases], "kind": "caseset",
    }

    solved: list[dict[str, Any]] = []
    data = _resolve_solved("caseset", spec, use_cache)
    if data is not None:
        for i, (name, kind, size) in enumerate(cases):
            solved.append(
                {
                    "name": name, "kind": kind, "size": size,
                    "ux": data[f"ux_{i}"], "uy": data[f"uy_{i}"], "rho": data[f"rho_{i}"],
                    "steps": int(data[f"steps_{i}"]), "converged": bool(data[f"conv_{i}"]),
                    "residual": float(data[f"resid_{i}"]),
                }
            )
        if progress_cb is not None:
            progress_cb(len(cases), len(cases))
    else:
        solved = _solve_caseset(nx, ny, u_in, reynolds, cases, max_workers, progress_cb)
        if use_cache:
            _save_solved(_cache_path("caseset", spec), spec, _caseset_blob(solved))

    datasets: list[CFDDataset] = []
    params = np.zeros((len(solved), 2), dtype=np.float64)
    names: list[str] = []
    for i, res in enumerate(solved):
        solid = shape_mask(nx, ny, kind=res["kind"], size=res["size"])
        mesh, keep = grid_with_hole(solid)
        vel, pres = _fields_from_state(res["ux"], res["uy"], res["rho"], keep)
        mesh.point_data["U"] = vel
        mesh.point_data["p"] = pres
        datasets.append(
            CFDDataset(
                mesh=mesh,
                time_steps=[0.0],
                field_names=["U", "p"],
                metadata={
                    "source": f"demo_karman_steady_{res['name']}",
                    "description": (
                        f"Steady flow past a {res['kind']} (frontal height {res['size']}, "
                        f"Re={reynolds:.0f}) — real hole, {mesh.n_cells} cells"
                    ),
                    "time_series_fields": {"U": vel[None, ...], "p": pres[None, ...]},
                    "time_series_locations": {"U": "point", "p": "point"},
                    "shape_kind": res["kind"],
                    "converged": bool(res["converged"]),
                    "residual": float(res["residual"]),
                    "solver": "lbm_obstacle_2d",
                },
            )
        )
        params[i, 0] = _SHAPE_SIDES[res["kind"]]
        params[i, 1] = float(res["size"])
        names.append(res["name"])

    not_converged = [r["name"] for r in solved if not r["converged"]]
    if not_converged:
        logger.warning("정상 수렴 실패 케이스: %s", ", ".join(not_converged))

    return {
        "datasets": datasets,
        "params": params,
        "param_names": ["shape_sides", "frontal_height"],
        "case_names": names,
        "params_source": f"demo (LBM 정상해, Re={reynolds:.0f})",
        "resampled": False,
        "grid_summary": (
            f"케이스마다 자기 격자 — 셀 수 {min(d.mesh.n_cells for d in datasets)}"
            f"~{max(d.mesh.n_cells for d in datasets)} (장애물 자리는 진짜 빈 공간)"
        ),
    }


# 번들 데이터 = 각 빌더의 **기본 인자**로 계산한 결과다. 기본값을 바꾸면 여기 스펙도
# 같이 바뀌어야 하므로, 스펙을 빌더 기본값에서 그대로 끌어온다 (드리프트 방지).
def _default_kwargs(func: Any) -> dict[str, Any]:
    import inspect

    return {
        name: p.default
        for name, p in inspect.signature(func).parameters.items()
        if p.default is not inspect.Parameter.empty and name not in ("use_cache", "progress_cb")
    }


def regenerate_bundled_data() -> dict[str, int]:
    """저장소에 커밋할 계산 완료 데이터를 (재)생성한다 — 유지보수용, 앱에선 안 부름.

    각 데모의 **기본 파라미터**로 실제 LBM 을 돌려 float32 로 저장한다. 물리나
    격자 규칙(또는 기본 인자)이 바뀌면 이 함수를 다시 돌려 파일을 갱신하고 커밋한다.

    Returns:
        ``{"unsteady": bytes, "caseset": bytes}`` — 생성한 파일 크기.
    """
    _BUNDLED_DIR.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}

    u = _default_kwargs(build_karman_unsteady)
    spec_u = {
        "nx": u["nx"], "ny": u["ny"], "size": u["size"], "u_in": u["u_in"],
        "re": u["reynolds"], "n": u["n_snapshots"], "kind": "unsteady",
    }
    solid = shape_mask(u["nx"], u["ny"], kind="circle", size=u["size"])
    vel, pres, times = _compute_unsteady_arrays(
        solid, u["u_in"], u["reynolds"], u["size"], u["n_snapshots"], None
    )
    path_u = _BUNDLED_DIR / "karman_unsteady.npz"
    _save_solved(path_u, spec_u, {"velocity": vel, "pressure": pres, "times": times})
    sizes["unsteady"] = path_u.stat().st_size

    c = _default_kwargs(build_karman_case_set)
    spec_c = {
        "nx": c["nx"], "ny": c["ny"], "u_in": c["u_in"], "re": c["reynolds"],
        "cases": [list(x) for x in c["cases"]], "kind": "caseset",
    }
    solved = _solve_caseset(
        c["nx"], c["ny"], c["u_in"], c["reynolds"], c["cases"], None, None
    )
    path_c = _BUNDLED_DIR / "karman_caseset.npz"
    _save_solved(path_c, spec_c, _caseset_blob(solved))
    sizes["caseset"] = path_c.stat().st_size
    return sizes
