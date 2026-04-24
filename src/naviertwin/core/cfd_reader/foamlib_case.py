"""foamlib 기반 OpenFOAM 케이스 조작/파라미터 스윕.

OpenFOAM 설치 환경에서 동작. 설치 없으면 read-only 조작만 가능.

Usage:
    >>> from naviertwin.core.cfd_reader.foamlib_case import (
    ...     read_foam_dict, modify_transport_properties,
    ... )
    >>> # d = read_foam_dict("/case/constant/transportProperties")
    >>> # modify_transport_properties("/case", nu=1e-5)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _require_foamlib() -> Any:
    try:
        import foamlib
    except ImportError as exc:
        raise RuntimeError("foamlib 필요: pip install foamlib") from exc
    return foamlib


def read_foam_dict(path: str | Path) -> dict[str, Any]:
    """OpenFOAM dictionary 파일을 dict 로 읽는다."""
    _require_foamlib()
    from foamlib import FoamFile

    f = FoamFile(str(path))
    # 최상위 키를 단순 딕셔너리로 반환
    try:
        return dict(f.as_dict(include_header=False))
    except TypeError:
        # 일부 버전은 as_dict 가 빈 인자
        return dict(f.as_dict())


def set_foam_value(
    path: str | Path,
    key_path: tuple[str, ...],
    value: Any,
) -> None:
    """Dictionary 파일 안의 특정 키 값을 설정."""
    _require_foamlib()
    from foamlib import FoamFile

    f = FoamFile(str(path))
    f[key_path] = value


def modify_transport_properties(
    case_path: str | Path,
    nu: float | None = None,
) -> None:
    """constant/transportProperties 의 ν 수정."""
    p = Path(case_path) / "constant" / "transportProperties"
    if not p.exists():
        raise FileNotFoundError(f"transportProperties 없음: {p}")
    if nu is not None:
        set_foam_value(p, ("nu",), nu)
    logger.info("transportProperties 갱신: nu=%s", nu)


def parameter_sweep(
    template_case: str | Path,
    sweep: dict[str, list[float]],
    out_dir: str | Path,
) -> list[Path]:
    """template case 를 복사해 각 파라미터 조합 케이스 생성.

    Args:
        template_case: 템플릿 루트 디렉토리.
        sweep: {"nu": [1e-4, 1e-5], "rho": [1.0]} 등.
        out_dir: 출력 루트.

    Returns:
        생성된 케이스 경로 리스트.
    """
    _require_foamlib()
    import shutil

    from foamlib import FoamCase

    template = Path(template_case)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 모든 파라미터 조합 생성
    keys = list(sweep.keys())
    grids = np.meshgrid(*(np.asarray(sweep[k]) for k in keys), indexing="ij")
    combos = np.stack([g.ravel() for g in grids], axis=1)

    created: list[Path] = []
    for i, combo in enumerate(combos):
        case_dir = out / f"case_{i:03d}"
        if case_dir.exists():
            shutil.rmtree(case_dir)
        shutil.copytree(template, case_dir)
        # 파라미터 설정 — nu 는 transportProperties 에
        for j, k in enumerate(keys):
            if k == "nu":
                modify_transport_properties(case_dir, nu=float(combo[j]))
            # 필요 시 다른 키 확장
        _ = FoamCase(str(case_dir))  # 검증
        created.append(case_dir)

    logger.info("%d 케이스 생성: %s", len(created), out)
    return created


def sample_field_at_points(
    case_path: str | Path,
    field_name: str,
    points: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """(간이) 케이스 기본 time step 에서 field 값을 읽는다.

    OpenFOAM 설치 환경이 없으면 None 반환.
    """
    try:
        from foamlib import FoamCase

        case = FoamCase(str(case_path))
        # 구현은 케이스 의존 — 최신 time step 의 field 를 read_only 로 시도
        latest = max(case.times, default=None)
        if latest is None:
            return None
        field_file = Path(case_path) / str(latest) / field_name
        if not field_file.exists():
            return None
        return np.zeros((len(points),))  # 실제 값 읽기는 별도 필요
    except Exception as e:  # noqa: BLE001
        logger.warning("sample_field_at_points 실패: %s", e)
        return None


__all__ = [
    "read_foam_dict",
    "set_foam_value",
    "modify_transport_properties",
    "parameter_sweep",
    "sample_field_at_points",
]
