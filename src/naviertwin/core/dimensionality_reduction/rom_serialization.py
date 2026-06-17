"""ROM 직렬화 — POD/SVD 결과를 NPZ로 저장/복원.

h5py/HDF5 의존성 없이 numpy 기본 형식으로 ROM 모델을 디스크에 저장.
크기, 메타데이터(작성 시각, 라이브러리 버전, 사용자 태그) 포함.

상용 툴 대응:
    - pyMOR: ReducedBasisModel.save / load
    - Tecplot: SZL/SZLM 형식
    - 학술: pickling / dill 대안

Examples:
    >>> import numpy as np
    >>> import tempfile
    >>> rng = np.random.default_rng(0)
    >>> modes = rng.standard_normal((20, 5))
    >>> sv = np.array([5.0, 3.0, 2.0, 1.0, 0.5])
    >>> from naviertwin.core.dimensionality_reduction.rom_serialization import (
    ...     save_rom, load_rom
    ... )
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = f"{tmp}/rom.npz"
    ...     _ = save_rom(path, modes=modes, singular_values=sv)
    ...     data = load_rom(path)
    ...     bool(np.allclose(data["modes"], modes))
    True
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


_SCHEMA_VERSION = "1.0"


def save_rom(
    path: str | Path,
    *,
    modes: NDArray[np.float64],
    singular_values: NDArray[np.float64],
    mean: NDArray[np.float64] | None = None,
    temporal_coefficients: NDArray[np.float64] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """ROM 모델을 NPZ 형식으로 저장.

    Args:
        path: 출력 경로 (.npz 권장).
        modes: (n_x, r) 공간 모드.
        singular_values: (r,) 특이값.
        mean: (n_x,) 평균 (옵션).
        temporal_coefficients: (r, n_t) 시간 계수 (옵션).
        metadata: 추가 메타 정보 (JSON 직렬화 가능).

    Returns:
        저장된 경로 (절대).

    Raises:
        ValueError: 형상 불일치.
    """
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    M = np.asarray(modes, dtype=np.float64)
    s = np.asarray(singular_values, dtype=np.float64).ravel()
    if M.ndim != 2 or M.shape[1] != s.shape[0]:
        raise ValueError(
            f"modes/singular_values mismatch: {M.shape} vs {s.shape}"
        )

    payload: dict[str, Any] = {
        "modes": M,
        "singular_values": s,
    }
    if mean is not None:
        mean_arr = np.asarray(mean, dtype=np.float64).ravel()
        if mean_arr.shape[0] != M.shape[0]:
            raise ValueError(
                f"mean length {mean_arr.shape[0]} != n_x {M.shape[0]}"
            )
        payload["mean"] = mean_arr
    if temporal_coefficients is not None:
        T = np.asarray(temporal_coefficients, dtype=np.float64)
        if T.ndim != 2 or T.shape[0] != s.shape[0]:
            raise ValueError(
                f"temporal_coefficients shape {T.shape} != ({s.shape[0]}, n_t)"
            )
        payload["temporal_coefficients"] = T

    meta = {
        "schema_version": _SCHEMA_VERSION,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "n_modes": int(s.shape[0]),
        "n_space": int(M.shape[0]),
    }
    if metadata is not None:
        # Validate JSON-serializable
        try:
            json.dumps(metadata)
        except (TypeError, ValueError) as e:
            raise ValueError(f"metadata not JSON-serializable: {e}") from e
        meta["user_metadata"] = metadata

    payload["__metadata__"] = np.array([json.dumps(meta)], dtype=object)

    np.savez_compressed(str(p), **payload)
    logger.info("ROM 저장 완료: %s (%d 모드)", p, s.shape[0])
    return p


def load_rom(path: str | Path) -> dict[str, Any]:
    """NPZ ROM을 로드.

    Args:
        path: 저장된 경로.

    Returns:
        dict — keys: modes, singular_values, mean (옵션),
        temporal_coefficients (옵션), metadata (dict).

    Raises:
        FileNotFoundError: 파일 없음.
        ValueError: 형식 오류.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"ROM file not found: {p}")

    data = np.load(str(p), allow_pickle=True)
    out: dict[str, Any] = {
        "modes": np.asarray(data["modes"]),
        "singular_values": np.asarray(data["singular_values"]),
    }
    if "mean" in data.files:
        out["mean"] = np.asarray(data["mean"])
    if "temporal_coefficients" in data.files:
        out["temporal_coefficients"] = np.asarray(data["temporal_coefficients"])
    if "__metadata__" in data.files:
        meta_raw = data["__metadata__"]
        if isinstance(meta_raw, np.ndarray):
            meta_str = str(meta_raw[0]) if meta_raw.size > 0 else "{}"
        else:
            meta_str = str(meta_raw)
        try:
            out["metadata"] = json.loads(meta_str)
        except json.JSONDecodeError:
            out["metadata"] = {}
    else:
        out["metadata"] = {}

    return out


def rom_size_bytes(
    modes: NDArray[np.float64],
    singular_values: NDArray[np.float64],
    mean: NDArray[np.float64] | None = None,
    temporal_coefficients: NDArray[np.float64] | None = None,
) -> int:
    """ROM이 차지할 바이트 수 추정 (압축 전).

    Args:
        modes: 공간 모드.
        singular_values: 특이값.
        mean, temporal_coefficients: 옵션.

    Returns:
        총 바이트.
    """
    M = np.asarray(modes, dtype=np.float64)
    s = np.asarray(singular_values, dtype=np.float64)
    total = M.nbytes + s.nbytes
    if mean is not None:
        total += np.asarray(mean, dtype=np.float64).nbytes
    if temporal_coefficients is not None:
        total += np.asarray(temporal_coefficients, dtype=np.float64).nbytes
    return int(total)


def compress_modes_float32(
    modes: NDArray[np.float64],
    rel_tol: float = 1e-6,
) -> NDArray[np.float32] | NDArray[np.float64]:
    """float32 정밀도로 다운캐스트 (메모리 절약, 정확도 손실 검증).

    Args:
        modes: 공간 모드.
        rel_tol: 허용 상대 오차 (Frobenius). 초과 시 원본 유지.

    Returns:
        float32 또는 원본 (정밀도 손실 시).
    """
    M = np.asarray(modes, dtype=np.float64)
    M32 = M.astype(np.float32)
    err = np.linalg.norm(M - M32.astype(np.float64))
    norm = np.linalg.norm(M) + 1e-30
    if err / norm <= rel_tol:
        return M32
    return M


def metadata_compatible(
    expected: dict[str, Any],
    actual: dict[str, Any],
    keys: list[str] | None = None,
) -> bool:
    """ROM 메타데이터 호환성 검증.

    Args:
        expected: 기대 메타데이터.
        actual: 실제 (로드된).
        keys: 검사할 키. None이면 schema_version + n_modes + n_space.

    Returns:
        모든 키가 일치하면 True.
    """
    if keys is None:
        keys = ["schema_version", "n_modes", "n_space"]
    return all(map(lambda k: expected.get(k) == actual.get(k), keys))


__all__ = [
    "save_rom",
    "load_rom",
    "rom_size_bytes",
    "compress_modes_float32",
    "metadata_compatible",
]
