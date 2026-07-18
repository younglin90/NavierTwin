"""CSV 파일 배열을 스냅샷 행렬로 로드.

각 CSV = 한 시간 스텝 또는 한 파라미터 포인트. 파일명 알파벳 순 정렬 후
각 파일의 지정 컬럼을 열 벡터로 쌓는다.

Examples:
    >>> # paths = ["t001.csv", "t002.csv", ...]
    >>> # X, coords = load_csv_snapshots(paths, column="U")
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def load_csv_snapshots(
    paths: Sequence[str | Path],
    *,
    column: str,
    coord_columns: tuple[str, ...] | None = ("x", "y", "z"),
    delimiter: str = ",",
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """CSV 시퀀스 → (X[n_points, n_snapshots], coords[n_points, 3] | None).

    Args:
        paths: CSV 경로 리스트.
        column: 스냅샷으로 쌓을 컬럼 이름.
        coord_columns: 첫 파일에서 추출할 좌표 컬럼. 없으면 None.
        delimiter: CSV 구분자.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas 필요") from exc

    if not paths:
        raise ValueError("paths 비어있음")

    def _read_snapshot(item: tuple[int, str | Path]) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        i, p = item
        df = pd.read_csv(p, delimiter=delimiter)
        if column not in df.columns:
            raise KeyError(f"{p}: '{column}' 컬럼 없음 (사용 가능: {list(df.columns)[:6]}…)")
        values = df[column].to_numpy(dtype=np.float64)
        first_coords = None
        if i == 0 and coord_columns:
            avail = tuple(filter(lambda c: c in df.columns, coord_columns))
            if avail:
                first_coords = df[list(avail)].to_numpy(dtype=np.float64)
        return values, first_coords

    cols, coord_candidates = map(list, zip(*map(_read_snapshot, enumerate(paths)), strict=True))
    coords = next(filter(lambda value: value is not None, coord_candidates), None)

    try:
        X = np.stack(cols, axis=1)
    except ValueError as e:
        sizes = list(map(lambda c: c.size, cols))
        raise ValueError(f"CSV 행 수 불일치: {sizes}") from e

    logger.info(
        "CSV snapshots 로드: %d files → %s (column=%s)",
        len(paths), X.shape, column,
    )
    return X, coords


__all__ = ["load_csv_snapshots"]
