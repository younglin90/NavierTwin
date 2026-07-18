"""CFD 필드 정합성 — NaN/Inf / 비현실 값 / 이상치 검출.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.validation.field_sanity import field_sanity_check
    >>> r = field_sanity_check(np.array([1., 2., np.nan, 3., np.inf]))
    >>> r["n_nan"]
    1
    >>> r["n_inf"]
    1
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def field_sanity_check(
    x: NDArray[np.float64],
    *,
    expected_range: tuple[float, float] | None = None,
) -> dict[str, int | float | bool]:
    """필드 통계 + 이상 현상 카운트."""
    x = np.asarray(x, dtype=np.float64)
    flat = x.ravel()
    finite = np.isfinite(flat)
    n_nan = int(np.isnan(flat).sum())
    n_inf = int(np.isinf(flat).sum())
    clean = flat[finite]
    out: dict[str, int | float | bool] = {
        "size": int(flat.size),
        "n_nan": n_nan,
        "n_inf": n_inf,
        "n_finite": int(finite.sum()),
        "all_finite": bool(n_nan == 0 and n_inf == 0),
    }
    if clean.size > 0:
        out.update({
            "min": float(clean.min()),
            "max": float(clean.max()),
            "mean": float(clean.mean()),
            "std": float(clean.std()),
        })
    if expected_range is not None:
        lo, hi = expected_range
        out["n_below"] = int((clean < lo).sum())
        out["n_above"] = int((clean > hi).sum())
        out["in_range"] = bool(out["n_below"] == 0 and out["n_above"] == 0)
    return out


def detect_outliers_iqr(
    x: NDArray[np.float64], k: float = 1.5,
) -> NDArray[np.bool_]:
    """IQR 방법으로 이상치 마스크."""
    a = np.asarray(x, dtype=np.float64)
    q1, q3 = np.quantile(a, [0.25, 0.75])
    iqr = q3 - q1
    return (a < q1 - k * iqr) | (a > q3 + k * iqr)


def detect_outliers_zscore(
    x: NDArray[np.float64], threshold: float = 3.0,
) -> NDArray[np.bool_]:
    a = np.asarray(x, dtype=np.float64)
    mu = a.mean()
    sd = a.std() + 1e-30
    return np.abs(a - mu) / sd > threshold


__all__ = [
    "field_sanity_check",
    "detect_outliers_iqr",
    "detect_outliers_zscore",
]
