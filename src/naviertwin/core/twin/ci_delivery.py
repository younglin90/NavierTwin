"""Confidence interval delivery — wrap predictions with CI bounds.

Examples:
    >>> from naviertwin.core.twin.ci_delivery import wrap_with_ci
    >>> wrap_with_ci(value=10.0, sigma=1.0, z=1.96)
    {'value': 10.0, 'lower': 8.04, 'upper': 11.96}
"""

from __future__ import annotations


def wrap_with_ci(*, value: float, sigma: float, z: float = 1.96) -> dict:
    return {
        "value": float(value),
        "lower": float(value - z * sigma),
        "upper": float(value + z * sigma),
    }


__all__ = ["wrap_with_ci"]
