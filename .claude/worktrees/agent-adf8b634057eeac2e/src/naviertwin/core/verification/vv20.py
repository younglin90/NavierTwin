"""ASME V&V20 — validation uncertainty u_val² = u_num² + u_input² + u_d².

Examples:
    >>> from naviertwin.core.verification.vv20 import validation_uncertainty
    >>> validation_uncertainty(u_num=0.1, u_input=0.05, u_data=0.02)
"""

from __future__ import annotations

import math


def validation_uncertainty(*, u_num: float, u_input: float, u_data: float) -> float:
    return float(math.sqrt(u_num ** 2 + u_input ** 2 + u_data ** 2))


def comparison_error(*, S: float, D: float) -> float:
    """E = S - D (simulation - data)."""
    return float(S - D)


def is_validated(*, E: float, u_val: float, k: float = 2.0) -> bool:
    return abs(E) <= k * u_val


__all__ = ["comparison_error", "is_validated", "validation_uncertainty"]
