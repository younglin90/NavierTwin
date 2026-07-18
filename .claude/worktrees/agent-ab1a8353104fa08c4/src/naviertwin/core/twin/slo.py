"""SLO tracker — burn rate of error budget.

Examples:
    >>> from naviertwin.core.twin.slo import burn_rate
    >>> burn_rate(error_count=10, total_count=1000, slo=0.99)
    1.0
"""

from __future__ import annotations


def burn_rate(*, error_count: int, total_count: int, slo: float) -> float:
    """error_rate / (1 - slo). >1 → consuming budget faster than allowed."""
    if total_count <= 0:
        return 0.0
    err_rate = error_count / total_count
    budget = 1.0 - slo
    return float(err_rate / max(budget, 1e-12))


__all__ = ["burn_rate"]
