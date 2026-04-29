"""Numerical methods public API."""

from naviertwin.core.numerics.chebyshev import (
    chebyshev_diff_matrix,
    chebyshev_points,
    lagrange_interp_1d,
)
from naviertwin.core.numerics.clenshaw_curtis import (
    clenshaw_curtis_weights,
    integrate_cc,
    integrate_cc_interval,
)

__all__ = [
    "chebyshev_diff_matrix",
    "chebyshev_points",
    "clenshaw_curtis_weights",
    "integrate_cc",
    "integrate_cc_interval",
    "lagrange_interp_1d",
]
