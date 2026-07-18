"""NTK (Neural Tangent Kernel) — empirical Jacobian-Jacobian product.

Examples:
    >>> import pytest
    >>> torch = pytest.importorskip('torch')
    >>> from naviertwin.utils.ntk import empirical_ntk
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def empirical_ntk(
    model_fn: Callable[[Any, Any], Any],
    params: Any,
    x1: Any,
    x2: Any,
) -> Any:
    """K(x1, x2) = J(x1) J(x2)ᵀ via torch.autograd.functional.jacobian."""
    if not has_torch():
        raise ImportError("torch not installed")
    from torch.autograd.functional import jacobian

    def f1(p):
        return model_fn(p, x1)

    def f2(p):
        return model_fn(p, x2)

    j1 = jacobian(f1, params)
    j2 = jacobian(f2, params)
    j1f = j1.reshape(j1.shape[0], -1)
    j2f = j2.reshape(j2.shape[0], -1)
    return j1f @ j2f.T


__all__ = ["empirical_ntk", "has_torch"]
