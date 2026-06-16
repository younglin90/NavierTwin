"""Optional C++ acceleration layer.

Import from this package only behind graceful fallbacks.  The extension is
optional so source checkouts and customer machines without a compiler keep
working with the NumPy implementations.
"""

from __future__ import annotations

try:
    from naviertwin._native import _kernels
except Exception:  # pragma: no cover - extension may be unavailable
    _kernels = None


HAS_NATIVE_KERNELS = _kernels is not None

__all__ = ["HAS_NATIVE_KERNELS", "_kernels"]
