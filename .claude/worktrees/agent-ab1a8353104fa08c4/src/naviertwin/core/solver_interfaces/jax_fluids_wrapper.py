"""JAX-Fluids 래퍼 — 있으면 미분가능 CFD, 없으면 RuntimeError.

Examples:
    >>> from naviertwin.core.solver_interfaces.jax_fluids_wrapper import jax_fluids_available
    >>> jax_fluids_available()  # doctest: +SKIP
    False
"""

from __future__ import annotations

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def jax_fluids_available() -> bool:
    try:
        import jaxfluids  # noqa: F401

        return True
    except ImportError:
        return False


def require_jax_fluids() -> None:
    """설치 요구 — 미설치 시 설치 안내 에러."""
    if not jax_fluids_available():
        raise RuntimeError(
            "JAX-Fluids 설치 필요: pip install jaxfluids (linux/CUDA 권장)"
        )


__all__ = ["jax_fluids_available", "require_jax_fluids"]
