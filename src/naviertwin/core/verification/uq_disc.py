"""Discretization uncertainty propagator — combine GCI across QoIs.

Examples:
    >>> from naviertwin.core.verification.uq_disc import combined_disc_uncertainty
    >>> combined_disc_uncertainty([0.01, 0.02, 0.005])
"""

from __future__ import annotations

from collections.abc import Sequence

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def combined_disc_uncertainty(uncs: Sequence[float]) -> float:
    """RSS of per-QoI discretization uncertainties."""
    return float(_kernels.combined_disc_uncertainty(list(uncs)))


__all__ = ["combined_disc_uncertainty"]
