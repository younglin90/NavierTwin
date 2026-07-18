"""Heat exchanger ε-NTU method.

Examples:
    >>> from naviertwin.core.applied.heat_exchanger import effectiveness
    >>> effectiveness(NTU=2.0, Cr=0.5, flow='counterflow')
"""

from __future__ import annotations

import numpy as np


def effectiveness(
    *, NTU: float, Cr: float, flow: str = "counterflow",
) -> float:
    if flow == "counterflow":
        if Cr < 1.0:
            return float((1 - np.exp(-NTU * (1 - Cr))) / (1 - Cr * np.exp(-NTU * (1 - Cr))))
        return float(NTU / (1 + NTU))  # Cr=1
    if flow == "parallel":
        return float((1 - np.exp(-NTU * (1 + Cr))) / (1 + Cr))
    raise ValueError(f"unknown flow: {flow}")


def heat_transfer_rate(
    *, eps: float, C_min: float, T_h_in: float, T_c_in: float,
) -> float:
    return float(eps * C_min * (T_h_in - T_c_in))


__all__ = ["effectiveness", "heat_transfer_rate"]
