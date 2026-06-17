"""CFD 단위 변환 — 속도/압력/온도/밀도/점도/길이.

Examples:
    >>> from naviertwin.utils.units import convert
    >>> convert(1.0, "bar", "Pa")
    100000.0
    >>> round(convert(100.0, "C", "K"), 2)
    373.15
"""

from __future__ import annotations

# 모든 단위를 SI 로 변환하는 계수 (value_in_SI = value * factor + offset)
_UNITS: dict[str, tuple[float, float, str]] = {
    # pressure (SI = Pa)
    "Pa": (1.0, 0.0, "pressure"),
    "kPa": (1e3, 0.0, "pressure"),
    "bar": (1e5, 0.0, "pressure"),
    "atm": (101325.0, 0.0, "pressure"),
    "psi": (6894.757293168, 0.0, "pressure"),
    # velocity (SI = m/s)
    "m/s": (1.0, 0.0, "velocity"),
    "km/h": (1 / 3.6, 0.0, "velocity"),
    "mph": (0.44704, 0.0, "velocity"),
    "ft/s": (0.3048, 0.0, "velocity"),
    "knot": (0.514444, 0.0, "velocity"),
    # temperature (SI = K)
    "K": (1.0, 0.0, "temperature"),
    "C": (1.0, 273.15, "temperature"),
    "F": (5.0 / 9.0, 459.67 * 5.0 / 9.0, "temperature"),
    # length (SI = m)
    "m": (1.0, 0.0, "length"),
    "mm": (1e-3, 0.0, "length"),
    "cm": (1e-2, 0.0, "length"),
    "km": (1e3, 0.0, "length"),
    "in": (0.0254, 0.0, "length"),
    "ft": (0.3048, 0.0, "length"),
    # density (SI = kg/m³)
    "kg/m3": (1.0, 0.0, "density"),
    "g/cm3": (1000.0, 0.0, "density"),
    # dynamic viscosity (SI = Pa·s)
    "Pa.s": (1.0, 0.0, "viscosity"),
    "cP": (1e-3, 0.0, "viscosity"),
}


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """value [from_unit] → [to_unit] (동일 카테고리 내)."""
    if from_unit not in _UNITS:
        raise ValueError(f"unknown unit: {from_unit}")
    if to_unit not in _UNITS:
        raise ValueError(f"unknown unit: {to_unit}")
    a, oa, ka = _UNITS[from_unit]
    b, ob, kb = _UNITS[to_unit]
    if ka != kb:
        raise ValueError(f"카테고리 불일치: {ka} vs {kb}")
    si = value * a + oa
    return (si - ob) / b


def list_units(category: str | None = None) -> list[str]:
    """사용 가능한 단위 리스트 (옵션 카테고리 필터)."""
    if category is None:
        return sorted(_UNITS.keys())
    units: list[str] = []
    items = list(_UNITS.items())
    idx = 0
    while idx < len(items):
        u, (_, _, k) = items[idx]
        if k == category:
            units.append(u)
        idx += 1
    return sorted(units)


__all__ = ["convert", "list_units"]
