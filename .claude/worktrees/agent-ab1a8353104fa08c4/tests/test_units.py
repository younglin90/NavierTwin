"""Round 93 — 단위 변환."""

from __future__ import annotations

import pytest


class TestUnits:
    def test_pressure(self) -> None:
        from naviertwin.utils.units import convert

        assert convert(1.0, "bar", "Pa") == pytest.approx(1e5)
        assert convert(1.0, "atm", "kPa") == pytest.approx(101.325)
        assert convert(14.5038, "psi", "bar") == pytest.approx(1.0, rel=1e-4)

    def test_velocity(self) -> None:
        from naviertwin.utils.units import convert

        assert convert(36.0, "km/h", "m/s") == pytest.approx(10.0)
        assert convert(100.0, "mph", "m/s") == pytest.approx(44.704, rel=1e-6)

    def test_temperature(self) -> None:
        from naviertwin.utils.units import convert

        assert convert(0.0, "C", "K") == pytest.approx(273.15)
        assert convert(32.0, "F", "C") == pytest.approx(0.0, abs=1e-6)
        assert convert(100.0, "C", "F") == pytest.approx(212.0, rel=1e-6)

    def test_length_density_viscosity(self) -> None:
        from naviertwin.utils.units import convert

        assert convert(1.0, "m", "mm") == pytest.approx(1000.0)
        assert convert(12.0, "in", "cm") == pytest.approx(30.48)
        assert convert(1.0, "g/cm3", "kg/m3") == pytest.approx(1000.0)
        assert convert(1.0, "cP", "Pa.s") == pytest.approx(1e-3)

    def test_errors(self) -> None:
        from naviertwin.utils.units import convert, list_units

        with pytest.raises(ValueError):
            convert(1.0, "bar", "m")  # 카테고리 불일치
        with pytest.raises(ValueError):
            convert(1.0, "foo", "Pa")
        assert "Pa" in list_units("pressure")
