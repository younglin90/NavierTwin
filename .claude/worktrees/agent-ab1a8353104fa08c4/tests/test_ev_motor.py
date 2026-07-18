"""Round 556 — EV motor envelope."""

from __future__ import annotations


class TestEV:
    def test_constant_torque_low(self) -> None:
        from naviertwin.core.applied.ev_motor import torque_envelope

        assert torque_envelope(omega=100, omega_base=300,
                                  T_max=200, P_max=60000) == 200

    def test_constant_power_high(self) -> None:
        from naviertwin.core.applied.ev_motor import torque_envelope

        # ω=600 > base; T = P/ω = 60000/600 = 100
        assert abs(torque_envelope(omega=600, omega_base=300,
                                     T_max=200, P_max=60000) - 100) < 1e-9

    def test_power(self) -> None:
        from naviertwin.core.applied.ev_motor import power_envelope

        # below base: P = ω T
        assert power_envelope(omega=200, omega_base=300,
                                T_max=100, P_max=99999) == 20000
