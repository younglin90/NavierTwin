"""Round 458 — solar PV."""

from __future__ import annotations


class TestPV:
    def test_iv_decreasing(self) -> None:
        from naviertwin.core.applied.solar_pv import iv_curve

        V, cur = iv_curve(I_ph=8.0, I_0=1e-9, V_T=0.0257, n=1.0, V_max=0.7, n_pts=50)
        # current monotone non-increasing
        assert (cur[:-1] >= cur[1:] - 1e-9).all()
        _ = V

    def test_mppt(self) -> None:
        from naviertwin.core.applied.solar_pv import iv_curve, mppt

        V, cur = iv_curve()
        Vm, Im, Pm = mppt(V, cur)
        # power = V*I positive
        assert Pm > 0
        # MPP voltage somewhere in middle (not at 0 or Voc)
        assert 0.3 < Vm < 0.7
