"""Round 559 — Jensen wake."""

from __future__ import annotations


class TestJensen:
    def test_at_rotor(self) -> None:
        from naviertwin.core.applied.jensen_wake import wake_velocity

        # x=0 → V=V0
        assert wake_velocity(V0=10, x=0, R=40) == 10.0

    def test_far(self) -> None:
        from naviertwin.core.applied.jensen_wake import wake_velocity

        # very far → recovery toward V0
        v_near = wake_velocity(V0=10, x=100, R=40)
        v_far = wake_velocity(V0=10, x=10000, R=40)
        assert v_far > v_near

    def test_farm(self) -> None:
        from naviertwin.core.applied.jensen_wake import farm_velocity

        out = farm_velocity(V0=10, distances=[200, 200, 200], R=40)
        assert len(out) == 4
        # downstream velocities should be ≤ V0
        assert all(o <= 10.0 for o in out)
