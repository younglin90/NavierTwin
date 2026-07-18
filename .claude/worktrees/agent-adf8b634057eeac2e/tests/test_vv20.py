"""Round 506 — V&V20."""

from __future__ import annotations


class TestVV20:
    def test_combined_uncertainty(self) -> None:
        from naviertwin.core.verification.vv20 import validation_uncertainty

        u = validation_uncertainty(u_num=0.3, u_input=0.4, u_data=0.0)
        assert abs(u - 0.5) < 1e-12

    def test_validation_pass(self) -> None:
        from naviertwin.core.verification.vv20 import (
            comparison_error,
            is_validated,
        )

        E = comparison_error(S=10.5, D=10.0)
        assert is_validated(E=E, u_val=0.5, k=2.0)
        assert not is_validated(E=10.0, u_val=0.5, k=2.0)
