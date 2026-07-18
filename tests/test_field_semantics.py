"""보존량 필드 판별(field_semantics) + coarsen 경고 표시 테스트.

점 보간 재샘플은 보존량(질량·유량 등)의 적분 총량을 보존하지 않으므로
(외부 검토 §6½ #5), 보존량 의심 필드가 있으면 경고를 띄운다.
"""

from __future__ import annotations

import pytest

from naviertwin.core.preprocessing.field_semantics import flag_conserved_fields


class TestFlagConservedFields:
    def test_intensive_fields_not_flagged(self) -> None:
        assert flag_conserved_fields(["p", "U", "T"]) == []

    def test_rho_flagged(self) -> None:
        assert flag_conserved_fields(["rho", "U"]) == ["rho"]

    def test_q_criterion_not_flagged(self) -> None:
        # "Q" 단독 패턴은 없다 — Q-criterion 과 혼동되기 때문.
        assert flag_conserved_fields(["Q-criterion", "q_criterion"]) == []

    def test_case_insensitive(self) -> None:
        assert flag_conserved_fields(["Density", "MASS_FLUX", "p"]) == [
            "Density",
            "MASS_FLUX",
        ]

    def test_explicit_flux_and_flow_rate(self) -> None:
        flagged = flag_conserved_fields(
            ["mass_flux", "volumetric_flux", "flowRate", "momentum_x"]
        )
        assert flagged == ["mass_flux", "volumetric_flux", "flowRate", "momentum_x"]

    def test_empty_input(self) -> None:
        assert flag_conserved_fields([]) == []

    def test_order_preserved(self) -> None:
        assert flag_conserved_fields(["mass_flow", "p", "rho"]) == ["mass_flow", "rho"]


class TestCoarsenConservedWarning:
    """웹 앱 coarsen 미리보기에 보존량 경고 state 가 채워지는지 검증."""

    @pytest.fixture()
    def app_factory(self):  # noqa: ANN201 — pytest fixture
        pytest.importorskip("trame", reason="웹 앱 테스트에는 trame 이 필요합니다.")
        pytest.importorskip("pyvista", reason="웹 앱 테스트에는 pyvista 가 필요합니다.")

        def make(name: str):  # noqa: ANN202
            from trame.app import get_server

            from naviertwin.web.app import NavierTwinWebApp

            server = get_server(name, client_type="vue3")
            return NavierTwinWebApp(server=server)

        return make

    def test_warning_set_when_conserved_field_present(self, app_factory) -> None:  # noqa: ANN001
        app = app_factory("nt-test-conserved-warn")
        app.load_demo()
        app.dataset.field_names.append("rho")
        app._update_coarsen_preview()
        warning = app.server.state.nt_coarsen_conserved_warning
        assert "rho" in warning
        assert "보존" in warning

    def test_warning_empty_without_conserved_fields(self, app_factory) -> None:  # noqa: ANN001
        app = app_factory("nt-test-conserved-clean")
        app.load_demo()  # demo 필드는 U, p 뿐 — 보존량 아님
        app._update_coarsen_preview()
        assert app.server.state.nt_coarsen_conserved_warning == ""

    def test_warning_cleared_without_dataset(self, app_factory) -> None:  # noqa: ANN001
        app = app_factory("nt-test-conserved-nodata")
        app._update_coarsen_preview()
        assert app.server.state.nt_coarsen_conserved_warning == ""
