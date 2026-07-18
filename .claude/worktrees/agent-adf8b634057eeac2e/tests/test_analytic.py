"""해석해 모듈 테스트.

해석해 자체의 수학적 정합성 + 수치해(해석해+잡음)와의 메트릭 상한을 검증한다.
Dedalus 기반 테스트는 @pytest.mark.optional 로 격리.
"""

from __future__ import annotations

import numpy as np
import pytest

pyvista = pytest.importorskip("pyvista", reason="pyvista 가 필요합니다")


class TestCouetteFlow:
    def test_linear_profile(self) -> None:
        """Couette 해석해가 y/H 비례하는 선형 프로파일이어야 한다."""
        from naviertwin.core.validation.analytic_solutions import couette_flow

        y = np.linspace(0.0, 1.0, 11)
        sol = couette_flow(U_top=2.0, H=1.0, y=y)

        assert sol.name == "couette"
        assert sol.velocity[0] == pytest.approx(0.0)
        assert sol.velocity[-1] == pytest.approx(2.0)
        # 중간값
        assert sol.velocity[5] == pytest.approx(1.0, rel=1e-6)

    def test_invalid_H_raises(self) -> None:
        from naviertwin.core.validation.analytic_solutions import couette_flow

        with pytest.raises(ValueError):
            couette_flow(U_top=1.0, H=0.0, y=np.array([0.0]))


class TestPoiseuille2D:
    def test_parabolic_profile(self) -> None:
        """Poiseuille 2D 해석해가 포물선 프로파일이어야 한다."""
        from naviertwin.core.validation.analytic_solutions import poiseuille_flow_2d

        H = 1.0
        y = np.linspace(0.0, H, 21)
        sol = poiseuille_flow_2d(dpdx=-1.0, mu=1.0, H=H, y=y)

        # 경계 속도 0
        assert sol.velocity[0] == pytest.approx(0.0, abs=1e-10)
        assert sol.velocity[-1] == pytest.approx(0.0, abs=1e-10)
        # 중심에서 최대: u_max = -(1/(2μ)) dp/dx · (H/2)² · ? 실제 u_max = H²/(8μ) · |dp/dx|
        u_max_expected = H * H / (8.0 * 1.0)  # |-1|=1
        u_max_actual = float(np.max(sol.velocity))
        assert u_max_actual == pytest.approx(u_max_expected, rel=1e-6)


class TestPoiseuillePipe:
    def test_hagen_poiseuille(self) -> None:
        from naviertwin.core.validation.analytic_solutions import poiseuille_pipe

        R = 0.5
        r = np.linspace(0.0, R, 20)
        sol = poiseuille_pipe(dpdx=-4.0, mu=1.0, R=R, r=r)
        # 벽면 속도 0
        assert sol.velocity[-1] == pytest.approx(0.0, abs=1e-10)
        # 중심 최대: u(0) = R²/(4μ) · |dp/dx| = 0.25·1·4/4 = 0.25
        assert sol.velocity[0] == pytest.approx(R**2, rel=1e-6)


class TestCompareAgainstAnalytic:
    def test_compare_with_noisy_numeric(self) -> None:
        """수치해 = 해석해 + 작은 노이즈 → 높은 R² 가 나와야 한다."""
        import pyvista as pv

        from naviertwin.core.validation.analytic_solutions import (
            compare_against_analytic,
            poiseuille_flow_2d,
        )

        # 2D 채널 유사 3D 구조격자 (cell 존재해야 sample() 가능)
        xs = np.linspace(0.0, 2.0, 4)
        ys = np.linspace(0.0, 1.0, 21)
        zs = np.linspace(-0.1, 0.1, 2)
        xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="ij")
        sgrid = pv.StructuredGrid(xg, yg, zg)
        grid = sgrid.cast_to_unstructured_grid()
        points = np.asarray(grid.points)

        # 수치해 = 해석해 + 노이즈
        rng = np.random.default_rng(0)
        mu = 1.0
        dpdx = -1.0
        H = 1.0
        u_true = -(1.0 / (2 * mu)) * dpdx * points[:, 1] * (H - points[:, 1])
        grid.point_data["U"] = u_true + 0.01 * rng.standard_normal(len(u_true))

        sol = poiseuille_flow_2d(dpdx=dpdx, mu=mu, H=H, y=ys)
        result = compare_against_analytic(grid, sol, field_name="U", axis="y")

        assert result["metrics"]["r2"] > 0.95
        assert result["metrics"]["relative_l2"] < 0.1
        assert len(result["analytic"]) == len(ys)
        assert len(result["numeric"]) == len(ys)

    def test_missing_field_raises(self) -> None:
        import pyvista as pv

        from naviertwin.core.validation.analytic_solutions import (
            compare_against_analytic,
            couette_flow,
        )

        grid = pv.PolyData(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])).cast_to_unstructured_grid()
        sol = couette_flow(1.0, 1.0, np.array([0.0, 0.5, 1.0]))
        with pytest.raises(ValueError, match="필드"):
            compare_against_analytic(grid, sol, field_name="NONE")


@pytest.mark.optional
class TestSpectralPoiseuille:
    def test_dedalus_backend_optional(self) -> None:
        pytest.importorskip("dedalus", reason="dedalus 가 필요합니다")
        from naviertwin.core.validation.analytic_solutions import spectral_poiseuille

        sol = spectral_poiseuille(dpdx=-1.0, mu=1.0, H=1.0, n_points=32)
        assert sol.velocity.shape == (32,)
        assert sol.params["method"] == "dedalus"

    def test_spectral_without_dedalus_raises(self) -> None:
        from unittest.mock import patch

        from naviertwin.core.validation.analytic_solutions import spectral_poiseuille

        # Dedalus import 를 실패시키기 위해 builtins.__import__ 후킹
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__  # noqa: E501

        def _blocker(name: str, *args: object, **kwargs: object) -> object:
            if name.startswith("dedalus"):
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_blocker):
            with pytest.raises(RuntimeError, match="dedalus"):
                spectral_poiseuille(dpdx=-1.0, mu=1.0, H=1.0, n_points=32)
