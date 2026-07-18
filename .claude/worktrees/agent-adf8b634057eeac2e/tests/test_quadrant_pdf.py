"""Round 609 — quadrant analysis + PDF estimation."""

from __future__ import annotations

import numpy as np
import pytest


class TestQuadrantSplit:
    def test_basic_quadrants(self) -> None:
        from naviertwin.core.flow_analysis.quadrant_pdf import quadrant_split

        rng = np.random.default_rng(0)
        up = rng.standard_normal(2000)
        vp = rng.standard_normal(2000)
        q = quadrant_split(up, vp)
        # 4 quadrants + hole
        assert {"Q1", "Q2", "Q3", "Q4", "hole"} <= set(q.keys())
        # 빈도 합 ≈ 1 (no hole)
        total = sum(q[k]["fraction"] for k in ("Q1", "Q2", "Q3", "Q4"))
        assert abs(total - 1.0) < 1e-10

    def test_hole_filter(self) -> None:
        from naviertwin.core.flow_analysis.quadrant_pdf import quadrant_split

        rng = np.random.default_rng(1)
        up = rng.standard_normal(2000)
        vp = rng.standard_normal(2000)
        q_no = quadrant_split(up, vp, hole=0.0)
        q_h = quadrant_split(up, vp, hole=2.0)
        # hole 활성화 시 hole 영역 카운트 > 0
        assert q_h["hole"]["count"] > 0
        # 4사분면 합은 줄어듦
        sum_no = sum(q_no[k]["count"] for k in ("Q1", "Q2", "Q3", "Q4"))
        sum_h = sum(q_h[k]["count"] for k in ("Q1", "Q2", "Q3", "Q4"))
        assert sum_h < sum_no

    def test_q2_q4_dominate_for_negative_uv(self) -> None:
        from naviertwin.core.flow_analysis.quadrant_pdf import quadrant_split

        # 인공: u' = -v' → 모든 점이 Q2 또는 Q4
        rng = np.random.default_rng(2)
        up = rng.standard_normal(1000)
        vp = -up
        q = quadrant_split(up, vp)
        assert q["Q2"]["fraction"] + q["Q4"]["fraction"] > 0.99

    def test_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.quadrant_pdf import quadrant_split

        with pytest.raises(ValueError, match="shape"):
            quadrant_split(np.zeros(10), np.zeros(20))

    def test_negative_hole_raises(self) -> None:
        from naviertwin.core.flow_analysis.quadrant_pdf import quadrant_split

        with pytest.raises(ValueError, match="hole"):
            quadrant_split(np.zeros(10), np.zeros(10), hole=-1.0)


class TestHistogramPDF:
    def test_normalize_integrates_to_one(self) -> None:
        from naviertwin.core.flow_analysis.quadrant_pdf import histogram_pdf

        rng = np.random.default_rng(3)
        x = rng.standard_normal(5000)
        c, p = histogram_pdf(x, bins=50)
        # 적분 ≈ 1
        bin_width = c[1] - c[0]
        integral = p.sum() * bin_width
        assert abs(integral - 1.0) < 0.05

    def test_no_normalize_returns_counts(self) -> None:
        from naviertwin.core.flow_analysis.quadrant_pdf import histogram_pdf

        x = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        c, counts = histogram_pdf(x, bins=5, normalize=False)
        assert int(counts.sum()) == 5

    def test_explicit_range(self) -> None:
        from naviertwin.core.flow_analysis.quadrant_pdf import histogram_pdf

        rng = np.random.default_rng(4)
        x = rng.standard_normal(1000)
        c, p = histogram_pdf(x, bins=20, range_=(-3, 3))
        assert c[0] >= -3
        assert c[-1] <= 3

    def test_invalid_bins_raises(self) -> None:
        from naviertwin.core.flow_analysis.quadrant_pdf import histogram_pdf

        with pytest.raises(ValueError, match="bins"):
            histogram_pdf(np.zeros(100), bins=0)


class TestKDE:
    def test_kde_normal_distribution(self) -> None:
        from naviertwin.core.flow_analysis.quadrant_pdf import kde_pdf

        rng = np.random.default_rng(5)
        x = rng.standard_normal(2000)
        eval_x, pdf = kde_pdf(x, n_eval=100)
        assert eval_x.shape == pdf.shape
        # 0 근처 피크
        peak = eval_x[pdf.argmax()]
        assert abs(peak) < 0.3

    def test_explicit_points(self) -> None:
        from naviertwin.core.flow_analysis.quadrant_pdf import kde_pdf

        rng = np.random.default_rng(6)
        x = rng.standard_normal(500)
        pts = np.linspace(-3, 3, 50)
        eval_x, pdf = kde_pdf(x, points=pts)
        np.testing.assert_array_equal(eval_x, pts)

    def test_explicit_bandwidth(self) -> None:
        from naviertwin.core.flow_analysis.quadrant_pdf import kde_pdf

        rng = np.random.default_rng(7)
        x = rng.standard_normal(500)
        eval_x, pdf = kde_pdf(x, bandwidth=0.3)
        assert pdf.shape == eval_x.shape

    def test_invalid_bandwidth_raises(self) -> None:
        from naviertwin.core.flow_analysis.quadrant_pdf import kde_pdf

        with pytest.raises(ValueError, match="bandwidth"):
            kde_pdf(np.zeros(10), bandwidth=0.0)

    def test_too_few_samples_raises(self) -> None:
        from naviertwin.core.flow_analysis.quadrant_pdf import kde_pdf

        with pytest.raises(ValueError, match="2 samples"):
            kde_pdf(np.array([1.0]))


class TestJointPDF:
    def test_basic_shape(self) -> None:
        from naviertwin.core.flow_analysis.quadrant_pdf import joint_pdf_2d

        rng = np.random.default_rng(8)
        x = rng.standard_normal(2000)
        y = rng.standard_normal(2000)
        xc, yc, pdf = joint_pdf_2d(x, y, bins=30)
        assert pdf.shape == (30, 30)
        # 정규화 적분 ≈ 1
        dx = xc[1] - xc[0]
        dy = yc[1] - yc[0]
        integral = pdf.sum() * dx * dy
        assert abs(integral - 1.0) < 0.1

    def test_explicit_range(self) -> None:
        from naviertwin.core.flow_analysis.quadrant_pdf import joint_pdf_2d

        rng = np.random.default_rng(9)
        x = rng.standard_normal(1000)
        y = rng.standard_normal(1000)
        xc, yc, pdf = joint_pdf_2d(x, y, bins=20, range_=((-3, 3), (-3, 3)))
        assert xc[0] >= -3
        assert yc[-1] <= 3

    def test_shape_mismatch_raises(self) -> None:
        from naviertwin.core.flow_analysis.quadrant_pdf import joint_pdf_2d

        with pytest.raises(ValueError, match="shape"):
            joint_pdf_2d(np.zeros(10), np.zeros(20))
