"""Round 627 — binary morphology + connected components."""

from __future__ import annotations

import numpy as np
import pytest


class TestDilation:
    def test_grows(self) -> None:
        from naviertwin.core.flow_analysis.morphology import binary_dilation_2d

        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        d = binary_dilation_2d(mask)
        assert d.sum() == 5  # 4-neighbor + 자신

    def test_8_connectivity(self) -> None:
        from naviertwin.core.flow_analysis.morphology import binary_dilation_2d

        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        d = binary_dilation_2d(mask, connectivity=2)
        assert d.sum() == 9  # 8-neighbor + self

    def test_iterations(self) -> None:
        from naviertwin.core.flow_analysis.morphology import binary_dilation_2d

        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        d1 = binary_dilation_2d(mask, iterations=1)
        d2 = binary_dilation_2d(mask, iterations=2)
        assert d2.sum() > d1.sum()

    def test_invalid_connectivity(self) -> None:
        from naviertwin.core.flow_analysis.morphology import binary_dilation_2d

        with pytest.raises(ValueError, match="connectivity"):
            binary_dilation_2d(np.zeros((5, 5), dtype=bool), connectivity=3)

    def test_invalid_iterations(self) -> None:
        from naviertwin.core.flow_analysis.morphology import binary_dilation_2d

        with pytest.raises(ValueError, match="iterations"):
            binary_dilation_2d(np.zeros((5, 5), dtype=bool), iterations=-1)

    def test_zero_iter(self) -> None:
        from naviertwin.core.flow_analysis.morphology import binary_dilation_2d

        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        d = binary_dilation_2d(mask, iterations=0)
        np.testing.assert_array_equal(d, mask)


class TestErosion:
    def test_shrinks(self) -> None:
        from naviertwin.core.flow_analysis.morphology import binary_erosion_2d

        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True  # 4x4 block
        e = binary_erosion_2d(mask)
        # 침식 후 작아짐
        assert e.sum() < mask.sum()

    def test_full_erodes_to_inner(self) -> None:
        from naviertwin.core.flow_analysis.morphology import binary_erosion_2d

        # 전체 True → 침식 후 가장자리 제거
        mask = np.ones((10, 10), dtype=bool)
        e = binary_erosion_2d(mask)
        assert not e[0, 0]
        assert e[5, 5]


class TestOpenClose:
    def test_opening_removes_small(self) -> None:
        from naviertwin.core.flow_analysis.morphology import binary_opening_2d

        mask = np.zeros((20, 20), dtype=bool)
        mask[10, 10] = True  # 단일 노이즈
        mask[3:8, 3:8] = True  # 5x5 큰 블록
        o = binary_opening_2d(mask)
        # 단일 픽셀은 사라짐
        assert not o[10, 10]
        # 큰 블록은 살아남음
        assert o[5, 5]

    def test_closing_fills_holes(self) -> None:
        from naviertwin.core.flow_analysis.morphology import binary_closing_2d

        mask = np.ones((10, 10), dtype=bool)
        mask[5, 5] = False  # 한 점 구멍
        c = binary_closing_2d(mask)
        assert c[5, 5]


class TestConnectedComponents:
    def test_two_components(self) -> None:
        from naviertwin.core.flow_analysis.morphology import (
            connected_components_2d,
        )

        mask = np.zeros((10, 10), dtype=bool)
        mask[1:4, 1:4] = True  # 컴포넌트 1
        mask[6:9, 6:9] = True  # 컴포넌트 2
        labels, n = connected_components_2d(mask)
        assert n == 2

    def test_diagonal_4_vs_8(self) -> None:
        from naviertwin.core.flow_analysis.morphology import (
            connected_components_2d,
        )

        mask = np.zeros((5, 5), dtype=bool)
        mask[1, 1] = True
        mask[2, 2] = True
        # 4-neighbor: 별개 (대각선 연결 안 됨)
        labels1, n1 = connected_components_2d(mask, connectivity=1)
        # 8-neighbor: 연결됨
        labels2, n2 = connected_components_2d(mask, connectivity=2)
        assert n1 == 2
        assert n2 == 1

    def test_empty_mask(self) -> None:
        from naviertwin.core.flow_analysis.morphology import (
            connected_components_2d,
        )

        labels, n = connected_components_2d(np.zeros((5, 5), dtype=bool))
        assert n == 0

    def test_invalid_connectivity(self) -> None:
        from naviertwin.core.flow_analysis.morphology import (
            connected_components_2d,
        )

        with pytest.raises(ValueError, match="connectivity"):
            connected_components_2d(np.zeros((5, 5), dtype=bool), connectivity=3)


class TestComponentSizes:
    def test_basic(self) -> None:
        from naviertwin.core.flow_analysis.morphology import (
            component_sizes,
            connected_components_2d,
        )

        mask = np.zeros((10, 10), dtype=bool)
        mask[1:4, 1:4] = True  # 9
        mask[6:8, 6:8] = True  # 4
        labels, _ = connected_components_2d(mask)
        sizes = component_sizes(labels)
        assert set(sizes.tolist()) == {9, 4}

    def test_empty(self) -> None:
        from naviertwin.core.flow_analysis.morphology import component_sizes

        sizes = component_sizes(np.zeros((5, 5), dtype=np.int32))
        assert len(sizes) == 0


class TestRemoveSmall:
    def test_keeps_large(self) -> None:
        from naviertwin.core.flow_analysis.morphology import remove_small_components

        mask = np.zeros((10, 10), dtype=bool)
        mask[1:4, 1:4] = True  # 9개
        mask[6, 6] = True  # 1개
        out = remove_small_components(mask, min_size=5)
        # 큰 것만 살아남음
        assert out[1, 1]
        assert not out[6, 6]

    def test_invalid_min_size(self) -> None:
        from naviertwin.core.flow_analysis.morphology import remove_small_components

        with pytest.raises(ValueError, match="min_size"):
            remove_small_components(np.zeros((5, 5), dtype=bool), min_size=0)


class TestThreshold:
    def test_above(self) -> None:
        from naviertwin.core.flow_analysis.morphology import threshold_to_mask

        f = np.array([1.0, 2.0, 3.0, 4.0])
        m = threshold_to_mask(f, threshold=2.5, mode="above")
        np.testing.assert_array_equal(m, [False, False, True, True])

    def test_below(self) -> None:
        from naviertwin.core.flow_analysis.morphology import threshold_to_mask

        f = np.array([1.0, 2.0, 3.0])
        m = threshold_to_mask(f, threshold=2.0, mode="below")
        np.testing.assert_array_equal(m, [True, False, False])

    def test_abs_above(self) -> None:
        from naviertwin.core.flow_analysis.morphology import threshold_to_mask

        f = np.array([-3.0, 1.0, 2.0])
        m = threshold_to_mask(f, threshold=2.5, mode="abs_above")
        np.testing.assert_array_equal(m, [True, False, False])

    def test_invalid_mode(self) -> None:
        from naviertwin.core.flow_analysis.morphology import threshold_to_mask

        with pytest.raises(ValueError, match="mode"):
            threshold_to_mask(np.zeros(5), threshold=0.5, mode="bogus")
