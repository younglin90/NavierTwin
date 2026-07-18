"""보존량 재샘플: point interpolation vs volume-weighted conservative resample 비교.

``field_semantics.conservative_resample_to_grid()`` 는 ESMPy/MEDCoupling 급
진짜 conservative remap 이 아니라 실용적 근사다(부피/점개수 가중 평균으로
target 격자점 "구역"에 원본 값을 모아 담는다). 그래도 총량 보존 관점에서
점 보간(VTK ``sample()``)보다 뚜렷하게 낫다는 걸 이 파일에서 직접 수치로
증명한다 (외부 검토 §6½ #5 후속).

핵심 테스트(``TestConservativeVsPointInterp``)는 상수 필드가 아니라 target
격자 셀 크기보다 훨씬 촘촘한 스케일로 값이 진동하는 필드(체커보드)를 쓴다.
상수 필드는 점 보간으로도 항상 완벽히 보존되므로(어디서 샘플링해도 같은
값) 두 방법의 차이를 보여주지 못한다 — 체커보드는 서브그리드 스케일 밀도
요동을 흉내낸, 점 보간이 가장 크게 실패하는 극단 사례다.
"""

from __future__ import annotations

import numpy as np
import pytest

from naviertwin.core.preprocessing.field_semantics import (
    conservative_resample_to_grid,
    total_field_integral,
)
from naviertwin.web import service

pv = pytest.importorskip("pyvista", reason="conservative resample 테스트에는 pyvista 가 필요합니다.")


def _checkerboard_source(n: int = 65, value: float = 2.0):
    """서브그리드 스케일(1칸마다 부호가 바뀜)로 진동하는 밀도장을 가진 소스 메쉬.

    평균값은 ``value / 2`` 다 — 상수장과 달리 점 하나만 샘플링하면 실제
    지역 평균과 완전히 다른 값을 뽑을 위험이 있다(정확히 이 테스트가
    증명하려는 것).
    """
    grid = pv.ImageData(
        dimensions=(n, n, 1), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)
    )
    coords = np.asarray(grid.points, dtype=np.float64)
    i = np.rint(coords[:, 0]).astype(np.int64)
    j = np.rint(coords[:, 1]).astype(np.int64)
    rho = np.where((i + j) % 2 == 0, value, 0.0)
    grid.point_data["rho"] = rho
    return grid


def _coarse_target(step: int = 8, n_fine: int = 65):
    """소스와 원점이 일치하고 정수 배수로 성긴 target 격자 (좌표 정확히 겹침).

    좌표를 정확히 겹치게 만드는 이유: target 노드가 소스 노드와 정확히
    일치하면 VTK 점 보간이 그 노드 값을 "그대로" 뽑는다 — 보간이 흐려주는
    효과 없이 "한 점만 보고 지역을 대표시키는" 점 보간의 문제를 순수하게
    드러낸다(실전에서도 성긴 target 격자가 소스 격자와 우연히 정렬되는 건
    드문 일이 아니다).
    """
    n_coarse = (n_fine - 1) // step + 1
    return pv.ImageData(
        dimensions=(n_coarse, n_coarse, 1),
        spacing=(float(step), float(step), 1.0),
        origin=(0.0, 0.0, 0.0),
    )


class TestConservativeVsPointInterp:
    """핵심 테스트 — conservative 가 point-interp 보다 총량 보존이 훨씬 낫다."""

    def test_conservative_resample_preserves_mass_far_better(self) -> None:
        source = _checkerboard_source()
        target = _coarse_target()

        truth = total_field_integral(source, "rho")
        assert truth > 0

        # 점 보간(기존 경로) — target 노드가 소스와 겹쳐서 항상 짝수 parity
        # 값(value)만 뽑는다. 실제 지역 평균은 value/2 이므로 총량이 2배로
        # 뻥튀기된다.
        point_interp = np.asarray(target.sample(source).point_data["rho"])
        grid_point = target.copy(deep=True)
        grid_point.point_data["rho"] = point_interp
        point_total = total_field_integral(grid_point, "rho")

        # conservative resample — target 셀 구역 안의 0/value 를 섞어 평균.
        conservative = conservative_resample_to_grid(source, "rho", target)
        assert conservative.shape == (target.n_points,)
        grid_cons = target.copy(deep=True)
        grid_cons.point_data["rho"] = conservative
        cons_total = total_field_integral(grid_cons, "rho")

        point_err = abs(point_total - truth) / truth
        cons_err = abs(cons_total - truth) / truth

        # 점 보간은 눈에 띄게 틀린다(이 구성에서는 정확히 2배 = 100% 오차).
        assert point_err > 0.5, f"점 보간 오차가 예상보다 작음: {point_err}"
        # conservative 는 총량을 훨씬 잘 보존한다.
        assert cons_err < 0.01, f"conservative 오차가 예상보다 큼: {cons_err}"
        # 핵심 주장: conservative 가 point-interp 보다 압도적으로 낫다.
        assert cons_err < point_err / 10, (point_err, cons_err)

    def test_vector_field_shapes_are_preserved(self) -> None:
        """벡터장(다성분)도 크래시 없이 성분별로 처리된다."""
        source = _checkerboard_source(n=17, value=1.0)
        u = np.asarray(source.point_data["rho"])
        source.point_data["U"] = np.stack([u, -u, np.zeros_like(u)], axis=1)
        target = _coarse_target(step=4, n_fine=17)

        result = conservative_resample_to_grid(source, "U", target)
        assert result.shape == (target.n_points, 3)
        assert np.all(np.isfinite(result))


class TestEmptyCellFallback:
    """target 격자 구역에 원본 점이 하나도 없을 때 크래시하지 않는다."""

    def test_no_crash_and_falls_back_to_nearest_neighbor(self, caplog) -> None:  # noqa: ANN001
        # 소스 점들이 도메인 한쪽 구석에만 몰려 있다 — target 격자 대부분의
        # 구역에는 배정될 원본 점이 없다.
        pts = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.0], [0.2, 0.0, 0.0]])
        cloud = pv.PolyData(pts)  # 셀 없는 point cloud → 부피 가중치도 못 구함
        cloud.point_data["p"] = np.array([1.0, 2.0, 3.0])
        target = pv.ImageData(
            dimensions=(5, 5, 1), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)
        )

        result = conservative_resample_to_grid(cloud, "p", target)

        assert result.shape == (target.n_points,)
        assert np.all(np.isfinite(result))  # NaN 없이 전부 최근접 폴백으로 채워짐
        # 폴백 값은 항상 원본 값 범위 안에 있어야 한다 (외삽 없음).
        assert result.min() >= 1.0 and result.max() <= 3.0

    def test_single_source_point_still_works(self) -> None:
        cloud = pv.PolyData(np.array([[0.5, 0.5, 0.0]]))
        cloud.point_data["p"] = np.array([7.0])
        target = pv.ImageData(
            dimensions=(3, 3, 1), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)
        )
        result = conservative_resample_to_grid(cloud, "p", target)
        assert np.allclose(result, 7.0)


class TestServiceWiringBackwardCompatible:
    """service.py 배선 — ``conservative_fields=()`` 기본값은 기존 동작과 완전히 동일."""

    def test_coarsen_dataset_default_matches_explicit_empty_tuple(self) -> None:
        ds = service.make_demo_dataset(nx=32, ny=32, n_steps=3, kind="advecting")
        baseline = service.coarsen_dataset(ds, resolution=12)
        explicit = service.coarsen_dataset(ds, resolution=12, conservative_fields=())

        assert baseline["points_after"] == explicit["points_after"]
        b_series = baseline["dataset"].metadata["time_series_fields"]
        e_series = explicit["dataset"].metadata["time_series_fields"]
        assert set(b_series) == set(e_series)
        for name in b_series:
            assert np.array_equal(
                np.asarray(b_series[name]), np.asarray(e_series[name])
            ), f"conservative_fields=() 인데 결과가 달라짐: {name}"

    def test_resample_cases_to_common_grid_default_matches_explicit_empty_tuple(
        self,
    ) -> None:
        ds1 = service.make_demo_dataset(nx=16, ny=16, n_steps=1, kind="advecting")
        ds2 = service.make_demo_dataset(nx=20, ny=14, n_steps=1, kind="advecting")

        baseline = service.resample_cases_to_common_grid(
            [ds1, ds2], resolution=10, parallel=False
        )
        explicit = service.resample_cases_to_common_grid(
            [ds1, ds2], resolution=10, parallel=False, conservative_fields=()
        )

        assert baseline["grid_summary"] == explicit["grid_summary"]
        for b_case, e_case in zip(baseline["datasets"], explicit["datasets"]):
            for name in b_case.field_names:
                bv = np.asarray(b_case.mesh.point_data[name])
                ev = np.asarray(e_case.mesh.point_data[name])
                assert np.array_equal(bv, ev), f"conservative_fields=() 인데 달라짐: {name}"

    def test_coarsen_dataset_with_conservative_field_still_returns_valid_result(
        self,
    ) -> None:
        """지정한 field 만 conservative 경로를 타고 결과는 여전히 유효하다."""
        ds = service.make_demo_dataset(nx=24, ny=24, n_steps=3, kind="advecting")
        result = service.coarsen_dataset(
            ds, resolution=10, conservative_fields=["p"]
        )
        out = result["dataset"]
        assert set(out.field_names) == {"U", "p"}
        p_series = np.asarray(out.metadata["time_series_fields"]["p"])
        assert np.all(np.isfinite(p_series))
        assert p_series.shape[0] == ds.n_time_steps
