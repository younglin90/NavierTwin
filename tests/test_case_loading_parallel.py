"""케이스 세트 병렬 로딩/재샘플 — 결정성 테스트 (roadmap v5.6 P1).

``load_case_set`` 의 파일 읽기 루프와 ``resample_cases_to_common_grid`` 의
케이스별 재샘플 루프를 스레드로 병렬화했다 (:mod:`naviertwin.utils.parallel`
의 ``thread_map`` — ``ThreadPoolExecutor.map`` 기반이라 완료 순서가 아니라
입력 순서로 결과를 돌려준다). 이 테스트는 그 병렬 경로가 순차 경로와
**바이트 단위로 동일한 결과**를 내는지만 검증한다 — 성능 변경이지 수치
변경이 아니어야 하기 때문이다. 벽시계 시간 하드 어서션은 CI 에서 불안정해
넣지 않는다.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyvista", reason="케이스 세트 테스트에는 pyvista 가 필요합니다.")

from naviertwin.web import service  # noqa: E402

# 헬퍼는 tests/test_web_service.py 의 것을 그대로 재사용한다 — 같은 폴더
# 레이아웃(케이스 .vtk/.vtu + params.csv)을 만드는 로직을 중복 유지하지 않는다.
from tests.test_web_service import _write_case_set, _write_varying_geometry_cases  # noqa: E402

# ──────────────────────────────────────────────────────────────────────
# load_case_set — 파일 읽기 병렬화
# ──────────────────────────────────────────────────────────────────────


def test_load_case_set_parallel_matches_sequential(tmp_path) -> None:
    """parallel=True/False 가 케이스 순서·파라미터·필드값 모두 동일해야 한다."""
    velocities = _write_case_set(tmp_path / "sweep", n_cases=5)

    par = service.load_case_set(tmp_path / "sweep", parallel=True)
    seq = service.load_case_set(tmp_path / "sweep", parallel=False)

    assert par["case_names"] == seq["case_names"] == [
        f"case_{i:02d}.vtk" for i in range(5)
    ]
    assert par["param_names"] == seq["param_names"]
    np.testing.assert_allclose(par["params"], seq["params"])
    np.testing.assert_allclose(par["params"][:, 0], velocities)
    assert len(par["datasets"]) == len(seq["datasets"]) == 5

    for i, (dp, ds) in enumerate(zip(par["datasets"], seq["datasets"])):
        assert dp.n_points == ds.n_points, f"case {i}: 점 수 불일치"
        p_par = np.asarray(dp.extract_field_snapshots("p"))
        p_seq = np.asarray(ds.extract_field_snapshots("p"))
        np.testing.assert_allclose(p_par, p_seq, err_msg=f"case {i}: p 필드 불일치")
        # 재샘플하지 않는 경로(같은 격자)이므로 좌표도 그대로 일치해야 한다.
        np.testing.assert_allclose(
            np.asarray(dp.mesh.points), np.asarray(ds.mesh.points),
            err_msg=f"case {i}: 좌표 불일치",
        )


def test_load_case_set_small_set_uses_sequential_fallback(tmp_path) -> None:
    """파일 2개(최소 허용치)는 parallel=True 를 줘도 순차 경로로 빠지되 결과는 정상."""
    velocities = _write_case_set(tmp_path / "small", n_cases=2)

    result = service.load_case_set(tmp_path / "small", parallel=True)

    assert len(result["datasets"]) == 2
    np.testing.assert_allclose(result["params"][:, 0], velocities)
    assert result["case_names"] == ["case_00.vtk", "case_01.vtk"]


def test_load_case_set_parallel_respects_max_workers(tmp_path) -> None:
    """max_workers 를 좁게 줘도(1) 결과는 순차 경로와 동일해야 한다."""
    _write_case_set(tmp_path / "sweep", n_cases=4)

    one_worker = service.load_case_set(tmp_path / "sweep", parallel=True, max_workers=1)
    seq = service.load_case_set(tmp_path / "sweep", parallel=False)

    assert one_worker["case_names"] == seq["case_names"]
    np.testing.assert_allclose(one_worker["params"], seq["params"])


# ──────────────────────────────────────────────────────────────────────
# resample_cases_to_common_grid — 케이스별 재샘플+EDT 병렬화
# ──────────────────────────────────────────────────────────────────────


def test_resample_cases_to_common_grid_parallel_matches_sequential(tmp_path) -> None:
    """형상이 다른 케이스(가변 메쉬)를 재샘플할 때 parallel=True/False 가 동일해야 한다."""
    radii = _write_varying_geometry_cases(
        tmp_path / "shapes", radii=(0.10, 0.15, 0.20, 0.25, 0.30)
    )
    # resample=False 로 원본(가변 메쉬) 그대로 로드한 뒤 재샘플 함수를 직접 호출한다.
    loaded = service.load_case_set(tmp_path / "shapes", resample=False, parallel=False)
    datasets = loaded["datasets"]
    assert len(datasets) == len(radii)

    par = service.resample_cases_to_common_grid(datasets, resolution=16, parallel=True)
    seq = service.resample_cases_to_common_grid(datasets, resolution=16, parallel=False)

    assert par["grid_summary"] == seq["grid_summary"]
    assert par["fields"] == seq["fields"]
    assert par["resolution"] == seq["resolution"] == 16
    assert len(par["datasets"]) == len(seq["datasets"]) == len(radii)

    for i, (dp, ds) in enumerate(zip(par["datasets"], seq["datasets"])):
        assert dp.n_points == ds.n_points, f"case {i}: 공통 격자 점 수 불일치"
        np.testing.assert_allclose(
            np.asarray(dp.mesh.points), np.asarray(ds.mesh.points),
            err_msg=f"case {i}: 공통 격자 좌표 불일치",
        )
        np.testing.assert_allclose(
            np.asarray(dp.mesh.point_data["sdf"]),
            np.asarray(ds.mesh.point_data["sdf"]),
            err_msg=f"case {i}: sdf 불일치",
        )
        np.testing.assert_allclose(
            np.asarray(dp.mesh.point_data["p"]),
            np.asarray(ds.mesh.point_data["p"]),
            err_msg=f"case {i}: p 필드 불일치",
        )


def test_resample_cases_to_common_grid_small_set_uses_sequential_fallback(tmp_path) -> None:
    """케이스 2개는 parallel=True 를 줘도 순차 경로로 빠지되 결과는 정상."""
    _write_varying_geometry_cases(tmp_path / "shapes", radii=(0.10, 0.30))
    loaded = service.load_case_set(tmp_path / "shapes", resample=False, parallel=False)

    result = service.resample_cases_to_common_grid(
        loaded["datasets"], resolution=16, parallel=True
    )
    assert len(result["datasets"]) == 2
    for ds in result["datasets"]:
        assert "sdf" in ds.mesh.point_data
        assert "p" in ds.mesh.point_data


def test_load_case_set_end_to_end_parallel_resample_matches_sequential(tmp_path) -> None:
    """load_case_set 전체 경로(읽기 + auto 재샘플)가 parallel=True/False 동일해야 한다."""
    radii = _write_varying_geometry_cases(
        tmp_path / "shapes", radii=(0.10, 0.15, 0.20, 0.25)
    )

    par = service.load_case_set(tmp_path / "shapes", resolution=16, parallel=True)
    seq = service.load_case_set(tmp_path / "shapes", resolution=16, parallel=False)

    assert par["resampled"] is seq["resampled"] is True
    assert par["grid_summary"] == seq["grid_summary"]
    np.testing.assert_allclose(par["params"][:, 0], radii)
    np.testing.assert_allclose(par["params"], seq["params"])

    for i, (dp, ds) in enumerate(zip(par["datasets"], seq["datasets"])):
        np.testing.assert_allclose(
            np.asarray(dp.mesh.point_data["sdf"]),
            np.asarray(ds.mesh.point_data["sdf"]),
            err_msg=f"case {i}: sdf 불일치",
        )
