"""능력 기반 전략 레지스트리 (v5.0) — 판정이 데이터 특성을 정확히 반영하는지.

지키려는 계약: "이 데이터에 무엇이 가능한가"를 학습 버튼을 누르기 **전에** 정확히
알 수 있어야 하고, 판정 이유가 실제 실행 결과(성공/거절)와 일치해야 한다.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyvista", reason="프로파일 계산에 pyvista 가 필요합니다.")

from naviertwin.web import service, strategies  # noqa: E402


def test_registry_covers_all_wired_methods() -> None:
    """앱의 nt_model_method 값과 레지스트리 키가 일치해야 카드 표시가 성립한다."""
    keys = {spec.key for spec in strategies.STRATEGIES}
    assert keys == {"rom", "physics", "dynamics", "operator"}


def test_single_snapshot_nothing_trainable() -> None:
    ds = service.make_demo_dataset(nx=8, ny=8, n_steps=1)
    profile = strategies.profile_data(ds)
    report = strategies.strategy_report(profile)
    assert not any(v["ok"] for v in report.values())
    assert strategies.recommend(profile)["method"] == "none"


def test_time_series_enables_rom_physics_dmd_not_operator() -> None:
    ds = service.make_demo_dataset(nx=10, ny=10, n_steps=12)
    profile = strategies.profile_data(ds)
    report = strategies.strategy_report(profile)
    assert report["rom"]["ok"] and report["physics"]["ok"] and report["dynamics"]["ok"]
    # operator 는 트윈 흐름에서 미배선 — 이유가 연산자 랩을 가리켜야 한다.
    assert not report["operator"]["ok"]
    assert "연산자 랩" in report["operator"]["reason"]
    assert strategies.recommend(profile)["method"] == "rom"


def test_short_series_rejects_dmd_with_step_count_reason() -> None:
    ds = service.make_demo_dataset(nx=8, ny=8, n_steps=3)
    report = strategies.strategy_report(strategies.profile_data(ds))
    assert report["rom"]["ok"]
    assert not report["dynamics"]["ok"]
    assert "타임스텝 3개" in report["dynamics"]["reason"]


def test_steady_sweep_enables_rom_and_physics_not_dmd() -> None:
    result = service.make_demo_case_set("sweep")
    profile = strategies.profile_data(result["datasets"][0], result["datasets"])
    report = strategies.strategy_report(profile)
    assert report["rom"]["ok"] and report["physics"]["ok"]
    assert not report["dynamics"]["ok"]
    assert "ParametricDMD" in report["dynamics"]["reason"]
    # v5.2: 정상 스윕은 GeometryFNO(FNO+SDF)도 가능 — few-shot 경고가 남아야 한다.
    assert report["operator"]["ok"]
    assert "수백 장이 문헌 기준" in report["operator"]["reason"]
    assert "정성적" in report["operator"]["reason"]
    assert strategies.recommend(profile)["method"] == "rom"


def test_varying_mesh_only_physics_and_recommends_it() -> None:
    """진짜 구멍(케이스마다 다른 격자) → Physics AI 만 가능해야 한다."""
    import pyvista as pv

    from naviertwin.core.cfd_reader.base import CFDDataset

    cases = []
    for side in (5, 6):
        grid = pv.ImageData(dimensions=(side, side, 1)).cast_to_unstructured_grid()
        grid.point_data["p"] = np.linspace(0.0, 1.0, grid.n_points)
        cases.append(
            CFDDataset(mesh=grid, time_steps=[0.0], field_names=["p"], metadata={})
        )
    profile = strategies.profile_data(cases[0], cases)
    assert profile.identical_mesh is False
    report = strategies.strategy_report(profile)
    assert report["physics"]["ok"]
    assert not report["rom"]["ok"]
    assert "격자가 달라" in report["rom"]["reason"]
    # GeometryFNO 는 케이스 3개 미만이면 거절한다 (여긴 2개).
    assert not report["operator"]["ok"]
    assert "최소 3개" in report["operator"]["reason"]
    rec = strategies.recommend(profile)
    assert rec["method"] == "physics"
    assert "형상 가변" in rec["reason"]


def test_unsteady_sweep_rom_physics_ok_with_time_note() -> None:
    """비정상 스윕(v5.0): ROM/Physics 가능 + 추천 문구에 (μ, t) 언급."""
    result = service.make_demo_case_set("sweep_unsteady")
    profile = strategies.profile_data(result["datasets"][0], result["datasets"])
    assert profile.n_time_steps > 1
    assert profile.total_snapshots == sum(
        d.n_time_steps for d in result["datasets"]
    )
    report = strategies.strategy_report(profile)
    assert report["rom"]["ok"] and report["physics"]["ok"]
    # v5.2: 비정상 스윕은 ParametricDMD 로 동역학 예보도 가능하다.
    assert report["dynamics"]["ok"]
    assert "ParametricDMD" in report["dynamics"]["reason"]
    # GeometryFNO 는 정상 스윕 전용 — 시간축이 있으면 거절한다.
    assert not report["operator"]["ok"]
    assert "미지원" in report["operator"]["reason"]
    rec = strategies.recommend(profile)
    assert rec["method"] == "rom"
    assert "(μ, t)" in rec["reason"]


def test_recommend_method_wrapper_keeps_contract(tmp_path) -> None:
    """service.recommend_method 의 기존 계약: 12스텝→rom+'ROM', 1스텝→none."""
    demo = service.make_demo_dataset(nx=10, ny=10, n_steps=12)
    rec = service.recommend_method(demo)
    assert rec["method"] == "rom"
    assert "ROM" in rec["reason"]

    single = service.make_demo_dataset(nx=8, ny=8, n_steps=1)
    assert service.recommend_method(single)["method"] == "none"


def test_profile_dims_detects_2d() -> None:
    ds = service.make_demo_dataset(nx=8, ny=8, n_steps=2)
    assert strategies.profile_data(ds).dims == 2
