"""재샘플 오차 바닥(remap error floor) — 검토 §11.2 반영.

핵심 계약: truth 를 공통 격자로 보냈다가 원본 좌표로 되돌린 왕복 오차가
**해상도가 오를수록 0 으로 수렴**해야 한다. 첫 구현은 VTK 가 경계 근처
무효점을 NaN 이 아니라 0-채움으로 표시(``vtkValidPointMask``)한다는 걸
놓쳐서, 해상도를 아무리 올려도 오차가 ~0.31 에서 안 떨어지는 버그가 있었다
— 이 파일의 단조 감소 테스트가 그 버그를 정확히 잡는다.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyvista", reason="재샘플에는 pyvista 가 필요합니다.")

from naviertwin.web import service  # noqa: E402


def test_remap_floor_shrinks_toward_zero_with_resolution() -> None:
    """해상도를 올릴수록 왕복 오차가 단조 감소해 0 에 가까워져야 한다.

    이 테스트가 없었다면 vtkValidPointMask 미필터링 버그(res→∞ 에도
    rel_l2≈0.31 고정)를 못 잡았다 — 원본보다 훨씬 촘촘한 격자로 왕복하면
    오차가 사실상 사라져야 하는 게 물리적으로 당연하기 때문이다.
    """
    demo = service.make_demo_dataset(nx=40, ny=40, n_steps=1)
    floors = [
        service.estimate_remap_floor(demo, "p", resolution=res)["rel_l2"]
        for res in (10, 20, 40, 160)
    ]
    assert all(f >= 0 for f in floors)
    # 엄격한 단조 감소(같은 방향으로만) — 노이즈 허용 없이.
    assert floors == sorted(floors, reverse=True), f"단조 감소가 아닙니다: {floors}"
    # 원본(40x40)보다 4배 촘촘한 격자로 왕복하면 사실상 원본을 복원해야 한다.
    assert floors[-1] < 0.02, f"고해상도 왕복 오차가 너무 큽니다: {floors[-1]}"
    # 아주 성긴 격자는 뚜렷한 손실이 있어야 한다(0 이면 테스트 자체가 무의미).
    assert floors[0] > 0.05


def test_remap_floor_excludes_invalid_boundary_points() -> None:
    """vtkValidPointMask 로 걸러진 무효점이 오차 계산에 안 들어가야 한다.

    직접 왕복을 재현해 무효점 비율을 확인하고, 그 점들을 포함/제외했을 때
    결과가 실제로 달라짐을 보여 필터링이 살아있는지 확인한다.
    """
    demo = service.make_demo_dataset(nx=30, ny=30, n_steps=1)
    mesh = demo.mesh
    grid = service._uniform_grid_over(mesh.bounds, 80)
    forward = grid.sample(mesh)
    back = mesh.sample(forward)
    mask = np.asarray(back.point_data["vtkValidPointMask"]).astype(bool)
    assert not mask.all(), "테스트 전제: 경계 근처에 무효점이 있어야 한다"

    original = np.asarray(mesh.point_data["p"])
    roundtrip = np.asarray(back.point_data["p"])
    filtered = service.compute_error_field(original[mask], roundtrip[mask])
    unfiltered = service.compute_error_field(original, roundtrip)
    assert filtered["rel_l2"] < unfiltered["rel_l2"], (
        "무효점을 걸러도 오차가 안 줄면 필터링이 실제로 동작하지 않는 것"
    )
    # service 함수 자체가 filtered 와 일치해야 한다(같은 마스킹을 내부에서 함).
    result = service.estimate_remap_floor(demo, "p", resolution=80)
    assert result["rel_l2"] == pytest.approx(filtered["rel_l2"], rel=1e-6)


def test_remap_floor_handles_missing_field() -> None:
    demo = service.make_demo_dataset(nx=16, ny=16, n_steps=1)
    result = service.estimate_remap_floor(demo, "no_such_field", resolution=20)
    assert result["rel_l2"] == 0.0
    assert result["note"]  # 이유가 남는다


def test_geometry_fno_reports_remap_floor(tmp_path) -> None:
    """GeometryFNO 학습 결과에 대표 케이스의 오차 바닥이 딸려 온다."""
    result = service.make_demo_case_set("shapes")
    twin = service.build_geometry_fno_twin(
        result["datasets"],
        "p",
        result["params"],
        param_names=result["param_names"],
        resolution=16,
        modes=6,
        width=12,
        epochs=20,
    )
    assert "remap_floor_rel_l2" in twin
    assert twin["remap_floor_rel_l2"] >= 0.0
    assert twin["engine"].training_metadata["remap_floor_rel_l2"] == twin["remap_floor_rel_l2"]
