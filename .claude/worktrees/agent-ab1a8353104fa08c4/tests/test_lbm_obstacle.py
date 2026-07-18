"""장애물 LBM 솔버 + 카르만 데모 — 물리와 "진짜 구멍" 계약 검증.

여기서 지키려는 것 두 가지:

1. **물리가 진짜인가.** 그림이 그럴듯한 것과 유동이 맞는 것은 다르다. Re 에 따라
   정상/셔딩이 갈리고 스트로할 수가 문헌 범위에 드는지 본다 (프로토타입 단계에서
   실제로 "셔딩이 전혀 안 나는" 상태를 이걸로 잡았다).
2. **벽 안쪽에 셀이 없는가.** SDF·0채움 같은 가짜 empty 가 아니라 격자에서 셀이
   실제로 빠졌는지 확인한다.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyvista", reason="격자 구성에 pyvista 가 필요합니다.")

from naviertwin.core.solvers.lbm_obstacle_2d import (  # noqa: E402
    SHAPE_KINDS,
    shape_mask,
    solve_obstacle_flow,
)


def _strouhal(
    probe: np.ndarray, size: float, u_in: float, *, sample_every: int = 1
) -> tuple[float, float]:
    """프로브 신호에서 (St, 진폭). 진폭이 0 이면 셔딩하지 않는 것.

    Args:
        probe: 시간 신호 (``sample_every`` 스텝 간격으로 뽑은 값).
        size: 기준 길이 (격자 단위).
        u_in: 기준 속도 (격자 단위).
        sample_every: 샘플 간격(스텝). rfftfreq 는 **샘플당** 주파수를 주므로 이걸로
            스텝당으로 환산해야 한다 — 안 하면 St 가 이 값만큼 뻥튀기된다.
    """
    sig = probe - probe.mean()
    amp = float(np.abs(sig).max())
    if amp < 1e-4:
        return 0.0, amp
    spec = np.abs(np.fft.rfft(sig * np.hanning(sig.size)))
    freqs = np.fft.rfftfreq(sig.size, d=float(sample_every))  # 스텝당 주파수
    k = int(np.argmax(spec[1:]) + 1)
    return float(freqs[k]) * size / u_in, amp


# ──────────────────────────────────────────────────────────────────────
# 형상 마스크
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", SHAPE_KINDS)
def test_shape_mask_has_requested_frontal_height(kind: str) -> None:
    """정면 높이는 형상과 무관하게 같아야 케이스끼리 폐색률이 비교된다."""
    size = 24
    mask = shape_mask(120, 80, kind=kind, size=size)
    rows = np.flatnonzero(mask.any(axis=1))
    frontal = int(rows.max() - rows.min() + 1)
    assert abs(frontal - size) <= 2, f"{kind}: 정면 높이 {frontal} != {size}"
    assert mask.any() and not mask.all()


def test_shape_masks_are_actually_different_shapes() -> None:
    """세모/네모/원이 실제로 다른 모양이어야 형상 스윕이 의미가 있다."""
    masks = {k: shape_mask(120, 80, kind=k, size=24) for k in SHAPE_KINDS}
    areas = {k: int(m.sum()) for k, m in masks.items()}
    # 같은 정면 높이면 면적은 삼각형 < 원 < 사각형 순 (기하학적 사실).
    assert areas["triangle"] < areas["circle"] < areas["square"], areas
    for a, b in [("circle", "square"), ("circle", "triangle"), ("square", "triangle")]:
        assert not np.array_equal(masks[a], masks[b])


def test_shape_mask_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="지원하지 않는 형상"):
        shape_mask(60, 40, kind="hexagon", size=10)
    with pytest.raises(ValueError, match="격자 높이"):
        shape_mask(60, 40, kind="circle", size=40)


# ──────────────────────────────────────────────────────────────────────
# 물리 — 이게 이 모듈의 존재 이유다
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_low_reynolds_is_genuinely_steady_with_recirculation() -> None:
    """Re=20 → 셔딩 없음 + 후류에 재순환 거품 (진짜 정상해).

    "정상 케이스" 데모가 정상이라고 부를 자격이 있는지를 여기서 정한다.
    """
    nx, ny, size = 200, 80, 16
    solid = shape_mask(nx, ny, kind="circle", size=size)
    # 수렴 판정이 요점이라 float64 — float32 는 잔차가 ~3e-5 에서 바닥을 친다.
    res = solve_obstacle_flow(
        solid,
        u_in=0.1,
        reynolds=20.0,
        max_steps=20_000,
        steady_tol=5e-6,
        dtype=np.float64,
    )
    assert res["converged"], f"정상해로 수렴하지 못했습니다 (잔차 {res['residual']:.2e})"
    assert np.isfinite(res["ux"]).all()

    cx, cy = nx // 4, ny // 2
    wake = res["ux"][cy, cx + size // 2 + 1 : cx + 4 * size]
    assert (wake < 0).any(), "후류에 재순환(역류)이 없습니다 — 유동이 물체를 못 느낀 것"


@pytest.mark.slow
def test_karman_street_sheds_at_the_right_strouhal_number() -> None:
    """Re=120 → 카르만 셔딩, St 가 문헌 범위.

    이 테스트가 없었다면 "셔딩이 전혀 없는" 유동을 카르만 데모라고 내보낼 뻔했다
    (교란 없이 돌리면 진폭이 5e-5 에 머문다 — 눈으로는 구분이 안 된다).
    """
    nx, ny, size, u_in = 240, 96, 20, 0.1
    record_every = 20
    solid = shape_mask(nx, ny, kind="circle", size=size)
    res = solve_obstacle_flow(
        solid,
        u_in=u_in,
        reynolds=120.0,
        max_steps=16_000,
        record_every=record_every,
        record_from=8_000,
        perturb=2e-2,
        perturb_steps=2_600,
    )
    assert np.isfinite(res["ux"]).all()
    snaps = res["snapshots"]
    assert len(snaps) > 100

    cx, cy = nx // 4, ny // 2
    probe = np.array([s["uy"][cy, cx + 3 * size] for s in snaps])
    st, amp = _strouhal(probe, size, u_in, sample_every=record_every)

    assert amp > 1e-3, f"셔딩하지 않습니다 (진폭 {amp:.2e}) — 카르만 와열이 아님"
    # 폐색률 21% 라 문헌 무구속값(~0.17)보다 높게 나온다 — 벽 간섭의 알려진 효과.
    assert 0.14 < st < 0.26, f"스트로할 수가 물리적 범위 밖입니다: {st:.3f}"

    # 와열은 위아래로 번갈아 떨어진다 → 중심선 uy 의 부호가 여러 번 바뀐다.
    sign_changes = int(np.sum(np.diff(np.sign(probe - probe.mean())) != 0))
    assert sign_changes >= 6, f"주기적 셔딩이 아닙니다 (부호 변화 {sign_changes}회)"


def test_solver_rejects_impossible_setups() -> None:
    """설정이 물리적으로 불가능하면 조용히 쓰레기를 뱉지 말고 즉시 거절한다."""
    solid = shape_mask(60, 40, kind="circle", size=10)
    # 마하수 초과 — LBM 은 약압축성이라 격자 속도에 상한이 있다.
    with pytest.raises(ValueError, match="마하수"):
        solve_obstacle_flow(solid, u_in=0.9, reynolds=100.0, max_steps=1)
    # 점성이 0 에 가까우면 omega 가 2 에 붙어 BGK 가 불안정해진다.
    with pytest.raises(ValueError, match="불안정"):
        solve_obstacle_flow(solid, u_in=0.1, reynolds=1e6, max_steps=1)
    with pytest.raises(ValueError, match="reynolds 는 양수"):
        solve_obstacle_flow(solid, reynolds=-5.0, max_steps=1)
    # float32 는 잔차가 ~3e-5 에서 바닥을 쳐 그 아래 tol 은 영원히 수렴하지 않는다
    # — 조용히 max_steps 까지 도는 대신 즉시 알려준다.
    with pytest.raises(ValueError, match="float32 로는"):
        solve_obstacle_flow(solid, steady_tol=1e-6, dtype=np.float32, max_steps=1)
    with pytest.raises(ValueError, match="고체 노드가 없습니다"):
        solve_obstacle_flow(np.zeros((20, 20), dtype=bool), max_steps=1)
    with pytest.raises(ValueError, match="2D"):
        solve_obstacle_flow(np.ones((4, 4, 4), dtype=bool), max_steps=1)


def test_float32_matches_float64_physics() -> None:
    """빠르다고 물리가 달라지면 안 된다 — f32/f64 가 같은 유동을 내야 한다.

    f32 는 반올림이 누적돼 f64 와 정확히 같을 수 없다(1500 스텝에 상대 ~2e-4).
    여기서 잡으려는 건 그 잡음이 아니라 **경로가 갈라지는 것** — bounce-back 이나
    경계 처리가 깨지면 O(1) 로 벌어지므로 1e-3 이면 충분히 걸린다.
    """
    solid = shape_mask(160, 64, kind="circle", size=14)
    kwargs = dict(u_in=0.1, reynolds=40.0, max_steps=1500)
    a = solve_obstacle_flow(solid, dtype=np.float64, **kwargs)
    b = solve_obstacle_flow(solid, dtype=np.float32, **kwargs)
    assert np.isfinite(b["ux"]).all()
    scale = float(np.abs(a["ux"]).max())
    assert np.abs(a["ux"] - b["ux"]).max() / scale < 1e-3
    assert np.abs(a["uy"] - b["uy"]).max() / scale < 1e-3
    # 벽면 no-slip 은 정밀도와 무관하게 지켜져야 한다.
    assert np.abs(b["ux"][solid]).max() < 1e-6


# ──────────────────────────────────────────────────────────────────────
# 진짜 구멍 — 가짜 empty 금지
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", SHAPE_KINDS)
def test_obstacle_interior_has_no_cells_at_all(kind: str) -> None:
    """벽 안쪽은 셀이 **없어야** 한다 — 0으로 채운 셀도, 마스크된 셀도 아니다."""
    from naviertwin.web.demo_karman import grid_with_hole

    nx, ny, size = 120, 80, 24
    solid = shape_mask(nx, ny, kind=kind, size=size)
    mesh, keep = grid_with_hole(solid)

    full_cells = (nx - 1) * (ny - 1)
    assert mesh.n_cells < full_cells, "셀이 하나도 안 빠졌습니다 — 구멍이 없음"

    # 장애물 중심은 셀 안이 아니라 진짜 빈 공간이어야 한다.
    rows = np.flatnonzero(solid.any(axis=1))
    cols = np.flatnonzero(solid.any(axis=0))
    cx = float((cols.min() + cols.max()) / 2.0)
    cy = float((rows.min() + rows.max()) / 2.0)
    assert mesh.find_closest_cell([cx, cy, 0.0]) >= 0  # 격자 자체는 존재
    centers = np.asarray(mesh.cell_centers().points)
    dist = np.hypot(centers[:, 0] - cx, centers[:, 1] - cy)
    assert dist.min() > 2.0, (
        f"{kind}: 장애물 중심에서 {dist.min():.1f} 셀 거리에 셀이 남아 있습니다 — 구멍이 아님"
    )

    # 내부 고체 노드는 어떤 셀도 참조하지 않으므로 점에서도 사라진다.
    # 셀은 2×2 블록이라 "내부"는 **8이웃**이 모두 고체인 노드다 — 대각선으로만
    # 유체와 닿는 노드는 걸친 셀에 속하므로 남는 게 맞다.
    kept = np.zeros(nx * ny, dtype=bool)
    kept[keep] = True
    kept_2d = kept.reshape(ny, nx)
    inner = solid.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            inner &= np.roll(np.roll(solid, dy, axis=0), dx, axis=1)
    assert inner.any(), "테스트 전제 오류: 내부 노드가 하나도 없습니다"
    assert not kept_2d[inner].any(), "장애물 내부 노드가 격자에 남아 있습니다"


def test_wall_nodes_survive_with_no_slip_values() -> None:
    """벽면 노드는 살아남아야 한다 — no-slip(u=0) 이 데이터에 실제로 있어야 한다."""
    from naviertwin.web.demo_karman import grid_with_hole

    nx, ny, size = 120, 80, 24
    solid = shape_mask(nx, ny, kind="circle", size=size)
    _mesh, keep = grid_with_hole(solid)

    kept = np.zeros(nx * ny, dtype=bool)
    kept[keep] = True
    kept_2d = kept.reshape(ny, nx)
    # 벽면 = 고체이면서 유체와 인접한 노드
    fluid = ~solid
    adjacent = np.zeros_like(solid)
    for axis in (0, 1):
        for shift in (1, -1):
            adjacent |= np.roll(fluid, shift, axis=axis)
    wall = solid & adjacent
    assert wall.any()
    assert kept_2d[wall].any(), "벽면 노드가 전부 사라졌습니다 — no-slip 경계가 없음"


def test_bundled_demo_data_is_committed_and_loads_without_solving(
    monkeypatch, tmp_path
) -> None:
    """기본 데모는 저장소 번들 데이터에서 즉시 로드 — 절대 계산하지 않는다.

    이게 이 데모의 핵심 계약이다. 계산이 로드 경로에 새어들면(번들 파일이 없거나
    스펙이 안 맞으면) solve_obstacle_flow 가 불릴 텐데, 그걸 폭발시켜 잡는다.
    """
    from naviertwin.web import demo_karman

    # 커밋된 파일이 실제로 있어야 한다 (패키지에 포함되는 자산).
    assert (demo_karman._BUNDLED_DIR / "karman_unsteady.npz").exists()
    assert (demo_karman._BUNDLED_DIR / "karman_caseset.npz").exists()

    # 사용자 캐시를 빈 임시 폴더로 돌려 "새 컴퓨터" 를 흉내낸다.
    monkeypatch.setenv("NAVIERTWIN_DEMO_CACHE", str(tmp_path / "fresh"))

    def _boom(*_a, **_k):
        raise AssertionError("번들 데이터가 있는데 LBM 을 새로 계산했습니다 — 로드 경로 누수")

    monkeypatch.setattr(demo_karman, "solve_obstacle_flow", _boom)
    monkeypatch.setattr(demo_karman, "_solve_caseset", _boom)

    ds = demo_karman.build_karman_unsteady()
    assert ds.n_time_steps >= 2
    assert ds.mesh.n_points > 40_000  # 고화질 유지
    snaps = ds.extract_field_snapshots("p")
    assert not np.allclose(snaps[:, 0], snaps[:, -1])  # 시간에 따라 변한다

    result = demo_karman.build_karman_case_set()
    assert len(result["datasets"]) == 6
    assert result["resampled"] is False
    assert len({d.mesh.n_points for d in result["datasets"]}) > 1  # 케이스마다 다른 격자
    assert all(d.metadata["converged"] for d in result["datasets"])


def test_bundled_data_rejected_when_spec_differs(monkeypatch, tmp_path) -> None:
    """스펙이 다르면(해상도 등) 번들을 쓰지 않고 계산 경로로 간다 — 잘못된 즉답 방지."""
    from naviertwin.web import demo_karman

    spec_default = {"kind": "unsteady", "nx": 400}
    spec_other = {"kind": "unsteady", "nx": 123}
    # 기본 스펙 파일을 만들어두고
    demo_karman._save_solved(tmp_path / "b.npz", spec_default, {"velocity": np.zeros((2, 3))})
    # 다른 스펙으로 읽으면 거부돼야 한다.
    assert demo_karman._load_solved(tmp_path / "b.npz", spec_other) is None
    assert demo_karman._load_solved(tmp_path / "b.npz", spec_default) is not None


def test_demo_catalog_advertises_the_solved_demos() -> None:
    """카탈로그에 등록돼야 앱에서 고를 수 있다 — 기능만 있고 데이터가 없으면 무용."""
    from naviertwin.web import service

    values = {d["value"] for d in service.DEMO_CATALOG}
    assert {"karman", "karman_shapes"} <= values
    assert "karman" in service.DEMO_TIME_SERIES_KINDS
    assert "karman_shapes" in service.DEMO_CASE_SET_KINDS
    # 실제 해석이라 오래 걸린다는 표시 — 앱이 이걸로 진행률을 켠다.
    assert set(service.DEMO_SOLVED_KINDS) == {"karman", "karman_shapes"}
    for entry in service.DEMO_CATALOG:
        if entry["value"] in service.DEMO_SOLVED_KINDS:
            assert "구멍" in entry["note"] or "셀" in entry["note"]


def test_case_set_load_keeps_different_meshes_when_asked(tmp_path) -> None:
    """resample=False 면 메쉬가 달라도 그대로 둔다 — 예전엔 무조건 거절했다.

    진짜 구멍이 뚫린 격자를 공통 격자로 옮기면 구멍이 가짜 empty 가 되므로,
    "재샘플하지 않음" 이 선택 가능해야 한다.
    """
    import pyvista as pv

    from naviertwin.web.service import load_case_set

    for i, side in enumerate((5, 6)):
        grid = pv.ImageData(dimensions=(side, side, 1))
        grid.point_data["p"] = np.linspace(0.0, 1.0, grid.n_points)
        grid.save(str(tmp_path / f"case_{i:02d}.vtk"))
    (tmp_path / "params.csv").write_text("mu\n1.0\n2.0\n", encoding="utf-8")

    result = load_case_set(tmp_path, resample=False)
    assert result["resampled"] is False
    counts = [d.mesh.n_points for d in result["datasets"]]
    assert len(set(counts)) > 1, "메쉬가 그대로 유지되지 않았습니다"
    assert "재샘플 안 함" in result["grid_summary"]


def test_hole_grows_with_obstacle_size() -> None:
    """장애물이 커지면 남는 셀이 줄어야 한다 — 구멍이 형상을 실제로 따라간다."""
    from naviertwin.web.demo_karman import grid_with_hole

    counts = []
    for size in (12, 24, 36):
        solid = shape_mask(160, 96, kind="circle", size=size)
        mesh, _ = grid_with_hole(solid)
        counts.append(mesh.n_cells)
    assert counts[0] > counts[1] > counts[2], f"구멍이 크기를 안 따라갑니다: {counts}"
