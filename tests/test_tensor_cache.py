"""ML 텐서 캐시(Zarr) 테스트 — 저장 계층 단계 2 (외부 검토 §6½ #6).

지키려는 계약:
    - put→get 왕복이 배열 3개(inputs/targets/valid_mask)를 bit-동일하게
      복원하고 meta 도 일치한다. grid 는 meta(dims/spacing/origin)에서
      재구성돼 원본과 같은 격자다.
    - 캐시 키는 콘텐츠 주소다 — 메쉬/파라미터/해상도/필드 중 하나만 바뀌어도
      키가 바뀌고, 같은 입력은 언제나 같은 키(결정적)다.
    - ``build_geometry_fno_twin`` 은 캐시 히트여도 결과(train_loss)가
      동일하다 — 캐시는 절대 정확성을 해치면 안 된다.
    - 손상된 캐시 항목은 조용히 miss 처리되고 재계산으로 폴백한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyvista", reason="케이스 텐서화에 pyvista 가 필요합니다.")
pytest.importorskip("zarr", reason="텐서 캐시는 zarr 선택 의존성이 필요합니다.")

from naviertwin.core.operator_learning.fno.case_tensorizer import (  # noqa: E402
    cases_to_grid_tensors,
)
from naviertwin.core.storage.tensor_cache import TensorCache  # noqa: E402
from naviertwin.web import service  # noqa: E402

# 스모크 스케일 — 해상도/epoch 를 최소로 (전체 3분 이내 목표).
_RESOLUTION = 12
_TINY = {"modes": 6, "width": 8, "epochs": 3}


@pytest.fixture(scope="module")
def shapes() -> dict:
    """반지름이 다른 원기둥 5케이스 데모 (module 공유 — 재생성 방지)."""
    return service.make_demo_case_set("shapes")


@pytest.fixture(scope="module")
def shapes_tensors(shapes: dict) -> dict:
    """shapes 데모의 텐서화 결과 (module 공유)."""
    return cases_to_grid_tensors(
        shapes["datasets"],
        shapes["params"],
        field_names=["p"],
        resolution=_RESOLUTION,
        param_names=shapes["param_names"],
    )


def _key(shapes: dict, **overrides) -> str:
    """기본 인자에서 일부만 바꿔 key_for 를 부르는 헬퍼."""
    kwargs = {
        "datasets": shapes["datasets"],
        "params": shapes["params"],
        "field_names": ["p"],
        "resolution": _RESOLUTION,
        "param_names": shapes["param_names"],
    }
    kwargs.update(overrides)
    return TensorCache.key_for(
        kwargs["datasets"],
        kwargs["params"],
        kwargs["field_names"],
        kwargs["resolution"],
        kwargs["param_names"],
    )


# ──────────────────────────────────────────────────────────────────────
# (a) put→get 왕복
# ──────────────────────────────────────────────────────────────────────


def test_put_get_roundtrip_bit_identical(
    tmp_path: Path, shapes: dict, shapes_tensors: dict
) -> None:
    cache = TensorCache(cache_dir=tmp_path / "cache")
    key = _key(shapes)
    assert cache.get(key) is None  # 빈 캐시 → miss

    cache.put(key, shapes_tensors)
    got = cache.get(key)
    assert got is not None

    # 배열 3개 bit-동일 (dtype 포함).
    for name in ("inputs", "targets", "valid_mask"):
        assert got[name].dtype == shapes_tensors[name].dtype
        assert np.array_equal(got[name], shapes_tensors[name]), name

    # meta 일치 — tuple 규약까지 복원된다.
    assert got["meta"] == shapes_tensors["meta"]
    assert got["channel_names"] == shapes_tensors["channel_names"]

    # grid 는 meta 에서 재구성 — 원본과 같은 격자다.
    original_grid = shapes_tensors["grid"]
    assert tuple(got["grid"].dimensions) == tuple(original_grid.dimensions)
    assert got["grid"].spacing == pytest.approx(original_grid.spacing)
    assert got["grid"].origin == pytest.approx(original_grid.origin)
    assert got["grid"].n_points == original_grid.n_points

    # stats: 항목 1개, 0 바이트 초과.
    stats = cache.stats()
    assert stats["n_entries"] == 1
    assert stats["total_bytes"] > 0

    # clear 후 miss.
    cache.clear()
    assert cache.get(key) is None
    assert cache.stats()["n_entries"] == 0


# ──────────────────────────────────────────────────────────────────────
# (b) 키가 입력 내용에 반응한다 / (c) 결정적
# ──────────────────────────────────────────────────────────────────────


def test_key_changes_when_any_input_changes(shapes: dict) -> None:
    base = _key(shapes)

    # 메쉬 변경 — 케이스 하나를 이동시킨 사본으로 교체.
    moved = [ds.mesh.copy() for ds in shapes["datasets"]]
    moved[0] = moved[0].translate((0.01, 0.0, 0.0))
    assert _key(shapes, datasets=moved) != base

    # 파라미터 변경 — 값 하나만 1 ULP 수준이 아니라 눈에 띄게 바꾼다.
    params2 = np.asarray(shapes["params"], dtype=np.float64).copy()
    params2.flat[0] += 0.001
    assert _key(shapes, params=params2) != base

    # 해상도 변경.
    assert _key(shapes, resolution=_RESOLUTION + 1) != base

    # 필드 변경.
    assert _key(shapes, field_names=["U"]) != base
    assert _key(shapes, field_names=["p", "U"]) != base

    # 파라미터명 변경.
    assert _key(shapes, param_names=["other_name"]) != base


def test_key_is_deterministic(shapes: dict) -> None:
    assert _key(shapes) == _key(shapes)
    # 형식: sha256 앞 16 hex.
    key = _key(shapes)
    assert len(key) == 16
    int(key, 16)  # hex 파싱 가능


# ──────────────────────────────────────────────────────────────────────
# (d) build_geometry_fno_twin: 캐시 히트여도 학습 결과 동일
# ──────────────────────────────────────────────────────────────────────


def test_build_twin_cache_hit_gives_identical_train_loss(
    tmp_path: Path, shapes: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch", reason="GeometryFNO 학습에 torch 가 필요합니다.")
    import naviertwin.core.operator_learning.fno.case_tensorizer as tensorizer_mod

    cache_dir = tmp_path / "twin_cache"
    calls = {"n": 0}
    real_fn = tensorizer_mod.cases_to_grid_tensors

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(tensorizer_mod, "cases_to_grid_tensors", counting)

    def build() -> dict:
        return service.build_geometry_fno_twin(
            shapes["datasets"],
            "p",
            shapes["params"],
            param_names=shapes["param_names"],
            resolution=_RESOLUTION,
            tensor_cache_dir=cache_dir,
            **_TINY,
        )

    first = build()
    assert calls["n"] == 1  # miss → 계산 1회 + put

    second = build()
    assert calls["n"] == 1  # hit → 텐서화 재계산 없음

    # 캐시 히트여도 학습 결과가 동일하다 (결정적 seed=0 기본값).
    assert second["train_loss"] == first["train_loss"]
    assert second["grid_summary"] == first["grid_summary"]
    assert second["target_names"] == first["target_names"]
    pred_first = first["engine"].predict(np.asarray([0.12]))
    pred_second = second["engine"].predict(np.asarray([0.12]))
    assert np.array_equal(pred_first, pred_second)

    # 캐시 항목은 1개뿐이다 (같은 입력 → 같은 키).
    assert TensorCache(cache_dir=cache_dir).stats()["n_entries"] == 1

    # use_tensor_cache=False 는 캐시를 건드리지 않고 재계산한다.
    third = service.build_geometry_fno_twin(
        shapes["datasets"],
        "p",
        shapes["params"],
        param_names=shapes["param_names"],
        resolution=_RESOLUTION,
        use_tensor_cache=False,
        tensor_cache_dir=cache_dir,
        **_TINY,
    )
    assert calls["n"] == 2
    assert third["train_loss"] == first["train_loss"]


# ──────────────────────────────────────────────────────────────────────
# (e) 손상 캐시 → 조용한 miss + 재계산
# ──────────────────────────────────────────────────────────────────────


def test_corrupted_entry_is_silent_miss(
    tmp_path: Path, shapes: dict, shapes_tensors: dict
) -> None:
    cache = TensorCache(cache_dir=tmp_path / "cache")
    key = _key(shapes)
    cache.put(key, shapes_tensors)
    assert cache.get(key) is not None

    # 항목 내부 파일을 일부 삭제해 손상시킨다 — inputs 배열부터 없앤다.
    entry = cache.cache_dir / key
    victims = sorted(entry.rglob("*inputs*"))
    assert victims, "손상 대상(inputs 관련 파일)이 있어야 한다"
    import shutil

    for victim in victims:
        if victim.is_dir():
            shutil.rmtree(victim)
        elif victim.exists():
            victim.unlink()

    # 예외 없이 miss.
    assert cache.get(key) is None

    # put 으로 다시 채우면 회복된다 (재계산 폴백과 같은 경로).
    cache.put(key, shapes_tensors)
    got = cache.get(key)
    assert got is not None
    assert np.array_equal(got["inputs"], shapes_tensors["inputs"])


def test_totally_bogus_entry_is_silent_miss(tmp_path: Path) -> None:
    cache = TensorCache(cache_dir=tmp_path / "cache")
    bogus = cache.cache_dir / "deadbeef00000000"
    bogus.mkdir(parents=True)
    (bogus / "zarr.json").write_text("this is not json{{{", encoding="utf-8")
    assert cache.get("deadbeef00000000") is None  # 예외 없이 miss
