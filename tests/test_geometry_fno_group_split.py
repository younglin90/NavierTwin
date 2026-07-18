"""GeometryFNO 학습 경로에 배선된 그룹 스플릿 (검토 §6½ #2, v5.6 P1+).

지키려는 계약:
    - ``group_split=False``(기본)는 이전과 동일하게 전체 케이스로 학습한다
      (``eval_split == {"enabled": False}``, 하위 호환).
    - ``group_split=True`` 면 케이스가 train/val/test 로 나뉘고, **train
      케이스만** 실제로 학습에 쓰인다(held-out 케이스 수만큼 엔진의 학습
      케이스 수가 줄어든다) — 그렇지 않으면 "그룹 스플릿"이 이름만 남는다.
    - held-out(val/test) 케이스는 순수 평가 전용 — 학습에 쓰인 공통 격자
      위에서 바로 오차를 재고, 4-way 일반화 라벨(``classify_query_split``)이
      함께 붙는다.
    - 같은 ``split_seed`` 는 항상 같은 분할을 재현한다.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyvista", reason="케이스 텐서화에 pyvista 가 필요합니다.")
pytest.importorskip("torch", reason="GeometryFNO 학습에 torch 가 필요합니다.")

from naviertwin.web import service  # noqa: E402

_TINY = {"modes": 6, "width": 8, "epochs": 5}
_RESOLUTION = 16


@pytest.fixture(scope="module")
def shapes_case_set() -> dict:
    """반지름이 다른 원기둥 5케이스 (정상, 공통 격자로 재샘플된 데모)."""
    return service.make_demo_case_set("shapes")


def test_group_split_disabled_by_default_matches_old_contract(shapes_case_set: dict) -> None:
    built = service.build_geometry_fno_twin(
        shapes_case_set["datasets"],
        "p",
        shapes_case_set["params"],
        param_names=shapes_case_set["param_names"],
        resolution=_RESOLUTION,
        **_TINY,
    )
    assert built["eval_split"] == {"enabled": False}
    # 전체 5케이스가 학습에 쓰인다 — 이전과 동일.
    assert built["engine"].training_metadata["n_cases"] == 5


def test_group_split_holds_out_cases_from_training(shapes_case_set: dict) -> None:
    built = service.build_geometry_fno_twin(
        shapes_case_set["datasets"],
        "p",
        shapes_case_set["params"],
        param_names=shapes_case_set["param_names"],
        resolution=_RESOLUTION,
        group_split=True,
        val_frac=0.2,
        test_frac=0.2,
        split_seed=0,
        **_TINY,
    )
    eval_split = built["eval_split"]
    assert eval_split["enabled"] is True
    train_idx = eval_split["train_idx"]
    val_idx = eval_split["val_idx"]
    test_idx = eval_split["test_idx"]
    # 5케이스 → 그룹 분할이 겹치지 않고 전부를 덮는다.
    assert sorted(train_idx + val_idx + test_idx) == list(range(5))
    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx).isdisjoint(test_idx)
    assert set(val_idx).isdisjoint(test_idx)
    # held-out 케이스가 실제로 있어야 이 테스트가 의미 있다.
    assert val_idx or test_idx
    # 엔진은 train 케이스만으로 학습됐다 — 전체(5)보다 적어야 한다.
    assert built["engine"].training_metadata["n_cases"] == len(train_idx)
    assert built["engine"].training_metadata["n_cases"] < 5


def test_group_split_holdout_metrics_are_finite_and_labeled(shapes_case_set: dict) -> None:
    built = service.build_geometry_fno_twin(
        shapes_case_set["datasets"],
        "p",
        shapes_case_set["params"],
        param_names=shapes_case_set["param_names"],
        resolution=_RESOLUTION,
        group_split=True,
        val_frac=0.2,
        test_frac=0.2,
        split_seed=0,
        **_TINY,
    )
    holdout = built["eval_split"]["holdout"]
    assert len(holdout) == len(built["eval_split"]["val_idx"]) + len(
        built["eval_split"]["test_idx"]
    )
    valid_classes = {
        "condition_interpolation",
        "condition_extrapolation",
        "geometry_ood",
        "joint_ood",
    }
    for row in holdout:
        assert row["split"] in {"val", "test"}
        assert row["query_split_class"] in valid_classes
        assert np.isfinite(row["rel_l2"])
        assert row["rel_l2"] >= 0.0
        assert np.isfinite(row["rmse"])
        assert row["rmse"] >= 0.0

    if built["eval_split"]["val_rel_l2_mean"] is not None:
        assert built["eval_split"]["val_rel_l2_mean"] >= 0.0
    if built["eval_split"]["test_rel_l2_mean"] is not None:
        assert built["eval_split"]["test_rel_l2_mean"] >= 0.0


def test_group_split_with_explicit_group_ids_flags_geometry_ood(shapes_case_set: dict) -> None:
    """geometry_ids 를 주면 held-out 케이스는 처음 보는 형상이라 *_ood 로 갈린다.

    shapes 데모는 케이스마다 다른 반지름(=다른 형상)이라, 케이스 자신을
    geometry_id 로 주면 held-out 케이스의 형상은 학습에 전혀 없던 것이다 —
    파라미터(반지름)가 학습 범위 안이면 geometry_ood, 밖이면 joint_ood 다.
    """
    n_cases = len(shapes_case_set["datasets"])
    built = service.build_geometry_fno_twin(
        shapes_case_set["datasets"],
        "p",
        shapes_case_set["params"],
        param_names=shapes_case_set["param_names"],
        resolution=_RESOLUTION,
        group_split=True,
        group_ids=list(range(n_cases)),  # 케이스 == 형상 == 그룹
        val_frac=0.2,
        test_frac=0.2,
        split_seed=0,
        **_TINY,
    )
    holdout = built["eval_split"]["holdout"]
    assert holdout, "이 시드에서는 held-out 케이스가 있어야 테스트가 의미 있다"
    for row in holdout:
        assert row["query_split_class"] in {"geometry_ood", "joint_ood"}


def test_group_split_seed_is_reproducible(shapes_case_set: dict) -> None:
    kwargs = dict(
        param_names=shapes_case_set["param_names"],
        resolution=_RESOLUTION,
        group_split=True,
        val_frac=0.2,
        test_frac=0.2,
        split_seed=7,
        **_TINY,
    )
    first = service.build_geometry_fno_twin(
        shapes_case_set["datasets"], "p", shapes_case_set["params"], **kwargs
    )
    second = service.build_geometry_fno_twin(
        shapes_case_set["datasets"], "p", shapes_case_set["params"], **kwargs
    )
    assert first["eval_split"]["train_idx"] == second["eval_split"]["train_idx"]
    assert first["eval_split"]["val_idx"] == second["eval_split"]["val_idx"]
    assert first["eval_split"]["test_idx"] == second["eval_split"]["test_idx"]


def test_group_split_too_few_cases_for_split_keeps_all_in_train(shapes_case_set: dict) -> None:
    """분할 비율이 반올림돼 val/test 그룹이 0개면 전부 train — 에러가 아니다."""
    two_cases = shapes_case_set["datasets"][:2]
    two_params = shapes_case_set["params"][:2]
    built = service.build_geometry_fno_twin(
        two_cases,
        "p",
        two_params,
        param_names=shapes_case_set["param_names"],
        resolution=_RESOLUTION,
        group_split=True,
        val_frac=0.15,
        test_frac=0.15,
        **_TINY,
    )
    assert built["eval_split"]["val_idx"] == []
    assert built["eval_split"]["test_idx"] == []
    assert built["eval_split"]["holdout"] == []
    assert built["eval_split"]["val_rel_l2_mean"] is None
    assert built["eval_split"]["test_rel_l2_mean"] is None
