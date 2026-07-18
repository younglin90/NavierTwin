"""그룹 인지 train/val/test 분할 + train-only 정규화 + 4-way 평가 분할 분류.

외부 검토 반영: "정규화 계수 ... 는 train set 만 사용해서 계산해야 한다 ... 비정상
데이터는 전체 trajectory 를 하나의 그룹으로 분할해야 한다." (로드맵 §6½ 검토).
"""

from __future__ import annotations

import numpy as np
import pytest


class TestGroupTrainValTestSplit:
    def test_no_group_straddles_splits(self) -> None:
        from naviertwin.core.preprocessing.group_split import group_train_val_test_split

        # 3 그룹 × 4 케이스 = 12 케이스. val/test 가 실제로 그룹을 하나씩 떼어가도록
        # 비율을 잡는다 (round(3*0.34)=1).
        group_ids = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
        split = group_train_val_test_split(
            n_cases=12, group_ids=group_ids, val_frac=0.34, test_frac=0.34, seed=0
        )

        def groups_of(idx: np.ndarray) -> set[int]:
            return {group_ids[i] for i in idx.tolist()}

        train_groups = groups_of(split.train_idx)
        val_groups = groups_of(split.val_idx)
        test_groups = groups_of(split.test_idx)

        assert train_groups & val_groups == set()
        assert train_groups & test_groups == set()
        assert val_groups & test_groups == set()
        # 모든 케이스가 정확히 한 곳에 배정됨
        all_idx = np.concatenate([split.train_idx, split.val_idx, split.test_idx])
        assert sorted(all_idx.tolist()) == list(range(12))
        # 실제로 val/test 도 비었지 않음 (그룹 나뉨을 증명)
        assert split.val_idx.size > 0
        assert split.test_idx.size > 0

    def test_no_group_ids_is_per_case_split(self) -> None:
        from naviertwin.core.preprocessing.group_split import group_train_val_test_split

        split = group_train_val_test_split(n_cases=10, group_ids=None, val_frac=0.2, test_frac=0.1, seed=0)
        # group_ids=None → 케이스 하나 = 그룹 하나이므로 그룹 수 == 케이스 수(10)
        # 이고, val_frac/test_frac 이 정확히 케이스 비율로 반영된다.
        assert split.train_idx.size == 7
        assert split.val_idx.size == 2
        assert split.test_idx.size == 1
        all_idx = np.concatenate([split.train_idx, split.val_idx, split.test_idx])
        assert sorted(all_idx.tolist()) == list(range(10))

    def test_deterministic(self) -> None:
        from naviertwin.core.preprocessing.group_split import group_train_val_test_split

        group_ids = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        a = group_train_val_test_split(10, group_ids, val_frac=0.2, test_frac=0.2, seed=7)
        b = group_train_val_test_split(10, group_ids, val_frac=0.2, test_frac=0.2, seed=7)
        assert np.array_equal(a.train_idx, b.train_idx)
        assert np.array_equal(a.val_idx, b.val_idx)
        assert np.array_equal(a.test_idx, b.test_idx)

    def test_zero_cases_raises(self) -> None:
        from naviertwin.core.preprocessing.group_split import group_train_val_test_split

        with pytest.raises(ValueError):
            group_train_val_test_split(0)

    def test_zero_train_groups_raises(self) -> None:
        from naviertwin.core.preprocessing.group_split import group_train_val_test_split

        # 4 그룹, val=0.5(=2그룹) + test=0.5(=2그룹) → train 그룹 0개.
        group_ids = [0, 1, 2, 3]
        with pytest.raises(ValueError):
            group_train_val_test_split(4, group_ids, val_frac=0.5, test_frac=0.5, seed=0)


class TestTrainOnlyNormalizer:
    def test_train_only_vs_global_fit_diverge(self) -> None:
        """핵심 누수 검증: train 부분집합이 전체보다 좁은 범위일 때, train-only
        정규화와 (잘못된) 전체 데이터 정규화의 결과가 수치적으로 명확히 달라야
        한다 — 다르지 않다면 정규화가 실제로 train 통계량만 쓰고 있다는 증거가
        안 된다.

        구체적 수치 (train=[1,2,3,4,5], 나머지 5개는 값 100인 이상치):
            train-only  mean=3.0,  std=1.41421356
            전체(global) mean=51.5, std=48.51030818
            train-only 로 정규화한 train 평균  = 0.0 (정의상 항상 0)
            global 로 정규화한 train 평균      = -0.9997875... (거의 -1, 0과 크게 다름)
        """
        from naviertwin.core.preprocessing.group_split import TrainOnlyNormalizer

        X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        X_extra = np.full((5, 1), 100.0)
        X_full = np.vstack([X_train, X_extra])

        train_only = TrainOnlyNormalizer().fit(X_train)
        assert train_only.mean_ is not None
        np.testing.assert_allclose(train_only.mean_, [3.0])
        np.testing.assert_allclose(train_only.std_, [1.41421356], atol=1e-6)

        global_fit = TrainOnlyNormalizer().fit(X_full)  # 일부러 "틀린" fit (전체 데이터)
        np.testing.assert_allclose(global_fit.mean_, [51.5])
        np.testing.assert_allclose(global_fit.std_, [48.51030818], atol=1e-4)

        train_only_norm = train_only.transform(X_train)
        global_norm_on_train = global_fit.transform(X_train)

        # train-only 정규화는 정의상 train 부분집합에서 평균 0/표준편차 1.
        np.testing.assert_allclose(train_only_norm.mean(), 0.0, atol=1e-10)
        np.testing.assert_allclose(train_only_norm.std(), 1.0, atol=1e-10)

        # 전체 데이터로 fit 한 정규화는 이상치에 끌려가 train 평균이 0 근처가
        # 전혀 아니다 (거의 -1) — 두 결과가 측정 가능하게 다르다는 증거.
        assert not np.allclose(global_norm_on_train.mean(), 0.0, atol=0.5)
        np.testing.assert_allclose(global_norm_on_train.mean(), -0.9997875, atol=1e-4)

        # 같은 입력(X_train)인데 두 정규화 결과 자체도 값 단위로 명확히 다르다.
        assert not np.allclose(train_only_norm, global_norm_on_train, atol=0.1)

    def test_fit_transform_and_zero_variance_guard(self) -> None:
        from naviertwin.core.preprocessing.group_split import TrainOnlyNormalizer

        X_train = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])  # 2번째 열은 분산 0
        norm = TrainOnlyNormalizer()
        out = norm.fit_transform(X_train)
        assert np.all(np.isfinite(out))
        # 분산 0인 열은 eps 가드로 나눗셈 폭발 없이 전부 0이 됨.
        np.testing.assert_allclose(out[:, 1], [0.0, 0.0, 0.0])

    def test_transform_before_fit_raises(self) -> None:
        from naviertwin.core.preprocessing.group_split import TrainOnlyNormalizer

        with pytest.raises(RuntimeError):
            TrainOnlyNormalizer().transform(np.array([[1.0]]))


class TestClassifyQuerySplit:
    def _train(self) -> tuple[np.ndarray, np.ndarray]:
        train_params = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        geometry_ids = np.array([10, 10, 20])
        return train_params, geometry_ids

    def test_condition_interpolation(self) -> None:
        from naviertwin.core.preprocessing.group_split import classify_query_split

        train_params, geometry_ids = self._train()
        result = classify_query_split(
            [1.0, 1.0], train_params, geometry_ids=geometry_ids, query_geometry_id=10
        )
        assert result == "condition_interpolation"

    def test_condition_extrapolation(self) -> None:
        from naviertwin.core.preprocessing.group_split import classify_query_split

        train_params, geometry_ids = self._train()
        result = classify_query_split(
            [5.0, 5.0], train_params, geometry_ids=geometry_ids, query_geometry_id=10
        )
        assert result == "condition_extrapolation"

    def test_geometry_ood(self) -> None:
        from naviertwin.core.preprocessing.group_split import classify_query_split

        train_params, geometry_ids = self._train()
        result = classify_query_split(
            [1.0, 1.0], train_params, geometry_ids=geometry_ids, query_geometry_id=99
        )
        assert result == "geometry_ood"

    def test_joint_ood(self) -> None:
        from naviertwin.core.preprocessing.group_split import classify_query_split

        train_params, geometry_ids = self._train()
        result = classify_query_split(
            [5.0, 5.0], train_params, geometry_ids=geometry_ids, query_geometry_id=99
        )
        assert result == "joint_ood"

    def test_no_geometry_axis(self) -> None:
        from naviertwin.core.preprocessing.group_split import classify_query_split

        train_params, _ = self._train()
        assert classify_query_split([1.0, 1.0], train_params) == "condition_interpolation"
        assert classify_query_split([5.0, 5.0], train_params) == "condition_extrapolation"

    def test_missing_query_geometry_id_raises(self) -> None:
        from naviertwin.core.preprocessing.group_split import classify_query_split

        train_params, geometry_ids = self._train()
        with pytest.raises(ValueError):
            classify_query_split([1.0, 1.0], train_params, geometry_ids=geometry_ids)
