"""그룹 인지(group-aware) train/val/test 분할 — 케이스/궤적 단위 데이터 누수 방지.

왜 필요한가 (외부 검토 반영): ``splitter.py`` 의 ``split_indices``/``split_snapshots``
는 스냅샷(행) 하나하나를 독립 샘플로 보고 무작위로 섞는다. 비정상(unsteady)
데이터에서 같은 궤적/케이스의 인접 시간 프레임은 서로 강하게 상관되어 있으므로,
이를 train 과 test 에 나눠 넣으면 시간 상관관계가 test 로 새어 들어가 성능이
과대평가된다. 이 모듈은 **케이스(궤적) 전체를 하나의 그룹**으로 묶어 분할한다 —
같은 그룹은 절대 train/val/test 를 가로지르지 않는다.

또한 정규화 계수·PCA basis·POD 모드 같은 전처리 통계량을 전체 데이터로 계산하면
test 정보가 전처리 단계에서부터 새어 들어간다 (train/test 분리 이전에 이미 test
분포를 "본" 것이 된다). :class:`TrainOnlyNormalizer` 는 이 원칙을 강제하는 얇은
래퍼다 — fit 은 반드시 train 인덱스만으로 호출해야 한다.

:func:`classify_query_split` 은 로드맵 검토가 요구하는 4가지 평가 분할 범주
(condition interpolation / condition extrapolation / geometry OOD / joint OOD)
를 판정하는 순수 분류기다. ``naviertwin.web.service.support_status`` 와는
**다른 개념**이니 혼동하지 말 것:

    - ``support_status`` (엔진 특화): 학습된 엔진의 파라미터 범위와 대조해
      IN_SUPPORT/NEAR_BOUNDARY/OUT_OF_SUPPORT 3단계 **연속** 마진을 준다 —
      "이 예측을 얼마나 믿을지"의 런타임 신호.
    - ``classify_query_split`` (엔진 무관): 이 검토가 요구하는 4-way **이산**
      평가 범주 이름을 준다 — "이 질의는 어떤 종류의 일반화 테스트인가"를
      평가 리포트에 라벨링하기 위한 것.

두 함수는 서로를 대체하지 않고 상호보완적이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "GroupSplit",
    "group_train_val_test_split",
    "TrainOnlyNormalizer",
    "classify_query_split",
]


@dataclass(frozen=True)
class GroupSplit:
    """케이스(그룹) 단위 train/val/test 분할 결과.

    Attributes:
        train_idx: train 에 속한 케이스 인덱스 (오름차순, ``int64``).
        val_idx: validation 에 속한 케이스 인덱스.
        test_idx: test 에 속한 케이스 인덱스.
    """

    train_idx: NDArray[np.int64]
    val_idx: NDArray[np.int64]
    test_idx: NDArray[np.int64]


def group_train_val_test_split(
    n_cases: int,
    group_ids: Sequence[int] | None = None,
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> GroupSplit:
    """케이스를 **그룹(group_ids) 단위**로 train/val/test 분할한다.

    같은 그룹(예: 같은 형상의 geometry_id, 또는 같은 비정상 궤적의 모든 시간
    프레임)은 절대 train/val/test 를 가로질러 배정되지 않는다 — 인접 시간
    프레임이 train 과 test 에 나뉘어 들어가는 시간 상관관계 누수를 막는다.

    ``group_ids`` 가 None 이면 케이스 각각이 자신만의 그룹이 된다 — 즉 표준
    케이스 단위 분할(그룹 크기 1)로 동작한다. 이는 스냅샷 단위로 섞는
    ``splitter.split_indices`` 보다 이미 안전하다: 최소한 "케이스" 라는 자연
    단위는 보존하기 때문이다.

    분할은 그룹 개수 기준으로 이뤄진다: 유니크 그룹을 ``seed`` 로 섞은 뒤
    앞에서부터 train, 다음 val, 나머지 test 로 자른다. 개별 그룹 크기가
    고르지 않으면 케이스 개수 비율이 ``val_frac``/``test_frac`` 과 정확히
    일치하지 않을 수 있다(그룹을 쪼갤 수 없으므로).

    Args:
        n_cases: 전체 케이스 수.
        group_ids: 케이스별 그룹 id (길이 ``n_cases``). 비정상 스윕이면 같은
            궤적의 모든 스냅샷에 같은 id 를, 형상 가변 스윕이면 같은 형상의
            모든 케이스에 같은 geometry_id 를 준다. None 이면 케이스=그룹.
        val_frac: validation 에 배정할 그룹 비율.
        test_frac: test 에 배정할 그룹 비율.
        seed: 그룹 셔플 시드 — 같은 시드는 항상 같은 분할을 재현한다.

    Returns:
        :class:`GroupSplit`.

    Raises:
        ValueError: ``n_cases == 0`` 이거나, ``group_ids`` 길이가 ``n_cases``
            와 다르거나, ``val_frac``/``test_frac`` 이 [0, 1) 밖이거나,
            그 합이 1 이상이거나, 배정 후 train 그룹이 0개가 되는 경우
            (그룹 수가 너무 적은데 val/test 비율이 너무 큰 경우).
    """
    if n_cases <= 0:
        raise ValueError("n_cases 는 1 이상이어야 합니다 (받음: 0).")
    if not (0.0 <= val_frac < 1.0) or not (0.0 <= test_frac < 1.0):
        raise ValueError("val_frac, test_frac 은 [0, 1) 범위여야 합니다.")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac 은 1 미만이어야 합니다 (train 이 남아야 함).")

    if group_ids is None:
        case_groups = np.arange(n_cases, dtype=np.int64)
    else:
        case_groups = np.asarray(group_ids)
        if case_groups.shape[0] != n_cases:
            raise ValueError(
                f"group_ids 길이({case_groups.shape[0]})가 n_cases({n_cases})와 다릅니다."
            )

    unique_groups = np.unique(case_groups)
    n_groups = unique_groups.shape[0]

    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)

    n_val_groups = int(round(n_groups * val_frac))
    n_test_groups = int(round(n_groups * test_frac))
    n_train_groups = n_groups - n_val_groups - n_test_groups

    if n_train_groups < 1:
        raise ValueError(
            f"그룹 {n_groups}개 중 val={n_val_groups}개, test={n_test_groups}개를 "
            "배정하면 train 그룹이 0개가 됩니다 — val_frac/test_frac 을 줄이거나 "
            "그룹 수를 늘리세요."
        )

    train_groups = set(shuffled[:n_train_groups].tolist())
    val_groups = set(shuffled[n_train_groups : n_train_groups + n_val_groups].tolist())
    test_groups = set(shuffled[n_train_groups + n_val_groups :].tolist())

    case_idx = np.arange(n_cases, dtype=np.int64)
    train_mask = np.array([g in train_groups for g in case_groups.tolist()])
    val_mask = np.array([g in val_groups for g in case_groups.tolist()])
    test_mask = np.array([g in test_groups for g in case_groups.tolist()])

    return GroupSplit(
        train_idx=np.sort(case_idx[train_mask]),
        val_idx=np.sort(case_idx[val_mask]),
        test_idx=np.sort(case_idx[test_mask]),
    )


class TrainOnlyNormalizer:
    """열(column) 단위 평균/표준편차 정규화 — **반드시 train 인덱스에만 fit**.

    전체(train+val+test) 데이터로 평균/표준편차를 계산하면 test 분포의 정보가
    전처리 단계에서 이미 모델에 스며든다(정규화 누수). 이 클래스는 그 실수를
    막기 위한 얇은 래퍼일 뿐이다 — ``fit`` 에 넘기는 배열이 train 부분집합인지
    확인하는 것은 호출자의 책임이다.

    Examples:
        틀린 사용 (전체 데이터로 fit — test 정보 유입)::

            >>> normalizer = TrainOnlyNormalizer()
            >>> normalizer.fit(X_full)  # doctest: +SKIP
            >>> X_all_norm = normalizer.transform(X_full)  # doctest: +SKIP

        올바른 사용 (train 부분집합으로만 fit)::

            >>> split = group_train_val_test_split(len(X_full))  # doctest: +SKIP
            >>> normalizer = TrainOnlyNormalizer().fit(X_full[split.train_idx])  # doctest: +SKIP
            >>> X_all_norm = normalizer.transform(X_full)  # doctest: +SKIP

    Attributes:
        mean_: fit 된 열별 평균 (fit 전에는 None).
        std_: fit 된 열별 표준편차 (0에 가까운 열은 ``eps`` 로 대체됨).
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = float(eps)
        self.mean_: NDArray[np.float64] | None = None
        self.std_: NDArray[np.float64] | None = None

    def fit(self, X_train: NDArray[np.float64]) -> "TrainOnlyNormalizer":
        """train 부분집합 ``X_train`` 에서 열별 평균/표준편차를 계산한다.

        Args:
            X_train: **train 인덱스만** 포함하는 (n_train, n_features) 배열.
                val/test/전체 데이터를 넘기면 정규화 누수가 발생한다.

        Returns:
            self (체이닝용).
        """
        arr = np.asarray(X_train, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std = np.where(std < self.eps, 1.0, std)
        self.mean_ = mean
        self.std_ = std
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """``(X - mean_) / std_`` — fit 에서 계산된 train 통계량으로 변환한다."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("fit() 을 먼저 호출해야 합니다 (train 부분집합으로).")
        arr = np.asarray(X, dtype=np.float64)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr.reshape(-1, 1)
        out = (arr - self.mean_) / self.std_
        return out.reshape(-1) if squeeze else out

    def fit_transform(self, X_train: NDArray[np.float64]) -> NDArray[np.float64]:
        """``fit(X_train).transform(X_train)`` 의 축약형."""
        return self.fit(X_train).transform(X_train)


def classify_query_split(
    query_params: NDArray[np.float64],
    train_params: NDArray[np.float64],
    *,
    geometry_ids: NDArray[np.int64] | None = None,
    query_geometry_id: int | None = None,
    tol_frac: float = 0.0,
) -> str:
    """평가 질의를 4가지 일반화 범주 중 하나로 분류한다.

    - ``"condition_interpolation"``: 파라미터가 학습 범위 안 **이고**(geometry_ids
      가 주어졌다면) 질의 형상이 학습 형상 집합에 존재.
    - ``"condition_extrapolation"``: 파라미터가 학습 범위 밖이지만 형상은 학습에
      쓰인 형상과 같음(또는 geometry_ids 를 안 쓰는 경우).
    - ``"geometry_ood"``: 형상은 학습에 없던 것이지만 파라미터는 학습 범위 안.
    - ``"joint_ood"``: 형상도 처음 보고 파라미터도 학습 범위 밖 — 가장 어려운
      일반화 테스트.

    ``geometry_ids`` 를 주지 않으면 형상 축을 아예 판정하지 않으므로
    ``"condition_interpolation"``/``"condition_extrapolation"`` 두 값만 나온다.

    Args:
        query_params: 질의 파라미터 벡터, shape (k,).
        train_params: 학습에 쓰인 파라미터 표, shape (n_train, k).
        geometry_ids: 학습 케이스별 geometry id, shape (n_train,). None 이면
            형상 축 판정을 생략한다.
        query_geometry_id: 질의의 geometry id. ``geometry_ids`` 를 줬다면
            반드시 함께 줘야 한다.
        tol_frac: 학습 범위를 얼마나 여유 있게 볼지 (범위 폭 대비 비율,
            :func:`naviertwin.web.service.support_status` 의 15% 마진과 같은
            개념이지만 여기서는 이산 판정이라 기본값이 0이다).

    Returns:
        위 4개 문자열 중 하나.

    Raises:
        ValueError: ``geometry_ids`` 는 줬는데 ``query_geometry_id`` 를 안 준 경우,
            또는 ``train_params`` 가 비어있는 경우.
    """
    train_arr = np.atleast_2d(np.asarray(train_params, dtype=np.float64))
    if train_arr.size == 0:
        raise ValueError("train_params 가 비어 있습니다.")
    if geometry_ids is not None and query_geometry_id is None:
        raise ValueError("geometry_ids 를 주면 query_geometry_id 도 함께 줘야 합니다.")

    query = np.asarray(query_params, dtype=np.float64).reshape(-1)
    lo = train_arr.min(axis=0)
    hi = train_arr.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    tol = tol_frac * span
    n = min(query.size, lo.size)
    in_range = bool(np.all((query[:n] >= lo[:n] - tol[:n]) & (query[:n] <= hi[:n] + tol[:n])))

    if geometry_ids is None:
        return "condition_interpolation" if in_range else "condition_extrapolation"

    known_geometries = set(np.asarray(geometry_ids).tolist())
    geometry_known = query_geometry_id in known_geometries

    if geometry_known and in_range:
        return "condition_interpolation"
    if geometry_known and not in_range:
        return "condition_extrapolation"
    if not geometry_known and in_range:
        return "geometry_ood"
    return "joint_ood"
