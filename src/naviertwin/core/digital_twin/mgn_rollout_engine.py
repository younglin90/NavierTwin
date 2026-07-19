"""MeshGraphNets 롤아웃 트윈 엔진 — 단일 케이스 시계열 자기회귀 예보 (Route 2, 원본 메쉬).

:mod:`~naviertwin.core.digital_twin.mgn_case_set_engine`
(:class:`MGNCaseSetTwinEngine`, 전략 키 ``mesh_gnn_mp``)와는 **완전히 다른
용도**다 — 그쪽은 정상(steady) 케이스 세트를 파라미터→필드 회귀로 감싸고,
이쪽은 원본 :class:`~naviertwin.core.gnn.meshgraphnets.meshgraphnets.
MeshGraphNets` 의 **진짜 자기회귀 시간 롤아웃**(``u_{t+1} = u_t +
MGN(u_t, edge)``)을 그대로 감싸 단일 케이스 시계열에서 학습 구간 밖까지
예보한다. ``predict(t)`` 는 시간값을 스텝 수로 바꿔 필요한 만큼만
:meth:`MeshGraphNets.predict` 를 호출하고, 이미 계산한 중간 상태는 캐시에서
바로 돌려준다(매번 t=0 부터 다시 굴리지 않음).

앱 배선의 핵심은 다른 Route 2 엔진과 동일하게
``training_metadata["varying_mesh"] = True`` 다: ``app.predict`` 의 기존
형상 가변 분기가 이 표시를 보고 ``service.predict_to_mesh(engine, [t], 보고
있는 케이스)`` 를 부르고, 그 함수는 ``engine.model.predict_at(coords, [t])``
덕타이핑만 요구한다. 다만 이 엔진은 케이스가 하나뿐이라 ``coords`` 는 항상
학습 케이스 점과 일치해야 한다(재샘플·kNN 폴백 없음 — 시간 롤아웃은 메쉬
위상 자체가 예측의 일부라 다른 격자로 옮기는 것이 의미가 없다).
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class _MGNRolloutModelFacade:
    """``predict_to_mesh`` 덕타이핑 파사드 (``predict_at``/``output_fields``/``field_names``)."""

    def __init__(self, engine: "MGNRolloutTwinEngine") -> None:
        self._engine = engine
        self.is_fitted = True

    @property
    def output_fields(self) -> list[dict[str, Any]]:
        return self._engine._build_output_specs()

    @property
    def field_names(self) -> list[str]:
        return list(self._engine.field_names)

    def predict_at(
        self,
        coords: NDArray[np.float64],
        params: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """시간 t에서 예측한다 — ``predict_to_mesh`` 소켓."""
        return self._engine._predict_at(coords, params)


class MGNRolloutTwinEngine:
    """단일 케이스 MeshGraphNets 롤아웃 트윈 — 진짜 자기회귀 시간 예보.

    ``params`` 는 시간값 t 하나다(케이스 세트 파라미터가 아니다). 학습 시
    저장한 t=0 초기 상태에서 시작해, 요청된 t 까지 필요한 스텝 수만큼
    :class:`~naviertwin.core.gnn.meshgraphnets.meshgraphnets.MeshGraphNets`
    의 델타 예측을 반복한다 — 학습 구간 밖(미래) t 도 그대로 외삽된다(이
    계열의 존재 이유, DMD 와 같은 "학습 구간 밖 예보" 범주지만 저랭크 선형
    가정이 없다).

    롤아웃 캐싱: 스텝 인덱스(정수) → 상태(N, C) 딕셔너리를 t=0 부터
    이어지는(contiguous) 구간으로 채운다. 이미 캐시된 스텝보다 먼 미래를
    요청하면 **캐시된 가장 먼 지점에서 이어서** 롤아웃하고(매번 처음부터
    다시 굴리지 않음), 캐시 범위 안이면 즉시 반환한다.

    Attributes:
        operator: fit 완료된 ``MeshGraphNets``.
        points: (N, 3) 학습 케이스 메쉬 좌표(재샘플 없음 — 이 점들과 일치할
            때만 예측 가능).
        edge_index: (2, 2E) 학습 그래프 에지(케이스 전체에서 고정).
        edge_features: (2E, e) 학습 그래프 에지 피처(케이스 전체에서 고정).
        field_names: 학습 요청 필드 이름.
        target_names: 채널 전개된 타깃 이름(예: ``["p", "U_x", "U_y"]``).
        training_metadata: 학습 범위(t)·필드 등 (웹 GUI 가 슬라이더 구성에 사용).
    """

    def __init__(
        self,
        operator: Any,
        *,
        points: NDArray[np.float64],
        edge_index: NDArray[np.int64],
        edge_features: NDArray[np.float32],
        initial_state: NDArray[np.float32],
        times: Sequence[float],
        field_names: Sequence[str],
        target_names: Sequence[str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """초기화.

        Args:
            operator: fit 완료된
                :class:`~naviertwin.core.gnn.meshgraphnets.meshgraphnets.MeshGraphNets`.
            points: (N, 3) 학습 케이스 메쉬 좌표.
            edge_index: (2, 2E) 학습 그래프 에지.
            edge_features: (2E, e) 학습 그래프 에지 피처.
            initial_state: (N, C) t=0(학습 첫 스냅샷) 상태 — 롤아웃 캐시의 시작점.
            times: 학습에 쓴 타임스텝 값(오름차순, 길이 T+1 ≥ 2).
            field_names: 학습 요청 필드 이름.
            target_names: 채널 전개된 타깃 이름.
            metadata: ``training_metadata`` 에 병합할 추가 항목.

        Raises:
            RuntimeError: operator 가 fit 되지 않은 경우.
            ValueError: 타임스텝이 2개 미만이거나 오름차순이 아닌 경우.
        """
        if not bool(getattr(operator, "is_fitted", False)):
            raise RuntimeError("MeshGraphNets 를 먼저 fit() 해야 합니다.")
        times_arr = np.asarray(times, dtype=np.float64).reshape(-1)
        if times_arr.size < 2:
            raise ValueError("롤아웃 트윈에는 타임스텝이 2개 이상 필요합니다.")
        diffs = np.diff(times_arr)
        if np.any(diffs <= 0):
            raise ValueError("타임스텝이 오름차순(엄격 증가)이어야 합니다.")

        self.operator = operator
        self.points = np.asarray(points, dtype=np.float64)
        self.edge_index = np.asarray(edge_index, dtype=np.int64)
        self.edge_features = np.asarray(edge_features, dtype=np.float32)
        self.field_names = [str(f) for f in field_names]
        self.target_names = [str(t) for t in target_names]

        self._times = times_arr
        self._t0 = float(times_arr[0])
        self._dt = float(np.median(diffs))

        # 롤아웃 캐시 — step index(0 부터 contiguous) → (N, C) 상태.
        self._cache: dict[int, NDArray[np.float64]] = {
            0: np.asarray(initial_state, dtype=np.float64)
        }
        self._max_cached_step = 0

        self.model = _MGNRolloutModelFacade(self)

        # TwinEngine 호출자 호환 덕타이핑.
        self.reducer_type = "mgn_rollout"
        self.surrogate_type = "meshgraphnets_rollout"
        self.model_type = "mgn_rollout"
        self.n_modes = 0

        self.training_metadata: dict[str, Any] = {
            "field_name": ",".join(self.field_names),
            "field_names": list(self.field_names),
            "target_names": list(self.target_names),
            "reducer": "mgn_rollout",
            "surrogate": "meshgraphnets_rollout",
            "problem_type": "rollout_forecast",
            "param_min": self._t0,
            "param_max": float(times_arr[-1]),
            "dt": self._dt,
            "n_train_steps": int(times_arr.size - 1),
            "n_points": int(self.points.shape[0]),
            # 원본 메쉬 표시 — app.predict 의 형상 가변 분기(predict_to_mesh)를
            # 그대로 태운다. 재샘플 없음(단일 케이스라 다른 케이스로 옮길 일도
            # 없다).
            "varying_mesh": True,
        }
        if metadata:
            self.training_metadata.update(metadata)

    # ------------------------------------------------------------------
    # 속성/헬퍼
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """감싼 operator 의 fit 여부."""
        return bool(getattr(self.operator, "is_fitted", False))

    def _build_output_specs(self) -> list[dict[str, Any]]:
        """채널별 field-major 경계 spec — ``split_multi_prediction`` 계약."""
        n_points = int(self.points.shape[0])
        specs: list[dict[str, Any]] = []
        for index, channel in enumerate(self.target_names):
            field_name = channel
            for requested in self.field_names:
                if channel == requested or channel.startswith(f"{requested}_"):
                    field_name = requested
                    break
            specs.append(
                {
                    "field_name": field_name,
                    "display_name": channel,
                    "start": index * n_points,
                    "end": (index + 1) * n_points,
                }
            )
        return specs

    def _step_for_time(self, t: float) -> int:
        """물리 시간 t 를 (반올림) 스텝 인덱스로 바꾼다 — t < t0 는 0 으로 clamp."""
        step = int(round((float(t) - self._t0) / self._dt))
        return max(0, step)

    def _state_at_step(self, step: int) -> NDArray[np.float64]:
        """캐시에서 바로 반환하거나, 캐시된 가장 먼 지점에서 이어서 롤아웃한다."""
        if step <= self._max_cached_step:
            return self._cache[step]

        start_step = self._max_cached_step
        start_state = self._cache[start_step]
        extra_steps = step - start_step
        rollout = self.operator.predict(
            {
                "x": start_state.astype(np.float32),
                "n_steps": int(extra_steps),
                "edge_index": self.edge_index,
                "edge_features": self.edge_features,
            }
        )  # (extra_steps+1, N, C) — index 0 은 start_state 재현
        offset = 1
        while offset < rollout.shape[0]:
            self._cache[start_step + offset] = np.asarray(rollout[offset], dtype=np.float64)
            offset += 1
        self._max_cached_step = step
        return self._cache[step]

    # ------------------------------------------------------------------
    # 예측
    # ------------------------------------------------------------------

    def _predict_rows(self, rows: NDArray[np.float64]) -> NDArray[np.float64]:
        """t 행들을 예측 — (n_rows, C×N) field-major."""
        outputs: list[NDArray[np.float64]] = []
        for row in rows:
            t = float(np.asarray(row, dtype=np.float64).reshape(-1)[0])
            step = self._step_for_time(t)
            state = self._state_at_step(step)  # (N, C)
            outputs.append(state.T.reshape(-1))  # 채널별 이어붙임 (field-major)
        return np.stack(outputs)

    def _predict_at(
        self, coords: NDArray[np.float64], params: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """파사드 :meth:`_MGNRolloutModelFacade.predict_at` 구현."""
        if not self.is_fitted:
            raise RuntimeError("MGNRolloutTwinEngine: operator 가 fit 되지 않았습니다.")
        coords_arr = np.asarray(coords, dtype=np.float64)
        if coords_arr.ndim != 2 or coords_arr.shape[1] < 3:
            raise ValueError(f"coords 는 (n_locations, 3) 이어야 합니다: {coords_arr.shape}")
        if coords_arr.shape[0] != self.points.shape[0]:
            raise ValueError(
                "이 트윈은 단일 케이스 시계열 전용입니다 — coords 점 수"
                f"({coords_arr.shape[0]})가 학습 케이스 점 수({self.points.shape[0]})와 "
                "달라 예측할 수 없습니다(재샘플·kNN 폴백 없음)."
            )
        params_arr = np.asarray(params, dtype=np.float64)
        single = params_arr.ndim <= 1
        rows = params_arr.reshape(1, -1) if single else params_arr

        outputs = self._predict_rows(rows)  # (n_rows, C×N)
        if single:
            return outputs[0]
        return outputs.T  # (C×N, n_rows) — predict_to_mesh/PhysicsNeMo 레이아웃

    def predict(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        """시간 t 에서 학습 케이스 메쉬 위 필드를 예측한다 (TwinEngine 계약).

        Args:
            params: 시간값. 스칼라 ``(1,)`` 또는 ``(n_times, 1)``.

        Returns:
            단일 시간이면 field-major 평탄 벡터(채널 수 × 점 수), 다중
            시간이면 (n_times, 채널 수 × 점 수).
        """
        if not self.is_fitted:
            raise RuntimeError("MGNRolloutTwinEngine: operator 가 fit 되지 않았습니다.")
        arr = np.asarray(params, dtype=np.float64)
        single = arr.ndim <= 1
        rows = arr.reshape(1, -1) if single else arr
        outputs = self._predict_rows(rows)  # (n_rows, C×N)
        return outputs[0] if single else outputs

    # ------------------------------------------------------------------
    # 저장/복원 (service.save_engine 호환)
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """엔진 전체(operator + 그래프 + 롤아웃 캐시)를 pickle 로 저장한다."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "MGNRolloutTwinEngine":
        """저장된 MeshGraphNets 롤아웃 트윈 엔진을 복원한다."""
        with Path(path).open("rb") as f:
            engine = pickle.load(f)
        if not isinstance(engine, cls):
            raise TypeError(f"MGNRolloutTwinEngine 파일이 아닙니다: {path}")
        return engine

    def get_params(self) -> dict[str, Any]:
        """TwinEngine 호출자 호환 메타데이터."""
        return {
            "reducer_type": self.reducer_type,
            "surrogate_type": self.surrogate_type,
            "model_type": self.model_type,
            "n_points": int(self.points.shape[0]),
            "n_train_steps": int(self._times.size - 1),
        }

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"MGNRolloutTwinEngine(n_points={self.points.shape[0]}, "
            f"channels={self.target_names}, t∈[{self._t0:.3g}, "
            f"{float(self._times[-1]):.3g}], status={status})"
        )

    # ------------------------------------------------------------------
    # pickle 지원 — 감싼 MeshGraphNets 의 ``_model`` 은 ``_build()`` 안에서
    # 로컬로 정의된 클래스(``_MGN``)의 인스턴스라 표준 pickle 이 못 다룬다.
    # meshgraphnets.py 는 건드리지 않고, 이 파일이 state_dict 바이트로
    # 직렬화한다 — CaseSetMGN.__getstate__/__setstate__ 와 같은 패턴.
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        operator = state.pop("operator")
        state.pop("model", None)  # 엔진 back-ref — 복원 시 재생성

        model = getattr(operator, "_model", None)
        model_bytes: bytes | None = None
        if model is not None:
            import torch

            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            model_bytes = buffer.getvalue()
        state["_operator_ctor"] = {
            "node_feat": operator.node_feat,
            "edge_feat": operator.edge_feat,
            "hidden": operator.hidden,
            "n_msgpass": operator.n_msgpass,
            "max_epochs": operator.max_epochs,
            "lr": operator.lr,
            "device": operator.device,
            "seed": operator.seed,
        }
        state["_operator_model_bytes"] = model_bytes
        state["_operator_train_losses"] = list(getattr(operator, "train_losses_", []))
        state["_operator_n_epochs"] = int(getattr(operator, "n_epochs", 0))
        state["_operator_is_fitted"] = bool(getattr(operator, "is_fitted", False))
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        ctor = state.pop("_operator_ctor")
        model_bytes = state.pop("_operator_model_bytes", None)
        train_losses = state.pop("_operator_train_losses", [])
        n_epochs = state.pop("_operator_n_epochs", 0)
        is_fitted = state.pop("_operator_is_fitted", False)
        self.__dict__.update(state)

        from naviertwin.core.gnn.meshgraphnets.meshgraphnets import MeshGraphNets

        operator = MeshGraphNets(**ctor)
        operator._device = operator._resolve_device()
        if model_bytes is not None:
            import torch

            operator._model = operator._build().to(operator._device)
            operator._model.load_state_dict(
                torch.load(io.BytesIO(model_bytes), map_location=operator._device)
            )
            operator._model.eval()
        operator.train_losses_ = list(train_losses)
        operator.n_epochs = n_epochs
        operator.is_fitted = is_fitted
        self.operator = operator
        self.model = _MGNRolloutModelFacade(self)


__all__ = ["MGNRolloutTwinEngine"]
