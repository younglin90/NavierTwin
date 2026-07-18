"""GeometryFNO(FNO+SDF) 케이스 세트 트윈 엔진 — 공통 격자 위 형상 인지 예측.

:class:`~naviertwin.core.operator_learning.fno.geometry_fno.GeometryFNO2D` 를
웹/데스크톱 트윈 계약(``predict(params)`` + ``training_metadata``)에 맞춰 감싼다.
학습 텐서는 :func:`naviertwin.core.operator_learning.fno.case_tensorizer.
cases_to_grid_tensors` 가 만든 것을 그대로 받는다 — 채널 = [sdf, mask, μ...].

예측 시 형상(SDF) 입력의 출처 — 정직한 한계:
    질의는 운전조건 벡터 μ 뿐이라 새 형상의 SDF 를 알 수 없다. 그래서 이 엔진은
    **정규화 μ 거리 기준 최근접 학습 케이스의 sdf/mask 채널을 재사용**하고 μ
    브로드캐스트 채널만 질의값으로 바꾼다. 형상 파라미터 스윕(예: 반지름)에서는
    "가장 가까운 학습 형상 위에서 새 μ 를 평가"하는 근사가 되고, 진짜 새 형상
    예측에는 SDF 입력 자체가 필요하다 — 후속 과제다.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


class _GeometryFNOModelFacade:
    """``service.split_multi_prediction`` 계약(``output_fields``)용 얇은 홀더.

    Physics AI 엔진의 ``engine.model.output_fields`` 덕타이핑을 그대로 만족시켜
    다중 채널 예측을 필드별 ``twin_<채널>`` 로 분해할 수 있게 한다.

    Attributes:
        output_fields: ``[{field_name, display_name, start, end}, ...]`` —
            field-major 로 이어붙인 예측 벡터의 채널 경계.
        is_fitted: 항상 True (fit 완료 후에만 만들어진다).
    """

    def __init__(self, output_fields: list[dict[str, Any]]) -> None:
        self.output_fields = list(output_fields)
        self.is_fitted = True


class GeometryFNOTwinEngine:
    """공통 격자 위 형상 인지 FNO 트윈 — 케이스 세트(정상 스윕) 전용.

    ``predict(μ)`` 는 **공통 격자의 점 순서**(``grid.points``)에 맞춘 field-major
    평탄 벡터를 돌려준다 (채널당 길이 = 격자 점 수). 예측은 학습에 쓴 공통
    격자 위에서만 의미가 있으므로, 앱은 :attr:`grid_dataset` 으로 뷰어를
    교체해 결과를 표시한다 (``training_metadata["common_grid"] = True`` 표시).

    Attributes:
        operator: 학습된 :class:`GeometryFNO2D`.
        model: ``output_fields`` 를 노출하는 파사드 —
            ``service.split_multi_prediction`` 이 그대로 쓴다.
        grid_dataset: 공통 격자 위 CFDDataset (sdf/mask 포함) — 예측 표시용.
        training_metadata: 앱/복원 경로가 읽는 학습 메타데이터.
    """

    def __init__(
        self,
        operator: Any,
        *,
        train_inputs: NDArray[np.float32],
        train_params: NDArray[np.float64],
        param_names: Sequence[str],
        field_names: Sequence[str],
        target_names: Sequence[str],
        grid: Any,
        backend: str = "builtin",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """초기화.

        Args:
            operator: fit 완료된 :class:`GeometryFNO2D`.
            train_inputs: 학습 입력 텐서 (N, H, W, 2 + k) — 예측 시 최근접
                케이스의 sdf/mask 채널을 재사용하기 위해 보관한다.
            train_params: 케이스별 운전조건 (N, k).
            param_names: 운전조건 이름 k 개.
            field_names: 학습에 요청한 필드 이름 (예: ``["p", "U"]``).
            target_names: 타깃 채널 이름 (벡터는 성분 전개, 예:
                ``["p", "U_x", "U_y", "U_z"]``).
            grid: 공통 pyvista ImageData 격자.
            backend: FNO 백엔드 ("builtin" | "neuralop").
            metadata: ``training_metadata`` 에 병합할 추가 항목.

        Raises:
            RuntimeError: ``operator`` 가 fit 되지 않은 경우.
            ValueError: 텐서/파라미터 shape 이 서로 맞지 않는 경우.
        """
        if not bool(getattr(operator, "is_fitted", False)):
            raise RuntimeError("GeometryFNO2D 를 먼저 fit() 해야 합니다.")
        inputs = np.asarray(train_inputs, dtype=np.float32)
        params = np.asarray(train_params, dtype=np.float64)
        if params.ndim == 1:
            params = params.reshape(-1, 1)
        if inputs.ndim != 4 or inputs.shape[0] != params.shape[0]:
            raise ValueError(
                f"train_inputs (N,H,W,C)={inputs.shape} 와 train_params "
                f"{params.shape} 의 케이스 수가 다릅니다."
            )
        if int(getattr(operator, "n_params", -1)) != params.shape[1]:
            raise ValueError(
                f"operator.n_params({getattr(operator, 'n_params', None)})와 "
                f"파라미터 차원({params.shape[1]})이 다릅니다."
            )

        self.operator = operator
        self._train_inputs = inputs
        self._train_params = params
        self.param_names = [str(n) for n in param_names]
        self.field_names = [str(f) for f in field_names]
        self.target_names = [str(t) for t in target_names]
        self.backend = str(backend)

        # 최근접 케이스 탐색용 정규화 스케일 (상수 파라미터는 1로 보호).
        span = params.max(axis=0) - params.min(axis=0)
        self._param_span = np.where(span > 0, span, 1.0)

        n_grid = int(grid.n_points)
        self._n_grid_points = n_grid
        self.model = _GeometryFNOModelFacade(
            self._build_output_specs(n_grid)
        )
        self.grid_dataset = self._build_grid_dataset(grid, inputs)

        # TwinEngine 덕타이핑 (save_engine/_restore_engine 경로 호환).
        self.reducer_type = "geometry_fno"
        self.surrogate_type = f"fno_sdf({self.backend})"
        self.model_type = "geometry_fno2d"
        self.n_modes = int(getattr(operator, "modes", 0))

        mins = [float(v) for v in params.min(axis=0)]
        maxs = [float(v) for v in params.max(axis=0)]
        self.training_metadata: dict[str, Any] = {
            "field_name": ",".join(self.field_names),
            "field_names": list(self.field_names),
            "target_names": list(self.target_names),
            "reducer": "geometry_fno",
            "surrogate": f"fno_sdf({self.backend})",
            "problem_type": "steady_sweep_operator",
            "param_names": list(self.param_names),
            "param_mins": mins,
            "param_maxs": maxs,
            "n_cases": int(params.shape[0]),
            # 예측이 (보고 있는 케이스 메쉬가 아니라) 공통 격자 위 벡터라는
            # 표시 — app.predict 가 이걸 보고 grid_dataset 으로 뷰어를 바꾼다.
            "common_grid": True,
        }
        if metadata:
            self.training_metadata.update(metadata)

    # ------------------------------------------------------------------
    # 구성 헬퍼
    # ------------------------------------------------------------------

    def _build_output_specs(self, n_grid: int) -> list[dict[str, Any]]:
        """field-major 평탄 벡터의 채널 경계 spec 을 만든다.

        채널명(예: ``U_x``)을 요청 필드(예: ``U``)에 되돌려 매핑해
        ``split_multi_prediction`` 의 벡터 크기(magnitude) 파생이 동작한다.
        """
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
                    "start": index * n_grid,
                    "end": (index + 1) * n_grid,
                }
            )
        return specs

    @staticmethod
    def _build_grid_dataset(grid: Any, inputs: NDArray[np.float32]) -> Any:
        """공통 격자 + 케이스 0 의 sdf/mask 를 담은 표시용 CFDDataset 을 만든다.

        (H, W) 배열의 C-order ``ravel()`` 은 두께 0 축을 뺀 VTK 점 순서와
        일치한다 — case_tensorizer 의 ``to_hw`` 역변환이다.
        """
        from naviertwin.core.cfd_reader.base import CFDDataset

        mesh = grid.copy(deep=True)
        mesh.point_data["sdf"] = np.asarray(
            inputs[0, :, :, 0], dtype=np.float64
        ).ravel()
        mesh.point_data["mask"] = np.asarray(
            inputs[0, :, :, 1], dtype=np.float64
        ).ravel()
        return CFDDataset(
            mesh=mesh,
            time_steps=[0.0],
            field_names=["sdf", "mask"],
            metadata={"source": "geometry_fno_common_grid"},
        )

    # ------------------------------------------------------------------
    # 예측
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """감싼 operator 의 fit 여부."""
        return bool(getattr(self.operator, "is_fitted", False))

    @property
    def n_params(self) -> int:
        """운전조건 파라미터 차원 k."""
        return int(self._train_params.shape[1])

    def nearest_case_index(self, mu: NDArray[np.float64]) -> int:
        """정규화 μ 거리 기준 최근접 학습 케이스 인덱스를 찾는다."""
        scaled = (self._train_params - mu.reshape(1, -1)) / self._param_span
        return int(np.argmin(np.linalg.norm(scaled, axis=1)))

    def _predict_single(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """μ 하나에 대한 field-major 평탄 예측 벡터를 만든다."""
        nearest = self.nearest_case_index(mu)
        x = self._train_inputs[nearest].copy()  # sdf/mask 는 최근접 케이스 것
        for j in range(mu.size):
            x[:, :, 2 + j] = np.float32(mu[j])
        pred = np.asarray(self.operator.predict(x), dtype=np.float64)  # (H, W, C)
        # 채널별 (H, W).ravel() = 공통 격자 점 순서 → field-major 로 이어붙인다.
        return np.concatenate(
            [pred[:, :, c].ravel() for c in range(pred.shape[-1])]
        )

    def predict(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        """운전조건 μ 에서 공통 격자 위 필드를 예측한다.

        형상 채널은 최근접 학습 케이스 것을 재사용한다 — 진짜 새 형상 예측은
        SDF 입력이 필요하다(후속). 자세한 근거는 모듈 docstring 참조.

        Args:
            params: (k,) 운전조건 벡터 또는 (N, k) 배치.

        Returns:
            (k,) 입력이면 field-major 평탄 벡터 (채널 수 × 격자 점 수),
            (N, k) 입력이면 (N, 채널 수 × 격자 점 수) 배열.

        Raises:
            RuntimeError: operator 가 fit 되지 않은 경우.
            ValueError: 파라미터 차원이 학습과 다른 경우.
        """
        if not self.is_fitted:
            raise RuntimeError("GeometryFNOTwinEngine: operator 가 fit 되지 않았습니다.")
        arr = np.asarray(params, dtype=np.float64)
        single = arr.ndim <= 1
        rows = arr.reshape(1, -1) if single else arr
        if rows.shape[1] != self.n_params:
            raise ValueError(
                f"파라미터 차원({rows.shape[1]})이 학습 차원({self.n_params})과 "
                f"다릅니다 — 학습 파라미터: {self.param_names}"
            )
        outputs = np.stack([self._predict_single(row) for row in rows])
        return outputs[0] if single else outputs

    # ------------------------------------------------------------------
    # 저장/복원 (service.save_engine 호환)
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """엔진 전체(operator + 격자 + 학습 텐서)를 pickle 로 저장한다."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "GeometryFNOTwinEngine":
        """저장된 GeometryFNO 트윈 엔진을 복원한다."""
        with Path(path).open("rb") as f:
            engine = pickle.load(f)
        if not isinstance(engine, cls):
            raise TypeError(f"GeometryFNOTwinEngine 파일이 아닙니다: {path}")
        return engine

    def get_params(self) -> dict[str, Any]:
        """TwinEngine 호출자 호환 메타데이터."""
        return {
            "reducer_type": self.reducer_type,
            "surrogate_type": self.surrogate_type,
            "model_type": self.model_type,
            "n_params": self.n_params,
            "n_grid_points": self._n_grid_points,
        }

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"GeometryFNOTwinEngine(backend={self.backend!r}, "
            f"n_cases={self._train_params.shape[0]}, "
            f"channels={self.target_names}, status={status})"
        )


__all__ = ["GeometryFNOTwinEngine"]
