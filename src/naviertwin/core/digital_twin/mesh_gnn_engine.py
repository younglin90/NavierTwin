"""메쉬 GNN 케이스 세트 트윈 엔진 — 재샘플 없는 원본 메쉬 위 예측 (Route 2).

:class:`~naviertwin.core.gnn.gnn_surrogate.case_set_gnn.CaseSetGNN` 을 웹/
데스크톱 트윈 계약(``predict(params)`` + ``training_metadata`` + ``save/load``)
에 맞춰 감싼다. GeometryFNO(Route 1)와 달리 공통 격자 텐서화가 없다 —
케이스마다 자기 메쉬(진짜 구멍 포함)를 그래프로 학습하고, 예측도 그 좌표
위에서 그대로 이뤄진다.

앱 배선의 핵심은 ``training_metadata["varying_mesh"] = True`` 다:
``app.predict`` 의 기존 형상 가변 분기가 이 표시를 보고
``service.predict_to_mesh(engine, μ, 보고 있는 케이스)`` 를 부르고, 그 함수는
``engine.model.predict_at(coords, params)`` 덕타이핑만 요구한다 — Physics AI
(:mod:`~naviertwin.core.physnemo.cfd_field_model`)가 세운 소켓에 두 번째
전략이 꽂히는 것이다. 앱 예측/뷰어 경로 수정은 없다.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class _MeshGNNModelFacade:
    """``predict_to_mesh``/``split_multi_prediction`` 덕타이핑 파사드.

    Attributes:
        output_fields: ``[{field_name, display_name, start, end}, ...]`` —
            field-major 예측 벡터의 채널 경계 (대표 케이스 점 수 기준).
        field_names: 학습 요청 필드 이름 (specs 가 빌 때의 폴백 라벨).
        is_fitted: 항상 True (fit 완료 후에만 만들어진다).
    """

    def __init__(self, engine: "MeshGNNTwinEngine") -> None:
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
        """임의 좌표에서 예측한다 — ``predict_to_mesh`` 소켓.

        좌표가 저장된 학습 케이스와 일치하면(지배 경로) 그 케이스의
        edge_index/노드 피처를 재사용하고, 불일치하면 scipy cKDTree kNN
        그래프로 폴백한다(메쉬 위상과 다른 연결이 생길 수 있어 경고 로그).

        Returns:
            단일 μ 행이면 (n_channels × n_locations,) field-major 평탄 벡터,
            (n_rows, k) 배치면 (n_channels × n_locations, n_rows) —
            PhysicsNeMoCFDFieldModel.predict_at 와 같은 레이아웃.
        """
        return self._engine._predict_at(coords, params)


class MeshGNNTwinEngine:
    """케이스-세트 메쉬 GNN 트윈 — 정상(steady) 파라미터 스윕 전용.

    ``predict(μ)`` 는 **대표(0번) 케이스 그래프** 위 field-major 평탄 벡터를
    돌려준다 — ``model.output_fields`` 의 start/end 경계가 이 길이에 맞춰져
    있어 ``split_multi_prediction`` 계약이 성립한다. 실제 표시 경로
    (형상 가변)는 ``predict_to_mesh`` → :meth:`_predict_at` 가 보고 있는
    케이스 좌표를 그대로 받아 그 메쉬 위에서 예측한다.
    """

    def __init__(
        self,
        operator: Any,
        *,
        cases: Sequence[dict[str, Any]],
        train_params: NDArray[np.float64],
        param_names: Sequence[str],
        field_names: Sequence[str],
        target_names: Sequence[str],
        norm: dict[str, Any],
        input_field_names: Sequence[str] = (),
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """초기화.

        Args:
            operator: fit 완료된 :class:`CaseSetGNN`.
            cases: 케이스별 그래프 정보 목록 — 각 항목은
                ``{"points": (N,3), "edge_index": (2,2E), "base_x": (N,f)}``.
                ``base_x`` 는 μ 채널을 뺀 노드 피처(정규화 좌표+입력 필드).
            train_params: (n_cases, k) 학습 운전조건.
            param_names: 운전조건 이름 k 개.
            field_names: 학습 요청 필드 (예: ``["p", "U"]``).
            target_names: 채널 전개된 타깃 이름 (예: ``["p","U_x","U_y","U_z"]``).
            norm: :func:`graph_norm_from_cases` 정규화 상수 (μ 정규화 재사용).
            input_field_names: 학습에 쓴 입력 필드 — 있으면 미지 좌표
                (kNN 폴백) 예측이 불가능해 명확히 거절한다.
            metadata: ``training_metadata`` 에 병합할 추가 항목.

        Raises:
            RuntimeError: operator 가 fit 되지 않은 경우.
            ValueError: 케이스 수와 파라미터 행 수가 다른 경우.
        """
        if not bool(getattr(operator, "is_fitted", False)):
            raise RuntimeError("CaseSetGNN 을 먼저 fit() 해야 합니다.")
        params = np.asarray(train_params, dtype=np.float64)
        if params.ndim == 1:
            params = params.reshape(-1, 1)
        if len(cases) != params.shape[0]:
            raise ValueError(
                f"케이스 수({len(cases)})와 파라미터 행 수({params.shape[0]})가 다릅니다."
            )

        self.operator = operator
        self._cases = [dict(c) for c in cases]
        self._train_params = params
        self.param_names = [str(n) for n in param_names]
        self.field_names = [str(f) for f in field_names]
        self.target_names = [str(t) for t in target_names]
        self.input_field_names = [str(n) for n in input_field_names]
        self._norm = dict(norm)

        # 최근접 케이스 탐색용 정규화 스케일 (상수 파라미터는 1로 보호).
        span = params.max(axis=0) - params.min(axis=0)
        self._param_span = np.where(span > 0, span, 1.0)

        self.model = _MeshGNNModelFacade(self)

        # TwinEngine 덕타이핑 (save_engine/_restore_engine 경로 호환).
        self.reducer_type = "mesh_gnn"
        self.surrogate_type = "case_set_gcn"
        self.model_type = "mesh_gnn"
        self.n_modes = 0

        mins = [float(v) for v in params.min(axis=0)]
        maxs = [float(v) for v in params.max(axis=0)]
        self.training_metadata: dict[str, Any] = {
            "field_name": ",".join(self.field_names),
            "field_names": list(self.field_names),
            "target_names": list(self.target_names),
            "reducer": "mesh_gnn",
            "surrogate": "case_set_gcn",
            "problem_type": "steady_sweep",
            "param_names": list(self.param_names),
            "param_mins": mins,
            "param_maxs": maxs,
            "n_cases": int(params.shape[0]),
            # 형상 가변 표시 — app.predict 가 이걸 보고 predict_to_mesh 경로
            # (보고 있는 원본 케이스 메쉬 위 표시)를 탄다. 재샘플 없음.
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

    @property
    def n_params(self) -> int:
        """운전조건 파라미터 차원 k."""
        return int(self._train_params.shape[1])

    def nearest_case_index(self, mu: NDArray[np.float64]) -> int:
        """정규화 μ 거리 기준 최근접 학습 케이스 인덱스를 찾는다."""
        scaled = (self._train_params - np.asarray(mu, dtype=np.float64).reshape(1, -1))
        scaled = scaled / self._param_span
        return int(np.argmin(np.linalg.norm(scaled, axis=1)))

    def _build_output_specs(self) -> list[dict[str, Any]]:
        """대표(0번) 케이스 점 수 기준 field-major 채널 경계 spec.

        채널명(예: ``U_x``)을 요청 필드(예: ``U``)에 되돌려 매핑해
        ``split_multi_prediction`` 의 크기(magnitude) 파생이 동작한다 —
        GeometryFNO 파사드와 같은 규칙.
        """
        n_points = int(self._cases[0]["points"].shape[0])
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

    def _mu_features(self, mu: NDArray[np.float64]) -> NDArray[np.float64]:
        """μ 를 학습 때와 같은 상수로 정규화한다 — (k,)."""
        center = np.asarray(self._norm["mu_center"], dtype=np.float64)
        scale = np.asarray(self._norm["mu_scale"], dtype=np.float64)
        return (np.asarray(mu, dtype=np.float64).reshape(-1) - center) / scale

    def _graph_for(
        self, base_x: NDArray[np.float32], edge_index: NDArray[np.int64],
        mu: NDArray[np.float64],
    ) -> dict[str, Any]:
        """base_x 에 μ 브로드캐스트 채널을 붙인 예측용 그래프를 만든다."""
        mu_n = self._mu_features(mu)
        if mu_n.size:
            x = np.hstack(
                [base_x, np.broadcast_to(mu_n, (base_x.shape[0], mu_n.size))]
            ).astype(np.float32)
        else:
            x = np.asarray(base_x, dtype=np.float32)
        return {"x": x, "edge_index": edge_index}

    def _match_case(self, coords: NDArray[np.float64]) -> dict[str, Any] | None:
        """coords 가 저장된 학습 케이스와 일치하면 그 케이스를 돌려준다."""
        for case in self._cases:
            points = case["points"]
            if coords.shape[0] != points.shape[0]:
                continue
            if np.allclose(coords, points, rtol=1e-8, atol=1e-10):
                return case
        return None

    def _knn_base(self, coords: NDArray[np.float64]) -> dict[str, Any]:
        """미지 좌표용 kNN(k=8) 그래프 폴백 — 위상 왜곡 가능성을 경고한다.

        Raises:
            ValueError: 학습에 입력 필드를 썼는데 미지 좌표라 그 값을 만들 수
                없는 경우.
        """
        if self.input_field_names:
            raise ValueError(
                "이 mesh_gnn 트윈은 입력 필드"
                f"({', '.join(self.input_field_names)})로 학습돼 학습 케이스와 "
                "다른 좌표에서는 예측할 수 없습니다 — 학습 케이스 메쉬에서 "
                "예측하세요."
            )
        from scipy.spatial import cKDTree

        n = coords.shape[0]
        k = min(8, max(1, n - 1))
        tree = cKDTree(coords)
        _dist, neighbors = tree.query(coords, k=k + 1)  # 자기 자신 포함
        neighbors = np.atleast_2d(neighbors)[:, 1:]  # 자기 자신 제거
        src = np.repeat(np.arange(n, dtype=np.int64), neighbors.shape[1])
        dst = neighbors.reshape(-1).astype(np.int64)
        pairs = np.stack([src, dst], axis=1)
        both = np.unique(np.concatenate([pairs, pairs[:, ::-1]], axis=0), axis=0)
        edge_index = np.ascontiguousarray(both.T)
        coords_n = (
            coords - np.asarray(self._norm["coord_center"], dtype=np.float64)
        ) / np.asarray(self._norm["coord_scale"], dtype=np.float64)
        logger.warning(
            "mesh_gnn: 좌표가 학습 케이스와 일치하지 않아 kNN(k=%d) 그래프로 "
            "예측합니다 — 메쉬 위상과 다른 연결(구멍을 가로지르는 에지 등)이 "
            "생길 수 있습니다.",
            k,
        )
        return {"base_x": coords_n.astype(np.float32), "edge_index": edge_index}

    # ------------------------------------------------------------------
    # 예측
    # ------------------------------------------------------------------

    def _predict_rows(
        self,
        base_x: NDArray[np.float32],
        edge_index: NDArray[np.int64],
        rows: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """μ 행들을 같은 그래프 위에서 예측 — (n_rows, C×N) field-major."""
        outputs: list[NDArray[np.float64]] = []
        for row in rows:
            graph = self._graph_for(base_x, edge_index, row)
            pred = self.operator.predict_graph(graph)  # (N, C)
            outputs.append(pred.T.reshape(-1))  # field-major: 채널별 이어붙임
        return np.stack(outputs)

    def _predict_at(
        self, coords: NDArray[np.float64], params: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """파사드 :meth:`_MeshGNNModelFacade.predict_at` 구현."""
        if not self.is_fitted:
            raise RuntimeError("MeshGNNTwinEngine: operator 가 fit 되지 않았습니다.")
        coords_arr = np.asarray(coords, dtype=np.float64)
        if coords_arr.ndim != 2 or coords_arr.shape[1] < 3:
            raise ValueError(f"coords 는 (n_locations, 3) 이어야 합니다: {coords_arr.shape}")
        coords_arr = coords_arr[:, :3]
        params_arr = np.asarray(params, dtype=np.float64)
        single = params_arr.ndim <= 1
        rows = params_arr.reshape(1, -1) if single else params_arr
        if rows.shape[1] != self.n_params:
            raise ValueError(
                f"파라미터 차원({rows.shape[1]})이 학습 차원({self.n_params})과 "
                f"다릅니다 — 학습 파라미터: {self.param_names}"
            )

        matched = self._match_case(coords_arr)
        if matched is not None:
            base_x, edge_index = matched["base_x"], matched["edge_index"]
        else:
            fallback = self._knn_base(coords_arr)
            base_x, edge_index = fallback["base_x"], fallback["edge_index"]

        outputs = self._predict_rows(base_x, edge_index, rows)  # (n_rows, C×N)
        if single:
            return outputs[0]
        return outputs.T  # (C×N, n_rows) — PhysicsNeMo predict_at 레이아웃

    def predict(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        """운전조건 μ 에서 **대표(0번) 케이스 메쉬** 위 필드를 예측한다.

        형상 가변 세트에는 "학습 격자"가 없으므로 이 벡터는 계약 유지용
        (output_fields 경계와 길이 일치)이다 — 실제 표시는 varying_mesh 분기의
        ``predict_to_mesh`` 가 보고 있는 케이스 좌표로 :meth:`_predict_at` 를
        부른다.

        Args:
            params: (k,) 운전조건 벡터 또는 (N, k) 배치.

        Returns:
            (k,) 입력이면 field-major 평탄 벡터 (채널 수 × 대표 케이스 점 수),
            (N, k) 입력이면 (N, 채널 수 × 점 수) 배열.
        """
        if not self.is_fitted:
            raise RuntimeError("MeshGNNTwinEngine: operator 가 fit 되지 않았습니다.")
        arr = np.asarray(params, dtype=np.float64)
        single = arr.ndim <= 1
        rows = arr.reshape(1, -1) if single else arr
        if rows.shape[1] != self.n_params:
            raise ValueError(
                f"파라미터 차원({rows.shape[1]})이 학습 차원({self.n_params})과 "
                f"다릅니다 — 학습 파라미터: {self.param_names}"
            )
        rep = self._cases[0]
        outputs = self._predict_rows(rep["base_x"], rep["edge_index"], rows)
        return outputs[0] if single else outputs

    # ------------------------------------------------------------------
    # 저장/복원 (service.save_engine 호환)
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """엔진 전체(operator + 케이스 그래프)를 pickle 로 저장한다."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "MeshGNNTwinEngine":
        """저장된 메쉬 GNN 트윈 엔진을 복원한다."""
        with Path(path).open("rb") as f:
            engine = pickle.load(f)
        if not isinstance(engine, cls):
            raise TypeError(f"MeshGNNTwinEngine 파일이 아닙니다: {path}")
        return engine

    def get_params(self) -> dict[str, Any]:
        """TwinEngine 호출자 호환 메타데이터."""
        return {
            "reducer_type": self.reducer_type,
            "surrogate_type": self.surrogate_type,
            "model_type": self.model_type,
            "n_params": self.n_params,
            "n_cases": int(self._train_params.shape[0]),
        }

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"MeshGNNTwinEngine(n_cases={self._train_params.shape[0]}, "
            f"channels={self.target_names}, status={status})"
        )


__all__ = ["MeshGNNTwinEngine"]
