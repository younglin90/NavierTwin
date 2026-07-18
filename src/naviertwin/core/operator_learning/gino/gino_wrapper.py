"""GINO 케이스 세트 래퍼 — 메쉬 네이티브/점군 학습, 재샘플 없음 (Route 2, 2번째 배선).

``mesh_gnn``(:mod:`naviertwin.core.gnn.case_set_gnn`, GCN)이 메쉬를 **그래프**
(edge_index 고정)로 바꿔 학습하는 것과 달리, 이 모듈은 ``neuraloperator`` 의
``GINO``(Li et al., 2023)를 감싸 케이스를 **순수 점군**으로 학습한다 — 고정
연결(edge_index)이 없으므로 예측 시 임의 좌표(학습 케이스와 다른 점 개수·
순서라도)에서 그대로 동작한다(그래프 재구성이 필요 없다). 내부적으로는

    입력 점군 --(GNO, 반경 이웃 적분)--> 균일 잠재 격자 --(FNO)--> 균일 잠재
    격자 --(GNO, 반경 이웃 적분)--> 출력 쿼리 점

세 단계로 전역 상호작용(FNO)과 임의 형상(GNO)을 결합한다.

GINO API 실측(``.omc/research/route2-mesh-native-wiring.md`` §2, neuralop
2.0.0, WSL): 생성자 반경 인자는 문서와 달리 ``in_gno_radius``/
``out_gno_radius``(``gno_radius`` 아님). 추가로 이 배선에서 실측한 것 —
기본(``in_gno_transform_type="linear"``) GNO 커널은 ``rep_features.mul_
(in_features)`` 로 입력 GNO 출력 채널과 입력 채널을 원소곱하므로 **입력 GNO
출력 채널(=fno_in_channels, transform_type="linear" 일 때)이 반드시
``in_channels`` 과 같아야 한다** — 다르면 ``IntegralTransform.forward`` 에서
브로드캐스트 shape 에러가 난다(문서/시그니처에 명시돼 있지 않음). 이 모듈은
``fno_in_channels=in_channels`` 를 항상 강제해 그 함정을 피한다.

규약(``mesh_gnn``/``case_tensorizer`` 와의 일치):
    - 벡터 필드 성분 전개는 ``case_tensorizer._field_channels`` 와 같은 규칙
      (``U`` → ``U_x, U_y, U_z``) — ``target_names`` 문자열이 두 루트 모두
      같다(테스트로 고정).
    - 운전조건 μ 는 **점 피처 브로드캐스트**(모든 점에 동일 값 k 채널) —
      GeometryFNO/mesh_gnn 의 μ 채널 규약과 동형.

좌표 정규화는 ``mesh_gnn``(z-score)과 다르다 — GINO 의 잠재 격자가
``[0,1]^3`` 균일 그리드라 점 좌표도 **min-max 로 [0,1]^3** 로 맞춘다(원본
케이스 point 순서/개수는 그대로, 좌표값만 스케일). ``coord_dim`` 은 항상 3
으로 고정한다(pyvista 점은 항상 (N,3)) — 2D 평면 메쉬는 z 축이 상수라
자동으로 그 축의 이웃 반경 판별력이 사라질 뿐 크래시하지 않는다(open3d
3D 가속 경로도 그대로 쓸 수 있다는 부수 이점).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.operator_learning.base import BaseOperator
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

_NEURALOP_MISSING = "neuraloperator 설치 필요: pip install naviertwin[full] (neuralop>=2.0)"

# case_tensorizer/case_graph 와 동일한 성분 접미사 규칙 — target_names 일치.
_COMPONENT_SUFFIXES = ("x", "y", "z")


def _require_gino() -> Any:
    try:
        from neuralop.models import GINO
    except ImportError as exc:  # pragma: no cover - 설치 환경 의존
        raise RuntimeError(_NEURALOP_MISSING) from exc
    return GINO


def _field_channels(mesh: Any, name: str) -> list[tuple[str, NDArray[np.float64]]]:
    """point_data 필드 하나를 (채널명, (N,) 배열) 목록으로 푼다.

    ``case_graph._field_channels``/``case_tensorizer._field_channels`` 와
    같은 접미사 규칙 — 세 루트 모두 같은 ``target_names`` 문자열을 낸다.
    """
    if name not in mesh.point_data:
        available = list(mesh.point_data.keys())
        raise ValueError(
            f"필드 '{name}' 이(가) 메쉬 point_data 에 없습니다. 사용 가능: {available}"
        )
    values = np.asarray(mesh.point_data[name], dtype=np.float64)
    if values.ndim == 1:
        return [(name, values)]
    channels: list[tuple[str, NDArray[np.float64]]] = []
    for j in range(values.shape[1]):
        suffix = _COMPONENT_SUFFIXES[j] if values.shape[1] <= 3 else str(j)
        channels.append((f"{name}_{suffix}", values[:, j]))
    return channels


def pointcloud_norm_from_cases(
    datasets: Sequence[Any],
    params: NDArray[np.float64],
    *,
    input_field_names: Sequence[str] = (),
) -> dict[str, Any]:
    """train 케이스들로 점군 정규화 상수를 계산한다.

    좌표는 **min-max** — GINO 의 잠재 격자가 ``[0,1]^3`` 균일 그리드라 좌표
    스케일이 그 범위와 맞아야 반경(``in_gno_radius``/``out_gno_radius``)이
    의미를 가진다(``mesh_gnn`` 의 z-score 정규화와 다른 지점). μ/입력 필드는
    ``mesh_gnn`` 과 동일하게 z-score.

    group split 시 train-only 정규화 원칙을 지키려면 빌더가 **train
    케이스만** 넘겨 계산하고, 반환 dict 를 val/test 의 :func:`case_to_pointcloud`
    에 그대로 주입한다.

    Returns:
        ``coord_min``/``coord_range`` (3,), ``mu_center``/``mu_scale`` (k,),
        ``input_names``/``input_center``/``input_scale`` 을 담은 dict.
        폭이 0 인 축의 range/scale 은 1 로 보호된다.
    """
    points = np.vstack(
        [np.asarray(getattr(d, "mesh", d).points, dtype=np.float64)[:, :3] for d in datasets]
    )
    coord_min = points.min(axis=0)
    coord_max = points.max(axis=0)
    coord_range = coord_max - coord_min
    coord_range = np.where(coord_range > 0, coord_range, 1.0)

    mu = np.asarray(params, dtype=np.float64)
    if mu.ndim == 1:
        mu = mu.reshape(-1, 1)
    mu_center = mu.mean(axis=0) if mu.size else np.zeros(mu.shape[1])
    mu_scale = mu.std(axis=0) if mu.size else np.ones(mu.shape[1])
    mu_scale = np.where(mu_scale > 0, mu_scale, 1.0)

    input_names = [str(n) for n in input_field_names]
    input_center = np.zeros(len(input_names))
    input_scale = np.ones(len(input_names))
    for j, name in enumerate(input_names):
        stacked = np.concatenate(
            [
                np.asarray(getattr(d, "mesh", d).point_data[name], dtype=np.float64).reshape(-1)
                for d in datasets
            ]
        )
        input_center[j] = float(stacked.mean())
        scale = float(stacked.std())
        input_scale[j] = scale if scale > 0 else 1.0

    return {
        "coord_min": coord_min,
        "coord_range": coord_range,
        "mu_center": mu_center,
        "mu_scale": mu_scale,
        "input_names": input_names,
        "input_center": input_center,
        "input_scale": input_scale,
    }


def case_to_pointcloud(
    dataset: Any,
    mu: NDArray[np.float64],
    field_names: Sequence[str],
    *,
    input_field_names: Sequence[str] = (),
    norm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """케이스 하나를 GINO 학습용 점군 dict 로 바꾼다 (재샘플 없음).

    점 피처 ``x`` = [정규화([0,1]) 좌표 3성분 | 정규화 입력 필드 | 정규화 μ
    브로드캐스트]. 타깃 ``y`` = 요청 필드의 채널 전개(물리 단위 그대로 —
    타깃 표준화는 :class:`GINOCaseSetOperator` 가 train 점군만으로 내부
    계산한다).

    Args:
        dataset: CFDDataset 또는 pyvista 메쉬.
        mu: (k,) 케이스 운전조건.
        field_names: 출력 필드 이름 목록 (벡터는 성분 전개).
        input_field_names: 점 피처로 함께 넣을 point 필드(예: ``wall_distance``).
        norm: :func:`pointcloud_norm_from_cases` 가 만든 정규화 상수. None 이면
            이 케이스 하나로 계산해 반환 dict 에 담는다.

    Returns:
        ``{"points", "coords01", "x", "y", "target_names", "norm"}`` —
        ``points`` (N, 3) float64 원 좌표(케이스 매칭·표시용), ``coords01``
        (N, 3) float32 [0,1] 정규화 좌표(GINO ``input_geom``/``output_queries``
        입력), ``x`` (N, f) float32, ``y`` (N, C) float32.

    Raises:
        ValueError: 필드가 없거나 μ 차원이 norm 과 다른 경우.
    """
    mesh = getattr(dataset, "mesh", dataset)
    mu_arr = np.asarray(mu, dtype=np.float64).reshape(-1)
    if norm is None:
        norm = pointcloud_norm_from_cases(
            [mesh], mu_arr.reshape(1, -1), input_field_names=input_field_names
        )
    if mu_arr.size != np.asarray(norm["mu_center"]).size:
        raise ValueError(
            f"μ 차원({mu_arr.size})이 정규화 상수 차원"
            f"({np.asarray(norm['mu_center']).size})과 다릅니다."
        )

    points = np.asarray(mesh.points, dtype=np.float64)[:, :3]
    coords01 = (points - norm["coord_min"]) / norm["coord_range"]

    blocks: list[NDArray[np.float64]] = [coords01]
    for j, name in enumerate([str(n) for n in input_field_names]):
        values = np.asarray(mesh.point_data[name], dtype=np.float64).reshape(-1, 1)
        blocks.append((values - norm["input_center"][j]) / norm["input_scale"][j])
    mu_n = (mu_arr - norm["mu_center"]) / norm["mu_scale"]
    if mu_n.size:
        blocks.append(np.broadcast_to(mu_n, (points.shape[0], mu_n.size)))
    x = np.hstack(blocks).astype(np.float32)

    target_names: list[str] = []
    y_channels: list[NDArray[np.float64]] = []
    for name in field_names:
        for channel_name, values in _field_channels(mesh, str(name)):
            target_names.append(channel_name)
            y_channels.append(values)
    if not y_channels:
        raise ValueError("출력 필드를 최소 1개 지정하세요.")
    y = np.stack(y_channels, axis=1).astype(np.float32)

    return {
        "points": points,
        "coords01": coords01.astype(np.float32),
        "x": x,
        "y": y,
        "target_names": target_names,
        "norm": norm,
    }


class GINOCaseSetOperator(BaseOperator):
    """케이스-세트 파라메트릭 GINO — 점 피처(좌표+μ) → 점 필드 회귀.

    ``mesh_gnn`` 의 :class:`~naviertwin.core.gnn.gnn_surrogate.case_set_gnn.
    CaseSetGNN` 과 같은 역할(케이스마다 크기 다른 표본을 한 모델로 학습)이지만
    고정 그래프가 없다 — 예측은 임의 점군(``coords01``)에서 그대로 동작한다.

    타깃 표준화는 fit 에 받은 점군(= train 케이스)만으로 내부 계산한다 —
    group split 시 train-only 정규화가 구조적으로 보장된다.

    케이스 수가 3~20 수준이라 케이스별 루프(batch=1)로 충분하다 — 텐서는
    케이스별로 device 에 올린다(전 케이스 GPU 상주 금지, mesh_gnn 과 동일
    메모리 원칙).

    Attributes:
        train_losses_: epoch 별 케이스 평균 MSE (표준화 y 기준).
        is_fitted: fit 완료 여부.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        in_gno_radius: float = 0.25,
        out_gno_radius: float = 0.25,
        fno_n_modes: tuple[int, int, int] = (4, 4, 4),
        fno_hidden_channels: int = 16,
        fno_n_layers: int = 2,
        latent_resolution: int = 6,
        gno_embed_channels: int = 16,
        in_gno_channel_mlp_hidden_layers: Sequence[int] = (16, 16),
        out_gno_channel_mlp_hidden_layers: Sequence[int] = (16, 16),
        max_epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.gno_coord_dim = 3  # pyvista 점은 항상 (N,3) — §모듈 docstring 근거.
        self.in_gno_radius = float(in_gno_radius)
        self.out_gno_radius = float(out_gno_radius)
        self.fno_n_modes = tuple(int(m) for m in fno_n_modes)
        self.fno_hidden_channels = int(fno_hidden_channels)
        self.fno_n_layers = int(fno_n_layers)
        self.latent_resolution = max(2, int(latent_resolution))
        self.gno_embed_channels = int(gno_embed_channels)
        self.in_gno_channel_mlp_hidden_layers = [int(v) for v in in_gno_channel_mlp_hidden_layers]
        self.out_gno_channel_mlp_hidden_layers = [
            int(v) for v in out_gno_channel_mlp_hidden_layers
        ]
        self.max_epochs = int(max_epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.seed = seed

        self._model: Any = None
        self._device: Any = None
        self._latent_queries: Any = None
        self._y_mean: NDArray[np.float32] | None = None
        self._y_std: NDArray[np.float32] | None = None
        self.train_losses_: list[float] = []

    def _resolve_device(self) -> Any:
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _build(self) -> Any:
        GINO = _require_gino()
        import torch

        if self.seed is not None:
            torch.manual_seed(self.seed)

        return GINO(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            gno_coord_dim=self.gno_coord_dim,
            in_gno_radius=self.in_gno_radius,
            out_gno_radius=self.out_gno_radius,
            fno_n_modes=self.fno_n_modes,
            fno_hidden_channels=self.fno_hidden_channels,
            fno_n_layers=self.fno_n_layers,
            # linear transform 커널은 in_gno 출력 채널(=fno_in_channels)이
            # in_channels 와 같아야 한다(§모듈 docstring 실측 함정).
            fno_in_channels=self.in_channels,
            gno_embed_channels=self.gno_embed_channels,
            in_gno_channel_mlp_hidden_layers=list(self.in_gno_channel_mlp_hidden_layers),
            out_gno_channel_mlp_hidden_layers=list(self.out_gno_channel_mlp_hidden_layers),
        )

    def _make_latent_queries(self, device: Any) -> Any:
        """``[0,1]^3`` 균일 잠재 격자 — 케이스 무관, fit 중 1회만 만든다."""
        import torch

        axis = torch.linspace(0.0, 1.0, self.latent_resolution)
        grids = torch.meshgrid(*([axis] * self.gno_coord_dim), indexing="ij")
        latent = torch.stack(grids, dim=-1).unsqueeze(0)  # (1, r,...,r, 3)
        return latent.to(device)

    @staticmethod
    def _validate_case(case: dict[str, Any], in_dim: int, out_dim: int) -> None:
        x = np.asarray(case["x"])
        y = np.asarray(case["y"])
        coords01 = np.asarray(case["coords01"])
        if x.ndim != 2 or x.shape[1] != in_dim:
            raise ValueError(f"case['x'] 는 (N, {in_dim}) 이어야 합니다: {x.shape}")
        if y.ndim != 2 or y.shape[1] != out_dim or y.shape[0] != x.shape[0]:
            raise ValueError(
                f"case['y'] 는 (N, {out_dim}) 이어야 합니다: {y.shape} (N={x.shape[0]})"
            )
        if coords01.ndim != 2 or coords01.shape[0] != x.shape[0] or coords01.shape[1] != 3:
            raise ValueError(f"case['coords01'] 는 (N, 3) 이어야 합니다: {coords01.shape}")

    def fit(self, dataset: dict[str, Any]) -> None:
        """점군 목록으로 학습한다.

        Args:
            dataset: ``{"cases": list[dict]}`` — :func:`case_to_pointcloud`
                반환 dict 목록. 케이스마다 점 수가 달라도 된다.

        Raises:
            ValueError: 케이스가 없거나 shape 이 선언 차원과 다른 경우.
            RuntimeError: neuraloperator 미설치.
        """
        import torch

        cases = list(dataset["cases"])
        if not cases:
            raise ValueError("학습할 케이스(점군)가 없습니다.")
        for case in cases:
            self._validate_case(case, self.in_channels, self.out_channels)

        all_y = np.concatenate([np.asarray(c["y"], dtype=np.float32) for c in cases], axis=0)
        self._y_mean = all_y.mean(axis=0)
        std = all_y.std(axis=0)
        self._y_std = np.where(std > 0, std, 1.0).astype(np.float32)

        self._device = self._resolve_device()
        self._model = self._build().to(self._device)
        self._latent_queries = self._make_latent_queries(self._device)
        optim = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = torch.nn.MSELoss()

        # CPU 텐서로 한 번만 준비 — device 전송은 케이스별 스텝에서(lazy),
        # mesh_gnn(CaseSetGNN)과 동일한 메모리 원칙.
        prepared = [
            (
                torch.tensor(np.asarray(c["coords01"], dtype=np.float32)).unsqueeze(0),
                torch.tensor(np.asarray(c["x"], dtype=np.float32)).unsqueeze(0),
                torch.tensor(
                    (np.asarray(c["y"], dtype=np.float32) - self._y_mean) / self._y_std
                ).unsqueeze(0),
            )
            for c in cases
        ]

        rng = np.random.default_rng(self.seed)
        self._model.train()
        self.train_losses_ = []
        for _epoch in range(self.max_epochs):
            order = rng.permutation(len(prepared))
            epoch_loss = 0.0
            for i in order:
                pts_cpu, x_cpu, y_cpu = prepared[i]
                pts = pts_cpu.to(self._device)
                x = x_cpu.to(self._device)
                y = y_cpu.to(self._device)
                optim.zero_grad()
                pred = self._model(
                    input_geom=pts,
                    latent_queries=self._latent_queries,
                    output_queries=pts,
                    x=x,
                )
                loss = loss_fn(pred, y)
                loss.backward()
                optim.step()
                epoch_loss += float(loss.item())
            self.train_losses_.append(epoch_loss / len(prepared))

        self.n_epochs = self.max_epochs
        self.is_fitted = True
        logger.info(
            "GINOCaseSetOperator 학습 완료: %d 케이스, loss=%.6g",
            len(cases),
            self.train_losses_[-1] if self.train_losses_ else float("nan"),
        )

    def predict_case(self, case: dict[str, Any]) -> NDArray[np.float64]:
        """점군 하나의 점별 필드를 예측한다 (물리 단위로 역표준화).

        Args:
            case: ``coords01`` (N, 3), ``x`` (N, in_dim) 을 담은 dict
                (``y`` 는 없어도 된다 — 예측 전용 점군). ``coords01`` 은
                입력 지오메트리이자 출력 쿼리로 그대로 쓰인다(예측 = 학습
                점과 동일 위치의 함수값 — 표면/전 필드 회귀 규약).

        Returns:
            (N, out_dim) float64 예측.
        """
        import torch

        self._check_fitted()
        coords01 = np.asarray(case["coords01"], dtype=np.float32)
        x = np.asarray(case["x"], dtype=np.float32)
        self._model.eval()
        with torch.no_grad():
            pts = torch.tensor(coords01, device=self._device).unsqueeze(0)
            xt = torch.tensor(x, device=self._device).unsqueeze(0)
            out = self._model(
                input_geom=pts,
                latent_queries=self._latent_queries,
                output_queries=pts,
                x=xt,
            )
        pred = out[0].cpu().numpy().astype(np.float64)
        assert self._y_mean is not None and self._y_std is not None
        return pred * self._y_std + self._y_mean

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        """BaseOperator 계약 — ``{"case": dict}`` 를 받아 점별 예측을 돌려준다."""
        return self.predict_case(inputs["case"])

    # ------------------------------------------------------------------
    # pickle 지원 — torch 모듈 대신 state_dict 바이트로 직렬화
    # (CaseSetGNN.__getstate__/__setstate__ 와 같은 패턴).
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        model = state.pop("_model", None)
        state.pop("_device", None)
        state.pop("_latent_queries", None)
        if model is not None:
            import io

            import torch

            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            state["_model_bytes"] = buffer.getvalue()
        else:
            state["_model_bytes"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        model_bytes = state.pop("_model_bytes", None)
        self.__dict__.update(state)
        self._model = None
        self._device = None
        self._latent_queries = None
        if model_bytes is not None:
            import io

            import torch

            self._device = self._resolve_device()
            self._model = self._build().to(self._device)
            # GINO 의 state_dict 에는 활성함수 등 비-텐서 항목이 섞여 들어가
            # torch>=2.6 기본값(weights_only=True)이 거부한다 — 이 바이트는
            # 우리가 방금 __getstate__ 로 만든 신뢰 데이터이므로 명시적으로
            # weights_only=False 로 로드한다(CaseSetGNN 은 순수 텐서 파라미터만
            # 있어 이 문제가 없었다).
            self._model.load_state_dict(
                torch.load(
                    io.BytesIO(model_bytes),
                    map_location=self._device,
                    weights_only=False,
                )
            )
            self._model.eval()
            self._latent_queries = self._make_latent_queries(self._device)

    def _check_fitted(self) -> None:
        if not self.is_fitted or self._model is None:
            raise RuntimeError("GINOCaseSetOperator.fit() 을 먼저 호출해야 합니다.")


__all__ = ["GINOCaseSetOperator", "case_to_pointcloud", "pointcloud_norm_from_cases"]
