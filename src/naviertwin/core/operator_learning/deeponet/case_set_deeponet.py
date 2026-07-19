"""DeepONet 케이스 세트 래퍼 — 순수 데이터 기반, 재샘플 없음 (operator 전략 3번째 백엔드).

``mesh_gnn``/``gino``(:mod:`naviertwin.core.operator_learning.gino.gino_wrapper`)와
나란한 Route 2 계열 전략이지만, 이 모듈은 branch/trunk 분리 구조(Lu et al.,
"Learning nonlinear operators via DeepONet", Nature MI 2021)를 그대로 쓴다 —

    branch(μ) → (p,) 잠재         (μ = 케이스 운전조건 벡터)
    trunk(y, channel) → (p,) 잠재  (y = 쿼리 좌표, channel = 출력 채널 one-hot)
    G(u)(y) = branch(μ) · trunk(y, channel) + bias

GINO(점군 GNO+FNO)나 mesh_gnn(그래프)과 달리 **PDE 잔차도, 점 단위 입력
피처도 필요 없다** — branch 입력이 케이스 전체를 대표하는 운전조건 μ 뿐이라
GeometryFNO 의 SDF 채널 공통 격자 텐서화가 아예 없다(:mod:`naviertwin.core.
digital_twin.deeponet_engine` 참고). trunk 는 좌표만 받으므로 예측 시에도
임의 좌표(학습 케이스와 점 개수·순서가 달라도)에서 그대로 동작한다 — GINO 와
같은 이유로 mesh_gnn 의 kNN 그래프 폴백이 구조적으로 필요 없다.

다중 채널(예: ``p, U_x, U_y, U_z``) 출력은 표준 DeepONet의 스칼라 쿼리를
그대로 두고, trunk 입력에 채널 one-hot 을 덧붙여 쿼리를 채널 수만큼
타일링한다(``[coords01 | channel_onehot]``, field-major 순서 —
``mesh_gnn``/``gino`` 의 ``target_names`` 전개 규약과 동일). 이렇게 하면
branch/trunk 한 쌍으로 다중 채널을 학습할 수 있다(``base.py`` 의 branch/
trunk 대칭성을 깨지 않음).

케이스마다 메쉬(쿼리 좌표)가 같으면(identical mesh) 학습 배치 전체가 trunk
쿼리 하나를 공유하므로, 표준 DeepONet 학습(§ :mod:`deeponet` 의
``DeepONet.fit``)과 동형인 배치 행렬곱 경로를 쓴다 — epoch 당 trunk 평가
1회로 케이스 수만큼 브랜치 배치를 처리한다. 케이스마다 메쉬가 다르면(형상
가변) 케이스별로 trunk 를 따로 평가하는 루프로 전환한다(``GINOCaseSetOperator``
와 같은 배치=1 케이스 루프 원칙).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.operator_learning.base import BaseOperator
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _mlp(sizes: list[int], activation: str = "gelu") -> Any:
    """:mod:`deeponet` 의 ``_mlp`` 와 같은 구성 — 이 파일을 독립적으로 유지하려고
    복제한다(``deeponet.py`` 를 건드리지 않기 위함)."""
    import torch.nn as nn

    acts = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
    act_cls = acts.get(activation.lower(), nn.GELU)
    layers: list[Any] = []
    i = 0
    while i < len(sizes) - 1:
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act_cls())
        i += 1
    return nn.Sequential(*layers)


class DeepONetCaseSetOperator(BaseOperator):
    """케이스-세트 파라메트릭 DeepONet — branch=μ, trunk=쿼리 좌표(+채널 one-hot).

    Attributes:
        branch_in: 운전조건 μ 차원 k (branch net 입력 폭).
        n_channels: 출력 채널 수 C(예: ``p, U_x, U_y, U_z`` 이면 4).
        trunk_in: trunk net 입력 폭 = 3(좌표) + ``n_channels``(one-hot).
        identical_mesh: 마지막 :meth:`fit` 이 감지한 "케이스 전부 같은 쿼리
            좌표" 여부 — 배치 학습 경로를 탔는지 기록(디버그/엔진 메타데이터용).
        train_losses_: epoch 별 평균 MSE(표준화하지 않은 물리 단위 손실).
        is_fitted: fit 완료 여부.
    """

    def __init__(
        self,
        branch_in: int,
        n_channels: int,
        *,
        hidden: int = 64,
        latent: int = 32,
        n_branch_layers: int = 3,
        n_trunk_layers: int = 3,
        activation: str = "gelu",
        max_epochs: int = 200,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        self.branch_in = int(branch_in)
        self.n_channels = int(n_channels)
        self.trunk_in = 3 + self.n_channels
        self.hidden = int(hidden)
        self.latent = int(latent)
        self.n_branch_layers = int(n_branch_layers)
        self.n_trunk_layers = int(n_trunk_layers)
        self.activation = activation
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.seed = seed

        self.identical_mesh = False
        self._branch: Any = None
        self._trunk: Any = None
        self._device: Any = None
        self.train_losses_: list[float] = []

    def _resolve_device(self) -> Any:
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _build(self) -> None:
        import torch
        import torch.nn as nn

        if self.seed is not None:
            torch.manual_seed(self.seed)
        b_sizes = [self.branch_in] + [self.hidden] * (self.n_branch_layers - 1) + [self.latent]
        t_sizes = [self.trunk_in] + [self.hidden] * (self.n_trunk_layers - 1) + [self.latent]
        self._branch = _mlp(b_sizes, self.activation)
        self._trunk = _mlp(t_sizes, self.activation)
        # bias 는 branch 모듈에 등록해 .to(device) 전파를 받는다 (deeponet.py 와 동형).
        self._branch.register_parameter("bias", nn.Parameter(torch.zeros(1)))

    @property
    def _bias(self) -> Any:
        return self._branch.bias

    def _trunk_query_for(self, coords01: NDArray[np.float32]) -> NDArray[np.float32]:
        """좌표 (N,3) 을 field-major 채널 타일링한 trunk 입력 (C*N, 3+C) 로 바꾼다.

        순서는 채널 우선(``channel * N + point``) — ``mesh_gnn``/``gino`` 의
        field-major 평탄화(``target_names`` 순서)와 동일한 규약이라 예측
        벡터를 그대로 이어붙이면 ``split_multi_prediction`` 경계가 맞는다.
        """
        coords = np.asarray(coords01, dtype=np.float32)
        n = coords.shape[0]
        eye = np.eye(self.n_channels, dtype=np.float32)
        blocks = [
            np.hstack([coords, np.broadcast_to(eye[c], (n, self.n_channels))])
            for c in range(self.n_channels)
        ]
        return np.concatenate(blocks, axis=0)  # (C*N, 3+C)

    @staticmethod
    def _target_flat(y: NDArray[np.float32]) -> NDArray[np.float32]:
        """(N,C) 타깃을 field-major (C*N,) 로 평탄화 — ``_trunk_query_for`` 와 짝."""
        return np.asarray(y, dtype=np.float32).T.reshape(-1)

    def _validate_case(self, case: dict[str, Any]) -> None:
        mu = np.asarray(case["mu"])
        coords01 = np.asarray(case["coords01"])
        y = np.asarray(case["y"])
        if mu.ndim != 1 or mu.shape[0] != self.branch_in:
            raise ValueError(f"case['mu'] 는 ({self.branch_in},) 이어야 합니다: {mu.shape}")
        if coords01.ndim != 2 or coords01.shape[1] != 3:
            raise ValueError(f"case['coords01'] 는 (N, 3) 이어야 합니다: {coords01.shape}")
        if (
            y.ndim != 2
            or y.shape[1] != self.n_channels
            or y.shape[0] != coords01.shape[0]
        ):
            raise ValueError(
                f"case['y'] 는 (N, {self.n_channels}) 이어야 합니다: {y.shape} "
                f"(N={coords01.shape[0]})"
            )

    def fit(self, dataset: dict[str, Any]) -> None:
        """케이스 목록으로 학습한다.

        Args:
            dataset: ``{"cases": list[dict]}`` — 각 항목은
                ``{"mu": (k,), "coords01": (N,3), "y": (N,C)}``. 케이스마다
                점 수가 달라도 된다(형상 가변).

        Raises:
            ValueError: 케이스가 없거나 shape 이 선언 차원과 다른 경우.
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        cases = list(dataset["cases"])
        if not cases:
            raise ValueError("학습할 케이스가 없습니다.")
        for case in cases:
            self._validate_case(case)

        ref_coords = np.asarray(cases[0]["coords01"], dtype=np.float32)
        identical_mesh = all(
            np.asarray(c["coords01"], dtype=np.float32).shape == ref_coords.shape
            and np.allclose(np.asarray(c["coords01"], dtype=np.float32), ref_coords)
            for c in cases[1:]
        )
        self.identical_mesh = bool(identical_mesh)

        self._device = self._resolve_device()
        self._build()
        self._branch.to(self._device)
        self._trunk.to(self._device)
        params = list(self._branch.parameters()) + list(self._trunk.parameters())
        optim = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()

        self.train_losses_ = []
        if self.identical_mesh:
            # 배치 경로 — trunk 쿼리가 케이스 전부 같으므로 epoch 당 1회만
            # 평가하고 branch 배치와 행렬곱한다(표준 DeepONet.fit 과 동형).
            trunk_np = self._trunk_query_for(ref_coords)
            trunk_t = torch.tensor(trunk_np, device=self._device)
            branch_all = np.stack(
                [np.asarray(c["mu"], dtype=np.float32) for c in cases]
            )
            y_all = np.stack([self._target_flat(c["y"]) for c in cases])
            loader = DataLoader(
                TensorDataset(torch.tensor(branch_all), torch.tensor(y_all)),
                batch_size=min(self.batch_size, len(cases)),
                shuffle=True,
            )
            for _epoch in range(self.max_epochs):
                epoch_loss = 0.0
                trunk_feat = self._trunk(trunk_t)
                for bb, yb in loader:
                    bb = bb.to(self._device)
                    yb = yb.to(self._device)
                    optim.zero_grad()
                    branch_feat = self._branch(bb)
                    pred = branch_feat @ trunk_feat.T + self._bias
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    optim.step()
                    trunk_feat = self._trunk(trunk_t)
                    epoch_loss += float(loss.item()) * bb.shape[0]
                self.train_losses_.append(epoch_loss / max(len(cases), 1))
        else:
            # 케이스별 루프 — 형상 가변이라 케이스마다 trunk 쿼리 길이가 다르다
            # (GINOCaseSetOperator 와 같은 batch=1 케이스 원칙).
            prepared = [
                (
                    torch.tensor(
                        np.asarray(c["mu"], dtype=np.float32)
                    ).unsqueeze(0),
                    torch.tensor(
                        self._trunk_query_for(np.asarray(c["coords01"], dtype=np.float32))
                    ),
                    torch.tensor(self._target_flat(c["y"])).unsqueeze(0),
                )
                for c in cases
            ]
            rng = np.random.default_rng(self.seed)
            for _epoch in range(self.max_epochs):
                order = rng.permutation(len(prepared))
                epoch_loss = 0.0
                for i in order:
                    mu_cpu, trunk_cpu, y_cpu = prepared[i]
                    mu_t = mu_cpu.to(self._device)
                    trunk_t = trunk_cpu.to(self._device)
                    y_t = y_cpu.to(self._device)
                    optim.zero_grad()
                    branch_feat = self._branch(mu_t)  # (1, p)
                    trunk_feat = self._trunk(trunk_t)  # (C*N_i, p)
                    pred = branch_feat @ trunk_feat.T + self._bias  # (1, C*N_i)
                    loss = loss_fn(pred, y_t)
                    loss.backward()
                    optim.step()
                    epoch_loss += float(loss.item())
                self.train_losses_.append(epoch_loss / max(len(prepared), 1))

        self.n_epochs = self.max_epochs
        self.is_fitted = True
        logger.info(
            "DeepONetCaseSetOperator 학습 완료: %d 케이스, identical_mesh=%s, loss=%.6g",
            len(cases),
            self.identical_mesh,
            self.train_losses_[-1] if self.train_losses_ else float("nan"),
        )

    def predict_case(self, case: dict[str, Any]) -> NDArray[np.float64]:
        """케이스 하나(μ + 쿼리 좌표)의 점별 필드를 예측한다.

        Args:
            case: ``{"mu": (k,), "coords01": (N,3)}`` — ``coords01`` 은 학습
                케이스와 점 개수·순서가 달라도 된다(임의 좌표 예측).

        Returns:
            (N, C) float64 예측.
        """
        import torch

        self._check_fitted()
        mu = np.asarray(case["mu"], dtype=np.float32).reshape(1, -1)
        coords01 = np.asarray(case["coords01"], dtype=np.float32)
        n = coords01.shape[0]
        trunk_np = self._trunk_query_for(coords01)
        self._branch.eval()
        self._trunk.eval()
        with torch.no_grad():
            mu_t = torch.tensor(mu, device=self._device)
            trunk_t = torch.tensor(trunk_np, device=self._device)
            branch_feat = self._branch(mu_t)
            trunk_feat = self._trunk(trunk_t)
            pred = (branch_feat @ trunk_feat.T + self._bias)[0]  # (C*N,)
        pred_np = pred.cpu().numpy().astype(np.float64)
        return pred_np.reshape(self.n_channels, n).T  # (N, C)

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        """BaseOperator 계약 — ``{"case": dict}`` 를 받아 점별 예측을 돌려준다."""
        return self.predict_case(inputs["case"])

    # ------------------------------------------------------------------
    # pickle 지원 — torch 모듈 대신 state_dict 바이트로 직렬화
    # (GINOCaseSetOperator.__getstate__/__setstate__ 와 같은 패턴).
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        branch = state.pop("_branch", None)
        trunk = state.pop("_trunk", None)
        state.pop("_device", None)
        if branch is not None and trunk is not None:
            import io

            import torch

            buf_b, buf_t = io.BytesIO(), io.BytesIO()
            torch.save(branch.state_dict(), buf_b)
            torch.save(trunk.state_dict(), buf_t)
            state["_branch_bytes"] = buf_b.getvalue()
            state["_trunk_bytes"] = buf_t.getvalue()
        else:
            state["_branch_bytes"] = None
            state["_trunk_bytes"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        branch_bytes = state.pop("_branch_bytes", None)
        trunk_bytes = state.pop("_trunk_bytes", None)
        self.__dict__.update(state)
        self._branch = None
        self._trunk = None
        self._device = None
        if branch_bytes is not None and trunk_bytes is not None:
            import io

            import torch

            self._device = self._resolve_device()
            self._build()
            self._branch.load_state_dict(
                torch.load(io.BytesIO(branch_bytes), map_location=self._device, weights_only=False)
            )
            self._trunk.load_state_dict(
                torch.load(io.BytesIO(trunk_bytes), map_location=self._device, weights_only=False)
            )
            self._branch.to(self._device)
            self._trunk.to(self._device)
            self._branch.eval()
            self._trunk.eval()

    def _check_fitted(self) -> None:
        if not self.is_fitted or self._branch is None or self._trunk is None:
            raise RuntimeError("DeepONetCaseSetOperator.fit() 을 먼저 호출해야 합니다.")


def deeponet_norm_from_cases(
    datasets: Sequence[Any],
    params: NDArray[np.float64],
) -> dict[str, Any]:
    """train 케이스들로 정규화 상수를 계산한다 — ``gino_wrapper.
    pointcloud_norm_from_cases`` 와 동일한 규약(좌표 min-max [0,1]^3, μ
    z-score)을 그대로 재사용한다(두 Route 2 백엔드가 같은 정규화를 쓰면
    비교가 쉬워진다).
    """
    from naviertwin.core.operator_learning.gino.gino_wrapper import (
        pointcloud_norm_from_cases,
    )

    return pointcloud_norm_from_cases(datasets, params)


def case_to_deeponet_sample(
    dataset: Any,
    mu: NDArray[np.float64],
    field_names: Sequence[str],
    *,
    norm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """케이스 하나를 DeepONet 학습/예측용 dict 로 바꾼다 (재샘플 없음).

    ``gino_wrapper.case_to_pointcloud`` 를 그대로 위임한다 — 벡터 성분 전개
    (``target_names``)와 좌표 정규화 규약이 mesh_gnn/GeometryFNO/GINO 전부와
    문자열까지 일치해야 하므로, 그 로직을 복제하지 않고 재사용한다. DeepONet
    은 점 단위 입력 피처(``x``)가 필요 없으므로(순수 branch=μ, trunk=좌표)
    반환 dict 의 ``x`` 필드는 그냥 버려도 된다 — 호출자는 ``coords01``,
    ``y``, ``target_names`` 만 쓴다.

    Returns:
        ``{"points", "coords01", "y", "target_names", "norm"}`` —
        :func:`~naviertwin.core.operator_learning.gino.gino_wrapper.case_to_pointcloud`
        와 같은 키(``x`` 포함, 미사용).
    """
    from naviertwin.core.operator_learning.gino.gino_wrapper import case_to_pointcloud

    return case_to_pointcloud(dataset, mu, field_names, norm=norm)


__all__ = [
    "DeepONetCaseSetOperator",
    "case_to_deeponet_sample",
    "deeponet_norm_from_cases",
]
