"""CaseSetMGN — 케이스별 그래프 목록으로 학습하는 MeshGraphNets 래퍼.

Route 2 세 번째 배선. :class:`~naviertwin.core.gnn.gnn_surrogate.case_set_gnn.
CaseSetGNN`(GCN)의 판박이지만, 메시지패싱이 진짜 ``edge_features`` 를 쓰는
:class:`~naviertwin.core.gnn.meshgraphnets.meshgraphnets.MeshGraphNets`
(Encode-Process-Decode)를 감싼다.

기존 ``MeshGraphNets`` 는 "전 trajectory 공유 edge_index" 를 전제해 케이스
세트(케이스마다 노드 수·에지가 다름)를 직접 학습할 수 없다 — 그래서
``MeshGraphNets.fit()`` 을 호출하지 않는다. 대신:

    1. ``MeshGraphNets`` 인스턴스를 만들어 아키텍처/디바이스만 구성한다
       (``_build``/``_resolve_device`` 재사용 — private 이지만 같은 패키지
       내부 조립이라 파일 수정은 아니다).
    2. 케이스별 그래프 루프(기존 ``CaseSetGNN`` 패턴)로 **직접** forward/
       backward 를 돌려 그 하나의 모델을 여러 케이스에 걸쳐 학습한다.
    3. 예측은 **수정된(더 이상 stale 하지 않은) 공개 API**
       :meth:`MeshGraphNets.predict` 를 ``edge_index``/``edge_features`` 를
       질의 케이스 것으로 오버라이드해서 부른다 — "학습된 MeshGraphNets 를
       감싸고 그 predict() 를 그대로 쓴다"는 배선 지시를 그대로 따른다.

정상(steady) 필드 회귀를 MGN 의 델타(트레젝토리) 프레임에 태우는 방법:
    - t=0 상태 ``x0`` = [정적 노드 피처(좌표+입력+μ 브로드캐스트) | 0(타깃 채널)]
    - t=1 상태 ``x1`` = [정적 노드 피처(동일)                     | 표준화 타깃]
    - MGN 은 ``u_{t+1} = u_t + MGN(u_t)`` 델타를 배우므로, 정적 채널의 델타는
      항상 0(모델이 쉽게 배우는 항등), 타깃 채널의 델타는 정확히 표준화 타깃
      값이 된다 — 한 스텝짜리 "트레젝토리 쌍"이 곧 회귀가 된다.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.gnn.meshgraphnets.meshgraphnets import MeshGraphNets
from naviertwin.core.operator_learning.base import BaseOperator
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

# torch_geometric 미설치 시 오류는 MeshGraphNets._build() 가 이미 일으킨다
# (RuntimeError, 같은 메시지) — 여기서 따로 감지하지 않는다.


class CaseSetMGN(BaseOperator):
    """케이스-세트 파라메트릭 MeshGraphNets — 메시지패싱(edge_features 사용) 회귀.

    ``CaseSetGNN`` 과 인터페이스가 동형이다(``fit({"graphs": [...]})`` /
    ``predict_graph(graph)``) — 차이는 내부 아키텍처(GCN vs Encode-Process-
    Decode 메시지패싱)뿐이라 :mod:`~naviertwin.core.digital_twin.
    mgn_case_set_engine` 이 ``mesh_gnn_engine`` 과 같은 얇은 어댑터로 감쌀 수
    있다.

    Attributes:
        train_losses_: epoch 별 케이스 평균 MSE(표준화 y 기준, 전체 상태
            델타에 대한 손실 — 정적 채널 델타는 0 이 정답이라 자연히 낮게
            수렴한다).
        is_fitted: fit 완료 여부.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_feat: int,
        hidden: int = 64,
        n_msgpass: int = 4,
        max_epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.edge_feat = int(edge_feat)
        self.hidden = int(hidden)
        self.n_msgpass = max(1, int(n_msgpass))
        self.max_epochs = int(max_epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.seed = seed

        self._mgn: MeshGraphNets | None = None
        self._y_mean: NDArray[np.float32] | None = None
        self._y_std: NDArray[np.float32] | None = None
        self.train_losses_: list[float] = []

    @property
    def _total_node_feat(self) -> int:
        """감싼 MeshGraphNets 의 ``node_feat`` — 정적 피처 + 타깃 채널."""
        return self.in_dim + self.out_dim

    @staticmethod
    def _validate_graph(
        graph: dict[str, Any], in_dim: int, out_dim: int, edge_feat: int
    ) -> None:
        x = np.asarray(graph["x"])
        y = np.asarray(graph["y"])
        edge = np.asarray(graph["edge_index"])
        edge_attr = np.asarray(graph["edge_attr"])
        if x.ndim != 2 or x.shape[1] != in_dim:
            raise ValueError(f"graph['x'] 는 (N, {in_dim}) 이어야 합니다: {x.shape}")
        if y.ndim != 2 or y.shape[1] != out_dim or y.shape[0] != x.shape[0]:
            raise ValueError(
                f"graph['y'] 는 (N, {out_dim}) 이어야 합니다: {y.shape} (N={x.shape[0]})"
            )
        if edge.ndim != 2 or edge.shape[0] != 2:
            raise ValueError(f"edge_index 는 (2, E) 이어야 합니다: {edge.shape}")
        if edge.size and int(edge.max()) >= x.shape[0]:
            raise ValueError(
                f"edge_index 최대값({int(edge.max())})이 노드 수({x.shape[0]})를 넘습니다."
            )
        if edge_attr.ndim != 2 or edge_attr.shape[1] != edge_feat:
            raise ValueError(
                f"graph['edge_attr'] 는 (E, {edge_feat}) 이어야 합니다: {edge_attr.shape}"
            )
        if edge_attr.shape[0] != edge.shape[1]:
            raise ValueError(
                f"edge_attr 개수({edge_attr.shape[0]})가 edge_index 개수"
                f"({edge.shape[1]})와 다릅니다."
            )

    def fit(self, dataset: dict[str, Any]) -> None:
        """그래프 목록으로 학습한다.

        Args:
            dataset: ``{"graphs": list[dict]}`` —
                :func:`~naviertwin.core.gnn.case_graph.case_to_graph` 반환
                dict 목록(``x``/``y``/``edge_index``/``edge_attr`` 필요).
                케이스마다 N, E 가 달라도 된다.

        Raises:
            ValueError: 그래프가 없거나 shape 이 선언 차원과 다른 경우.
            RuntimeError: torch_geometric 미설치(``MeshGraphNets._build`` 경유).
        """
        import torch

        graphs = list(dataset["graphs"])
        if not graphs:
            raise ValueError("학습할 그래프가 없습니다.")
        for graph in graphs:
            self._validate_graph(graph, self.in_dim, self.out_dim, self.edge_feat)

        # 타깃 표준화 — fit 에 받은(=train) 그래프만으로 계산한다.
        all_y = np.concatenate(
            [np.asarray(g["y"], dtype=np.float32) for g in graphs], axis=0
        )
        self._y_mean = all_y.mean(axis=0)
        std = all_y.std(axis=0)
        self._y_std = np.where(std > 0, std, 1.0).astype(np.float32)

        # 감싸는 MeshGraphNets — 아키텍처/디바이스 조립만 위임하고, fit() 은
        # 절대 부르지 않는다(전 trajectory 공유 edge_index 전제라 케이스
        # 세트에 못 씀). 대신 케이스별 루프로 이 하나의 모델을 직접 학습한다.
        mgn = MeshGraphNets(
            node_feat=self._total_node_feat,
            edge_feat=self.edge_feat,
            hidden=self.hidden,
            n_msgpass=self.n_msgpass,
            max_epochs=self.max_epochs,
            lr=self.lr,
            device=self.device,
            seed=self.seed,
        )
        mgn._device = mgn._resolve_device()
        mgn._model = mgn._build().to(mgn._device)

        optim = torch.optim.Adam(
            mgn._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = torch.nn.MSELoss()

        # CPU 텐서로 한 번만 준비 — device 전송은 케이스별 스텝에서(lazy),
        # CaseSetGNN 과 동일하게 전 케이스 GPU 상주를 피한다.
        prepared: list[tuple[Any, Any, Any, Any]] = []
        for g in graphs:
            x_static = np.asarray(g["x"], dtype=np.float32)
            y_norm = (
                np.asarray(g["y"], dtype=np.float32) - self._y_mean
            ) / self._y_std
            x0 = np.concatenate([x_static, np.zeros_like(y_norm)], axis=1)
            x1 = np.concatenate([x_static, y_norm], axis=1)
            prepared.append(
                (
                    torch.tensor(x0),
                    torch.tensor(x1),
                    torch.tensor(np.asarray(g["edge_index"], dtype=np.int64)),
                    torch.tensor(np.asarray(g["edge_attr"], dtype=np.float32)),
                )
            )

        rng = np.random.default_rng(self.seed)
        mgn._model.train()
        self.train_losses_ = []
        for _epoch in range(self.max_epochs):
            order = rng.permutation(len(prepared))
            epoch_loss = 0.0
            for i in order:
                x0_cpu, x1_cpu, edge_cpu, eattr_cpu = prepared[i]
                x0 = x0_cpu.to(mgn._device)
                x1 = x1_cpu.to(mgn._device)
                edge = edge_cpu.to(mgn._device)
                eattr = eattr_cpu.to(mgn._device)
                optim.zero_grad()
                delta_pred = mgn._model(x0, edge, eattr)
                loss = loss_fn(delta_pred, x1 - x0)
                loss.backward()
                optim.step()
                epoch_loss += float(loss.item())
            self.train_losses_.append(epoch_loss / len(prepared))

        mgn.n_epochs = self.max_epochs
        mgn.is_fitted = True
        self._mgn = mgn

        self.n_epochs = self.max_epochs
        self.is_fitted = True
        logger.info(
            "CaseSetMGN 학습 완료: %d 그래프, msgpass=%d, loss=%.6g",
            len(graphs),
            self.n_msgpass,
            self.train_losses_[-1] if self.train_losses_ else float("nan"),
        )

    def predict_graph(self, graph: dict[str, Any]) -> NDArray[np.float64]:
        """그래프 하나의 노드별 필드를 예측한다 (물리 단위로 역표준화).

        ``graph["edge_index"]``/``graph["edge_attr"]`` 를 감싼
        ``MeshGraphNets.predict()`` 에 오버라이드로 넘긴다 — 수정된(더 이상
        stale edge_features 를 재사용하지 않는) 경로를 그대로 탄다.

        Args:
            graph: ``x`` (N, in_dim), ``edge_index`` (2, E), ``edge_attr``
                (E, edge_feat) 를 담은 dict(``y`` 는 없어도 된다).

        Returns:
            (N, out_dim) float64 예측.
        """
        self._check_fitted()
        assert self._mgn is not None and self._y_mean is not None and self._y_std is not None
        x_static = np.asarray(graph["x"], dtype=np.float32)
        n = x_static.shape[0]
        x0 = np.concatenate(
            [x_static, np.zeros((n, self.out_dim), dtype=np.float32)], axis=1
        )
        edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
        edge_attr = np.asarray(graph["edge_attr"], dtype=np.float32)
        out = self._mgn.predict(
            {
                "x": x0,
                "n_steps": 1,
                "edge_index": edge_index,
                "edge_features": edge_attr,
            }
        )  # (2, N, total_dim) — [t=0, t=1]
        y_norm = np.asarray(out[-1], dtype=np.float64)[:, self.in_dim :]
        return y_norm * self._y_std + self._y_mean

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        """BaseOperator 계약 — ``{"graph": dict}`` 를 받아 노드 예측을 돌려준다."""
        return self.predict_graph(inputs["graph"])

    # ------------------------------------------------------------------
    # pickle 지원 — 감싼 MeshGraphNets 의 ``_model`` 은 ``_build()`` 안에서
    # 로컬로 정의된 클래스(``_MGN``)의 인스턴스라 표준 pickle 이 못 다룬다.
    # meshgraphnets.py 는 건드리지 않고, 이 파일이 state_dict 바이트로
    # 직렬화한다 — CaseSetGNN.__getstate__/__setstate__ 와 같은 패턴.
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        mgn = state.pop("_mgn", None)
        if mgn is not None:
            import torch

            model = getattr(mgn, "_model", None)
            model_bytes: bytes | None = None
            if model is not None:
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                model_bytes = buffer.getvalue()
            state["_mgn_ctor"] = {
                "node_feat": mgn.node_feat,
                "edge_feat": mgn.edge_feat,
                "hidden": mgn.hidden,
                "n_msgpass": mgn.n_msgpass,
                "max_epochs": mgn.max_epochs,
                "lr": mgn.lr,
                "device": mgn.device,
                "seed": mgn.seed,
            }
            state["_mgn_model_bytes"] = model_bytes
            state["_mgn_train_losses"] = list(getattr(mgn, "train_losses_", []))
            state["_mgn_n_epochs"] = int(getattr(mgn, "n_epochs", 0))
            state["_mgn_is_fitted"] = bool(getattr(mgn, "is_fitted", False))
        else:
            state["_mgn_ctor"] = None
            state["_mgn_model_bytes"] = None
            state["_mgn_train_losses"] = []
            state["_mgn_n_epochs"] = 0
            state["_mgn_is_fitted"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        ctor = state.pop("_mgn_ctor", None)
        model_bytes = state.pop("_mgn_model_bytes", None)
        train_losses = state.pop("_mgn_train_losses", [])
        n_epochs = state.pop("_mgn_n_epochs", 0)
        is_fitted = state.pop("_mgn_is_fitted", False)
        self.__dict__.update(state)
        self._mgn = None
        if ctor is not None:
            import torch

            mgn = MeshGraphNets(**ctor)
            mgn._device = mgn._resolve_device()
            if model_bytes is not None:
                mgn._model = mgn._build().to(mgn._device)
                mgn._model.load_state_dict(
                    torch.load(io.BytesIO(model_bytes), map_location=mgn._device)
                )
                mgn._model.eval()
            mgn.train_losses_ = list(train_losses)
            mgn.n_epochs = n_epochs
            mgn.is_fitted = is_fitted
            self._mgn = mgn

    def _check_fitted(self) -> None:
        if not self.is_fitted or self._mgn is None:
            raise RuntimeError("CaseSetMGN.fit() 을 먼저 호출해야 합니다.")


__all__ = ["CaseSetMGN"]
