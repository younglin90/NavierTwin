"""GNN surrogate / MeshGraphNets 테스트.

PyTorch Geometric 필요 — 미설치 시 전체 skip.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch 필요")
pytest.importorskip("torch_geometric", reason="PyTorch Geometric 필요")


class TestGNNSurrogate:
    def test_fit_predict_shapes(self) -> None:
        from naviertwin.core.gnn.gnn_surrogate.gnn_surrogate import GNNSurrogate

        rng = np.random.default_rng(0)
        n_samples, n_nodes = 10, 25
        X = rng.standard_normal((n_samples, n_nodes, 2)).astype(np.float32)
        Y = (X ** 2).astype(np.float32)
        edge = np.stack([np.arange(n_nodes), np.roll(np.arange(n_nodes), -1)]).astype(np.int64)

        op = GNNSurrogate(in_dim=2, out_dim=2, hidden=16, n_layers=2, max_epochs=3)
        op.fit({"node_features": X, "outputs": Y, "edge_index": edge})
        assert op.is_fitted
        assert len(op.train_losses_) == 3

        y_hat = op.predict({"x": X[:2]})
        assert y_hat.shape == (2, n_nodes, 2)

    def test_predict_unfitted_raises(self) -> None:
        from naviertwin.core.gnn.gnn_surrogate.gnn_surrogate import GNNSurrogate

        op = GNNSurrogate(in_dim=1, out_dim=1, hidden=4, n_layers=1, max_epochs=1)
        with pytest.raises(RuntimeError, match="fit"):
            op.predict({"x": np.zeros((5, 1))})

    def test_edge_validation(self) -> None:
        from naviertwin.core.gnn.gnn_surrogate.gnn_surrogate import GNNSurrogate

        op = GNNSurrogate(in_dim=1, out_dim=1, hidden=4, n_layers=1, max_epochs=1)
        X = np.zeros((2, 5, 1), dtype=np.float32)
        Y = np.zeros((2, 5, 1), dtype=np.float32)
        with pytest.raises(ValueError, match="edge_index"):
            op.fit({
                "node_features": X,
                "outputs": Y,
                "edge_index": np.array([0, 1, 2]),  # 잘못된 shape
            })


class TestMeshGraphNets:
    def test_fit_and_rollout(self) -> None:
        from naviertwin.core.gnn.meshgraphnets.meshgraphnets import MeshGraphNets

        rng = np.random.default_rng(0)
        n_traj, Tp1, n_nodes, f = 3, 5, 20, 2
        traj = rng.standard_normal((n_traj, Tp1, n_nodes, f)).astype(np.float32)
        edge = np.stack([np.arange(n_nodes), np.roll(np.arange(n_nodes), -1)]).astype(np.int64)

        op = MeshGraphNets(
            node_feat=f, edge_feat=1, hidden=8, n_msgpass=2, max_epochs=2,
        )
        op.fit({"trajectories": traj, "edge_index": edge})

        rollout = op.predict({"x": traj[0, 0], "n_steps": 3})
        assert rollout.shape == (4, n_nodes, f)

    def test_insufficient_timesteps_raises(self) -> None:
        from naviertwin.core.gnn.meshgraphnets.meshgraphnets import MeshGraphNets

        op = MeshGraphNets(node_feat=1, edge_feat=1, hidden=4, n_msgpass=1, max_epochs=1)
        traj = np.zeros((2, 1, 5, 1), dtype=np.float32)
        edge = np.stack([np.arange(5), np.roll(np.arange(5), -1)]).astype(np.int64)
        with pytest.raises(ValueError, match="타임스텝"):
            op.fit({"trajectories": traj, "edge_index": edge})

    def test_edge_feature_dim_mismatch(self) -> None:
        from naviertwin.core.gnn.meshgraphnets.meshgraphnets import MeshGraphNets

        op = MeshGraphNets(node_feat=1, edge_feat=3, hidden=4, n_msgpass=1, max_epochs=1)
        traj = np.zeros((1, 2, 4, 1), dtype=np.float32)
        edge = np.stack([np.arange(4), np.roll(np.arange(4), -1)]).astype(np.int64)
        edge_feat = np.ones((edge.shape[1], 2), dtype=np.float32)  # 잘못된 차원
        with pytest.raises(ValueError, match="edge_features"):
            op.fit({
                "trajectories": traj,
                "edge_index": edge,
                "edge_features": edge_feat,
            })

    def test_predict_edge_features_override_used(self) -> None:
        """predict()에 새 edge_index/edge_features 를 넘기면 롤아웃이 이를 반영해야 한다.

        버그(수정 전): predict() 는 inputs["edge_index"] 오버라이드는 받으면서도
        inputs["edge_features"] 는 아예 읽지 않고 fit() 시점의 ``self._edge_features``
        를 그대로 재사용했다 — 서로 다른 edge_features 를 넘겨도 롤아웃 결과가 동일했다.
        """
        from naviertwin.core.gnn.meshgraphnets.meshgraphnets import MeshGraphNets

        rng = np.random.default_rng(0)
        n_traj, Tp1, n_nodes, f = 2, 3, 6, 2
        traj = rng.standard_normal((n_traj, Tp1, n_nodes, f)).astype(np.float32)
        edge = np.stack([np.arange(n_nodes), np.roll(np.arange(n_nodes), -1)]).astype(np.int64)

        op = MeshGraphNets(
            node_feat=f, edge_feat=1, hidden=8, n_msgpass=2, max_epochs=2, seed=0,
        )
        op.fit({"trajectories": traj, "edge_index": edge})

        x0 = traj[0, 0]
        edge_feat_a = np.zeros((edge.shape[1], 1), dtype=np.float32)
        edge_feat_b = np.full((edge.shape[1], 1), 5.0, dtype=np.float32)

        rollout_a = op.predict(
            {"x": x0, "n_steps": 2, "edge_index": edge, "edge_features": edge_feat_a}
        )
        rollout_b = op.predict(
            {"x": x0, "n_steps": 2, "edge_index": edge, "edge_features": edge_feat_b}
        )

        assert not np.allclose(rollout_a, rollout_b), (
            "서로 다른 edge_features 를 넘겼는데 롤아웃 결과가 같다 — "
            "predict() 가 edge_features 오버라이드를 무시하고 fit() 시점 값을 "
            "재사용하는 버그"
        )

    def test_predict_edge_index_size_change_does_not_reuse_stale_edge_features(
        self,
    ) -> None:
        """predict() 에서 edge_index 만 바뀌어도(에지 개수 변경) fit() 시점
        edge_features 를 억지로 재사용해 크래시하지 않고, 새 그래프에 맞는 기본값을
        다시 만들어야 한다.
        """
        from naviertwin.core.gnn.meshgraphnets.meshgraphnets import MeshGraphNets

        rng = np.random.default_rng(1)
        n_traj, Tp1, n_nodes, f = 2, 3, 6, 2
        traj = rng.standard_normal((n_traj, Tp1, n_nodes, f)).astype(np.float32)
        edge_fit = np.stack(
            [np.arange(n_nodes), np.roll(np.arange(n_nodes), -1)]
        ).astype(np.int64)

        op = MeshGraphNets(
            node_feat=f, edge_feat=1, hidden=8, n_msgpass=2, max_epochs=2, seed=0,
        )
        op.fit({"trajectories": traj, "edge_index": edge_fit})

        # predict 시점 그래프는 노드/에지 개수가 다른 별개 메쉬(형상 가변 시나리오)
        n_nodes2 = 9
        edge_predict = np.stack(
            [np.arange(n_nodes2), np.roll(np.arange(n_nodes2), -1)]
        ).astype(np.int64)
        x0 = rng.standard_normal((n_nodes2, f)).astype(np.float32)

        rollout = op.predict({"x": x0, "n_steps": 1, "edge_index": edge_predict})
        assert rollout.shape == (2, n_nodes2, f)
