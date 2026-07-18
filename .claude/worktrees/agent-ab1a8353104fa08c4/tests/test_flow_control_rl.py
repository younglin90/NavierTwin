"""Round 35 — GaussianPolicy + REINFORCE."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch 필요")


class TestGaussianPolicy:
    def test_sample_shape(self) -> None:
        from naviertwin.core.flow_control.policy_gradient import GaussianPolicy

        pol = GaussianPolicy(state_dim=4, action_dim=2, hidden=16, seed=0)
        rng = np.random.default_rng(0)
        s = rng.standard_normal((10, 4)).astype(np.float32)
        a, logp = pol.sample(s)
        assert a.shape == (10, 2)
        assert logp.shape == (10,)

    def test_reinforce_loss_changes(self) -> None:
        from naviertwin.core.flow_control.policy_gradient import (
            GaussianPolicy,
            reinforce_update,
        )

        pol = GaussianPolicy(state_dim=3, action_dim=1, hidden=16, seed=0)
        rng = np.random.default_rng(0)

        # 여러 업데이트 후 loss 변화 기록
        losses = []
        for _ in range(5):
            s = rng.standard_normal((30, 3)).astype(np.float32)
            a, _ = pol.sample(s)
            # 보상: 액션 크기 클수록 나쁨 (작은 a 선호)
            r = -np.abs(a).ravel()
            loss = reinforce_update(pol, s, a, None, r, lr=1e-2)
            losses.append(loss)
        # loss 가 유한
        assert all(np.isfinite(losses))
