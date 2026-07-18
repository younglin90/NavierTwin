"""간단한 REINFORCE policy gradient — flow control 기본 골격.

목적: 상태 s_t (유동장 또는 축약 표현) → 액션 a_t (경계조건, 힘 등).
에피소드 후 누적 보상을 사용해 stochastic policy 를 업데이트.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.flow_control.policy_gradient import (
    ...     GaussianPolicy, reinforce_update,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> policy = GaussianPolicy(state_dim=4, action_dim=1, hidden=16, seed=0)
    >>> # 임의 에피소드 데이터
    >>> states = rng.standard_normal((20, 4)).astype(np.float32)
    >>> actions, logprobs = policy.sample(states)
    >>> rewards = np.abs(actions).ravel() * -1
    >>> policy_loss = reinforce_update(policy, states, actions, logprobs, rewards, lr=1e-2)
    >>> isinstance(policy_loss, float)
    True
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class GaussianPolicy:
    """상태 s → (μ(s), σ) 로 행동 샘플링."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 64,
        log_std_init: float = -0.5,
        seed: int | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.log_std_init = log_std_init
        self.seed = seed

        import torch
        import torch.nn as nn

        if seed is not None:
            torch.manual_seed(seed)
        self.mu_net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
        self.log_std = nn.Parameter(log_std_init * torch.ones(action_dim))
        self._optim: Any = None

    def parameters(self) -> Any:
        yield from self.mu_net.parameters()
        yield self.log_std

    def sample(
        self, states: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """행동 샘플 + 로그 확률 반환."""
        import torch

        s = torch.tensor(np.asarray(states, dtype=np.float32))
        mu = self.mu_net(s)
        std = torch.exp(self.log_std)
        eps = torch.randn_like(mu)
        a = mu + std * eps
        logprob = -0.5 * (((a - mu) / std) ** 2 + 2 * self.log_std + np.log(2 * np.pi))
        logprob = logprob.sum(dim=-1)
        return a.detach().cpu().numpy(), logprob.detach().cpu().numpy()


def reinforce_update(
    policy: GaussianPolicy,
    states: NDArray[np.float64],
    actions: NDArray[np.float64],
    logprobs_sampled: NDArray[np.float64],
    rewards: NDArray[np.float64],
    lr: float = 1e-3,
    baseline: bool = True,
) -> float:
    """REINFORCE loss = - Σ (G_t - b) · log π(a|s).

    시간할인 없이 whitened reward 사용.
    """
    import torch

    if policy._optim is None:
        policy._optim = torch.optim.Adam(list(policy.parameters()), lr=lr)

    s = torch.tensor(np.asarray(states, dtype=np.float32))
    a = torch.tensor(np.asarray(actions, dtype=np.float32))
    r = torch.tensor(np.asarray(rewards, dtype=np.float32))

    if baseline:
        r = r - r.mean()
        std = float(r.std()) + 1e-6
        r = r / std

    mu = policy.mu_net(s)
    std_out = torch.exp(policy.log_std)
    logp = -0.5 * (((a - mu) / std_out) ** 2 + 2 * policy.log_std + np.log(2 * np.pi))
    logp = logp.sum(dim=-1)
    loss = -(r * logp).mean()

    policy._optim.zero_grad()
    loss.backward()
    policy._optim.step()
    return float(loss.item())


__all__ = ["GaussianPolicy", "reinforce_update"]
