"""Mean-field VI — diagonal Gaussian q(z) = N(μ, diag σ²)."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def mean_field_vi(
    log_p: Callable[[NDArray[np.float64]], float],
    dim: int,
    *, n_iter: int = 1000, lr: float = 0.05, mc_samples: int = 16,
    seed: int | None = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """diag Gaussian q 의 μ, log σ 반환 (FD gradient + Adam ascent on ELBO)."""
    rng = np.random.default_rng(seed)
    mu = np.zeros(dim)
    log_sigma = np.zeros(dim)

    m_mu = np.zeros(dim)
    v_mu = np.zeros(dim)
    m_ls = np.zeros(dim)
    v_ls = np.zeros(dim)
    b1, b2, eps = 0.9, 0.999, 1e-8

    def elbo(mu_, log_s_, n_mc=mc_samples):
        sigma = np.exp(log_s_)
        eps_s = rng.standard_normal((n_mc, dim))
        z = mu_[None, :] + sigma[None, :] * eps_s
        values = np.fromiter(map(log_p, z), dtype=np.float64, count=n_mc)
        total = float(values.mean())
        entropy = float(np.sum(log_s_) + 0.5 * dim * (1 + np.log(2 * np.pi)))
        return total + entropy

    eye = np.eye(dim)
    t = 1
    while t <= n_iter:
        fd_eps = 1e-3
        e0 = elbo(mu, log_sigma)
        mu_eval = np.repeat(mu[None, :], 2 * dim, axis=0)
        ls_eval = np.repeat(log_sigma[None, :], 2 * dim, axis=0)
        mu_eval[0::2] += fd_eps * eye
        ls_eval[1::2] += fd_eps * eye
        fd_values = np.fromiter(
            map(lambda pair: elbo(pair[0], pair[1]), zip(mu_eval, ls_eval, strict=True)),
            dtype=np.float64,
            count=2 * dim,
        )
        g_mu = (fd_values[0::2] - e0) / fd_eps
        g_ls = (fd_values[1::2] - e0) / fd_eps

        m_mu = b1 * m_mu + (1 - b1) * g_mu
        v_mu = b2 * v_mu + (1 - b2) * (g_mu * g_mu)
        mh = m_mu / (1 - b1 ** t)
        vh = v_mu / (1 - b2 ** t)
        mu = mu + lr * mh / (np.sqrt(vh) + eps)

        m_ls = b1 * m_ls + (1 - b1) * g_ls
        v_ls = b2 * v_ls + (1 - b2) * (g_ls * g_ls)
        mh2 = m_ls / (1 - b1 ** t)
        vh2 = v_ls / (1 - b2 ** t)
        log_sigma = log_sigma + lr * mh2 / (np.sqrt(vh2) + eps)
        t += 1

    return mu, log_sigma


__all__ = ["mean_field_vi"]
