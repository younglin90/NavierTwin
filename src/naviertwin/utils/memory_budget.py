"""Memory budget estimator — model + activations + optimizer state.

Examples:
    >>> from naviertwin.utils.memory_budget import estimate_memory
    >>> r = estimate_memory(n_params=1e6, batch=32, seq_len=128, hidden=64,
    ...                     bytes_per_param=4, act_factor=2.0, opt_states=2)
"""

from __future__ import annotations


def estimate_memory(
    *, n_params: float, batch: int, seq_len: int, hidden: int,
    bytes_per_param: int = 4, act_factor: float = 2.0,
    opt_states: int = 2,
) -> dict[str, float]:
    """Returns bytes per category."""
    params_bytes = n_params * bytes_per_param
    grads_bytes = params_bytes
    opt_bytes = params_bytes * opt_states
    act_bytes = batch * seq_len * hidden * bytes_per_param * act_factor
    total = params_bytes + grads_bytes + opt_bytes + act_bytes
    return {
        "params": float(params_bytes),
        "grads": float(grads_bytes),
        "optimizer": float(opt_bytes),
        "activations": float(act_bytes),
        "total": float(total),
    }


__all__ = ["estimate_memory"]
