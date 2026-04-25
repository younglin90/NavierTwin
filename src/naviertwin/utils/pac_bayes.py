"""PAC-Bayes margin bound — McAllester, simplified.

bound = train_loss + sqrt((KL + log(n/δ)) / (2(n-1))).

Examples:
    >>> from naviertwin.utils.pac_bayes import pac_bayes_bound
    >>> pac_bayes_bound(train_loss=0.05, KL=10.0, n=10000, delta=0.05) > 0.05
    True
"""

from __future__ import annotations

import math


def pac_bayes_bound(
    *, train_loss: float, KL: float, n: int, delta: float = 0.05,
) -> float:
    return float(train_loss + math.sqrt((KL + math.log(n / delta)) / (2 * (n - 1))))


__all__ = ["pac_bayes_bound"]
