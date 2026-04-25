"""FLIP/PIC blend — particle velocity update with mixing parameter α.

v_p ← α (v_grid_new) + (1-α) (v_p + (v_grid_new - v_grid_old)).
α=1 → PIC (smooth), α=0 → FLIP (noisy but conserves energy).

Examples:
    >>> from naviertwin.core.meshless.flip_pic import flip_pic_blend
    >>> flip_pic_blend(v_p=1.0, v_old=1.0, v_new=2.0, alpha=0.5)
    1.5
"""

from __future__ import annotations


def flip_pic_blend(*, v_p: float, v_old: float, v_new: float, alpha: float = 0.05) -> float:
    pic = v_new
    flip = v_p + (v_new - v_old)
    return float(alpha * pic + (1 - alpha) * flip)


__all__ = ["flip_pic_blend"]
