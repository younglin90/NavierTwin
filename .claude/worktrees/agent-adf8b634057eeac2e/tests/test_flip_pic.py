"""Round 474 — FLIP/PIC."""

from __future__ import annotations


class TestFLIPPIC:
    def test_pure_pic(self) -> None:
        from naviertwin.core.meshless.flip_pic import flip_pic_blend

        # alpha=1 → PIC = v_new
        assert flip_pic_blend(v_p=99, v_old=0, v_new=5.0, alpha=1.0) == 5.0

    def test_pure_flip(self) -> None:
        from naviertwin.core.meshless.flip_pic import flip_pic_blend

        # alpha=0 → FLIP: v_p + delta
        assert flip_pic_blend(v_p=1.0, v_old=2.0, v_new=4.0, alpha=0.0) == 3.0
