"""Round 537 — FLOPs counter."""

from __future__ import annotations


class TestFLOPs:
    def test_linear(self) -> None:
        from naviertwin.utils.flops_counter import linear_flops

        assert linear_flops(in_dim=10, out_dim=5, batch=1) == 100

    def test_conv(self) -> None:
        from naviertwin.utils.flops_counter import conv2d_flops

        assert conv2d_flops(
            in_ch=3, out_ch=4, kernel=3, H_out=2, W_out=2, batch=1,
        ) == 2 * 3 * 4 * 9 * 4

    def test_attention(self) -> None:
        from naviertwin.utils.flops_counter import attention_flops

        f = attention_flops(batch=1, seq=8, d_model=16, n_heads=4)
        assert f > 0
