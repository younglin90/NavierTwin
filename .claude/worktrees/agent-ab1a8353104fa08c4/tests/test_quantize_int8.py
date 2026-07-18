"""Round 417 — INT8 quantize."""

from __future__ import annotations

import numpy as np


class TestQuantizeINT8:
    def test_round_trip(self) -> None:
        from naviertwin.utils.quantize import dequantize_int8, quantize_int8

        x = np.linspace(-1.0, 1.0, 256)
        q, s = quantize_int8(x)
        x_rec = dequantize_int8(q, s)
        assert np.max(np.abs(x - x_rec)) < 0.02

    def test_zero(self) -> None:
        from naviertwin.utils.quantize import quantize_int8

        q, s = quantize_int8(np.zeros(5))
        assert (q == 0).all()
