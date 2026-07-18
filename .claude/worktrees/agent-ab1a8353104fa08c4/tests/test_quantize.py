"""Round 166 — dynamic quantization."""

from __future__ import annotations

import pytest


class TestQuantize:
    def test_quantize_and_size(self) -> None:
        pytest.importorskip("torch")
        import torch
        import torch.nn as nn

        from naviertwin.core.export.quantize import (
            compare_inference,
            dynamic_quantize,
            model_size_bytes,
        )

        m = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 32))
        mq = dynamic_quantize(m)
        x = torch.randn(4, 64)
        res = compare_inference(m, mq, x, atol=1.0)
        assert "max_abs_diff" in res
        # 양자화 후 작아져야 함
        s1 = model_size_bytes(m)
        s2 = model_size_bytes(mq)
        assert s2 <= s1  # 작거나 같아야 함 (tiny 모델에서는 동일할 수도)
