"""Round 420 — P category milestone: train acceleration utils."""

from __future__ import annotations

import numpy as np


class TestMilestoneP:
    def test_imports(self) -> None:
        from naviertwin.utils import (  # noqa: F401
            activation_offload,
            distillation,
            grad_checkpoint,
            mixed_precision,
            pipeline_parallel,
            pruning,
            quantize,
            tensor_parallel,
            zero_shard,
        )

    def test_quantize_then_prune(self) -> None:
        from naviertwin.utils.pruning import prune_magnitude
        from naviertwin.utils.quantize import dequantize_int8, quantize_int8

        rng = np.random.default_rng(0)
        w = rng.standard_normal(100)
        wp = prune_magnitude(w, sparsity=0.5)
        q, s = quantize_int8(wp)
        wr = dequantize_int8(q, s)
        # half values zero, others approximately preserved
        assert (wr == 0).sum() >= 40
