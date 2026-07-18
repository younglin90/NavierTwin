"""Round 540 — BB category milestone: edge/mobile (R531-R539) e2e."""

from __future__ import annotations


class TestMilestoneBB:
    def test_imports(self) -> None:
        from naviertwin.core.export import (  # noqa: F401
            coreml_stub,
            onnx_wrap,
            tflite_stub,
            torchscript_wrap,
        )
        from naviertwin.utils import (  # noqa: F401
            flops_counter,
            latency_profiler,
            memory_budget,
            neon_dot,
            power_throttle,
        )

    def test_flops_memory_report(self) -> None:
        from naviertwin.utils.flops_counter import linear_flops
        from naviertwin.utils.memory_budget import estimate_memory

        flops = linear_flops(in_dim=128, out_dim=64, batch=32)
        mem = estimate_memory(n_params=128 * 64, batch=32, seq_len=1, hidden=64,
                                bytes_per_param=4, act_factor=1.0)
        assert flops > 0
        assert mem["total"] > 0
