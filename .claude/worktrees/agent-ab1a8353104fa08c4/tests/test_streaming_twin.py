"""Round 27 — StreamingDigitalTwin 테스트."""

from __future__ import annotations

import numpy as np
import pytest


class TestStreamingDigitalTwin:
    def test_converges_to_truth(self) -> None:
        from naviertwin.core.digital_twin.streaming_twin import StreamingDigitalTwin

        rng = np.random.default_rng(0)
        A = np.eye(2)  # 정지 시스템 → 관측에 수렴해야
        H = np.eye(2)
        R = 0.01 * np.eye(2)
        true = np.array([1.0, -0.5])

        twin = StreamingDigitalTwin(
            state_dim=2, n_ensemble=80,
            model_fn=lambda x: A @ x,
            H=H, R=R, process_noise=0.01, rng=rng,
        )
        twin.initialize(5.0 * rng.standard_normal((80, 2)))

        for _ in range(10):
            twin.step()
            y = true + 0.05 * rng.standard_normal(2)
            twin.assimilate(y)

        est = twin.estimate()
        assert np.linalg.norm(est - true) < 0.3
        # 불확실성도 초기보다 작아져야
        unc = twin.uncertainty()
        assert float(unc.max()) < 2.0

    def test_history_tracked(self) -> None:
        from naviertwin.core.digital_twin.streaming_twin import StreamingDigitalTwin

        rng = np.random.default_rng(0)
        twin = StreamingDigitalTwin(
            state_dim=2, n_ensemble=20,
            model_fn=lambda x: x,
            H=np.eye(2), R=np.eye(2),
            history_size=5,
            rng=rng,
        )
        twin.initialize(rng.standard_normal((20, 2)))
        for _ in range(10):
            twin.step()
        # history_size 제한
        assert len(twin.history) <= 5

    def test_wrong_ensemble_shape(self) -> None:
        from naviertwin.core.digital_twin.streaming_twin import StreamingDigitalTwin

        twin = StreamingDigitalTwin(
            state_dim=3, n_ensemble=10,
            model_fn=lambda x: x,
            H=np.eye(3), R=np.eye(3),
        )
        with pytest.raises(ValueError):
            twin.initialize(np.zeros((8, 3)))

    def test_step_before_initialize_raises(self) -> None:
        from naviertwin.core.digital_twin.streaming_twin import StreamingDigitalTwin

        twin = StreamingDigitalTwin(
            state_dim=2, n_ensemble=10,
            model_fn=lambda x: x,
            H=np.eye(2), R=np.eye(2),
        )
        with pytest.raises(RuntimeError):
            twin.step()
