"""Round 578 — coverage uplift for state_space.base (was 0%)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


class _StubSSM:
    """Local stub since BaseSSM is abstract."""


def _make_concrete():
    from naviertwin.core.state_space.base import BaseSSM

    class ConcreteSSM(BaseSSM):
        def fit(self, dataset: dict[str, Any]) -> None:
            self.state_dim = dataset["sequences"].shape[-1]
            self.seq_len = dataset["sequences"].shape[1]
            self.is_fitted = True

        def predict(self, inputs: dict[str, Any]) -> np.ndarray:
            self._check_fitted()
            x0 = np.asarray(inputs["initial_state"])
            n = inputs.get("n_steps", self.seq_len)
            return np.tile(x0, (n, 1))

    return ConcreteSSM


class TestBaseSSM:
    def test_repr_states(self) -> None:
        Cls = _make_concrete()
        m = Cls(device="cpu")
        s = repr(m)
        assert "not fitted" in s
        assert "cpu" in s
        m.fit({"sequences": np.zeros((1, 5, 3))})
        assert "fitted" in repr(m)
        assert m.state_dim == 3
        assert m.seq_len == 5

    def test_check_fitted_raises(self) -> None:
        Cls = _make_concrete()
        m = Cls()
        with pytest.raises(RuntimeError, match="fit"):
            m.predict({"initial_state": np.array([1.0, 2.0, 3.0])})

    def test_predict_after_fit(self) -> None:
        Cls = _make_concrete()
        m = Cls()
        m.fit({"sequences": np.zeros((2, 4, 2))})
        out = m.predict({"initial_state": np.array([1.0, 2.0]), "n_steps": 3})
        assert out.shape == (3, 2)
        assert (out == [1.0, 2.0]).all()

    def test_cannot_instantiate_abstract(self) -> None:
        from naviertwin.core.state_space.base import BaseSSM

        with pytest.raises(TypeError):
            BaseSSM()  # abstract
        _ = _StubSSM  # silence
