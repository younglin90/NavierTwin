"""Round 254 — activations."""

from __future__ import annotations

import numpy as np


class TestAct:
    def test_relu(self) -> None:
        from naviertwin.core.neural.activations import relu

        assert relu(np.array([-1, 0, 1])).tolist() == [0, 0, 1]

    def test_sigmoid_range(self) -> None:
        from naviertwin.core.neural.activations import sigmoid

        y = sigmoid(np.array([-100, 0, 100]))
        assert abs(float(y[0])) < 1e-10
        assert abs(float(y[1]) - 0.5) < 1e-10
        assert abs(float(y[2]) - 1.0) < 1e-10

    def test_gelu(self) -> None:
        from naviertwin.core.neural.activations import gelu

        assert abs(float(gelu(np.zeros(1))[0])) < 1e-12
        assert gelu(np.array([10.0]))[0] > 9.5

    def test_softmax_sum(self) -> None:
        from naviertwin.core.neural.activations import softmax

        y = softmax(np.array([1.0, 2.0, 3.0]))
        assert abs(float(y.sum()) - 1.0) < 1e-12

    def test_softplus_nonneg(self) -> None:
        from naviertwin.core.neural.activations import softplus

        assert np.all(softplus(np.array([-10.0, 0.0, 10.0])) >= 0)

    def test_elu(self) -> None:
        from naviertwin.core.neural.activations import elu

        y = elu(np.array([2.0, -2.0]))
        assert y[0] == 2.0
        assert y[1] > -1.0
