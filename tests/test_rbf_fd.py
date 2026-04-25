"""Round 476 — RBF-FD weights."""

from __future__ import annotations

import numpy as np


class TestRBFFD:
    def test_first_deriv(self) -> None:
        from naviertwin.core.meshless.rbf_fd import rbf_fd_weights_1d

        s = np.array([-1.0, 0.0, 1.0])
        w = rbf_fd_weights_1d(s, eps=0.1, order=1)
        # Apply to f=x → derivative=1
        assert abs(np.dot(w, s) - 1.0) < 1e-3
        # Apply to f=1 (constant) → derivative=0
        assert abs(np.dot(w, np.ones(3))) < 1e-6

    def test_second_deriv(self) -> None:
        from naviertwin.core.meshless.rbf_fd import rbf_fd_weights_1d

        s = np.array([-1.0, 0.0, 1.0])
        w = rbf_fd_weights_1d(s, eps=0.1, order=2)
        # Apply to f=x² → derivative=2
        assert abs(np.dot(w, s ** 2) - 2.0) < 0.5
