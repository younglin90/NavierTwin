"""Round 647 — batch surrogate evaluation utilities."""

from __future__ import annotations

import numpy as np
import pytest


class TestBatchPredict:
    def test_basic_scalar_output(self) -> None:
        from naviertwin.core.surrogate.batch_evaluation import batch_predict

        def f(X):
            return np.sum(X ** 2, axis=1)

        rng = np.random.default_rng(0)
        X = rng.standard_normal((1000, 5))
        y = batch_predict(f, X, chunk_size=100)
        # 직접 비교
        np.testing.assert_allclose(y, f(X))

    def test_vector_output(self) -> None:
        from naviertwin.core.surrogate.batch_evaluation import batch_predict

        def f(X):
            return np.column_stack([X[:, 0], X[:, 1]])

        rng = np.random.default_rng(1)
        X = rng.standard_normal((500, 3))
        y = batch_predict(f, X, chunk_size=50)
        assert y.shape == (500, 2)

    def test_progress_callback(self) -> None:
        from naviertwin.core.surrogate.batch_evaluation import batch_predict

        def f(X):
            return np.sum(X, axis=1)

        calls = []

        def cb(cur, total):
            calls.append((cur, total))

        X = np.random.default_rng(2).standard_normal((100, 3))
        batch_predict(f, X, chunk_size=25, progress_callback=cb)
        assert len(calls) == 4
        assert calls[-1] == (100, 100)

    def test_1d_input_treated_as_column(self) -> None:
        from naviertwin.core.surrogate.batch_evaluation import batch_predict

        def f(X):
            return X[:, 0] ** 2

        x = np.array([1.0, 2.0, 3.0])
        y = batch_predict(f, x, chunk_size=2)
        np.testing.assert_allclose(y, [1.0, 4.0, 9.0])

    def test_output_dtype(self) -> None:
        from naviertwin.core.surrogate.batch_evaluation import batch_predict

        def f(X):
            return np.sum(X, axis=1)

        X = np.random.default_rng(3).standard_normal((50, 2))
        y = batch_predict(f, X, chunk_size=10, output_dtype=np.float32)
        assert y.dtype == np.float32

    def test_invalid_chunk_size(self) -> None:
        from naviertwin.core.surrogate.batch_evaluation import batch_predict

        with pytest.raises(ValueError, match="chunk_size"):
            batch_predict(lambda X: X[:, 0], np.zeros((10, 2)), chunk_size=0)

    def test_empty_X_raises(self) -> None:
        from naviertwin.core.surrogate.batch_evaluation import batch_predict

        with pytest.raises(ValueError, match="empty"):
            batch_predict(lambda X: X[:, 0], np.zeros((0, 2)), chunk_size=10)


class TestBatchPredictUncertainty:
    def test_returns_two_arrays(self) -> None:
        from naviertwin.core.surrogate.batch_evaluation import (
            batch_predict_with_uncertainty,
        )

        def mean_fn(X):
            return np.sum(X, axis=1)

        def std_fn(X):
            return np.std(X, axis=1)

        rng = np.random.default_rng(4)
        X = rng.standard_normal((200, 3))
        m, s = batch_predict_with_uncertainty(mean_fn, std_fn, X, chunk_size=50)
        assert m.shape == s.shape == (200,)


class TestBatchPredictSafe:
    def test_no_failure(self) -> None:
        from naviertwin.core.surrogate.batch_evaluation import batch_predict_safe

        def f(X):
            return np.sum(X, axis=1)

        X = np.random.default_rng(5).standard_normal((100, 3))
        y, success = batch_predict_safe(f, X, chunk_size=20)
        assert success.all()
        np.testing.assert_allclose(y, f(X))

    def test_partial_failure(self) -> None:
        from naviertwin.core.surrogate.batch_evaluation import batch_predict_safe

        # 첫 chunk만 성공, 나머지 실패
        call_count = [0]

        def f(X):
            call_count[0] += 1
            if call_count[0] > 1:
                raise RuntimeError("simulated")
            return np.sum(X, axis=1)

        X = np.random.default_rng(6).standard_normal((100, 3))
        y, success = batch_predict_safe(f, X, chunk_size=20, fallback_value=-99.0)
        # 첫 20개만 성공
        assert success[:20].all()
        assert not success[20:].any()
        assert (y[20:] == -99.0).all()

    def test_total_failure_returns_fallback(self) -> None:
        from naviertwin.core.surrogate.batch_evaluation import batch_predict_safe

        def f(X):
            raise RuntimeError("always fails")

        X = np.random.default_rng(7).standard_normal((50, 3))
        y, success = batch_predict_safe(f, X, chunk_size=10, fallback_value=0.0)
        assert not success.any()
        assert (y == 0.0).all()


class TestSplitIntoChunks:
    def test_exact_division(self) -> None:
        from naviertwin.core.surrogate.batch_evaluation import split_into_chunks

        X = np.zeros((100, 3))
        chunks = split_into_chunks(X, chunk_size=20)
        assert chunks == [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]

    def test_remainder(self) -> None:
        from naviertwin.core.surrogate.batch_evaluation import split_into_chunks

        X = np.zeros((10, 2))
        chunks = split_into_chunks(X, chunk_size=3)
        assert chunks == [(0, 3), (3, 6), (6, 9), (9, 10)]

    def test_chunk_larger_than_X(self) -> None:
        from naviertwin.core.surrogate.batch_evaluation import split_into_chunks

        X = np.zeros((5, 2))
        chunks = split_into_chunks(X, chunk_size=100)
        assert chunks == [(0, 5)]

    def test_invalid_chunk_size(self) -> None:
        from naviertwin.core.surrogate.batch_evaluation import split_into_chunks

        with pytest.raises(ValueError, match="chunk_size"):
            split_into_chunks(np.zeros(10), chunk_size=-1)
