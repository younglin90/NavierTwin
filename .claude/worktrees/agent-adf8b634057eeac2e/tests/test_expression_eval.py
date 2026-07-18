"""Round 616 — safe expression evaluator (CFD-Post-style custom expressions)."""

from __future__ import annotations

import numpy as np
import pytest


class TestSafeEval:
    def test_basic_arithmetic(self) -> None:
        from naviertwin.core.flow_analysis.expression_eval import safe_eval

        result = safe_eval("2 + 3 * 4", {})
        assert result == 14

    def test_with_arrays(self) -> None:
        from naviertwin.core.flow_analysis.expression_eval import safe_eval

        u = np.array([1.0, 2.0, 3.0])
        v = np.array([0.0, 1.0, 0.0])
        result = safe_eval("sqrt(u**2 + v**2)", {"u": u, "v": v})
        np.testing.assert_allclose(result, [1.0, np.sqrt(5), 3.0])

    def test_constants_pi(self) -> None:
        from naviertwin.core.flow_analysis.expression_eval import safe_eval

        result = safe_eval("2 * pi", {})
        np.testing.assert_allclose(result, 2 * np.pi)

    def test_unary_minus(self) -> None:
        from naviertwin.core.flow_analysis.expression_eval import safe_eval

        u = np.array([1.0, -2.0])
        result = safe_eval("-u", {"u": u})
        np.testing.assert_allclose(result, [-1.0, 2.0])

    def test_comparison(self) -> None:
        from naviertwin.core.flow_analysis.expression_eval import safe_eval

        u = np.array([1.0, 2.0, 3.0])
        result = safe_eval("u > 1.5", {"u": u})
        np.testing.assert_array_equal(result, [False, True, True])

    def test_chained_comparison(self) -> None:
        from naviertwin.core.flow_analysis.expression_eval import safe_eval

        u = np.array([1.0, 2.0, 3.0])
        result = safe_eval("0 < u", {"u": u})
        np.testing.assert_array_equal(result, [True, True, True])

    def test_bool_and_or(self) -> None:
        from naviertwin.core.flow_analysis.expression_eval import safe_eval

        a = np.array([True, True, False])
        b = np.array([True, False, False])
        result_and = safe_eval("a & b", {"a": a, "b": b})
        np.testing.assert_array_equal(result_and, [True, False, False])
        result_or = safe_eval("a | b", {"a": a, "b": b})
        np.testing.assert_array_equal(result_or, [True, True, False])

    def test_function_calls(self) -> None:
        from naviertwin.core.flow_analysis.expression_eval import safe_eval

        u = np.array([1.0, 4.0, 9.0])
        np.testing.assert_allclose(safe_eval("sqrt(u)", {"u": u}), [1, 2, 3])
        np.testing.assert_allclose(safe_eval("abs(-u)", {"u": u}), [1, 4, 9])
        np.testing.assert_allclose(safe_eval("mean(u)", {"u": u}), 14 / 3)
        np.testing.assert_allclose(safe_eval("std(u)", {"u": u}), np.std(u))

    def test_norm(self) -> None:
        from naviertwin.core.flow_analysis.expression_eval import safe_eval

        v = np.array([3.0, 4.0])
        result = safe_eval("norm(v)", {"v": v})
        np.testing.assert_allclose(result, 5.0)

    def test_where(self) -> None:
        from naviertwin.core.flow_analysis.expression_eval import safe_eval

        u = np.array([-1.0, 2.0, -3.0, 4.0])
        result = safe_eval("where(u > 0, u, 0.0)", {"u": u})
        np.testing.assert_allclose(result, [0, 2, 0, 4])

    def test_ternary(self) -> None:
        from naviertwin.core.flow_analysis.expression_eval import safe_eval

        u = np.array([1.0, 2.0, 3.0])
        result = safe_eval("u * 2 if mean(u) > 1 else u", {"u": u})
        np.testing.assert_allclose(result, [2, 4, 6])

    def test_unknown_name_raises(self) -> None:
        from naviertwin.core.flow_analysis.expression_eval import (
            ExpressionError,
            safe_eval,
        )

        with pytest.raises(ExpressionError, match="unknown name"):
            safe_eval("missing + 1", {})

    def test_unknown_function_raises(self) -> None:
        from naviertwin.core.flow_analysis.expression_eval import (
            ExpressionError,
            safe_eval,
        )

        with pytest.raises(ExpressionError, match="unknown function"):
            safe_eval("foobar(1)", {})

    def test_attribute_access_blocked(self) -> None:
        from naviertwin.core.flow_analysis.expression_eval import (
            ExpressionError,
            safe_eval,
        )

        with pytest.raises(ExpressionError):
            safe_eval("u.sum()", {"u": np.zeros(3)})

    def test_subscript_blocked(self) -> None:
        from naviertwin.core.flow_analysis.expression_eval import (
            ExpressionError,
            safe_eval,
        )

        # subscript는 unsupported AST node
        with pytest.raises(ExpressionError):
            safe_eval("u[0]", {"u": np.zeros(3)})

    def test_invalid_input_raises(self) -> None:
        from naviertwin.core.flow_analysis.expression_eval import (
            ExpressionError,
            safe_eval,
        )

        with pytest.raises(ExpressionError, match="string"):
            safe_eval(123, {})  # type: ignore[arg-type]

    def test_syntax_error_raises(self) -> None:
        from naviertwin.core.flow_analysis.expression_eval import (
            ExpressionError,
            safe_eval,
        )

        with pytest.raises(ExpressionError, match="syntax"):
            safe_eval("2 + ", {})

    def test_clip(self) -> None:
        from naviertwin.core.flow_analysis.expression_eval import safe_eval

        u = np.array([-2.0, 0.0, 2.0, 5.0])
        result = safe_eval("clip(u, 0, 3)", {"u": u})
        np.testing.assert_allclose(result, [0, 0, 2, 3])

    def test_kwargs_supported(self) -> None:
        from naviertwin.core.flow_analysis.expression_eval import safe_eval

        u = np.array([[1.0, 2.0], [3.0, 4.0]])
        # axis kwarg
        result = safe_eval("mean(u, axis=0)", {"u": u})
        np.testing.assert_allclose(result, [2.0, 3.0])


class TestListAPI:
    def test_list_funcs(self) -> None:
        from naviertwin.core.flow_analysis.expression_eval import (
            list_supported_functions,
        )

        funcs = list_supported_functions()
        assert "sqrt" in funcs
        assert "norm" in funcs
        assert isinstance(funcs, list)

    def test_list_consts(self) -> None:
        from naviertwin.core.flow_analysis.expression_eval import (
            list_supported_constants,
        )

        consts = list_supported_constants()
        assert "pi" in consts
        assert "e" in consts
