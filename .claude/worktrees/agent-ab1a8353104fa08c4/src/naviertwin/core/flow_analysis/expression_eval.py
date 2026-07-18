"""사용자 정의 표현식 평가기 — 안전 sandbox AST 기반.

CFD-Post의 Custom Expressions나 ParaView의 Calculator 같이, 사용자가
"sqrt(u**2 + v**2 + w**2)" 등을 입력해 새 필드를 만들 수 있게 한다.
보안: AST 화이트리스트로 안전 노드만 허용 (eval()은 금지).

지원:
    - 산술: + - * / // % **
    - 비교: == != < <= > >=
    - 논리: and or not (numpy bitwise & | ~로 매핑)
    - 함수: sqrt, exp, log, sin, cos, tan, abs, min, max,
            mean, sum, std, where, dot, norm

Examples:
    >>> import numpy as np
    >>> u = np.array([1.0, 2.0, 3.0])
    >>> v = np.array([0.0, 1.0, 0.0])
    >>> from naviertwin.core.flow_analysis.expression_eval import safe_eval
    >>> safe_eval("sqrt(u**2 + v**2)", {"u": u, "v": v}).tolist()
    [1.0, 2.23606797749979, 3.0]
"""

from __future__ import annotations

import ast
import operator as op
from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.BitAnd: op.and_,
    ast.BitOr: op.or_,
    ast.BitXor: op.xor,
}

_UNARY_OPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
    ast.Invert: op.invert,
}

_CMP_OPS = {
    ast.Eq: op.eq,
    ast.NotEq: op.ne,
    ast.Lt: op.lt,
    ast.LtE: op.le,
    ast.Gt: op.gt,
    ast.GtE: op.ge,
}

_FUNCS: dict[str, Any] = {
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "atan2": np.arctan2,
    "abs": np.abs,
    "min": np.minimum,
    "max": np.maximum,
    "mean": np.mean,
    "sum": np.sum,
    "std": np.std,
    "var": np.var,
    "where": np.where,
    "dot": np.dot,
    "norm": np.linalg.norm,
    "clip": np.clip,
    "floor": np.floor,
    "ceil": np.ceil,
    "round": np.round,
    "sign": np.sign,
    "tanh": np.tanh,
    "sinh": np.sinh,
    "cosh": np.cosh,
}


_CONSTS: dict[str, float] = {
    "pi": float(np.pi),
    "e": float(np.e),
    "True": True,  # type: ignore[dict-item]
    "False": False,  # type: ignore[dict-item]
}


class ExpressionError(ValueError):
    """표현식 파싱/평가 오류."""


def _eval_node(node: Any, vars_: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in vars_:
            return vars_[node.id]
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ExpressionError(f"unknown name: {node.id}")
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, vars_)
        right = _eval_node(node.right, vars_)
        f = _BIN_OPS.get(type(node.op))
        if f is None:
            raise ExpressionError(f"unsupported binary op: {type(node.op).__name__}")
        return f(left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, vars_)
        f = _UNARY_OPS.get(type(node.op))
        if f is None:
            raise ExpressionError(f"unsupported unary op: {type(node.op).__name__}")
        return f(operand)
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, vars_)
        result = None
        cmp_idx = 0
        while cmp_idx < len(node.ops):
            cmp_op = node.ops[cmp_idx]
            comp = node.comparators[cmp_idx]
            right = _eval_node(comp, vars_)
            f = _CMP_OPS.get(type(cmp_op))
            if f is None:
                raise ExpressionError(f"unsupported cmp op: {type(cmp_op).__name__}")
            this = f(left, right)
            result = this if result is None else result & this
            left = right
            cmp_idx += 1
        return result
    if isinstance(node, ast.BoolOp):
        values = list(map(lambda v: _eval_node(v, vars_), node.values))
        if isinstance(node.op, ast.And):
            out = values[0]
            value_idx = 1
            while value_idx < len(values):
                out = out & values[value_idx]
                value_idx += 1
            return out
        if isinstance(node.op, ast.Or):
            out = values[0]
            value_idx = 1
            while value_idx < len(values):
                out = out | values[value_idx]
                value_idx += 1
            return out
        raise ExpressionError(f"unsupported bool op: {type(node.op).__name__}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("only direct function calls allowed")
        fname = node.func.id
        if fname not in _FUNCS:
            raise ExpressionError(f"unknown function: {fname}")
        args = list(map(lambda a: _eval_node(a, vars_), node.args))
        if node.keywords:
            kwargs = dict(map(lambda kw: (kw.arg, _eval_node(kw.value, vars_)), node.keywords))
            return _FUNCS[fname](*args, **kwargs)
        return _FUNCS[fname](*args)
    if isinstance(node, ast.IfExp):
        cond = _eval_node(node.test, vars_)
        return np.where(cond, _eval_node(node.body, vars_), _eval_node(node.orelse, vars_))
    raise ExpressionError(f"unsupported AST node: {type(node).__name__}")


def safe_eval(
    expression: str,
    variables: dict[str, NDArray[np.float64] | float],
) -> Any:
    """사용자 표현식을 안전하게 평가한다.

    Args:
        expression: Python 식 문자열 (예: "sqrt(u**2 + v**2)").
        variables: 변수 이름 → numpy 배열 또는 스칼라.

    Returns:
        평가 결과 (스칼라 또는 ndarray).

    Raises:
        ExpressionError: 파싱 실패, 미지 변수, 비허용 노드.
    """
    if not isinstance(expression, str):
        raise ExpressionError("expression must be a string")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"syntax error: {e}") from e
    return _eval_node(tree.body, variables)


def list_supported_functions() -> list[str]:
    """지원하는 함수 이름 목록."""
    return sorted(_FUNCS.keys())


def list_supported_constants() -> list[str]:
    """지원하는 상수 이름 목록."""
    return sorted(_CONSTS.keys())


__all__ = [
    "safe_eval",
    "list_supported_functions",
    "list_supported_constants",
    "ExpressionError",
]
