"""Mutation-test smoke — apply a known mutation to a function and verify a
caller-supplied checker rejects the mutated output.

Examples:
    >>> from naviertwin.utils.mutation_smoke import mutate_op
    >>> def add(a, b): return a + b
    >>> mutated = mutate_op(add, op='+ → -')
    >>> mutated(2, 3)
    -1
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

_MUT = {
    "+ → -": (operator.add, operator.sub),
    "* → /": (operator.mul, operator.truediv),
    "< → >": (operator.lt, operator.gt),
}


def mutate_op(fn: Callable, *, op: str) -> Callable:
    """Returns a function that swaps the first matching operator usage.

    Strategy: rebuild fn with replaced binary operator using closure.
    Limitation: only handles single-line `lambda` or two-arg `def f(a,b)`.
    """
    if op not in _MUT:
        raise ValueError(f"unknown mutation {op}")
    orig, replacement = _MUT[op]

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        # Naive: assume fn is "a OP b"; just call replacement directly.
        return replacement(args[0], args[1])

    wrapped.__name__ = f"mutated_{fn.__name__}_{op}"
    return wrapped


def assert_kills_mutant(
    fn_under_test: Callable, mutated_fn: Callable,
    test_inputs: list[tuple],
) -> bool:
    """Returns True if at least one input produces different output → mutant killed."""
    idx = 0
    while idx < len(test_inputs):
        args = test_inputs[idx]
        if fn_under_test(*args) != mutated_fn(*args):
            return True
        idx += 1
    return False


__all__ = ["assert_kills_mutant", "mutate_op"]
