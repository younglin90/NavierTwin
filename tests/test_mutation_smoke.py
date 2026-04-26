"""Round 569 — mutation smoke."""

from __future__ import annotations


class TestMutation:
    def test_kills_mutant(self) -> None:
        from naviertwin.utils.mutation_smoke import assert_kills_mutant, mutate_op

        def add(a, b):
            return a + b

        mut = mutate_op(add, op="+ → -")
        # 2-3 vs 2+3 → caught
        assert assert_kills_mutant(add, mut, [(2, 3)])

    def test_survives_when_inputs_dont_distinguish(self) -> None:
        from naviertwin.utils.mutation_smoke import assert_kills_mutant, mutate_op

        def add(a, b):
            return a + b

        mut = mutate_op(add, op="+ → -")
        # 2 + 0 == 2 - 0 → mutant survives this single input
        assert not assert_kills_mutant(add, mut, [(2, 0)])
