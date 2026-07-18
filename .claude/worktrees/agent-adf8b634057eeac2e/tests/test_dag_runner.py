"""Round 521 — DAG runner."""

from __future__ import annotations


class TestDAG:
    def test_chain(self) -> None:
        from naviertwin.utils.workflow.dag_runner import DAGRunner

        r = DAGRunner()
        r.add("a", lambda inputs: 1)
        r.add("b", lambda inputs: inputs["a"] + 1, deps=["a"])
        r.add("c", lambda inputs: inputs["b"] * 2, deps=["b"])
        out = r.run()
        assert out == {"a": 1, "b": 2, "c": 4}

    def test_diamond(self) -> None:
        from naviertwin.utils.workflow.dag_runner import DAGRunner

        r = DAGRunner()
        r.add("root", lambda i: 10)
        r.add("l", lambda i: i["root"] + 1, deps=["root"])
        r.add("r", lambda i: i["root"] - 1, deps=["root"])
        r.add("end", lambda i: i["l"] * i["r"], deps=["l", "r"])
        out = r.run()
        assert out["end"] == 11 * 9
