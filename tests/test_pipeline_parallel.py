"""Round 414 — pipeline parallel."""

from __future__ import annotations


class TestPipeline:
    def test_simple_schedule(self) -> None:
        from naviertwin.utils.pipeline_parallel import gpipe_schedule

        sched = gpipe_schedule(n_stages=2, n_microbatch=2)
        # all (stage, microbatch) pairs covered
        assert (0, 0) in sched
        assert (1, 0) in sched
        assert (0, 1) in sched
        assert (1, 1) in sched

    def test_total_count(self) -> None:
        from naviertwin.utils.pipeline_parallel import gpipe_schedule

        sched = gpipe_schedule(n_stages=4, n_microbatch=3)
        assert len(sched) == 4 * 3
