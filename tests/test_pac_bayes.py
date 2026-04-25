"""Round 494 — PAC-Bayes."""

from __future__ import annotations


class TestPB:
    def test_positive_above_loss(self) -> None:
        from naviertwin.utils.pac_bayes import pac_bayes_bound

        b = pac_bayes_bound(train_loss=0.1, KL=5.0, n=1000, delta=0.05)
        assert b > 0.1

    def test_more_data_tightens(self) -> None:
        from naviertwin.utils.pac_bayes import pac_bayes_bound

        b1 = pac_bayes_bound(train_loss=0.1, KL=5.0, n=100, delta=0.05)
        b2 = pac_bayes_bound(train_loss=0.1, KL=5.0, n=10000, delta=0.05)
        assert b2 < b1
