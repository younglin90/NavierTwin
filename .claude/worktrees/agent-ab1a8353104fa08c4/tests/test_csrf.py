"""Round 446 — CSRF tokens."""

from __future__ import annotations


class TestCSRF:
    def test_round_trip(self) -> None:
        from naviertwin.utils.csrf import generate_token, verify_token

        t = generate_token("sess", secret="key")
        assert verify_token("sess", t, secret="key")

    def test_wrong_session(self) -> None:
        from naviertwin.utils.csrf import generate_token, verify_token

        t = generate_token("s1", secret="k")
        assert not verify_token("s2", t, secret="k")

    def test_wrong_secret(self) -> None:
        from naviertwin.utils.csrf import generate_token, verify_token

        t = generate_token("s", secret="k1")
        assert not verify_token("s", t, secret="k2")
