"""Round 449 — JWT-lite."""

from __future__ import annotations

import pytest


class TestJWT:
    def test_round_trip(self) -> None:
        from naviertwin.utils.jwt_lite import decode_jwt, encode_jwt

        claims = {"sub": "alice", "exp": 9999}
        tok = encode_jwt(claims, secret="key")
        assert decode_jwt(tok, secret="key") == claims

    def test_tamper_payload(self) -> None:
        from naviertwin.utils.jwt_lite import decode_jwt, encode_jwt

        tok = encode_jwt({"sub": "alice"}, secret="k")
        h, p, s = tok.split(".")
        tampered = f"{h}.eyJzdWIiOiJib2IifQ.{s}"
        with pytest.raises(ValueError):
            decode_jwt(tampered, secret="k")
