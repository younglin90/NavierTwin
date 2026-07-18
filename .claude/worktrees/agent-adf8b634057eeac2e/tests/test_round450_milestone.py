"""Round 450 — S category milestone: security stack (R441-R449) e2e."""

from __future__ import annotations


class TestMilestoneS:
    def test_imports(self) -> None:
        from naviertwin.utils import (  # noqa: F401
            backoff,
            csrf,
            hash_chain,
            jwt_lite,
            path_sanitize,
            rbac,
            secret_redact,
            signed_config,
            timing_safe,
        )

    def test_security_e2e(self) -> None:
        """Sign config → JWT for user → RBAC permission check."""
        from naviertwin.utils.jwt_lite import decode_jwt, encode_jwt
        from naviertwin.utils.rbac import RBAC
        from naviertwin.utils.signed_config import sign_config, verify_config

        cfg = {"epochs": 5}
        sig = sign_config(cfg, key="server")
        assert verify_config(cfg, sig, key="server")

        tok = encode_jwt({"sub": "alice", "role": "admin"}, secret="srv")
        claims = decode_jwt(tok, secret="srv")
        rbac = RBAC()
        rbac.add_role("admin", {"deploy"})
        rbac.assign(claims["sub"], claims["role"])
        assert rbac.can("alice", "deploy")
