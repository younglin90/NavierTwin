"""Round 445 — secret redaction."""

from __future__ import annotations


class TestRedact:
    def test_token(self) -> None:
        from naviertwin.utils.secret_redact import redact

        assert "abcdef" not in redact("token=abcdef123")
        assert "REDACTED" in redact("token=abcdef123")

    def test_bearer(self) -> None:
        from naviertwin.utils.secret_redact import redact

        out = redact("Authorization: Bearer abc.def.ghi")
        assert "abc.def.ghi" not in out

    def test_clean_passes_through(self) -> None:
        from naviertwin.utils.secret_redact import redact

        assert redact("hello world") == "hello world"

    def test_env_assignment_token(self) -> None:
        from naviertwin.utils.secret_redact import redact

        out = redact("NAVIER_TWIN_TEST_TOKEN=secret123")
        assert "secret123" not in out
        assert "REDACTED" in out

    def test_redact_object_sensitive_key(self) -> None:
        from naviertwin.utils.secret_redact import redact_object

        payload = {
            "auth_token": "secret123",
            "nested": {"password": "pw-123"},
            "note": "ok",
        }
        redacted = redact_object(payload)
        assert redacted["auth_token"] == "***REDACTED***"
        assert redacted["nested"]["password"] == "***REDACTED***"
        assert redacted["note"] == "ok"
