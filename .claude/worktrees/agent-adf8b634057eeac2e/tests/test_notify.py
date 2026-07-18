"""Round 527 — notify."""

from __future__ import annotations


class TestNotify:
    def test_slack(self) -> None:
        from naviertwin.utils.workflow.notify import slack_payload

        p = slack_payload(text="hello", channel="#ml")
        assert p["text"] == "hello"
        assert p["channel"] == "#ml"

    def test_email(self) -> None:
        from naviertwin.utils.workflow.notify import email_payload

        e = email_payload(subject="run", body="done", to=["a@b.com"])
        assert e["to"] == ["a@b.com"]
