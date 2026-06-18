"""Notification webhook — Slack-style payload formatter.

Examples:
    >>> from naviertwin.utils.workflow.notify import slack_payload
    >>> p = slack_payload(text='Run done', channel='#ml')
    >>> p['text']
    'Run done'
"""

from __future__ import annotations

from typing import Any


def slack_payload(*, text: str, channel: str = "", username: str = "naviertwin") -> dict:
    p: dict[str, Any] = {"text": text, "username": username}
    if channel:
        p["channel"] = channel
    return p


def email_payload(*, subject: str, body: str, to: list[str]) -> dict:
    return {"subject": subject, "body": body, "to": list(to)}


__all__ = ["email_payload", "slack_payload"]
