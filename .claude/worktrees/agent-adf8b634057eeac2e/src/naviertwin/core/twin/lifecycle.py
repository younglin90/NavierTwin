"""Twin lifecycle FSM — created → trained → deployed → drifting → retraining → deployed.

Examples:
    >>> from naviertwin.core.twin.lifecycle import TwinFSM
    >>> f = TwinFSM()
    >>> f.transition('train'); f.state
    'trained'
"""

from __future__ import annotations

_T = {
    "created": {"train": "trained", "archive": "archived"},
    "trained": {"deploy": "deployed", "archive": "archived"},
    "deployed": {"drift": "drifting", "archive": "archived"},
    "drifting": {"retrain": "retraining", "archive": "archived"},
    "retraining": {"deploy": "deployed", "archive": "archived"},
    "archived": {},
}


class TwinFSM:
    def __init__(self) -> None:
        self.state = "created"

    def transition(self, event: str) -> str:
        nxt = _T.get(self.state, {}).get(event)
        if nxt is None:
            raise ValueError(f"invalid event {event} from {self.state}")
        self.state = nxt
        return self.state


__all__ = ["TwinFSM"]
