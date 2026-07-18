"""Undo/Redo Command 스택 — GUI 공통 유틸.

Qt 에 의존하지 않는 순수 파이썬 구현. GUI 패널에서 do/undo 콜백 쌍을
등록하면 스택 기반 되돌리기/재실행을 제공한다.

Examples:
    >>> from naviertwin.utils.undo_redo import Command, UndoRedoStack
    >>> stack = UndoRedoStack()
    >>> state = [0]
    >>> def do(): state[0] += 1
    >>> def undo(): state[0] -= 1
    >>> stack.execute(Command(do=do, undo=undo, label="increment"))
    >>> state
    [1]
    >>> stack.undo(); state
    [0]
    >>> stack.redo(); state
    [1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Command:
    """do/undo 쌍 + 레이블 한 단위."""

    do: Callable[[], None]
    undo: Callable[[], None]
    label: str = ""

    def execute(self) -> None:
        self.do()

    def revert(self) -> None:
        self.undo()


class UndoRedoStack:
    """LIFO Command 스택."""

    def __init__(self, max_size: int = 100) -> None:
        self.max_size = max_size
        self._undo: list[Command] = []
        self._redo: list[Command] = []

    def execute(self, cmd: Command) -> None:
        """명령을 실행하고 undo 스택에 push, redo 스택 clear."""
        cmd.execute()
        self._undo.append(cmd)
        self._redo.clear()
        if len(self._undo) > self.max_size:
            self._undo.pop(0)

    def undo(self) -> Command | None:
        if not self._undo:
            return None
        cmd = self._undo.pop()
        cmd.revert()
        self._redo.append(cmd)
        return cmd

    def redo(self) -> Command | None:
        if not self._redo:
            return None
        cmd = self._redo.pop()
        cmd.execute()
        self._undo.append(cmd)
        return cmd

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()

    def __len__(self) -> int:
        return len(self._undo)


__all__ = ["Command", "UndoRedoStack"]
