"""실행 이력 뷰어 다이얼로그 — RunHistory 항목을 테이블로 표시.

각 항목 더블클릭 시 panel에 op + 파라미터를 다시 로드 (replay).

Signals:
    replay_requested(int): 사용자가 인덱스 i 항목을 재실행 요청.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class HistoryDialog(QDialog):
    """RunHistory 항목 테이블 뷰어.

    Examples:
        dlg = HistoryDialog(history_entries, parent=panel)
        dlg.replay_requested.connect(panel._replay_from_history)
        dlg.exec()
    """

    replay_requested = Signal(int)

    def __init__(
        self,
        entries: list[dict[str, Any]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("실행 이력")
        self.resize(900, 500)
        self._entries = list(entries)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        info = QLabel(
            f"총 {len(self._entries)}개 실행 기록 (최신이 마지막). "
            f"항목 더블클릭 또는 '재실행' 버튼으로 복원."
        )
        layout.addWidget(info)

        # 테이블
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            ["#", "Timestamp", "Op", "Status", "Error/Summary"],
        )
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive,
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows,
        )
        self._table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection,
        )
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.itemDoubleClicked.connect(self._on_double_clicked)
        layout.addWidget(self._table, stretch=2)

        self._populate()

        # 상세 panel
        self._detail = QTextEdit()
        self._detail.setReadOnly(True)
        layout.addWidget(self._detail, stretch=1)

        # 버튼 row
        btn_row = QHBoxLayout()
        self._replay_btn = QPushButton("선택 항목 재실행")
        self._replay_btn.clicked.connect(self._on_replay_clicked)
        self._replay_btn.setEnabled(False)
        btn_row.addWidget(self._replay_btn)
        btn_row.addStretch()
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _populate(self) -> None:
        self._table.setRowCount(len(self._entries))
        i = 0
        while i < len(self._entries):
            e = self._entries[i]
            self._table.setItem(i, 0, QTableWidgetItem(str(i)))
            self._table.setItem(
                i, 1, QTableWidgetItem(str(e.get("timestamp", "-"))),
            )
            self._table.setItem(i, 2, QTableWidgetItem(str(e.get("op", "?"))))
            status = str(e.get("status", "?"))
            status_item = QTableWidgetItem(status)
            if status == "error":
                status_item.setForeground(Qt.GlobalColor.red)
            else:
                status_item.setForeground(Qt.GlobalColor.darkGreen)
            self._table.setItem(i, 3, status_item)
            err_or_summary = ""
            if status == "error":
                err_or_summary = str(e.get("error", ""))
            else:
                rs = e.get("result_summary") or {}
                if isinstance(rs, dict):
                    err_or_summary = ", ".join(list(rs.keys())[:6])
            self._table.setItem(i, 4, QTableWidgetItem(err_or_summary))
            i += 1
        self._table.resizeColumnsToContents()

    def _on_selection_changed(self) -> None:
        idx = self.selected_index()
        if idx < 0:
            self._detail.clear()
            self._replay_btn.setEnabled(False)
            return
        self._replay_btn.setEnabled(True)
        e = self._entries[idx]
        lines = [
            f"# {idx}: {e.get('op', '?')}",
            f"Time: {e.get('timestamp', '-')}",
            f"Status: {e.get('status', '?')}",
        ]
        if e.get("error"):
            lines.append(f"Error: {e['error']}")
        kw = e.get("kwargs_summary") or {}
        if kw:
            lines.append("\nKwargs:")
            kw_items = tuple(kw.items())
            kw_idx = 0
            while kw_idx < len(kw_items):
                k, v = kw_items[kw_idx]
                lines.append(f"  {k}: {v}")
                kw_idx += 1
        rs = e.get("result_summary") or {}
        if rs:
            lines.append("\nResult:")
            rs_items = tuple(rs.items())
            rs_idx = 0
            while rs_idx < len(rs_items):
                k, v = rs_items[rs_idx]
                lines.append(f"  {k}: {v}")
                rs_idx += 1
        self._detail.setPlainText("\n".join(lines))

    def _on_double_clicked(self, _item: QTableWidgetItem) -> None:
        idx = self.selected_index()
        if idx >= 0:
            self.replay_requested.emit(idx)

    def _on_replay_clicked(self) -> None:
        idx = self.selected_index()
        if idx >= 0:
            self.replay_requested.emit(idx)

    def selected_index(self) -> int:
        rows = self._table.selectionModel().selectedRows()
        return rows[0].row() if rows else -1


__all__ = ["HistoryDialog"]
