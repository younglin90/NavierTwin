"""Minimal PyInstaller entry point for the Windows desktop app.

This module intentionally avoids importing :mod:`naviertwin.main`.  The CLI
module contains many command handlers whose optional dependencies are not
needed just to start the packaged GUI, and PyInstaller would otherwise pull
those packages into the installer.
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

from naviertwin import __version__


def _setup_qt_runtime_defaults() -> None:
    """Set Qt/VTK platform defaults before QApplication is created."""
    is_wsl = "WSL_DISTRO_NAME" in os.environ or Path("/usr/lib/wsl").exists()
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if is_wsl and has_display and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("QT_X11_NO_MITSHM", "1")


def main() -> int:
    """Run the NavierTwin desktop GUI."""
    _setup_qt_runtime_defaults()

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("NavierTwin")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("NavierTwin")

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    except (ValueError, OSError):
        pass

    sigint_pump = QTimer()
    sigint_pump.setInterval(500)
    sigint_pump.timeout.connect(lambda: None)
    sigint_pump.start()
    app._naviertwin_sigint_pump = sigint_pump  # type: ignore[attr-defined]

    from naviertwin.gui.main_window import MainWindow
    from naviertwin.gui.utils.combo_fix import apply_to_widget_tree, install_combo_close_filter

    install_combo_close_filter(app)
    window = MainWindow(config_path=None)
    apply_to_widget_tree(window)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
