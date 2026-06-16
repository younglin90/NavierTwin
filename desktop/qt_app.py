"""Desktop GUI entry point.

Run from the repository root:
    python3 -m desktop.qt_app
    python3 -m desktop.qt_app --config path/to/config.json
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _setup_qt_defaults() -> None:
    """Set Qt platform defaults before QApplication/QtInteractor are created."""
    is_wsl = "WSL_DISTRO_NAME" in os.environ or os.path.exists("/usr/lib/wsl")
    if is_wsl and os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("QT_X11_NO_MITSHM", "1")


_setup_qt_defaults()


_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main() -> int:
    """Run the NavierTwin desktop GUI."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="desktop.qt_app",
        description="NavierTwin Desktop (PySide6 GUI)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="설정 파일 경로 (생략 시 기본 경로)",
    )
    args = parser.parse_args()

    from naviertwin.main import _run_gui

    return _run_gui(args.config)


if __name__ == "__main__":
    sys.exit(main())
