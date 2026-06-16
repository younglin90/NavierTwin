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


def _attach_visible_console(title: str) -> bool:
    """Windows GUI 모드 (console=False) 에서도 콘솔 창을 띄워 stdout/stderr 가
    실시간으로 보이게 한다.

    Returns:
        True if a visible console is attached (parent or newly allocated).
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes  # noqa: PLC0415

        kernel32 = ctypes.windll.kernel32
        ATTACH_PARENT_PROCESS = -1
        attached = bool(kernel32.AttachConsole(ATTACH_PARENT_PROCESS))
        if not attached:
            attached = bool(kernel32.AllocConsole())
        if not attached:
            return False
        try:
            kernel32.SetConsoleTitleW(title)
        except Exception:
            pass
        # stdout/stderr/stdin 을 새 콘솔 핸들로 리바인드.
        import os  # noqa: PLC0415

        for fd_name, mode, flag in (
            ("stdin", "r", os.O_RDONLY),
            ("stdout", "w", os.O_WRONLY),
            ("stderr", "w", os.O_WRONLY),
        ):
            try:
                handle = open("CONOUT$" if fd_name != "stdin" else "CONIN$", mode, buffering=1)
                setattr(sys, fd_name, handle)
            except Exception:
                pass
        # UTF-8 출력 (한국어 메시지 보존).
        try:
            kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass
        return True
    except Exception:
        return False


def _print_banner(text: str) -> None:
    """진행 메시지 — 콘솔이 있을 때만 출력 (없으면 noop)."""
    try:
        print(text, flush=True)
    except Exception:
        pass


def _run_feature_pack_install_mode(argv: list[str]) -> int | None:
    """Handle installer-invoked online optional feature installation."""
    if len(argv) < 2 or argv[1] != "--install-feature-pack":
        return None

    import argparse
    import json
    import time

    parser = argparse.ArgumentParser(prog="NavierTwin.exe --install-feature-pack")
    parser.add_argument("--install-feature-pack", required=True)
    parser.add_argument("--feature-pack-root", default=None)
    parser.add_argument("--feature-pack-log", default=None)
    parser.add_argument("--json", action="store_true", default=False)
    parser.add_argument(
        "--no-console",
        action="store_true",
        default=False,
        help="콘솔 창을 띄우지 않음 (GUI 내부 호출용).",
    )
    args = parser.parse_args(argv[1:])

    pack_id = args.install_feature_pack
    if not args.no_console:
        _attach_visible_console(f"NavierTwin — '{pack_id}' Feature Pack 설치 중")

    _print_banner(
        "\n" + "=" * 70 + "\n"
        f"NavierTwin Feature Pack 설치 → '{pack_id}'\n"
        + "=" * 70
    )
    _print_banner(f"진행 로그: {args.feature_pack_log or '(없음)'}")
    _print_banner("PyPI 에서 의존성을 다운로드합니다. 인터넷 속도에 따라 수 분 소요됩니다.\n")
    start_ts = time.time()

    from naviertwin.utils.feature_packs import (
        get_feature_pack_spec,
        install_feature_pack_online,
    )

    try:
        spec = get_feature_pack_spec(pack_id)
        _print_banner(f"[1/3] 패키지 목록: {', '.join(spec.packages)}")
        if spec.extra_index_urls:
            _print_banner(f"      추가 인덱스: {', '.join(spec.extra_index_urls)}")
        _print_banner("[2/3] pip install 실행 중 — 다운로드 진행률이 아래에 표시됩니다 ...\n")

        status = install_feature_pack_online(
            pack_id,
            root=Path(args.feature_pack_root) if args.feature_pack_root else None,
            log_file=Path(args.feature_pack_log) if args.feature_pack_log else None,
        )
        elapsed = time.time() - start_ts
        _print_banner(
            f"\n[3/3] 완료 — '{pack_id}' 설치 성공 ({elapsed:.0f}초)"
        )
        _print_banner(f"      설치 경로: {status.get('path')}")
        if status.get("missing_modules"):
            _print_banner(
                f"      ⚠ 일부 모듈 미감지: {status['missing_modules']}"
            )
        else:
            _print_banner("      모든 모듈 import 가능 상태입니다.")
    except Exception as exc:  # noqa: BLE001 - this runs unattended from Setup.
        elapsed = time.time() - start_ts
        _print_banner(
            f"\n❌ 실패 — '{pack_id}' 설치 중 오류 ({elapsed:.0f}초): {exc}"
        )
        _print_banner(
            "인터넷 / 프록시 / pip 인덱스 접근을 확인하세요.\n"
            "수동 재시도: NavierTwin.exe --install-feature-pack " + pack_id
        )
        if args.feature_pack_log:
            log_path = Path(args.feature_pack_log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"ERROR: {exc}\n")
        # 사용자가 콘솔 메시지를 읽을 시간 확보.
        _print_banner("\n5초 후 창이 닫힙니다.")
        try:
            time.sleep(5)
        except Exception:
            pass
        return 2

    if args.json:
        print(json.dumps(status, ensure_ascii=False, sort_keys=True))
    return 0


def main() -> int:
    """Run the NavierTwin desktop GUI."""
    feature_pack_exit = _run_feature_pack_install_mode(sys.argv)
    if feature_pack_exit is not None:
        return feature_pack_exit

    _setup_qt_runtime_defaults()
    from naviertwin.utils.feature_packs import activate_installed_feature_packs

    activate_installed_feature_packs()

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
