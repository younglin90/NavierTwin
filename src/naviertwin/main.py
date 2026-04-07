"""NavierTwin CLI 진입점.

Examples:
    GUI 실행::

        $ naviertwin --gui

    도움말 출력::

        $ naviertwin --help
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    """CLI 인수 파서를 구성하여 반환한다.

    Returns:
        구성된 ArgumentParser 인스턴스.
    """
    parser = argparse.ArgumentParser(
        prog="naviertwin",
        description=(
            "NavierTwin — CFD 후처리 결과를 AI/ROM 디지털 트윈으로 변환하는 툴\n"
            "버전: 0.1.0"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=False,
        help="GUI 모드로 실행한다 (PySide6 필요)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="사용할 설정 파일 경로 (JSON). 기본값: ~/.naviertwin/config.json",
    )
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="로그 레벨 (기본값: INFO)",
    )
    return parser


def _run_gui(config_path: str | None) -> int:
    """PySide6 GUI 애플리케이션을 실행한다.

    Args:
        config_path: 설정 파일 경로. None이면 기본 경로를 사용한다.

    Returns:
        프로세스 종료 코드.
    """
    try:
        from PySide6.QtWidgets import QApplication  # noqa: PLC0415
    except ImportError:
        print(
            "오류: PySide6를 찾을 수 없습니다.\n"
            "설치 방법: pip install 'naviertwin[core]'",
            file=sys.stderr,
        )
        return 1

    app = QApplication(sys.argv)
    app.setApplicationName("NavierTwin")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("NavierTwin")

    # TODO(v1.0): MainWindow 구현 후 연결
    from PySide6.QtWidgets import QLabel, QMainWindow  # noqa: PLC0415

    window = QMainWindow()
    window.setWindowTitle("NavierTwin v0.1.0")
    window.resize(1280, 800)
    label = QLabel("NavierTwin — 스캐폴딩 단계 (v0.1.0)\nGUI는 v1.0에서 구현됩니다.")
    label.setAlignment(label.alignment())
    window.setCentralWidget(label)
    window.show()

    return app.exec()


def main() -> None:
    """CLI 진입점 함수.

    ``--gui`` 플래그가 없으면 도움말을 출력하고 종료한다.
    """
    parser = _build_parser()
    args = parser.parse_args()

    # 로거 초기화
    from naviertwin.utils.logger import get_logger  # noqa: PLC0415

    logger = get_logger(__name__)
    logger.debug("NavierTwin 시작. 인수: %s", args)

    if args.gui:
        sys.exit(_run_gui(args.config))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
