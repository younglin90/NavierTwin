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

from naviertwin import __version__


def _build_parser() -> argparse.ArgumentParser:
    """CLI 인수 파서를 구성하여 반환한다.

    Returns:
        구성된 ArgumentParser 인스턴스.
    """
    parser = argparse.ArgumentParser(
        prog="naviertwin",
        description=(
            "NavierTwin — CFD 후처리 결과를 AI/ROM 디지털 트윈으로 변환하는 툴\n"
            f"버전: {__version__}"
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
        version=f"%(prog)s {__version__}",
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

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="cavity 또는 Burgers 벤치마크 실행")
    p_bench.add_argument("--kind", choices=["cavity", "burgers", "lbm"], default="cavity")

    # server
    p_serv = sub.add_parser("server", help="FastAPI REST 서버 실행")
    p_serv.add_argument("--host", default="127.0.0.1")
    p_serv.add_argument("--port", type=int, default=8000)

    # pipeline
    p_pipe = sub.add_parser("pipeline", help="합성 end-to-end 파이프라인 실행")
    p_pipe.add_argument("--n-modes", type=int, default=5)
    p_pipe.add_argument("--reducer", choices=["pod", "ae"], default="pod")
    p_pipe.add_argument("--surrogate", choices=["kriging", "rbf"], default="kriging")

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
    app.setApplicationVersion(__version__)
    app.setOrganizationName("NavierTwin")

    from naviertwin.gui.main_window import MainWindow  # noqa: PLC0415

    window = MainWindow()
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
    elif args.command == "benchmark":
        sys.exit(_run_benchmark(args.kind))
    elif args.command == "server":
        sys.exit(_run_server(args.host, args.port))
    elif args.command == "pipeline":
        sys.exit(_run_pipeline(args.reducer, args.n_modes, args.surrogate))
    else:
        parser.print_help()
        sys.exit(0)


def _run_benchmark(kind: str) -> int:
    """단일 벤치마크 실행."""
    import runpy
    from pathlib import Path

    base = Path(__file__).resolve().parent.parent.parent / "examples"
    scripts = {
        "cavity": base / "cavity_benchmark.py",
        "burgers": base / "burgers_fno.py",
        "lbm": base / "lbm_rom_pipeline.py",
    }
    script = scripts.get(kind)
    if script is None or not script.exists():
        print(f"벤치마크 스크립트 없음: {kind}", file=sys.stderr)
        return 1
    runpy.run_path(str(script), run_name="__main__")
    return 0


def _run_server(host: str, port: int) -> int:
    """FastAPI 서버 실행 (uvicorn)."""
    try:
        import uvicorn

        from naviertwin.api.server import app
    except ImportError as e:
        print(f"오류: FastAPI/uvicorn 설치 필요: {e}", file=sys.stderr)
        return 1
    if app is None:
        print("app 생성 실패 (FastAPI 미설치)", file=sys.stderr)
        return 1
    uvicorn.run(app, host=host, port=port)
    return 0


def _run_pipeline(reducer: str, n_modes: int, surrogate: str) -> int:
    """합성 파이프라인 실행."""
    import numpy as np

    from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline

    rng = np.random.default_rng(0)
    r = max(1, n_modes)
    U = rng.standard_normal((60, r))
    V = rng.standard_normal((r, 30))
    X = U @ V + 0.01 * rng.standard_normal((60, 30))

    pipe = NavierTwinPipeline(
        reducer_kind=reducer, n_modes=n_modes, surrogate_kind=surrogate,
    )
    pipe.load_snapshots(X, field_name="U")
    pipe.reduce()
    params = np.linspace(0, 1, 30).reshape(-1, 1)
    pipe.fit_surrogate(params)
    metrics = pipe.validate(params[-5:], pipe.state.coeffs[-5:])
    print(f"파이프라인 완료: {metrics}")
    return 0


if __name__ == "__main__":
    main()
