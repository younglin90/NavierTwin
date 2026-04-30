"""NavierTwin CLI 진입점.

Examples:
    GUI 실행::

        $ naviertwin --gui

    도움말 출력::

        $ naviertwin --help
"""

from __future__ import annotations

import argparse
import json
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
            "NavierTwin - CFD 후처리 결과를 AI/ROM 디지털 트윈으로 변환하는 툴\n"
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

    # pipeline-demo
    p_demo = sub.add_parser("pipeline-demo", help="합성 데모를 실행하고 리포트 산출물 저장")
    p_demo.add_argument("--outdir", required=True, help="metrics.json/report.html 출력 디렉토리")
    p_demo.add_argument("--n-modes", type=int, default=3)
    p_demo.add_argument("--surrogate", choices=["kriging", "rbf"], default="rbf")

    # model-sweep
    p_sweep = sub.add_parser("model-sweep", help="여러 ROM/surrogate 후보를 자동 비교")
    p_sweep.add_argument("--reducers", default="pod", help="쉼표 구분 reducer 목록: pod,ae")
    p_sweep.add_argument("--n-modes", default="2,3,5", help="쉼표 구분 모드 수 목록")
    p_sweep.add_argument(
        "--surrogates",
        default="rbf,kriging",
        help="쉼표 구분 surrogate 목록: rbf,kriging",
    )
    p_sweep.add_argument("--samples", type=int, default=24, help="합성 snapshot 개수")
    p_sweep.add_argument("--features", type=int, default=48, help="합성 field feature 개수")
    p_sweep.add_argument("--seed", type=int, default=7, help="재현 가능한 sweep seed")
    p_sweep.add_argument("--json", dest="as_json", action="store_true", help="JSON으로 출력")

    # preflight
    p_preflight = sub.add_parser("preflight", help="CFD 입력 데이터 readiness 점검")
    p_preflight.add_argument("path", help="점검할 CFD 파일 또는 케이스 디렉토리")
    p_preflight.add_argument("--json", dest="as_json", action="store_true", help="JSON으로 출력")
    p_preflight.add_argument("--output", default=None, metavar="PATH", help="readiness JSON 리포트 저장 경로")

    # support-bundle
    p_support = sub.add_parser("support-bundle", help="고객 지원용 진단 번들 생성")
    p_support.add_argument("--outdir", required=True, help="지원 번들 출력 디렉토리")
    p_support.add_argument("--preflight", default=None, help="선택적으로 readiness 점검할 CFD 입력 경로")
    p_support.add_argument(
        "--zip",
        dest="zip_bundle",
        action="store_true",
        default=False,
        help="지원 번들 디렉토리를 유지하면서 support-bundle.zip을 추가 생성",
    )
    p_support.add_argument(
        "--include-optional",
        action="store_true",
        default=False,
        help="doctor 리포트에 GUI/API/GPU 선택 의존성 점검 포함",
    )

    # autorefine
    p_refine = sub.add_parser(
        "autorefine",
        help="ROADMAP를 주기 분석해 자동으로 완료 가능한 항목을 반영",
    )
    p_refine.add_argument("--interval-sec", type=int, default=60)
    p_refine.add_argument("--iterations", type=int, default=1)
    p_refine.add_argument("--dry-run", action="store_true", default=False)
    p_refine.add_argument(
        "--project-root",
        default=".",
        help="프로젝트 루트 경로 (기본값: 현재 디렉토리)",
    )
    p_refine.add_argument(
        "--artifact-dir",
        default=None,
        help="리포트 출력 디렉토리 (기본값: <root>/verify_artifacts/autorefine)",
    )

    # update-check
    p_update = sub.add_parser(
        "update-check",
        help="로컬 릴리스 메타데이터로 업데이트 가능 여부 확인",
    )
    p_update.add_argument("--metadata", required=True, help="릴리스 메타데이터 JSON 경로")
    p_update.add_argument("--channel", default="stable", choices=["stable", "beta", "nightly"])
    p_update.add_argument("--current-version", default=__version__)

    # doctor
    p_doctor = sub.add_parser("doctor", help="설치/런타임 환경 진단 리포트 출력")
    p_doctor.add_argument("--json", dest="as_json", action="store_true", help="JSON으로 출력")
    p_doctor.add_argument("--output", default=None, metavar="PATH", help="진단 리포트 저장 경로")
    p_doctor.add_argument(
        "--include-optional",
        action="store_true",
        default=False,
        help="GUI/API/GPU 선택 의존성까지 점검",
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
    app.setApplicationVersion(__version__)
    app.setOrganizationName("NavierTwin")

    from naviertwin.gui.main_window import MainWindow  # noqa: PLC0415

    window = MainWindow(config_path=config_path)
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
    elif args.command == "pipeline-demo":
        sys.exit(
            _run_pipeline_demo(
                outdir=args.outdir,
                n_modes=args.n_modes,
                surrogate=args.surrogate,
            )
        )
    elif args.command == "model-sweep":
        sys.exit(
            _run_model_sweep(
                reducers=args.reducers,
                n_modes=args.n_modes,
                surrogates=args.surrogates,
                samples=args.samples,
                features=args.features,
                seed=args.seed,
                as_json=args.as_json,
            )
        )
    elif args.command == "preflight":
        sys.exit(_run_preflight(path=args.path, as_json=args.as_json, output=args.output))
    elif args.command == "support-bundle":
        sys.exit(
            _run_support_bundle(
                outdir=args.outdir,
                preflight=args.preflight,
                include_optional=args.include_optional,
                zip_bundle=args.zip_bundle,
            )
        )
    elif args.command == "autorefine":
        sys.exit(
            _run_autorefine(
                interval_sec=args.interval_sec,
                iterations=args.iterations,
                apply=not args.dry_run,
                project_root=args.project_root,
                artifact_dir=args.artifact_dir,
            )
        )
    elif args.command == "update-check":
        sys.exit(
            _run_update_check(
                metadata=args.metadata,
                channel=args.channel,
                current_version=args.current_version,
            )
        )
    elif args.command == "doctor":
        sys.exit(
            _run_doctor(
                as_json=args.as_json,
                include_optional=args.include_optional,
                output=args.output,
            )
        )
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


def _run_pipeline_demo(*, outdir: str, n_modes: int, surrogate: str) -> int:
    """합성 데이터 기반 첫 실행 데모를 수행하고 산출물을 저장한다."""
    from pathlib import Path

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import numpy as np

        from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline

        rng = np.random.default_rng(42)
        rank = max(1, n_modes)
        n_features = 48
        n_snapshots = 24
        params = np.linspace(0, 1, n_snapshots).reshape(-1, 1)
        t = params[:, 0]
        basis = rng.standard_normal((n_features, rank))
        coeff_rows = []
        for mode in range(rank):
            frequency = mode + 1
            if mode % 3 == 0:
                coeff_rows.append(np.sin(frequency * np.pi * t))
            elif mode % 3 == 1:
                coeff_rows.append(np.cos(frequency * np.pi * t))
            else:
                coeff_rows.append((t - 0.5) ** frequency)
        coeffs = np.vstack(coeff_rows)
        snapshots = basis @ coeffs + 0.005 * rng.standard_normal((n_features, n_snapshots))

        pipe = NavierTwinPipeline(
            reducer_kind="pod",
            n_modes=rank,
            surrogate_kind=surrogate,
        )
        pipe.load_snapshots(snapshots, field_name="U")
        pipe.reduce()
        pipe.fit_surrogate(params)
        metrics = pipe.validate(params[-6:], pipe.state.coeffs[-6:])

        metrics_path = output_dir / "metrics.json"
        report_path = output_dir / "report.html"
        payload = {
            "status": "ok",
            "artifacts": {
                "metrics": str(metrics_path),
                "report": str(report_path),
            },
            "demo": {
                "reducer": "pod",
                "surrogate": surrogate,
                "n_modes": rank,
                "n_features": n_features,
                "n_snapshots": n_snapshots,
            },
            "metrics": metrics,
        }
        metrics_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        pipe.export_report(str(report_path), project="NavierTwin Pipeline Demo")
    except ImportError as exc:
        print(
            "pipeline-demo error: missing optional dependency. "
            "Install with pip install 'naviertwin[core]'. "
            f"Details: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def _parse_csv_tokens(value: str, *, allowed: set[str], label: str) -> list[str]:
    """쉼표 구분 문자열을 검증된 토큰 목록으로 변환한다."""
    parsed = [token.strip().lower() for token in value.split(",") if token.strip()]
    if not parsed:
        raise ValueError(f"{label} must include at least one value")
    invalid = sorted(set(parsed) - allowed)
    if invalid:
        allowed_text = ",".join(sorted(allowed))
        raise ValueError(f"unsupported {label}: {','.join(invalid)} (allowed: {allowed_text})")
    return parsed


def _parse_csv_ints(value: str, *, label: str) -> list[int]:
    """쉼표 구분 문자열을 양의 정수 목록으로 변환한다."""
    parsed: list[int] = []
    for token in value.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        try:
            number = int(stripped)
        except ValueError as exc:
            raise ValueError(f"{label} must be positive integers: {stripped}") from exc
        if number < 1:
            raise ValueError(f"{label} must be positive integers: {number}")
        parsed.append(number)
    if not parsed:
        raise ValueError(f"{label} must include at least one value")
    return parsed


def _run_model_sweep(
    *,
    reducers: str,
    n_modes: str,
    surrogates: str,
    samples: int,
    features: int,
    seed: int,
    as_json: bool,
) -> int:
    """여러 reducer/surrogate 후보를 같은 합성 데이터셋에서 비교한다."""
    try:
        import numpy as np

        from naviertwin.core.digital_twin.pipeline_compare import (
            compare_models,
            rank_table,
        )

        reducer_values = _parse_csv_tokens(
            reducers,
            allowed={"pod", "ae"},
            label="reducers",
        )
        mode_values = _parse_csv_ints(n_modes, label="n-modes")
        surrogate_values = _parse_csv_tokens(
            surrogates,
            allowed={"kriging", "rbf"},
            label="surrogates",
        )
        if samples < 8:
            raise ValueError("samples must be at least 8")
        if features < 4:
            raise ValueError("features must be at least 4")

        rng = np.random.default_rng(seed)
        rank = max(max(mode_values), 2)
        params = np.linspace(0, 1, samples).reshape(-1, 1)
        t = params[:, 0]
        basis = rng.standard_normal((features, rank))
        coeff_rows = []
        for mode in range(rank):
            frequency = mode + 1
            if mode % 3 == 0:
                coeff_rows.append(np.sin(frequency * np.pi * t))
            elif mode % 3 == 1:
                coeff_rows.append(np.cos(frequency * np.pi * t))
            else:
                coeff_rows.append((t - 0.5) ** frequency)
        coeffs = np.vstack(coeff_rows)
        snapshots = basis @ coeffs + 0.005 * rng.standard_normal((features, samples))

        configs = [
            (reducer, mode_count, surrogate)
            for reducer in reducer_values
            for mode_count in mode_values
            for surrogate in surrogate_values
        ]
        rows = compare_models(snapshots, params, configs, seed=seed)
    except (ImportError, ValueError) as exc:
        print(f"model-sweep error: {exc}", file=sys.stderr)
        return 2

    payload = {
        "status": "ok",
        "configs": len(rows),
        "best": rows[0] if rows else None,
        "rows": rows,
    }
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(rank_table(rows))
        if rows:
            best = rows[0]
            print(
                "best: "
                f"{best['reducer_kind']} n_modes={best['n_modes']} "
                f"{best['surrogate_kind']} rmse={best['rmse']:.6g}"
            )
    return 0


def _run_preflight(*, path: str, as_json: bool, output: str | None = None) -> int:
    """CFD 입력 데이터 readiness 리포트를 출력한다."""
    from pathlib import Path

    from naviertwin.core.validation.dataset_preflight import (
        build_dataset_preflight_report,
        format_preflight_report,
        report_to_json,
    )

    report = build_dataset_preflight_report(path)
    json_report = report_to_json(report)
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_report + "\n", encoding="utf-8")
    print(json_report if as_json else format_preflight_report(report))
    return 1 if report.get("status") == "error" else 0


def _run_support_bundle(
    *,
    outdir: str,
    preflight: str | None,
    include_optional: bool,
    zip_bundle: bool,
) -> int:
    """고객 지원용 진단 번들을 생성한다."""
    from naviertwin.utils.support_bundle import build_support_bundle, report_to_json

    try:
        report = build_support_bundle(
            outdir=outdir,
            preflight=preflight,
            include_optional=include_optional,
            zip_bundle=zip_bundle,
        )
    except OSError as exc:
        print(f"support-bundle error: {exc}", file=sys.stderr)
        return 2
    print(report_to_json(report))
    return 0


def _run_autorefine(
    *,
    interval_sec: int,
    iterations: int,
    apply: bool,
    project_root: str,
    artifact_dir: str | None,
) -> int:
    """ROADMAP 자동 고도화 루프 실행."""
    from naviertwin.utils.workflow.autorefine import run_autorefine_loop

    reports = run_autorefine_loop(
        project_root=project_root,
        interval_sec=interval_sec,
        iterations=iterations,
        apply=apply,
        artifact_dir=artifact_dir,
    )
    if not reports:
        print("autorefine: 실행 결과가 없습니다.")
        return 1
    last = reports[-1]
    print(
        "autorefine 완료: "
        f"pending={last['pending_count']}, "
        f"auto_candidates={last['auto_candidate_count']}, "
        f"applied={last['applied_count']}"
    )
    return 0


def _run_update_check(*, metadata: str, channel: str, current_version: str) -> int:
    """로컬 릴리스 메타데이터를 사용해 업데이트 가능 여부를 출력한다."""
    from pathlib import Path

    from naviertwin.utils.updater import check_for_update

    try:
        result = check_for_update(
            Path(metadata),
            channel=channel,
            current_version=current_version,
        )
    except (OSError, ValueError) as exc:
        print(f"update-check error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


def _run_doctor(*, as_json: bool, include_optional: bool, output: str | None = None) -> int:
    """설치/런타임 환경 진단 리포트를 출력한다."""
    from pathlib import Path

    from naviertwin.utils.doctor import (
        build_doctor_report,
        format_doctor_report,
        report_to_json,
    )

    report = build_doctor_report(include_optional=include_optional)
    json_report = report_to_json(report)
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_report + "\n", encoding="utf-8")
    print(json_report if as_json else format_doctor_report(report))
    return 1 if report.get("status") == "error" else 0


if __name__ == "__main__":
    main()
