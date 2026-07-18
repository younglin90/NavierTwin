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
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

from naviertwin import __version__


def _csv_items(value: str, *, lowercase: bool = False) -> list[str]:
    """쉼표 구분 값을 공백 제거 후 비어 있지 않은 항목으로 반환한다."""
    items = filter(None, map(str.strip, value.split(",")))
    if lowercase:
        return list(map(str.lower, items))
    return list(items)


def _nonempty_strings(values: Any) -> list[str]:
    """값 컬렉션을 비어 있지 않은 문자열 목록으로 변환한다."""
    return list(filter(None, map(str, values)))


def _sample_header(width: int) -> str:
    """sample_N CSV 헤더를 만든다."""
    return ",".join(map(lambda idx: f"sample_{idx}", range(width)))


def _checks_passed(checks: list[dict[str, Any]]) -> bool:
    """checks payload의 passed 값을 모두 확인한다."""
    return all(map(lambda check: check["passed"], checks))


def _metadata_without_artifact_integrity(metadata: dict[str, Any]) -> dict[str, Any]:
    """artifact_integrity 항목만 제거한 metadata 사본을 만든다."""
    copied = dict(metadata)
    copied.pop("artifact_integrity", None)
    return copied


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

    # web
    p_web = sub.add_parser("web", help="trame 기반 웹 GUI 실행 (브라우저)")
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8080)
    p_web.add_argument(
        "--no-browser",
        action="store_true",
        help="시작 시 기본 브라우저 자동 열기 비활성화",
    )

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

    # build-twin
    p_build = sub.add_parser("build-twin", help="CFD/CSV 데이터셋에서 디지털 트윈 산출물 생성")
    source = p_build.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", default=None, help="ReaderFactory로 읽을 CFD 파일/케이스 경로")
    source.add_argument(
        "--csv-snapshots",
        default=None,
        help="쉼표 구분 CSV 파일/글롭/디렉토리. 각 CSV는 하나의 snapshot",
    )
    p_build.add_argument("--field", default=None, help="CFD 입력에서 사용할 필드명")
    p_build.add_argument("--field-column", default=None, help="CSV snapshot에서 사용할 컬럼명")
    p_build.add_argument("--params", default=None, help="선택적 파라미터 CSV 경로")
    p_build.add_argument(
        "--param-columns",
        default=None,
        help="쉼표 구분 파라미터 컬럼명. 생략하면 numeric 컬럼 전체 사용",
    )
    p_build.add_argument("--outdir", required=True, help="metrics/report/checkpoint 출력 디렉토리")
    p_build.add_argument(
        "--reducer",
        choices=["pod", "incremental_pod", "mrpod", "ae"],
        default="pod",
    )
    p_build.add_argument("--n-modes", type=int, default=3)
    p_build.add_argument("--surrogate", choices=["kriging", "rbf"], default="rbf")
    p_build.add_argument("--validation-count", type=int, default=3)
    p_build.add_argument("--json", dest="as_json", action="store_true", help="JSON으로 출력")

    # predict-twin
    p_predict = sub.add_parser("predict-twin", help="저장된 TwinEngine으로 즉시 예측")
    predict_engine = p_predict.add_mutually_exclusive_group(required=True)
    predict_engine.add_argument("--engine", default=None, help="build-twin이 생성한 engine.pkl 경로")
    predict_engine.add_argument(
        "--artifacts-dir",
        default=None,
        help="engine.pkl을 포함한 build/extract 산출물 디렉토리",
    )
    predict_source = p_predict.add_mutually_exclusive_group(required=True)
    predict_source.add_argument("--params", default=None, help="쉼표 구분 단일 입력 파라미터")
    predict_source.add_argument("--params-csv", default=None, help="배치 입력 파라미터 CSV 경로")
    p_predict.add_argument(
        "--param-columns",
        default=None,
        help="쉼표 구분 파라미터 컬럼명. 생략하면 numeric 컬럼 전체 사용",
    )
    p_predict.add_argument("--output", default=None, help="예측 필드 CSV 저장 경로")
    p_predict.add_argument("--json", dest="as_json", action="store_true", help="JSON으로 출력")

    # benchmark-twin
    p_benchmark_twin = sub.add_parser(
        "benchmark-twin",
        help="저장/배포된 TwinEngine 예측 지연시간 측정",
    )
    benchmark_engine = p_benchmark_twin.add_mutually_exclusive_group(required=True)
    benchmark_engine.add_argument("--engine", default=None, help="측정할 engine.pkl 경로")
    benchmark_engine.add_argument(
        "--artifacts-dir",
        default=None,
        help="engine.pkl을 포함한 build/extract 산출물 디렉토리",
    )
    benchmark_source = p_benchmark_twin.add_mutually_exclusive_group(required=True)
    benchmark_source.add_argument("--params", default=None, help="쉼표 구분 단일 입력 파라미터")
    benchmark_source.add_argument("--params-csv", default=None, help="배치 입력 파라미터 CSV 경로")
    p_benchmark_twin.add_argument(
        "--param-columns",
        default=None,
        help="쉼표 구분 파라미터 컬럼명. 생략하면 numeric 컬럼 전체 사용",
    )
    p_benchmark_twin.add_argument("--warmup", type=int, default=2, help="측정 전 warmup 예측 횟수")
    p_benchmark_twin.add_argument("--repeat", type=int, default=20, help="latency 측정 반복 횟수")
    p_benchmark_twin.add_argument("--max-mean-ms", type=float, default=None, help="허용 최대 mean latency(ms)")
    p_benchmark_twin.add_argument("--max-p50-ms", type=float, default=None, help="허용 최대 p50 latency(ms)")
    p_benchmark_twin.add_argument("--max-p95-ms", type=float, default=None, help="허용 최대 p95 latency(ms)")
    p_benchmark_twin.add_argument("--max-p99-ms", type=float, default=None, help="허용 최대 p99 latency(ms)")
    p_benchmark_twin.add_argument(
        "--min-throughput-hz",
        type=float,
        default=None,
        help="허용 최소 예측 처리량(Hz)",
    )
    p_benchmark_twin.add_argument("--output", default=None, help="latency JSON 리포트 저장 경로")
    p_benchmark_twin.add_argument("--json", dest="as_json", action="store_true", help="JSON으로 출력")

    # validate-twin
    p_validate = sub.add_parser("validate-twin", help="저장된 TwinEngine을 기준 CFD/CSV 데이터로 검증")
    validate_engine = p_validate.add_mutually_exclusive_group(required=True)
    validate_engine.add_argument("--engine", default=None, help="검증할 engine.pkl 경로")
    validate_engine.add_argument(
        "--artifacts-dir",
        default=None,
        help="engine.pkl을 포함한 build/extract 산출물 디렉토리",
    )
    validate_source = p_validate.add_mutually_exclusive_group(required=True)
    validate_source.add_argument("--input", default=None, help="ReaderFactory로 읽을 CFD 파일/케이스 경로")
    validate_source.add_argument(
        "--csv-snapshots",
        default=None,
        help="쉼표 구분 CSV 파일/글롭/디렉토리. 각 CSV는 하나의 기준 snapshot",
    )
    p_validate.add_argument("--field", default=None, help="CFD 입력에서 사용할 필드명")
    p_validate.add_argument("--field-column", default=None, help="CSV snapshot에서 사용할 컬럼명")
    p_validate.add_argument("--params", default=None, help="선택적 검증 파라미터 CSV 경로")
    p_validate.add_argument(
        "--param-columns",
        default=None,
        help="쉼표 구분 파라미터 컬럼명. 생략하면 numeric 컬럼 전체 사용",
    )
    p_validate.add_argument("--max-rmse", type=float, default=None, help="허용 최대 RMSE")
    p_validate.add_argument("--min-r2", type=float, default=None, help="허용 최소 R²")
    p_validate.add_argument(
        "--max-relative-l2",
        type=float,
        default=None,
        help="허용 최대 relative L2 error",
    )
    p_validate.add_argument("--output", default=None, help="검증 metrics JSON 저장 경로")
    p_validate.add_argument("--json", dest="as_json", action="store_true", help="JSON으로 출력")

    # package-twin
    p_package = sub.add_parser("package-twin", help="트윈 산출물을 고객 전달용 ZIP으로 패키징")
    p_package.add_argument("--artifacts-dir", required=True, help="build-twin 산출물 디렉토리")
    p_package.add_argument("--output", required=True, help="생성할 ZIP 경로")
    p_package.add_argument(
        "--include-validation",
        default=None,
        help="선택적 validate-twin JSON 리포트 경로",
    )
    p_package.add_argument("--max-mean-ms", type=float, default=None, help="delivery SLO 최대 mean latency(ms)")
    p_package.add_argument("--max-p50-ms", type=float, default=None, help="delivery SLO 최대 p50 latency(ms)")
    p_package.add_argument("--max-p95-ms", type=float, default=100.0, help="delivery SLO 최대 p95 latency(ms)")
    p_package.add_argument("--max-p99-ms", type=float, default=None, help="delivery SLO 최대 p99 latency(ms)")
    p_package.add_argument(
        "--min-throughput-hz",
        type=float,
        default=10.0,
        help="delivery SLO 최소 예측 처리량(Hz)",
    )
    p_package.add_argument(
        "--no-latency-slo",
        action="store_true",
        default=False,
        help="delivery.json에 latency SLO 정책을 기록하지 않음",
    )
    p_package.add_argument("--json", dest="as_json", action="store_true", help="JSON으로 출력")

    # verify-twin-package
    p_verify_package = sub.add_parser(
        "verify-twin-package",
        help="고객 전달용 트윈 ZIP 무결성 검증",
    )
    p_verify_package.add_argument("--package", required=True, help="검증할 package-twin ZIP 경로")
    p_verify_package.add_argument(
        "--extract-to",
        default=None,
        help="검증 성공 시 ZIP 내용을 안전하게 추출할 디렉토리",
    )
    p_verify_package.add_argument("--json", dest="as_json", action="store_true", help="JSON으로 출력")

    # inspect-twin-package
    p_inspect_package = sub.add_parser(
        "inspect-twin-package",
        help="고객 전달용 트윈 ZIP 구성과 delivery metadata 조회",
    )
    p_inspect_package.add_argument("--package", required=True, help="조회할 package-twin ZIP 경로")
    p_inspect_package.add_argument("--json", dest="as_json", action="store_true", help="JSON으로 출력")

    # accept-twin-package
    p_accept_package = sub.add_parser(
        "accept-twin-package",
        help="전달용 트윈 ZIP을 검증, 예측, latency gate까지 원샷 수락 검사",
    )
    p_accept_package.add_argument("--package", required=True, help="검사할 package-twin ZIP 경로")
    p_accept_package.add_argument(
        "--extract-to",
        default=None,
        help="검증 성공 시 ZIP 내용을 안전하게 추출할 디렉토리. 생략하면 임시 디렉토리 사용",
    )
    p_accept_package.add_argument(
        "--prediction-output",
        default=None,
        help="샘플 입력 예측 필드 CSV 저장 경로",
    )
    p_accept_package.add_argument("--warmup", type=int, default=2, help="latency 측정 전 warmup 횟수")
    p_accept_package.add_argument("--repeat", type=int, default=20, help="latency 측정 반복 횟수")
    p_accept_package.add_argument("--max-mean-ms", type=float, default=None, help="허용 최대 mean latency(ms)")
    p_accept_package.add_argument("--max-p50-ms", type=float, default=None, help="허용 최대 p50 latency(ms)")
    p_accept_package.add_argument("--max-p95-ms", type=float, default=None, help="허용 최대 p95 latency(ms)")
    p_accept_package.add_argument("--max-p99-ms", type=float, default=None, help="허용 최대 p99 latency(ms)")
    p_accept_package.add_argument(
        "--min-throughput-hz",
        type=float,
        default=None,
        help="허용 최소 예측 처리량(Hz)",
    )
    p_accept_package.add_argument(
        "--skip-benchmark",
        action="store_true",
        default=False,
        help="무결성/샘플 예측만 수행하고 latency 측정은 생략",
    )
    p_accept_package.add_argument("--output", default=None, help="acceptance JSON 리포트 저장 경로")
    p_accept_package.add_argument(
        "--summary-output",
        default=None,
        help="사람이 읽는 acceptance Markdown 요약 리포트 저장 경로",
    )
    p_accept_package.add_argument("--json", dest="as_json", action="store_true", help="JSON으로 출력")

    # preflight
    p_preflight = sub.add_parser("preflight", help="CFD 입력 데이터 readiness 점검")
    p_preflight.add_argument("path", help="점검할 CFD 파일 또는 케이스 디렉토리")
    p_preflight.add_argument("--json", dest="as_json", action="store_true", help="JSON으로 출력")
    p_preflight.add_argument("--output", default=None, metavar="PATH", help="readiness JSON 리포트 저장 경로")

    # support-bundle
    p_support = sub.add_parser("support-bundle", help="고객 지원용 진단 번들 생성")
    p_support.add_argument("--outdir", required=True, help="지원 번들 출력 디렉토리")
    p_support.add_argument("--preflight", default=None, help="선택적으로 readiness 점검할 CFD 입력 경로")
    p_support.add_argument("--acceptance-json", default=None, help="포함할 acceptance JSON 리포트 경로")
    p_support.add_argument(
        "--acceptance-summary",
        default=None,
        help="포함할 acceptance Markdown 요약 리포트 경로",
    )
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

    # inspect-support-bundle
    p_inspect_support = sub.add_parser(
        "inspect-support-bundle",
        help="기존 고객 지원 번들 디렉토리/ZIP 요약 및 무결성 점검",
    )
    p_inspect_support.add_argument("path", help="점검할 support-bundle 디렉토리 또는 ZIP")
    p_inspect_support.add_argument("--json", dest="as_json", action="store_true", help="JSON으로 출력")

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
    p_update.add_argument(
        "--verify-artifact",
        default=None,
        metavar="PATH",
        help="다운로드한 설치 파일을 릴리스 메타데이터 SHA256으로 검증",
    )

    # feature-pack
    p_feature = sub.add_parser(
        "feature-pack",
        help="대형 선택 기능 팩 다운로드/설치/상태 조회",
    )
    feature_sub = p_feature.add_subparsers(dest="feature_pack_command", metavar="<action>")
    p_feature_list = feature_sub.add_parser("list", help="Feature Pack 상태 조회")
    p_feature_list.add_argument("--json", dest="as_json", action="store_true", help="JSON으로 출력")
    p_feature_install = feature_sub.add_parser("install", help="다운로드한 Feature Pack ZIP 설치")
    p_feature_install.add_argument("--archive", required=True, help="설치할 Feature Pack ZIP 경로")
    p_feature_install.add_argument("--pack", default=None, help="예상 pack id 검증")
    p_feature_install.add_argument("--sha256", default=None, help="선택적 ZIP SHA256 검증값")
    p_feature_install.add_argument("--json", dest="as_json", action="store_true", help="JSON으로 출력")
    p_feature_download = feature_sub.add_parser("download", help="Feature Pack ZIP 다운로드")
    p_feature_download.add_argument("--pack", required=True, help="다운로드할 pack id")
    p_feature_download.add_argument("--url", default=None, help="기본 GitHub Release URL 대신 사용할 URL")
    p_feature_download.add_argument("--sha256", default=None, help="선택적 ZIP SHA256 검증값")
    p_feature_download.add_argument("--output-dir", default=None, help="다운로드 출력 디렉토리")
    p_feature_download.add_argument(
        "--install",
        action="store_true",
        default=False,
        help="다운로드 후 즉시 설치",
    )
    p_feature_download.add_argument("--json", dest="as_json", action="store_true", help="JSON으로 출력")

    # batch-train
    p_batch = sub.add_parser(
        "batch-train",
        help="헤드리스 배치 트윈 학습 (mpirun 분산 지원, GUI 비의존)",
    )
    p_batch.add_argument(
        "--config",
        dest="batch_config",
        required=True,
        metavar="PATH",
        help="배치 잡 정의 JSON 경로 (jobs 목록)",
    )
    p_batch.add_argument("--json", dest="as_json", action="store_true", help="JSON으로 출력")

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


def _setup_qt_runtime_defaults() -> None:
    """QApplication 생성 전 Qt/VTK viewer 플랫폼 기본값을 잡는다."""
    is_wsl = "WSL_DISTRO_NAME" in os.environ or Path("/usr/lib/wsl").exists()
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if is_wsl and has_display and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("QT_X11_NO_MITSHM", "1")


def _run_gui(config_path: str | None) -> int:
    """PySide6 GUI 애플리케이션을 실행한다.

    Args:
        config_path: 설정 파일 경로. None이면 기본 경로를 사용한다.

    Returns:
        프로세스 종료 코드.
    """
    _setup_qt_runtime_defaults()

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

    # 터미널에서 Ctrl+C (SIGINT) 로 즉시 종료 가능하도록 핸들러 복원.
    # PyQt/PySide 는 기본적으로 Python signal 핸들러가 GUI 이벤트 루프에
    # 진입하는 동안 호출되지 않으므로, 명시적으로 SIG_DFL 로 되돌리고 0.5s
    # 주기 더미 타이머로 인터프리터에 제어권을 넘긴다.
    import signal as _signal  # noqa: PLC0415

    _signal.signal(_signal.SIGINT, _signal.SIG_DFL)
    try:
        _signal.signal(_signal.SIGTERM, _signal.SIG_DFL)
    except (ValueError, OSError):
        pass

    from PySide6.QtCore import QTimer  # noqa: PLC0415

    _sigint_pump = QTimer()
    _sigint_pump.setInterval(500)
    _sigint_pump.timeout.connect(lambda: None)
    _sigint_pump.start()
    app._naviertwin_sigint_pump = _sigint_pump  # GC 방지로 보관

    # QSS 가 QComboBox QAbstractItemView 를 스타일링하면 popup 닫힘 핸들러가
    # 누락되는 Qt6 버그를 글로벌 이벤트 필터로 보정한다.
    from naviertwin.gui.utils.combo_fix import install_combo_close_filter  # noqa: PLC0415

    install_combo_close_filter(app)

    from naviertwin.gui.main_window import MainWindow  # noqa: PLC0415

    window = MainWindow(config_path=config_path)

    # MainWindow 생성 시점에 만들어진 모든 콤보박스에도 닫힘 보정 일괄 적용.
    from naviertwin.gui.utils.combo_fix import apply_to_widget_tree  # noqa: PLC0415

    apply_to_widget_tree(window)

    window.show()

    return app.exec()


def main() -> None:
    """CLI 진입점 함수.

    ``--gui`` 플래그가 없으면 도움말을 출력하고 종료한다.
    """
    from naviertwin.utils.feature_packs import activate_installed_feature_packs  # noqa: PLC0415

    activate_installed_feature_packs()

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
    elif args.command == "web":
        sys.exit(_run_web_gui(args.host, args.port, not args.no_browser))
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
    elif args.command == "build-twin":
        sys.exit(
            _run_build_twin(
                input_path=args.input,
                csv_snapshots=args.csv_snapshots,
                field=args.field,
                field_column=args.field_column,
                params=args.params,
                param_columns=args.param_columns,
                outdir=args.outdir,
                reducer=args.reducer,
                n_modes=args.n_modes,
                surrogate=args.surrogate,
                validation_count=args.validation_count,
                as_json=args.as_json,
            )
        )
    elif args.command == "predict-twin":
        sys.exit(
            _run_predict_twin(
                engine_path=args.engine,
                artifacts_dir=args.artifacts_dir,
                params=args.params,
                params_csv=args.params_csv,
                param_columns=args.param_columns,
                output=args.output,
                as_json=args.as_json,
            )
        )
    elif args.command == "benchmark-twin":
        sys.exit(
            _run_benchmark_twin(
                engine_path=args.engine,
                artifacts_dir=args.artifacts_dir,
                params=args.params,
                params_csv=args.params_csv,
                param_columns=args.param_columns,
                warmup=args.warmup,
                repeat=args.repeat,
                max_mean_ms=args.max_mean_ms,
                max_p50_ms=args.max_p50_ms,
                max_p95_ms=args.max_p95_ms,
                max_p99_ms=args.max_p99_ms,
                min_throughput_hz=args.min_throughput_hz,
                output=args.output,
                as_json=args.as_json,
            )
        )
    elif args.command == "validate-twin":
        sys.exit(
            _run_validate_twin(
                engine_path=args.engine,
                artifacts_dir=args.artifacts_dir,
                input_path=args.input,
                csv_snapshots=args.csv_snapshots,
                field=args.field,
                field_column=args.field_column,
                params=args.params,
                param_columns=args.param_columns,
                max_rmse=args.max_rmse,
                min_r2=args.min_r2,
                max_relative_l2=args.max_relative_l2,
                output=args.output,
                as_json=args.as_json,
            )
        )
    elif args.command == "package-twin":
        sys.exit(
            _run_package_twin(
                artifacts_dir=args.artifacts_dir,
                output=args.output,
                include_validation=args.include_validation,
                max_mean_ms=args.max_mean_ms,
                max_p50_ms=args.max_p50_ms,
                max_p95_ms=args.max_p95_ms,
                max_p99_ms=args.max_p99_ms,
                min_throughput_hz=args.min_throughput_hz,
                no_latency_slo=args.no_latency_slo,
                as_json=args.as_json,
            )
        )
    elif args.command == "verify-twin-package":
        sys.exit(
            _run_verify_twin_package(
                package_path=args.package,
                extract_to=args.extract_to,
                as_json=args.as_json,
            )
        )
    elif args.command == "inspect-twin-package":
        sys.exit(_run_inspect_twin_package(package_path=args.package, as_json=args.as_json))
    elif args.command == "accept-twin-package":
        sys.exit(
            _run_accept_twin_package(
                package_path=args.package,
                extract_to=args.extract_to,
                prediction_output=args.prediction_output,
                warmup=args.warmup,
                repeat=args.repeat,
                max_mean_ms=args.max_mean_ms,
                max_p50_ms=args.max_p50_ms,
                max_p95_ms=args.max_p95_ms,
                max_p99_ms=args.max_p99_ms,
                min_throughput_hz=args.min_throughput_hz,
                skip_benchmark=args.skip_benchmark,
                output=args.output,
                summary_output=args.summary_output,
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
                acceptance_json=args.acceptance_json,
                acceptance_summary=args.acceptance_summary,
            )
        )
    elif args.command == "inspect-support-bundle":
        sys.exit(_run_inspect_support_bundle(path=args.path, as_json=args.as_json))
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
                verify_artifact=args.verify_artifact,
            )
        )
    elif args.command == "feature-pack":
        sys.exit(_run_feature_pack(args))
    elif args.command == "batch-train":
        sys.exit(_run_batch_train(config=args.batch_config, as_json=args.as_json))
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


def _run_web_gui(host: str, port: int, open_browser: bool) -> int:
    """trame 기반 웹 GUI 실행 (브라우저)."""
    try:
        from naviertwin.web.app import run_web
    except ImportError as e:
        print(
            "오류: trame 설치 필요: pip install naviertwin[web]\n"
            f"  {e}",
            file=sys.stderr,
        )
        return 1
    return run_web(host=host, port=port, open_browser=open_browser)


def _run_batch_train(*, config: str, as_json: bool) -> int:
    """헤드리스 배치 트윈 학습 실행 (MPI 는 이 경로에서만 초기화된다).

    GUI 이벤트 루프와 mpi4py 초기화가 충돌하지 않도록, MPI 감지/사용은
    :mod:`naviertwin.cli.batch_train` 헤드리스 경로 전용이다. mpi4py 가 없거나
    ``mpirun`` 없이 실행돼도 rank 0 / size 1 순차 실행으로 폴백한다.

    Returns:
        0: 모든 잡 성공. 1: 하나 이상 잡 실패. 2: config/런타임 오류.
    """
    from naviertwin.cli.batch_train import run_batch  # noqa: PLC0415

    try:
        payload = run_batch(Path(config))
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"batch-train error: {exc}", file=sys.stderr)
        return 2

    failed = [r for r in payload["results"] if r.get("status") != "ok"]
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(
            f"batch-train 완료: rank {payload['rank']}/{payload['size']}, "
            f"{len(payload['results'])}개 잡 실행, {len(failed)}개 실패"
        )
        print(f"results: {payload['results_path']}")
    return 1 if failed else 0


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
        mode = 0
        while mode < rank:
            frequency = mode + 1
            if mode % 3 == 0:
                coeff_rows.append(np.sin(frequency * np.pi * t))
            elif mode % 3 == 1:
                coeff_rows.append(np.cos(frequency * np.pi * t))
            else:
                coeff_rows.append((t - 0.5) ** frequency)
            mode += 1
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
    parsed = _csv_items(value, lowercase=True)
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
    tokens = value.split(",")
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
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

        from itertools import product

        rng = np.random.default_rng(seed)
        rank = max(max(mode_values), 2)
        params = np.linspace(0, 1, samples).reshape(-1, 1)
        t = params[:, 0]
        basis = rng.standard_normal((features, rank))
        coeff_rows = []
        mode = 0
        while mode < rank:
            frequency = mode + 1
            if mode % 3 == 0:
                coeff_rows.append(np.sin(frequency * np.pi * t))
            elif mode % 3 == 1:
                coeff_rows.append(np.cos(frequency * np.pi * t))
            else:
                coeff_rows.append((t - 0.5) ** frequency)
            mode += 1
        coeffs = np.vstack(coeff_rows)
        snapshots = basis @ coeffs + 0.005 * rng.standard_normal((features, samples))

        configs = list(product(reducer_values, mode_values, surrogate_values))
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


def _expand_csv_snapshot_paths(value: str) -> list[Path]:
    """CSV snapshot 입력 문자열을 정렬된 경로 목록으로 확장한다."""
    from glob import glob

    paths: list[Path] = []
    tokens = value.split(",")
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        raw = token.strip()
        if not raw:
            continue
        candidate = Path(raw).expanduser()
        if candidate.is_dir():
            matches = sorted(candidate.glob("*.csv"))
        else:
            matches = list(map(Path, sorted(glob(str(candidate)))))
            if not matches:
                matches = [candidate]
        paths.extend(matches)

    unique_paths = list(dict.fromkeys(map(Path.resolve, paths)))
    if not unique_paths:
        raise ValueError("csv-snapshots must resolve to at least one CSV file")
    missing = list(map(str, filter(lambda path: not path.exists(), unique_paths)))
    if missing:
        raise FileNotFoundError(f"CSV snapshot file not found: {missing[0]}")
    return unique_paths


def _load_build_twin_params(
    *,
    params_path: str | None,
    param_columns: str | None,
    n_snapshots: int,
) -> tuple[Any, dict[str, Any]]:
    """사용자 파라미터 CSV 또는 기본 normalized index를 로드한다."""
    import numpy as np

    if params_path is None:
        values = np.linspace(0.0, 1.0, n_snapshots).reshape(-1, 1)
        return values, _build_parameter_contract(
            values,
            names=["normalized_index"],
            source="default-normalized-index",
        )

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas required to read --params CSV") from exc

    df = pd.read_csv(params_path)
    if param_columns:
        columns = _csv_items(param_columns)
        invalid = sorted(set(columns) - set(map(str, df.columns)))
        if invalid:
            raise ValueError(f"unsupported param-columns: {','.join(invalid)}")
    else:
        columns = list(df.select_dtypes(include=["number"]).columns)
    if not columns:
        raise ValueError("--params must contain at least one numeric column")

    values = df[columns].to_numpy(dtype=np.float64)
    if values.shape[0] != n_snapshots:
        raise ValueError(
            f"params row count mismatch: {values.shape[0]} vs snapshots {n_snapshots}"
        )
    return values, _build_parameter_contract(
        values,
        names=list(map(str, columns)),
        source=str(Path(params_path).expanduser()),
    )


def _build_parameter_contract(
    values: Any,
    *,
    names: list[str],
    source: str,
) -> dict[str, Any]:
    """학습 파라미터의 고객-facing 입력 contract를 만든다."""
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"params must be 2D to build contract, got shape {array.shape}")
    if len(names) != array.shape[1]:
        raise ValueError("parameter names must match parameter dimension")
    mins = np.min(array, axis=0)
    maxs = np.max(array, axis=0)
    ranges = list(
        map(
            lambda item: {
                "name": item[1],
                "observed_min": float(mins[item[0]]),
                "observed_max": float(maxs[item[0]]),
            },
            enumerate(names),
        )
    )
    return {
        "schema": "naviertwin-parameter-contract-v1",
        "source": source,
        "dim": int(array.shape[1]),
        "names": list(names),
        "ranges": ranges,
    }


def _load_build_twin_snapshots(
    *,
    input_path: str | None,
    csv_snapshots: str | None,
    field: str | None,
    field_column: str | None,
) -> tuple[Any, str, dict[str, object]]:
    """CFD reader 또는 CSV 시퀀스에서 snapshot 행렬을 로드한다."""
    import numpy as np

    if csv_snapshots is not None:
        from naviertwin.core.cfd_reader.csv_snapshots import load_csv_snapshots

        paths = _expand_csv_snapshot_paths(csv_snapshots)
        column = field_column or field
        if not column:
            raise ValueError("--field-column or --field is required with --csv-snapshots")
        snapshots, coords = load_csv_snapshots(paths, column=column)
        metadata: dict[str, object] = {
            "source": "csv-snapshots",
            "files": list(map(str, paths)),
            "field_column": column,
        }
        if coords is not None:
            metadata["coord_shape"] = list(coords.shape)
        return np.asarray(snapshots, dtype=np.float64), column, metadata

    if input_path is None:
        raise ValueError("--input or --csv-snapshots is required")

    from naviertwin.core.cfd_reader import ReaderFactory

    dataset = ReaderFactory.create_and_read(Path(input_path).expanduser())
    selected_field = field or (dataset.field_names[0] if dataset.field_names else "")
    if not selected_field:
        raise ValueError("--field is required when the CFD input has no named fields")
    snapshots = np.asarray(dataset.extract_field_snapshots(selected_field), dtype=np.float64)
    metadata = {
        "source": "cfd-reader",
        "input": str(input_path),
        "field": selected_field,
        "n_points": int(dataset.n_points),
        "n_cells": int(dataset.n_cells),
        "n_time_steps": int(dataset.n_time_steps),
    }
    return snapshots, selected_field, metadata


def _build_twin_payload(
    *,
    input_path: str | None,
    csv_snapshots: str | None,
    field: str | None,
    field_column: str | None,
    params: str | None,
    param_columns: str | None,
    outdir: str,
    reducer: str,
    n_modes: int,
    surrogate: str,
    validation_count: int,
) -> dict[str, Any]:
    """CFD/CSV dataset을 학습 가능한 디지털 트윈 산출물 payload로 변환한다."""
    from naviertwin.core.digital_twin.manifest import build_manifest, save_manifest
    from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline
    from naviertwin.core.digital_twin.pipeline_checkpoint import save_pipeline_state
    from naviertwin.core.digital_twin.twin_engine import TwinEngine

    snapshots, selected_field, source_meta = _load_build_twin_snapshots(
        input_path=input_path,
        csv_snapshots=csv_snapshots,
        field=field,
        field_column=field_column,
    )
    if snapshots.ndim != 2:
        raise ValueError(f"snapshots must be 2D, got shape {snapshots.shape}")
    n_features, n_snapshots = snapshots.shape
    if n_snapshots < 4:
        raise ValueError("build-twin requires at least 4 snapshots")
    if n_modes < 1:
        raise ValueError("n-modes must be positive")

    params_array, parameter_contract = _load_build_twin_params(
        params_path=params,
        param_columns=param_columns,
        n_snapshots=n_snapshots,
    )
    val_count = min(max(1, validation_count), max(1, n_snapshots // 3))
    train_count = n_snapshots - val_count
    if train_count < 3:
        raise ValueError("not enough training snapshots after validation split")

    train_snapshots = snapshots[:, :train_count]
    val_snapshots = snapshots[:, train_count:]
    train_params = params_array[:train_count]
    val_params = params_array[train_count:]

    pipe = NavierTwinPipeline(
        reducer_kind=reducer,
        n_modes=n_modes,
        surrogate_kind=surrogate,
    )
    pipe.load_snapshots(train_snapshots, field_name=selected_field)
    pipe.reduce()
    pipe.fit_surrogate(train_params)
    y_true = pipe.state.reducer.encode(val_snapshots)
    metrics = pipe.validate(val_params, y_true)

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    checkpoint_path = output_dir / "pipeline.h5"
    engine_path = output_dir / "engine.pkl"
    manifest_path = output_dir / "manifest.json"
    report_path = output_dir / "report.html"

    save_pipeline_state(pipe, checkpoint_path)
    engine = TwinEngine.from_fitted_components(pipe.state.reducer, pipe.state.surrogate)
    engine.save(engine_path)
    pipe.export_report(str(report_path), project="NavierTwin Build Twin")

    payload = {
        "status": "ok",
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "engine": str(engine_path),
            "manifest": str(manifest_path),
            "metrics": str(metrics_path),
            "report": str(report_path),
        },
        "source": source_meta,
        "field": selected_field,
        "training": {
            "reducer": reducer,
            "n_modes": int(n_modes),
            "surrogate": surrogate,
            "n_features": int(n_features),
            "n_snapshots": int(n_snapshots),
            "train_count": int(train_count),
            "validation_count": int(val_count),
            "param_dim": int(params_array.shape[1]),
            "parameter_contract": parameter_contract,
        },
        "metrics": metrics,
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    artifact_integrity = _artifact_integrity_map(
        {
            "metrics": metrics_path,
            "checkpoint": checkpoint_path,
            "engine": engine_path,
            "report": report_path,
        }
    )
    manifest = build_manifest(
        reducer=reducer,
        n_modes=n_modes,
        surrogate=surrogate,
        metrics=metrics,
        extra={
            "command": "build-twin",
            "manifest_schema": "naviertwin-build-twin-v1",
            "source": source_meta,
            "field": selected_field,
            "n_features": n_features,
            "n_snapshots": n_snapshots,
            "train_count": train_count,
            "validation_count": val_count,
            "param_dim": int(params_array.shape[1]),
            "parameter_contract": parameter_contract,
            "has_engine": True,
            "engine_path": str(engine_path),
            "artifact_integrity": artifact_integrity,
        },
    )
    save_manifest(manifest, manifest_path)
    return payload


def _run_build_twin(
    *,
    input_path: str | None,
    csv_snapshots: str | None,
    field: str | None,
    field_column: str | None,
    params: str | None,
    param_columns: str | None,
    outdir: str,
    reducer: str,
    n_modes: int,
    surrogate: str,
    validation_count: int,
    as_json: bool,
) -> int:
    """CFD/CSV dataset을 학습 가능한 디지털 트윈 산출물로 변환한다."""
    try:
        payload = _build_twin_payload(
            input_path=input_path,
            csv_snapshots=csv_snapshots,
            field=field,
            field_column=field_column,
            params=params,
            param_columns=param_columns,
            outdir=outdir,
            reducer=reducer,
            n_modes=n_modes,
            surrogate=surrogate,
            validation_count=validation_count,
        )
    except (ImportError, RuntimeError, OSError, ValueError, KeyError) as exc:
        print(f"build-twin error: {exc}", file=sys.stderr)
        return 2

    if as_json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        training = payload["training"]
        metrics = payload["metrics"]
        print(
            "build-twin 완료: "
            f"field={payload['field']}, snapshots={training['n_snapshots']}, "
            f"rmse={metrics.get('rmse', float('nan')):.6g}"
        )
        print(f"artifacts: {Path(outdir)}")
    return 0


def _artifact_integrity_map(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """산출물 파일의 크기와 SHA256을 manifest-friendly dict로 반환한다."""
    from naviertwin.utils.hashing import hash_file

    records: dict[str, dict[str, Any]] = {}
    path_items = list(paths.items())
    index = 0
    while index < len(path_items):
        name, path = path_items[index]
        index += 1
        if path.exists():
            records[name] = {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": hash_file(path),
            }
    return records


def _load_predict_twin_params(
    *,
    params: str | None,
    params_csv: str | None,
    param_columns: str | None,
) -> Any:
    """predict-twin 입력 파라미터를 numpy 배열로 로드한다."""
    import numpy as np

    if params is not None:
        values = _parse_csv_floats(params, label="params")
        return np.asarray(values, dtype=np.float64)

    if params_csv is None:
        raise ValueError("--params or --params-csv is required")
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas required to read --params-csv") from exc

    df = pd.read_csv(params_csv)
    if param_columns:
        columns = _csv_items(param_columns)
        invalid = sorted(set(columns) - set(map(str, df.columns)))
        if invalid:
            raise ValueError(f"unsupported param-columns: {','.join(invalid)}")
    else:
        columns = list(df.select_dtypes(include=["number"]).columns)
    if not columns:
        raise ValueError("--params-csv must contain at least one numeric column")
    return df[columns].to_numpy(dtype=np.float64)


def _parse_csv_floats(value: str, *, label: str) -> list[float]:
    """쉼표 구분 문자열을 float 목록으로 변환한다."""
    parsed: list[float] = []
    tokens = value.split(",")
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        stripped = token.strip()
        if not stripped:
            continue
        try:
            parsed.append(float(stripped))
        except ValueError as exc:
            raise ValueError(f"{label} must be comma-separated floats: {stripped}") from exc
    if not parsed:
        raise ValueError(f"{label} must include at least one value")
    return parsed


def _resolve_twin_engine_path(
    *,
    engine_path: str | None,
    artifacts_dir: str | None,
) -> Path:
    """engine.pkl 직접 경로 또는 산출물 디렉토리에서 TwinEngine 경로를 찾는다."""
    if engine_path:
        return Path(engine_path).expanduser()
    if artifacts_dir:
        root = Path(artifacts_dir).expanduser()
        engine_file = root / "engine.pkl"
        if not engine_file.exists():
            raise FileNotFoundError(f"missing engine.pkl in artifacts-dir: {root}")
        return engine_file
    raise ValueError("--engine or --artifacts-dir is required")


def _load_twin_parameter_contract(engine_file: Path) -> dict[str, Any] | None:
    """engine.pkl 옆 manifest.json에서 optional parameter contract를 읽는다."""
    manifest_path = engine_file.parent / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    extra = manifest.get("extra", {})
    if not isinstance(extra, dict):
        return None
    contract = extra.get("parameter_contract")
    return contract if isinstance(contract, dict) else None


def _check_twin_parameter_contract(
    params_array: Any,
    contract: dict[str, Any] | None,
) -> dict[str, Any]:
    """입력 파라미터 배열이 저장된 트윈 contract와 호환되는지 확인한다."""
    import numpy as np

    array = np.asarray(params_array, dtype=np.float64)
    if contract is None:
        return {
            "available": False,
            "passed": True,
            "warnings": ["parameter contract unavailable"],
        }
    if array.ndim == 1:
        input_dim = int(array.shape[0])
        batch = array.reshape(1, -1)
    elif array.ndim == 2:
        input_dim = int(array.shape[1])
        batch = array
    else:
        raise ValueError(f"params must be 1D or 2D, got shape {array.shape}")

    expected_dim = int(contract.get("dim", -1))
    names = list(map(str, contract.get("names", [])))
    source = str(contract.get("source", "unknown"))
    if input_dim != expected_dim:
        display_names = ", ".join(names) if names else "-"
        raise ValueError(
            "parameter dimension mismatch: "
            f"expected {expected_dim} from contract source={source} "
            f"names=[{display_names}], got {input_dim} with input shape {list(array.shape)}"
        )

    warnings: list[dict[str, Any]] = []
    ranges = contract.get("ranges", [])
    if isinstance(ranges, list):
        limited_ranges = ranges[:input_dim]
        index = 0
        while index < len(limited_ranges):
            record = limited_ranges[index]
            if not isinstance(record, dict):
                index += 1
                continue
            name = str(record.get("name", names[index] if index < len(names) else index))
            observed_min = float(record.get("observed_min", float("-inf")))
            observed_max = float(record.get("observed_max", float("inf")))
            actual_min = float(np.min(batch[:, index]))
            actual_max = float(np.max(batch[:, index]))
            if actual_min < observed_min or actual_max > observed_max:
                warnings.append(
                    {
                        "name": name,
                        "observed_min": observed_min,
                        "observed_max": observed_max,
                        "input_min": actual_min,
                        "input_max": actual_max,
                    }
                )
            index += 1

    return {
        "available": True,
        "passed": True,
        "expected_dim": expected_dim,
        "input_dim": input_dim,
        "source": source,
        "names": names,
        "warnings": warnings,
    }


def _run_predict_twin(
    *,
    engine_path: str | None,
    artifacts_dir: str | None = None,
    params: str | None,
    params_csv: str | None,
    param_columns: str | None,
    output: str | None,
    as_json: bool,
) -> int:
    """저장된 TwinEngine으로 입력 파라미터의 유동장 예측을 수행한다."""
    try:
        import numpy as np

        from naviertwin.core.digital_twin.twin_engine import TwinEngine

        engine_file = _resolve_twin_engine_path(
            engine_path=engine_path,
            artifacts_dir=artifacts_dir,
        )
        engine = TwinEngine.load(engine_file)
        params_array = _load_predict_twin_params(
            params=params,
            params_csv=params_csv,
            param_columns=param_columns,
        )
        parameter_contract = _load_twin_parameter_contract(engine_file)
        parameter_check = _check_twin_parameter_contract(params_array, parameter_contract)
        prediction = np.asarray(engine.predict(params_array), dtype=np.float64)
        output_path = Path(output).expanduser() if output else None
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            matrix = prediction.reshape(-1, 1) if prediction.ndim == 1 else prediction
            header = _sample_header(matrix.shape[1])
            np.savetxt(output_path, matrix, delimiter=",", header=header, comments="")

        preview_array = prediction.reshape(-1)[: min(8, prediction.size)]
        payload = {
            "status": "ok",
            "engine": str(engine_file),
            "artifacts_dir": str(Path(artifacts_dir).expanduser()) if artifacts_dir else None,
            "input_shape": list(params_array.shape),
            "prediction_shape": list(prediction.shape),
            "output": str(output_path) if output_path is not None else None,
            "parameter_contract": parameter_contract,
            "parameter_check": parameter_check,
            "preview": list(map(float, preview_array)),
        }
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        print(f"predict-twin error: {exc}", file=sys.stderr)
        return 2

    if as_json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(
            "predict-twin 완료: "
            f"input_shape={payload['input_shape']}, "
            f"prediction_shape={payload['prediction_shape']}"
        )
        if output_path is not None:
            print(f"output: {output_path}")
    return 0


def _benchmark_twin_payload(
    *,
    engine_path: str | None,
    artifacts_dir: str | None,
    params_array: Any,
    warmup: int,
    repeat: int,
    max_mean_ms: float | None = None,
    max_p50_ms: float | None = None,
    max_p95_ms: float | None = None,
    max_p99_ms: float | None = None,
    min_throughput_hz: float | None = None,
) -> dict[str, Any]:
    """저장된 TwinEngine 예측 latency를 측정하고 CLI/API 공통 payload를 만든다."""
    from time import perf_counter

    import numpy as np

    from naviertwin.core.digital_twin.twin_engine import TwinEngine

    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if repeat < 1:
        raise ValueError("repeat must be >= 1")

    engine_file = _resolve_twin_engine_path(
        engine_path=engine_path,
        artifacts_dir=artifacts_dir,
    )
    engine = TwinEngine.load(engine_file)
    params = np.asarray(params_array, dtype=np.float64)
    if params.ndim not in {1, 2}:
        raise ValueError(f"params must be 1D or 2D, got shape {params.shape}")
    parameter_contract = _load_twin_parameter_contract(engine_file)
    parameter_check = _check_twin_parameter_contract(params, parameter_contract)

    warmup_index = 0
    while warmup_index < warmup:
        engine.predict(params)
        warmup_index += 1

    samples_ms: list[float] = []
    prediction_shape: list[int] = []
    repeat_index = 0
    while repeat_index < repeat:
        started = perf_counter()
        prediction = np.asarray(engine.predict(params), dtype=np.float64)
        samples_ms.append(float((perf_counter() - started) * 1000.0))
        prediction_shape = list(prediction.shape)
        repeat_index += 1

    durations = np.asarray(samples_ms, dtype=np.float64)
    mean_ms = float(np.mean(durations))
    latency = {
        "min": float(np.min(durations)),
        "mean": mean_ms,
        "p50": float(np.percentile(durations, 50)),
        "p95": float(np.percentile(durations, 95)),
        "p99": float(np.percentile(durations, 99)),
        "max": float(np.max(durations)),
    }
    throughput_hz = float(1000.0 / mean_ms) if mean_ms > 0 else None
    acceptance = _benchmark_twin_acceptance(
        latency,
        throughput_hz=throughput_hz,
        max_mean_ms=max_mean_ms,
        max_p50_ms=max_p50_ms,
        max_p95_ms=max_p95_ms,
        max_p99_ms=max_p99_ms,
        min_throughput_hz=min_throughput_hz,
    )
    return {
        "status": "ok" if acceptance["passed"] else "failed",
        "engine": str(engine_file),
        "artifacts_dir": str(Path(artifacts_dir).expanduser()) if artifacts_dir else None,
        "input_shape": list(params.shape),
        "prediction_shape": prediction_shape,
        "warmup": int(warmup),
        "repeat": int(repeat),
        "latency_ms": latency,
        "samples_ms": samples_ms,
        "throughput_hz": throughput_hz,
        "parameter_contract": parameter_contract,
        "parameter_check": parameter_check,
        "acceptance": acceptance,
    }


def _run_benchmark_twin(
    *,
    engine_path: str | None,
    artifacts_dir: str | None = None,
    params: str | None,
    params_csv: str | None,
    param_columns: str | None,
    warmup: int,
    repeat: int,
    max_mean_ms: float | None = None,
    max_p50_ms: float | None = None,
    max_p95_ms: float | None = None,
    max_p99_ms: float | None = None,
    min_throughput_hz: float | None = None,
    output: str | None,
    as_json: bool,
) -> int:
    """저장된 TwinEngine의 반복 예측 latency를 측정한다."""
    try:
        params_array = _load_predict_twin_params(
            params=params,
            params_csv=params_csv,
            param_columns=param_columns,
        )
        payload = _benchmark_twin_payload(
            engine_path=engine_path,
            artifacts_dir=artifacts_dir,
            params_array=params_array,
            warmup=warmup,
            repeat=repeat,
            max_mean_ms=max_mean_ms,
            max_p50_ms=max_p50_ms,
            max_p95_ms=max_p95_ms,
            max_p99_ms=max_p99_ms,
            min_throughput_hz=min_throughput_hz,
        )
        output_path = Path(output).expanduser() if output else None
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        print(f"benchmark-twin error: {exc}", file=sys.stderr)
        return 2

    if as_json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        latency = payload["latency_ms"]
        acceptance = payload["acceptance"]
        print(
            "benchmark-twin 완료: "
            f"repeat={repeat}, p50={latency['p50']:.6g} ms, "
            f"p95={latency['p95']:.6g} ms"
        )
        if not acceptance["passed"]:
            print("acceptance: failed")
        if output_path is not None:
            print(f"output: {output_path}")
    return 0 if payload["acceptance"]["passed"] else 1


def _benchmark_twin_acceptance(
    latency_ms: dict[str, float],
    *,
    throughput_hz: float | None,
    max_mean_ms: float | None,
    max_p50_ms: float | None,
    max_p95_ms: float | None,
    max_p99_ms: float | None,
    min_throughput_hz: float | None,
) -> dict[str, Any]:
    """benchmark-twin SLO 설정을 pass/fail 결과로 변환한다."""
    checks: list[dict[str, Any]] = []
    latency_specs = [
        ("latency_ms.mean", "mean", max_mean_ms),
        ("latency_ms.p50", "p50", max_p50_ms),
        ("latency_ms.p95", "p95", max_p95_ms),
        ("latency_ms.p99", "p99", max_p99_ms),
    ]
    latency_index = 0
    while latency_index < len(latency_specs):
        metric, key, threshold = latency_specs[latency_index]
        latency_index += 1
        if threshold is None:
            continue
        value = float(latency_ms[key])
        checks.append(
            {
                "metric": metric,
                "op": "<=",
                "threshold": float(threshold),
                "value": value,
                "passed": bool(value <= threshold),
            }
        )

    if min_throughput_hz is not None:
        value = float(throughput_hz) if throughput_hz is not None else 0.0
        checks.append(
            {
                "metric": "throughput_hz",
                "op": ">=",
                "threshold": float(min_throughput_hz),
                "value": value,
                "passed": bool(value >= min_throughput_hz),
            }
        )

    return {
        "configured": bool(checks),
        "passed": _checks_passed(checks),
        "checks": checks,
    }


def _run_validate_twin(
    *,
    engine_path: str | None,
    artifacts_dir: str | None = None,
    input_path: str | None,
    csv_snapshots: str | None,
    field: str | None,
    field_column: str | None,
    params: str | None,
    param_columns: str | None,
    max_rmse: float | None = None,
    min_r2: float | None = None,
    max_relative_l2: float | None = None,
    output: str | None,
    as_json: bool,
) -> int:
    """저장된 TwinEngine 예측장을 기준 CFD/CSV snapshot과 비교 검증한다."""
    try:
        import numpy as np

        from naviertwin.core.digital_twin.twin_engine import TwinEngine
        from naviertwin.core.validation.metrics import compute_all_metrics

        engine_file = _resolve_twin_engine_path(
            engine_path=engine_path,
            artifacts_dir=artifacts_dir,
        )
        snapshots, selected_field, source_meta = _load_build_twin_snapshots(
            input_path=input_path,
            csv_snapshots=csv_snapshots,
            field=field,
            field_column=field_column,
        )
        truth = np.asarray(snapshots, dtype=np.float64)
        if truth.ndim != 2:
            raise ValueError(f"validation snapshots must be 2D, got shape {truth.shape}")

        n_features, n_snapshots = truth.shape
        params_array, _parameter_contract = _load_build_twin_params(
            params_path=params,
            param_columns=param_columns,
            n_snapshots=n_snapshots,
        )

        engine = TwinEngine.load(engine_file)
        prediction = _align_twin_prediction(engine.predict(params_array), truth.shape)
        metrics = compute_all_metrics(truth, prediction)
        per_sample_rmse = np.sqrt(np.mean((truth - prediction) ** 2, axis=0))
        worst_index = int(np.argmax(per_sample_rmse)) if per_sample_rmse.size else 0
        acceptance = _validate_twin_acceptance(
            metrics,
            max_rmse=max_rmse,
            min_r2=min_r2,
            max_relative_l2=max_relative_l2,
        )

        payload = {
            "status": "ok" if acceptance["passed"] else "failed",
            "engine": str(engine_file),
            "artifacts_dir": str(Path(artifacts_dir).expanduser()) if artifacts_dir else None,
            "field": selected_field,
            "source": source_meta,
            "acceptance": acceptance,
            "validation": {
                "n_features": int(n_features),
                "n_snapshots": int(n_snapshots),
                "param_dim": int(params_array.shape[1]),
                "truth_shape": list(truth.shape),
                "prediction_shape": list(prediction.shape),
                "worst_sample_index": worst_index,
                "worst_sample_rmse": float(per_sample_rmse[worst_index])
                if per_sample_rmse.size
                else 0.0,
            },
            "metrics": metrics,
        }
        output_path = Path(output).expanduser() if output else None
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
    except (ImportError, RuntimeError, OSError, ValueError, KeyError) as exc:
        print(f"validate-twin error: {exc}", file=sys.stderr)
        return 2

    if as_json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(
            "validate-twin 완료: "
            f"field={selected_field}, snapshots={n_snapshots}, "
            f"rmse={metrics.get('rmse', float('nan')):.6g}, "
            f"r2={metrics.get('r2', float('nan')):.6g}"
        )
        if not acceptance["passed"]:
            print("acceptance: failed")
        if output_path is not None:
            print(f"output: {output_path}")
    return 0 if acceptance["passed"] else 1


def _validate_twin_acceptance(
    metrics: dict[str, float],
    *,
    max_rmse: float | None,
    min_r2: float | None,
    max_relative_l2: float | None,
) -> dict[str, Any]:
    """validate-twin threshold 설정을 pass/fail 결과로 변환한다."""
    checks: list[dict[str, Any]] = []
    specs = [
        ("rmse", "<=", max_rmse),
        ("r2", ">=", min_r2),
        ("relative_l2", "<=", max_relative_l2),
    ]
    spec_index = 0
    while spec_index < len(specs):
        metric, op, threshold = specs[spec_index]
        spec_index += 1
        if threshold is None:
            continue
        value = float(metrics.get(metric, float("nan")))
        passed = value <= threshold if op == "<=" else value >= threshold
        checks.append(
            {
                "metric": metric,
                "op": op,
                "threshold": float(threshold),
                "value": value,
                "passed": bool(passed),
            }
        )
    return {
        "configured": bool(checks),
        "passed": _checks_passed(checks),
        "checks": checks,
    }


def _align_twin_prediction(prediction: Any, expected_shape: tuple[int, int]) -> Any:
    """TwinEngine 예측 결과를 snapshot 행렬 shape=(features, samples)에 맞춘다."""
    import numpy as np

    array = np.asarray(prediction, dtype=np.float64)
    if array.shape == expected_shape:
        return array
    if array.T.shape == expected_shape:
        return array.T
    if expected_shape[1] == 1 and array.reshape(-1).shape[0] == expected_shape[0]:
        return array.reshape(expected_shape)
    raise ValueError(
        f"prediction shape mismatch: got {array.shape}, expected {expected_shape}"
    )


def _package_twin_payload(
    *,
    artifacts_dir: str,
    output: str,
    include_validation: str | None,
    max_mean_ms: float | None = None,
    max_p50_ms: float | None = None,
    max_p95_ms: float | None = 100.0,
    max_p99_ms: float | None = None,
    min_throughput_hz: float | None = 10.0,
    no_latency_slo: bool = False,
) -> dict[str, Any]:
    """build-twin 산출물을 고객 전달 ZIP payload로 패키징한다."""
    from naviertwin.utils.workflow.artifact_zip import read_manifest, zip_artifacts

    root = Path(artifacts_dir).expanduser()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"artifacts-dir not found: {root}")

    source_integrity = _verify_twin_source_integrity(root)
    files = _collect_twin_package_artifacts(root, include_validation=include_validation)
    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latency_slo = None
    if not no_latency_slo:
        latency_slo = _build_latency_slo_policy(
            max_mean_ms=max_mean_ms,
            max_p50_ms=max_p50_ms,
            max_p95_ms=max_p95_ms,
            max_p99_ms=max_p99_ms,
            min_throughput_hz=min_throughput_hz,
            source="package-twin",
        )
    delivery_entries = _build_twin_delivery_entries(
        root,
        files=files,
        source_integrity=source_integrity,
        latency_slo=latency_slo,
    )
    zip_artifacts(files, output_path, extra_entries=delivery_entries)
    zip_manifest = read_manifest(output_path)
    return {
        "status": "ok",
        "output": str(output_path),
        "artifacts_dir": str(root),
        "files": list(map(lambda path: path.name, files)),
        "generated_entries": list(delivery_entries),
        "source_integrity": source_integrity,
        "latency_slo": latency_slo,
        "manifest_entries": zip_manifest,
    }


def _run_package_twin(
    *,
    artifacts_dir: str,
    output: str,
    include_validation: str | None,
    max_mean_ms: float | None = None,
    max_p50_ms: float | None = None,
    max_p95_ms: float | None = 100.0,
    max_p99_ms: float | None = None,
    min_throughput_hz: float | None = 10.0,
    no_latency_slo: bool = False,
    as_json: bool,
) -> int:
    """build-twin 산출물을 무결성 manifest가 포함된 ZIP으로 패키징한다."""
    try:
        payload = _package_twin_payload(
            artifacts_dir=artifacts_dir,
            output=output,
            include_validation=include_validation,
            max_mean_ms=max_mean_ms,
            max_p50_ms=max_p50_ms,
            max_p95_ms=max_p95_ms,
            max_p99_ms=max_p99_ms,
            min_throughput_hz=min_throughput_hz,
            no_latency_slo=no_latency_slo,
        )
    except (ImportError, RuntimeError, OSError, ValueError, KeyError) as exc:
        print(f"package-twin error: {exc}", file=sys.stderr)
        return 2

    if as_json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(
            "package-twin 완료: "
            f"files={len(payload['files'])}, output={payload['output']}"
        )
    return 0


def _build_twin_delivery_entries(
    root: Path,
    *,
    files: list[Path],
    source_integrity: dict[str, Any],
    latency_slo: dict[str, Any] | None,
) -> dict[str, str]:
    """고객 전달 ZIP 내부에 생성할 안내/요약 파일을 만든다."""
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    extra_meta = manifest.get("extra", {})
    if not isinstance(extra_meta, dict):
        extra_meta = {}
    parameter_contract = extra_meta.get("parameter_contract")
    if not isinstance(parameter_contract, dict):
        parameter_contract = None
    parameter_input_args = _parameter_input_args_from_contract(parameter_contract)
    latency_slo_args = _latency_slo_command_args(latency_slo)
    predict_command = _command_with_parts(
        "naviertwin predict-twin --artifacts-dir <extracted-dir>",
        parameter_input_args,
        "--output prediction.csv --json",
    )
    benchmark_command = _command_with_parts(
        "naviertwin benchmark-twin --artifacts-dir <extracted-dir>",
        parameter_input_args,
        "--warmup 2 --repeat 20",
        latency_slo_args,
        "--output latency.json --json",
    )
    validate_command = _command_with_parts(
        "naviertwin validate-twin --artifacts-dir <extracted-dir> "
        '--csv-snapshots "case/validation/*.csv" '
        "--field-column U --max-rmse 0.05 --min-r2 0.98 "
        "--output validation.json --json"
    )
    verify_command = _command_with_parts(
        "naviertwin verify-twin-package --package naviertwin-twin.zip "
        "--extract-to ./naviertwin-twin --json"
    )
    accept_command = _command_with_parts(
        "naviertwin accept-twin-package --package naviertwin-twin.zip",
        "--extract-to ./naviertwin-twin",
        latency_slo_args,
        "--output acceptance.json --summary-output acceptance.md --json",
    )
    sample_params_csv = _sample_params_csv_from_contract(parameter_contract)
    generated_entries = ["README.txt", "delivery.json"]
    if sample_params_csv is not None:
        generated_entries.append("sample_params.csv")
    delivery = {
        "format": "NavierTwin delivery package",
        "schema": "naviertwin-delivery-v1",
        "artifacts_dir": str(root),
        "files": list(map(lambda path: path.name, files)),
        "generated_entries": generated_entries,
        "source_integrity": source_integrity,
        "parameter_contract": parameter_contract,
        "latency_slo": latency_slo,
        "build_manifest": {
            "config": manifest.get("config", {}),
            "metrics": manifest.get("metrics", {}),
            "extra": _metadata_without_artifact_integrity(extra_meta),
        },
        "commands": {
            "accept_package": accept_command,
            "benchmark": benchmark_command,
            "predict": predict_command,
            "validate": validate_command,
            "verify_package": verify_command,
        },
    }
    if parameter_contract is None:
        contract_lines = [
            "Expected input parameters:",
            "- parameter contract unavailable (legacy artifact)",
        ]
    else:
        ranges = parameter_contract.get("ranges", [])
        range_lines: list[str] = []
        if isinstance(ranges, list):
            range_index = 0
            while range_index < len(ranges):
                record = ranges[range_index]
                range_index += 1
                if not isinstance(record, dict):
                    continue
                range_lines.append(
                    "- {name}: observed [{lo}, {hi}]".format(
                        name=record.get("name", "-"),
                        lo=record.get("observed_min", "-"),
                        hi=record.get("observed_max", "-"),
                    )
                )
        contract_lines = [
            "Expected input parameters:",
            f"- dimension: {parameter_contract.get('dim', '-')}",
            f"- names: {', '.join(map(str, parameter_contract.get('names', []))) or '-'}",
            *range_lines,
        ]
    slo_lines = _latency_slo_readme_lines(latency_slo)
    readme = "\n".join(
        [
            "NavierTwin Digital Twin Delivery Package",
            "========================================",
            "",
            "Included core artifacts:",
            "- engine.pkl: loadable parameter-to-field prediction TwinEngine",
            "- manifest.json: build configuration, metrics, and artifact integrity",
            "- metrics.json: build/validation metrics from build-twin",
            "- report.html: customer-readable build report",
            "- validation.json: optional independent validation report",
            "- sample_params.csv: generated example input parameters used by this twin",
            "",
            *contract_lines,
            "",
            *slo_lines,
            "",
            "Recommended checks:",
            "1. Run the one-command delivery acceptance smoke:",
            f"   {accept_command}",
            "2. Or verify and extract this ZIP manually:",
            f"   {verify_command}",
            "3. Run a prediction:",
            f"   {predict_command}",
            "4. Benchmark prediction latency:",
            f"   {benchmark_command}",
            "   Adjust --max-p95-ms/--min-throughput-hz to your deployment SLO.",
            "5. Validate against held-out snapshots:",
            f"   {validate_command}",
            "",
            "See delivery.json to read machine-readable package metadata.",
            "",
        ]
    )
    entries = {
        "README.txt": readme,
        "delivery.json": json.dumps(delivery, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    }
    if sample_params_csv is not None:
        entries["sample_params.csv"] = sample_params_csv
    return entries


def _command_with_parts(*parts: str | None) -> str:
    """비어 있지 않은 CLI 조각만 공백 하나로 이어 붙인다."""
    stripped = map(lambda part: part.strip() if part else "", parts)
    return " ".join(filter(None, stripped))


def _build_latency_slo_policy(
    *,
    max_mean_ms: float | None,
    max_p50_ms: float | None,
    max_p95_ms: float | None,
    max_p99_ms: float | None,
    min_throughput_hz: float | None,
    source: str,
) -> dict[str, Any] | None:
    """delivery metadata에 기록할 latency SLO 정책을 만든다."""
    raw_thresholds = {
        "max_mean_ms": max_mean_ms,
        "max_p50_ms": max_p50_ms,
        "max_p95_ms": max_p95_ms,
        "max_p99_ms": max_p99_ms,
        "min_throughput_hz": min_throughput_hz,
    }
    threshold_items = filter(lambda item: item[1] is not None, raw_thresholds.items())
    thresholds = dict(map(lambda item: (item[0], float(item[1])), threshold_items))
    if not thresholds:
        return None
    return {
        "schema": "naviertwin-latency-slo-v1",
        "source": source,
        "thresholds": thresholds,
    }


def _latency_slo_command_args(policy: dict[str, Any] | None) -> str:
    """latency SLO 정책을 benchmark/accept CLI 인자로 직렬화한다."""
    if not isinstance(policy, dict):
        return ""
    thresholds = policy.get("thresholds")
    if not isinstance(thresholds, dict):
        return ""
    flag_map = [
        ("max_mean_ms", "--max-mean-ms"),
        ("max_p50_ms", "--max-p50-ms"),
        ("max_p95_ms", "--max-p95-ms"),
        ("max_p99_ms", "--max-p99-ms"),
        ("min_throughput_hz", "--min-throughput-hz"),
    ]
    args: list[str] = []
    flag_index = 0
    while flag_index < len(flag_map):
        key, flag = flag_map[flag_index]
        flag_index += 1
        if key not in thresholds:
            continue
        args.extend([flag, f"{float(thresholds[key]):.12g}"])
    return " ".join(args)


def _latency_slo_readme_lines(policy: dict[str, Any] | None) -> list[str]:
    """delivery README에 넣을 latency SLO 설명을 만든다."""
    if not isinstance(policy, dict):
        return [
            "Recommended latency SLO:",
            "- not specified; pass benchmark/accept thresholds explicitly",
        ]
    thresholds = policy.get("thresholds")
    if not isinstance(thresholds, dict) or not thresholds:
        return [
            "Recommended latency SLO:",
            "- not specified; pass benchmark/accept thresholds explicitly",
        ]
    label_map = {
        "max_mean_ms": "mean latency <=",
        "max_p50_ms": "p50 latency <=",
        "max_p95_ms": "p95 latency <=",
        "max_p99_ms": "p99 latency <=",
        "min_throughput_hz": "throughput >=",
    }
    lines = ["Recommended latency SLO:"]
    label_keys = list(label_map)
    label_index = 0
    while label_index < len(label_keys):
        key = label_keys[label_index]
        label_index += 1
        if key not in thresholds:
            continue
        unit = "Hz" if key == "min_throughput_hz" else "ms"
        lines.append(f"- {label_map[key]} {float(thresholds[key]):.12g} {unit}")
    return lines


def _latency_slo_thresholds(policy: dict[str, Any] | None) -> dict[str, float | None]:
    """latency SLO 정책에서 benchmark gate 임계값만 추출한다."""
    values: dict[str, float | None] = {
        "max_mean_ms": None,
        "max_p50_ms": None,
        "max_p95_ms": None,
        "max_p99_ms": None,
        "min_throughput_hz": None,
    }
    if not isinstance(policy, dict):
        return values
    thresholds = policy.get("thresholds")
    if not isinstance(thresholds, dict):
        return values
    keys = list(values)
    key_index = 0
    while key_index < len(keys):
        key = keys[key_index]
        key_index += 1
        if key not in thresholds:
            continue
        try:
            values[key] = float(thresholds[key])
        except (TypeError, ValueError):
            continue
    return values


def _merge_latency_slo_policy(
    policy: dict[str, Any] | None,
    *,
    max_mean_ms: float | None,
    max_p50_ms: float | None,
    max_p95_ms: float | None,
    max_p99_ms: float | None,
    min_throughput_hz: float | None,
) -> dict[str, float | None]:
    """패키지 latency SLO 기본값에 명시 CLI 임계값을 덮어쓴다."""
    thresholds = _latency_slo_thresholds(policy)
    overrides = {
        "max_mean_ms": max_mean_ms,
        "max_p50_ms": max_p50_ms,
        "max_p95_ms": max_p95_ms,
        "max_p99_ms": max_p99_ms,
        "min_throughput_hz": min_throughput_hz,
    }
    override_items = list(overrides.items())
    override_index = 0
    while override_index < len(override_items):
        key, value = override_items[override_index]
        override_index += 1
        if value is not None:
            thresholds[key] = float(value)
    return thresholds


def _parameter_input_args_from_contract(contract: dict[str, Any] | None) -> str:
    """delivery command에 넣을 contract-aware 입력 인자 예시를 만든다."""
    if not isinstance(contract, dict):
        return "--params 0.25"
    names = _nonempty_strings(contract.get("names", []))
    try:
        dim = int(contract.get("dim", 0))
    except (TypeError, ValueError):
        dim = 0
    if dim > 0 and len(names) == dim and _sample_params_csv_from_contract(contract):
        return (
            "--params-csv <extracted-dir>/sample_params.csv "
            f"--param-columns {','.join(names)}"
        )
    return f"--params {_example_params_from_contract(contract)}"


def _sample_params_csv_from_contract(contract: dict[str, Any] | None) -> str | None:
    """parameter contract에서 1-row sample_params.csv 내용을 만든다."""
    if not isinstance(contract, dict):
        return None
    names = _nonempty_strings(contract.get("names", []))
    try:
        dim = int(contract.get("dim", 0))
    except (TypeError, ValueError):
        return None
    if dim <= 0 or len(names) != dim:
        return None
    return f"{','.join(names)}\n{_example_params_from_contract(contract)}\n"


def _example_params_from_contract(contract: dict[str, Any] | None) -> str:
    """parameter contract에서 copy-paste 가능한 --params 예시를 만든다."""
    if not isinstance(contract, dict):
        return "0.25"
    try:
        dim = int(contract.get("dim", 1))
    except (TypeError, ValueError):
        dim = 1
    ranges = contract.get("ranges", [])
    values: list[float] = []
    if isinstance(ranges, list):
        limited_ranges = ranges[:dim]
        range_index = 0
        while range_index < len(limited_ranges):
            record = limited_ranges[range_index]
            range_index += 1
            if not isinstance(record, dict):
                break
            try:
                observed_min = float(record["observed_min"])
                observed_max = float(record["observed_max"])
            except (KeyError, TypeError, ValueError):
                break
            values.append((observed_min + observed_max) / 2.0)
    if len(values) != dim:
        values = [0.25] * max(1, dim)
    return ",".join(map(lambda value: f"{value:.6g}", values))


def _verify_twin_source_integrity(root: Path) -> dict[str, Any]:
    """build-twin manifest에 기록된 artifact hash와 현재 파일을 대조한다."""
    from naviertwin.utils.hashing import hash_file

    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("missing required twin artifact: manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("extra", {}).get("artifact_integrity", {})
    if not isinstance(records, dict) or not records:
        return {"configured": False, "passed": True, "checks": []}

    checks: list[dict[str, Any]] = []
    record_items = list(records.items())
    record_index = 0
    while record_index < len(record_items):
        name, record = record_items[record_index]
        record_index += 1
        if not isinstance(record, dict):
            continue
        recorded_path = Path(str(record.get("path", name)))
        candidate = root / recorded_path.name
        if not candidate.exists():
            raise FileNotFoundError(f"missing manifest artifact: {candidate.name}")
        actual_bytes = int(candidate.stat().st_size)
        actual_sha256 = hash_file(candidate)
        expected_bytes = int(record.get("bytes", -1))
        expected_sha256 = str(record.get("sha256", ""))
        passed = actual_bytes == expected_bytes and actual_sha256 == expected_sha256
        checks.append(
            {
                "artifact": str(name),
                "path": str(candidate),
                "bytes": actual_bytes,
                "expected_bytes": expected_bytes,
                "sha256": actual_sha256,
                "expected_sha256": expected_sha256,
                "passed": passed,
            }
        )
        if not passed:
            raise ValueError(f"integrity mismatch: {candidate.name}")
    return {"configured": True, "passed": True, "checks": checks}


def _collect_twin_package_artifacts(
    root: Path,
    *,
    include_validation: str | None,
) -> list[Path]:
    """고객 전달 ZIP에 포함할 build-twin 산출물 목록을 결정한다."""
    required = ["engine.pkl", "manifest.json"]
    optional = ["metrics.json", "pipeline.h5", "report.html"]
    missing = list(filter(lambda name: not (root / name).exists(), required))
    if missing:
        raise FileNotFoundError(f"missing required twin artifact: {missing[0]}")

    artifact_names = filter(lambda name: (root / name).exists(), [*optional, *required])
    files = list(map(lambda name: root / name, artifact_names))
    validation_path = Path(include_validation).expanduser() if include_validation else root / "validation.json"
    if validation_path.exists():
        files.append(validation_path)
    return list(dict.fromkeys(map(Path.resolve, files)))


def _run_verify_twin_package(
    *,
    package_path: str,
    extract_to: str | None = None,
    as_json: bool,
) -> int:
    """package-twin ZIP의 MANIFEST.json 무결성 기록을 검증한다."""
    try:
        payload = _verify_twin_package_archive(
            Path(package_path).expanduser(),
            extract_to=Path(extract_to).expanduser() if extract_to else None,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"verify-twin-package error: {exc}", file=sys.stderr)
        return 2

    if as_json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(
            "verify-twin-package 완료: "
            f"status={payload['status']}, checks={len(payload['checks'])}"
        )
        if payload.get("extracted_to"):
            print(f"extracted_to: {payload['extracted_to']}")
        if payload["errors"]:
            print("errors: " + "; ".join(payload["errors"]))
    return 0 if payload["status"] == "ok" else 1


def _run_inspect_twin_package(*, package_path: str, as_json: bool) -> int:
    """package-twin ZIP의 구성/메타데이터를 읽기 전용으로 조회한다."""
    try:
        payload = _inspect_twin_package_archive(Path(package_path).expanduser())
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"inspect-twin-package error: {exc}", file=sys.stderr)
        return 2

    if as_json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(
            "inspect-twin-package 완료: "
            f"status={payload['status']}, entries={payload['manifest_entry_count']}, "
            f"format={payload.get('format') or '-'}"
        )
        metrics = payload.get("metrics") or {}
        if isinstance(metrics, dict) and metrics:
            rmse = metrics.get("rmse", "-")
            r2 = metrics.get("r2", "-")
            print(f"metrics: rmse={rmse}, r2={r2}")
        if payload["errors"]:
            print("errors: " + "; ".join(payload["errors"]))
    return 0 if payload["status"] == "ok" else 1


def _run_accept_twin_package(
    *,
    package_path: str,
    extract_to: str | None,
    prediction_output: str | None,
    warmup: int,
    repeat: int,
    max_mean_ms: float | None = None,
    max_p50_ms: float | None = None,
    max_p95_ms: float | None = None,
    max_p99_ms: float | None = None,
    min_throughput_hz: float | None = None,
    skip_benchmark: bool,
    output: str | None,
    summary_output: str | None,
    as_json: bool,
) -> int:
    """고객 전달 ZIP을 수락 가능한 디지털 트윈 패키지인지 원샷 점검한다."""
    try:
        import tempfile

        package = Path(package_path).expanduser()
        if extract_to:
            payload = _accept_twin_package_archive(
                package,
                extract_to=Path(extract_to).expanduser(),
                temporary_extraction=False,
                prediction_output=Path(prediction_output).expanduser()
                if prediction_output
                else None,
                warmup=warmup,
                repeat=repeat,
                max_mean_ms=max_mean_ms,
                max_p50_ms=max_p50_ms,
                max_p95_ms=max_p95_ms,
                max_p99_ms=max_p99_ms,
                min_throughput_hz=min_throughput_hz,
                skip_benchmark=skip_benchmark,
            )
        else:
            with tempfile.TemporaryDirectory(prefix="naviertwin-accept-") as tmp_raw:
                payload = _accept_twin_package_archive(
                    package,
                    extract_to=Path(tmp_raw) / "twin",
                    temporary_extraction=True,
                    prediction_output=Path(prediction_output).expanduser()
                    if prediction_output
                    else None,
                    warmup=warmup,
                    repeat=repeat,
                    max_mean_ms=max_mean_ms,
                    max_p50_ms=max_p50_ms,
                    max_p95_ms=max_p95_ms,
                    max_p99_ms=max_p99_ms,
                    min_throughput_hz=min_throughput_hz,
                    skip_benchmark=skip_benchmark,
                )

        summary_output_path = Path(summary_output).expanduser() if summary_output else None
        if summary_output_path is not None:
            payload["summary_output"] = str(summary_output_path)
            summary_output_path.parent.mkdir(parents=True, exist_ok=True)
            summary_output_path.write_text(
                _format_accept_twin_package_summary(payload),
                encoding="utf-8",
            )
        output_path = Path(output).expanduser() if output else None
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
    except (ImportError, RuntimeError, OSError, ValueError, KeyError) as exc:
        print(f"accept-twin-package error: {exc}", file=sys.stderr)
        return 2

    if as_json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(
            "accept-twin-package 완료: "
            f"status={payload['status']}, extracted_to={payload['extracted_to']}"
        )
        benchmark = payload.get("benchmark", {})
        if isinstance(benchmark, dict) and benchmark.get("latency_ms"):
            latency = benchmark["latency_ms"]
            print(
                "latency: "
                f"p50={latency['p50']:.6g} ms, p95={latency['p95']:.6g} ms"
            )
        if output_path is not None:
            print(f"output: {output_path}")
        if summary_output_path is not None:
            print(f"summary_output: {summary_output_path}")
    return 0 if payload["status"] == "ok" else 1


def _format_accept_twin_package_summary(payload: dict[str, Any]) -> str:
    """accept-twin-package 결과를 고객 검수용 Markdown으로 요약한다."""
    acceptance = payload.get("acceptance", {})
    prediction = payload.get("prediction") or {}
    benchmark = payload.get("benchmark") or {}
    latency = benchmark.get("latency_ms") if isinstance(benchmark, dict) else {}
    latency = latency if isinstance(latency, dict) else {}
    throughput_hz = benchmark.get("throughput_hz") if isinstance(benchmark, dict) else None
    benchmark_acceptance = benchmark.get("acceptance", {}) if isinstance(benchmark, dict) else {}
    checks = benchmark_acceptance.get("checks", []) if isinstance(benchmark_acceptance, dict) else []
    parameter_input = payload.get("parameter_input") or {}
    latency_slo = payload.get("latency_slo") or {}
    effective_slo = latency_slo.get("effective", {}) if isinstance(latency_slo, dict) else {}

    lines = [
        "# NavierTwin Package Acceptance Summary",
        "",
        f"- Status: {_summary_status(payload.get('status') == 'ok')}",
        f"- Package: {payload.get('package', '-')}",
        f"- Extracted to: {payload.get('extracted_to', '-')}",
        f"- Summary report: {payload.get('summary_output', '-')}",
        "",
        "## Gate Results",
        "",
        "| Gate | Result |",
        "| --- | --- |",
        f"| Package integrity and metadata | {_summary_status(_bool_from_mapping(acceptance, 'package'))} |",
        f"| Sample prediction | {_summary_status(_bool_from_mapping(acceptance, 'prediction'))} |",
        f"| Latency benchmark | {_summary_status(_bool_from_mapping(acceptance, 'benchmark'))} |",
        "",
        "## Prediction",
        "",
        f"- Parameter source: {parameter_input.get('source', '-')}",
        f"- Input shape: {prediction.get('input_shape', '-')}",
        f"- Prediction shape: {prediction.get('prediction_shape', '-')}",
    ]
    if prediction.get("output"):
        lines.append(f"- Prediction output: {prediction.get('output')}")

    lines.extend(
        [
            "",
            "## Latency",
            "",
            f"- Warmup: {benchmark.get('warmup', '-') if isinstance(benchmark, dict) else '-'}",
            f"- Repeat: {benchmark.get('repeat', '-') if isinstance(benchmark, dict) else '-'}",
            f"- p50 ms: {_summary_float(latency.get('p50'))}",
            f"- p95 ms: {_summary_float(latency.get('p95'))}",
            f"- p99 ms: {_summary_float(latency.get('p99'))}",
            f"- Throughput Hz: {_summary_float(throughput_hz)}",
            "",
            "## Effective SLO",
            "",
            "| Metric | Threshold |",
            "| --- | --- |",
        ]
    )
    slo_keys = ("max_mean_ms", "max_p50_ms", "max_p95_ms", "max_p99_ms", "min_throughput_hz")
    slo_index = 0
    while slo_index < len(slo_keys):
        key = slo_keys[slo_index]
        slo_index += 1
        value = effective_slo.get(key) if isinstance(effective_slo, dict) else None
        lines.append(f"| {key} | {_summary_float(value)} |")

    lines.extend(["", "## SLO Checks", "", "| Metric | Rule | Threshold | Value | Result |", "| --- | --- | --- | --- | --- |"])
    if checks:
        check_index = 0
        while check_index < len(checks):
            check = checks[check_index]
            check_index += 1
            if not isinstance(check, dict):
                continue
            lines.append(
                "| {metric} | {op} | {threshold} | {value} | {result} |".format(
                    metric=check.get("metric", "-"),
                    op=check.get("op", "-"),
                    threshold=_summary_float(check.get("threshold")),
                    value=_summary_float(check.get("value")),
                    result=_summary_status(bool(check.get("passed"))),
                )
            )
    else:
        lines.append("| - | - | - | - | No latency SLO configured |")

    errors = payload.get("verification", {}).get("errors", [])
    if isinstance(errors, list) and errors:
        lines.extend(["", "## Verification Errors", ""])
        lines.extend(map(lambda error: f"- {error}", errors))

    return "\n".join(lines) + "\n"


def _bool_from_mapping(mapping: Any, key: str) -> bool:
    """summary helper: dict-like 값에서 bool을 안전하게 읽는다."""
    return bool(mapping.get(key)) if isinstance(mapping, dict) else False


def _summary_status(passed: bool) -> str:
    """Markdown summary의 pass/fail 문자열을 통일한다."""
    return "PASS" if passed else "FAIL"


def _summary_float(value: Any) -> str:
    """Markdown summary에 넣을 float 값을 compact하게 표시한다."""
    if value is None:
        return "-"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _accept_twin_package_archive(
    package_path: Path,
    *,
    extract_to: Path,
    temporary_extraction: bool,
    prediction_output: Path | None,
    warmup: int,
    repeat: int,
    max_mean_ms: float | None,
    max_p50_ms: float | None,
    max_p95_ms: float | None,
    max_p99_ms: float | None,
    min_throughput_hz: float | None,
    skip_benchmark: bool,
) -> dict[str, Any]:
    """전달 ZIP의 검증, 샘플 예측, latency SLO를 단일 payload로 묶는다."""
    if warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if repeat < 1:
        raise ValueError("--repeat must be >= 1")

    verification = _verify_twin_package_archive(package_path)
    inspection = _inspect_twin_package_archive(package_path)
    package_latency_slo = inspection.get("latency_slo")
    if not isinstance(package_latency_slo, dict):
        package_latency_slo = None
    effective_latency_slo = _merge_latency_slo_policy(
        package_latency_slo,
        max_mean_ms=max_mean_ms,
        max_p50_ms=max_p50_ms,
        max_p95_ms=max_p95_ms,
        max_p99_ms=max_p99_ms,
        min_throughput_hz=min_throughput_hz,
    )
    package_passed = verification["status"] == "ok" and inspection["status"] == "ok"
    payload: dict[str, Any] = {
        "status": "failed",
        "package": str(package_path),
        "extracted_to": str(extract_to),
        "temporary_extraction": temporary_extraction,
        "verification": verification,
        "inspection": inspection,
        "latency_slo": {
            "package": package_latency_slo,
            "effective": effective_latency_slo,
        },
        "parameter_input": None,
        "prediction": None,
        "benchmark": None,
        "acceptance": {
            "package": package_passed,
            "prediction": False,
            "benchmark": False,
            "passed": False,
        },
    }
    if not package_passed:
        return payload

    verification = _verify_twin_package_archive(package_path, extract_to=extract_to)
    payload["verification"] = verification

    from time import perf_counter

    import numpy as np

    from naviertwin.core.digital_twin.twin_engine import TwinEngine

    engine_file = _resolve_twin_engine_path(engine_path=None, artifacts_dir=str(extract_to))
    engine = TwinEngine.load(engine_file)
    parameter_contract = _load_twin_parameter_contract(engine_file)
    if parameter_contract is None:
        raw_contract = inspection.get("parameter_contract")
        parameter_contract = raw_contract if isinstance(raw_contract, dict) else None
    params_array, parameter_input = _accept_twin_package_params(
        extract_to,
        contract=parameter_contract,
    )
    parameter_check = _check_twin_parameter_contract(params_array, parameter_contract)
    prediction = np.asarray(engine.predict(params_array), dtype=np.float64)
    prediction_path = prediction_output
    if prediction_path is not None:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        matrix = prediction.reshape(-1, 1) if prediction.ndim == 1 else prediction
        header = _sample_header(matrix.shape[1])
        np.savetxt(prediction_path, matrix, delimiter=",", header=header, comments="")

    preview_array = prediction.reshape(-1)[: min(8, prediction.size)]
    payload["parameter_input"] = parameter_input
    payload["prediction"] = {
        "engine": str(engine_file),
        "input_shape": list(params_array.shape),
        "prediction_shape": list(prediction.shape),
        "output": str(prediction_path) if prediction_path is not None else None,
        "parameter_contract": parameter_contract,
        "parameter_check": parameter_check,
        "preview": list(map(float, preview_array)),
    }
    payload["acceptance"]["prediction"] = True

    if skip_benchmark:
        benchmark_acceptance = {
            "configured": False,
            "passed": True,
            "checks": [],
            "skipped": True,
        }
        payload["benchmark"] = {
            "skipped": True,
            "warmup": int(warmup),
            "repeat": int(repeat),
            "acceptance": benchmark_acceptance,
        }
    else:
        warmup_index = 0
        while warmup_index < warmup:
            engine.predict(params_array)
            warmup_index += 1

        durations_ms: list[float] = []
        benchmark_prediction_shape: list[int] = []
        repeat_index = 0
        while repeat_index < repeat:
            started = perf_counter()
            benchmark_prediction = np.asarray(engine.predict(params_array), dtype=np.float64)
            durations_ms.append(float((perf_counter() - started) * 1000.0))
            benchmark_prediction_shape = list(benchmark_prediction.shape)
            repeat_index += 1

        durations = np.asarray(durations_ms, dtype=np.float64)
        mean_ms = float(np.mean(durations))
        latency = {
            "min": float(np.min(durations)),
            "mean": mean_ms,
            "p50": float(np.percentile(durations, 50)),
            "p95": float(np.percentile(durations, 95)),
            "p99": float(np.percentile(durations, 99)),
            "max": float(np.max(durations)),
        }
        throughput_hz = float(1000.0 / mean_ms) if mean_ms > 0 else None
        benchmark_acceptance = _benchmark_twin_acceptance(
            latency,
            throughput_hz=throughput_hz,
            max_mean_ms=effective_latency_slo.get("max_mean_ms"),
            max_p50_ms=effective_latency_slo.get("max_p50_ms"),
            max_p95_ms=effective_latency_slo.get("max_p95_ms"),
            max_p99_ms=effective_latency_slo.get("max_p99_ms"),
            min_throughput_hz=effective_latency_slo.get("min_throughput_hz"),
        )
        payload["benchmark"] = {
            "skipped": False,
            "input_shape": list(params_array.shape),
            "prediction_shape": benchmark_prediction_shape,
            "warmup": int(warmup),
            "repeat": int(repeat),
            "latency_ms": latency,
            "samples_ms": durations_ms,
            "throughput_hz": throughput_hz,
            "parameter_check": parameter_check,
            "acceptance": benchmark_acceptance,
        }
    payload["acceptance"]["benchmark"] = bool(benchmark_acceptance["passed"])
    payload["acceptance"]["passed"] = all(
        map(lambda name: bool(payload["acceptance"][name]), ("package", "prediction", "benchmark"))
    )
    payload["status"] = "ok" if payload["acceptance"]["passed"] else "failed"
    return payload


def _accept_twin_package_params(
    artifacts_dir: Path,
    *,
    contract: dict[str, Any] | None,
) -> tuple[Any, dict[str, Any]]:
    """전달 패키지 내부의 sample_params.csv 또는 contract 예시값을 입력으로 선택한다."""
    sample_params = artifacts_dir / "sample_params.csv"
    param_columns = _param_columns_from_contract(contract)
    if sample_params.exists():
        params_array = _load_predict_twin_params(
            params=None,
            params_csv=str(sample_params),
            param_columns=param_columns,
        )
        return params_array, {
            "source": "sample_params.csv",
            "path": str(sample_params),
            "param_columns": param_columns,
        }

    params = _example_params_from_contract(contract)
    params_array = _load_predict_twin_params(
        params=params,
        params_csv=None,
        param_columns=None,
    )
    return params_array, {
        "source": "parameter_contract_example",
        "params": params,
        "param_columns": None,
    }


def _param_columns_from_contract(contract: dict[str, Any] | None) -> str | None:
    """계약에 명확한 파라미터 이름이 있으면 CSV 컬럼 인자로 변환한다."""
    if not isinstance(contract, dict):
        return None
    names = _nonempty_strings(contract.get("names", []))
    try:
        dim = int(contract.get("dim", 0))
    except (TypeError, ValueError):
        return None
    return ",".join(names) if dim > 0 and len(names) == dim else None


def _is_safe_zip_member_name(name: str) -> bool:
    """ZIP member 이름이 대상 디렉토리 밖을 가리키지 않는지 검사한다."""
    from pathlib import PurePosixPath

    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def _extract_verified_twin_package(package_path: Path, extract_to: Path) -> list[str]:
    """검증 완료된 ZIP을 path traversal 없이 추출한다."""
    import tempfile
    import zipfile

    root = extract_to.resolve()
    if root.exists() and not root.is_dir():
        raise ValueError(f"extract target must be a directory: {root}")
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"extract target must be empty: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)

    extracted: list[str] = []
    with tempfile.TemporaryDirectory(prefix=f".{root.name}.", dir=root.parent) as staging_raw:
        staging = Path(staging_raw).resolve()
        with zipfile.ZipFile(package_path) as archive:
            archive_infos = archive.infolist()
            archive_index = 0
            while archive_index < len(archive_infos):
                info = archive_infos[archive_index]
                archive_index += 1
                name = info.filename
                if info.is_dir():
                    continue
                if not _is_safe_zip_member_name(name):
                    raise ValueError(f"unsafe archive entry: {name}")
                target = (staging / name).resolve()
                if staging != target and staging not in target.parents:
                    raise ValueError(f"unsafe archive entry: {name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(info))
                extracted.append(name)
        if root.exists():
            root.rmdir()
        staging.rename(root)
    return extracted


def _verify_twin_package_archive(
    package_path: Path,
    *,
    extract_to: Path | None = None,
) -> dict[str, Any]:
    """ZIP 내부 MANIFEST.json의 bytes/SHA256과 실제 archive entry를 대조한다."""
    import zipfile
    from collections import Counter
    from hashlib import sha256

    if not package_path.exists():
        raise FileNotFoundError(f"package not found: {package_path}")

    required_entries = {"engine.pkl", "manifest.json"}
    errors: list[str] = []
    checks: list[dict[str, Any]] = []
    duplicate_archive_entries: list[str] = []
    duplicate_manifest_entries: list[str] = []
    with zipfile.ZipFile(package_path) as archive:
        archived_names = archive.namelist()
        names = set(archived_names)
        archive_counts = sorted(Counter(archived_names).items())
        archive_count_index = 0
        while archive_count_index < len(archive_counts):
            name, count = archive_counts[archive_count_index]
            archive_count_index += 1
            if count > 1:
                duplicate_archive_entries.append(name)
                errors.append(f"duplicate archive entry: {name}")
        sorted_names = sorted(names)
        sorted_name_index = 0
        while sorted_name_index < len(sorted_names):
            name = sorted_names[sorted_name_index]
            sorted_name_index += 1
            if not _is_safe_zip_member_name(name):
                errors.append(f"unsafe archive entry: {name}")
        if "MANIFEST.json" not in names:
            errors.append("missing MANIFEST.json")
            manifest_entries: list[Any] = []
        else:
            manifest_entries = json.loads(archive.read("MANIFEST.json").decode("utf-8"))
            if not isinstance(manifest_entries, list):
                raise ValueError("MANIFEST.json must contain a list")

        manifest_names: set[str] = set()
        manifest_entry_index = 0
        while manifest_entry_index < len(manifest_entries):
            entry = manifest_entries[manifest_entry_index]
            manifest_entry_index += 1
            if not isinstance(entry, dict):
                errors.append("invalid MANIFEST.json entry")
                continue
            name = str(entry.get("name", ""))
            if name in manifest_names:
                duplicate_manifest_entries.append(name)
                errors.append(f"duplicate MANIFEST.json entry: {name}")
            manifest_names.add(name)
            if name not in names:
                errors.append(f"missing archived file: {name}")
                checks.append({"name": name, "passed": False, "error": "missing"})
                continue
            data = archive.read(name)
            actual_bytes = len(data)
            actual_sha256 = sha256(data).hexdigest()
            expected_bytes = int(entry.get("bytes", -1))
            expected_sha256 = str(entry.get("sha256", ""))
            passed = actual_bytes == expected_bytes and actual_sha256 == expected_sha256
            if not passed:
                errors.append(f"integrity mismatch: {name}")
            checks.append(
                {
                    "name": name,
                    "bytes": actual_bytes,
                    "expected_bytes": expected_bytes,
                    "sha256": actual_sha256,
                    "expected_sha256": expected_sha256,
                    "passed": passed,
                }
            )

    extra_names = sorted((names - {"MANIFEST.json"}) - manifest_names)
    extra_name_index = 0
    while extra_name_index < len(extra_names):
        name = extra_names[extra_name_index]
        extra_name_index += 1
        errors.append(f"unmanifested archive entry: {name}")
    missing_required_names = sorted(required_entries - manifest_names)
    missing_required_index = 0
    while missing_required_index < len(missing_required_names):
        name = missing_required_names[missing_required_index]
        missing_required_index += 1
        errors.append(f"missing required artifact in MANIFEST.json: {name}")
    passed = not errors and all(map(lambda check: check.get("passed") is True, checks))
    payload = {
        "status": "ok" if passed else "failed",
        "package": str(package_path),
        "required_entries": sorted(required_entries),
        "manifest_entry_count": len(checks),
        "duplicate_archive_entries": duplicate_archive_entries,
        "duplicate_manifest_entries": duplicate_manifest_entries,
        "checks": checks,
        "errors": errors,
    }
    if extract_to is not None:
        payload["extracted_to"] = str(extract_to)
        payload["extracted_entries"] = (
            _extract_verified_twin_package(package_path, extract_to) if passed else []
        )
    return payload


def _inspect_twin_package_archive(package_path: Path) -> dict[str, Any]:
    """트윈 전달 ZIP의 검증 상태와 delivery metadata를 요약한다."""
    import zipfile

    verification = _verify_twin_package_archive(package_path)
    errors = list(verification.get("errors", []))
    with zipfile.ZipFile(package_path) as archive:
        names = set(archive.namelist())
        if "MANIFEST.json" in names:
            manifest_entries = json.loads(archive.read("MANIFEST.json").decode("utf-8"))
            if not isinstance(manifest_entries, list):
                manifest_entries = []
        else:
            manifest_entries = []

        delivery: dict[str, Any] | None = None
        manifest_contract: dict[str, Any] | None = None
        if "manifest.json" in names:
            try:
                build_manifest_raw = json.loads(archive.read("manifest.json").decode("utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"invalid manifest.json: {exc.msg}")
            else:
                if isinstance(build_manifest_raw, dict):
                    manifest_extra = build_manifest_raw.get("extra", {})
                    if isinstance(manifest_extra, dict):
                        raw_contract = manifest_extra.get("parameter_contract")
                        if isinstance(raw_contract, dict):
                            manifest_contract = raw_contract
                else:
                    errors.append("manifest.json must contain an object")
        if "delivery.json" in names:
            try:
                delivery_raw = json.loads(archive.read("delivery.json").decode("utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"invalid delivery.json: {exc.msg}")
            else:
                if isinstance(delivery_raw, dict):
                    delivery = delivery_raw
                else:
                    errors.append("delivery.json must contain an object")

    build_manifest = delivery.get("build_manifest", {}) if delivery else {}
    if not isinstance(build_manifest, dict):
        build_manifest = {}
    metrics = build_manifest.get("metrics", {})
    config = build_manifest.get("config", {})
    extra = build_manifest.get("extra", {})
    if not isinstance(extra, dict):
        extra = {}
    delivery_contract = delivery.get("parameter_contract") if delivery else None
    if not isinstance(delivery_contract, dict):
        delivery_contract = extra.get("parameter_contract")
    if isinstance(delivery_contract, dict) and manifest_contract and delivery_contract != manifest_contract:
        errors.append("delivery.json parameter_contract differs from manifest.json")
    parameter_contract = manifest_contract
    if parameter_contract is None and isinstance(delivery_contract, dict):
        parameter_contract = delivery_contract
    latency_slo = delivery.get("latency_slo") if delivery else None
    if latency_slo is not None and not isinstance(latency_slo, dict):
        errors.append("delivery.json latency_slo must contain an object")
        latency_slo = None
    commands = delivery.get("commands", {}) if delivery else {}
    files = delivery.get("files", []) if delivery else []
    generated_entries = delivery.get("generated_entries", []) if delivery else []
    payload = {
        "status": "ok" if verification.get("status") == "ok" and not errors else "failed",
        "package": str(package_path),
        "format": delivery.get("format") if delivery else None,
        "schema": delivery.get("schema") if delivery else None,
        "delivery_metadata_present": delivery is not None,
        "manifest_entry_count": len(manifest_entries) or verification.get("manifest_entry_count", 0),
        "entries": manifest_entries,
        "files": files if isinstance(files, list) else [],
        "generated_entries": generated_entries if isinstance(generated_entries, list) else [],
        "commands": commands if isinstance(commands, dict) else {},
        "metrics": metrics if isinstance(metrics, dict) else {},
        "config": config if isinstance(config, dict) else {},
        "parameter_contract": parameter_contract,
        "latency_slo": latency_slo,
        "validation_included": "validation.json" in names,
        "readme_present": "README.txt" in names,
        "verification": verification,
        "errors": errors,
    }
    return payload


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
    acceptance_json: str | None = None,
    acceptance_summary: str | None = None,
) -> int:
    """고객 지원용 진단 번들을 생성한다."""
    from naviertwin.utils.support_bundle import build_support_bundle, report_to_json

    try:
        report = build_support_bundle(
            outdir=outdir,
            preflight=preflight,
            include_optional=include_optional,
            zip_bundle=zip_bundle,
            acceptance_json=acceptance_json,
            acceptance_summary=acceptance_summary,
        )
    except (OSError, ValueError) as exc:
        print(f"support-bundle error: {exc}", file=sys.stderr)
        return 2
    print(report_to_json(report))
    return 0


def _run_inspect_support_bundle(*, path: str, as_json: bool) -> int:
    """기존 고객 지원 번들을 read-only로 점검한다."""
    from naviertwin.utils.support_bundle import (
        format_support_bundle_inspection,
        inspect_support_bundle,
        report_to_json,
    )

    try:
        report = inspect_support_bundle(path)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"inspect-support-bundle error: {exc}", file=sys.stderr)
        return 2
    print(report_to_json(report) if as_json else format_support_bundle_inspection(report))
    return 0 if report.get("status") == "ok" else 1


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


def _run_update_check(
    *,
    metadata: str,
    channel: str,
    current_version: str,
    verify_artifact: str | None = None,
    trusted_public_keys: dict[str, str] | None = None,
) -> int:
    """로컬 릴리스 메타데이터를 사용해 업데이트 가능 여부를 출력한다."""
    from pathlib import Path

    from naviertwin.utils.updater import (
        check_for_update,
        load_release_metadata,
        verify_release_artifact,
    )

    try:
        result = check_for_update(
            Path(metadata),
            channel=channel,
            current_version=current_version,
            trusted_public_keys=trusted_public_keys,
        )
        payload = result.to_dict()
        exit_code = 0
        if verify_artifact:
            release = load_release_metadata(
                Path(metadata),
                trusted_public_keys=trusted_public_keys,
            )
            verification = verify_release_artifact(
                Path(verify_artifact),
                expected_sha256=release.sha256,
                installer_signing=release.installer_signing,
            )
            payload["artifact_verification"] = verification.to_dict()
            if not verification.verified:
                exit_code = 3
            authenticode = verification.authenticode or {}
            if (
                authenticode.get("authenticode_required") is True
                and authenticode.get("checked") is True
                and authenticode.get("status") != "verified"
            ):
                exit_code = 4
    except (OSError, ValueError) as exc:
        print(f"update-check error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return exit_code


def _run_feature_pack(args: argparse.Namespace) -> int:
    """Feature Pack 상태 조회, 다운로드, 설치를 처리한다."""
    from pathlib import Path

    from naviertwin.utils.feature_packs import (
        all_feature_pack_statuses,
        download_feature_pack,
        install_feature_pack_archive,
    )

    action = getattr(args, "feature_pack_command", None)
    if action in (None, "list"):
        payload = {
            "status": "ok",
            "feature_packs": all_feature_pack_statuses(),
        }
        if getattr(args, "as_json", False):
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        else:
            feature_packs = payload["feature_packs"]
            pack_index = 0
            while pack_index < len(feature_packs):
                item = feature_packs[pack_index]
                pack_index += 1
                state = "installed" if item["installed"] else "missing"
                missing = ",".join(item["missing_modules"]) or "-"
                print(f"{item['id']}: {state} | missing={missing} | {item['download_url']}")
        return 0

    try:
        if action == "install":
            payload = install_feature_pack_archive(
                args.archive,
                expected_pack_id=args.pack,
                expected_sha256=args.sha256,
            )
        elif action == "download":
            archive = download_feature_pack(
                args.pack,
                url=args.url,
                output_dir=Path(args.output_dir) if args.output_dir else None,
                expected_sha256=args.sha256,
            )
            payload = {
                "status": "ok",
                "archive": str(archive),
            }
            if args.install:
                payload["install"] = install_feature_pack_archive(
                    archive,
                    expected_pack_id=args.pack,
                    expected_sha256=args.sha256,
                )
        else:
            raise ValueError(f"unknown feature-pack action: {action}")
    except (OSError, ValueError, KeyError) as exc:
        print(f"feature-pack error: {exc}", file=sys.stderr)
        return 2

    if getattr(args, "as_json", False):
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
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
