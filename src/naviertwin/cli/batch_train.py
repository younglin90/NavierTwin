"""헤드리스 MPI 배치 트윈 학습 CLI (v5.6 P1+).

여러 트윈 학습 잡(JSON config)을 클러스터/멀티프로세스에서 라운드로빈으로
분배해 실행한다. 학습 로직은 전부 :mod:`naviertwin.web.service` 를 재사용하고,
이 모듈은 잡 분배·실행·결과 저장 오케스트레이션만 담당한다.

설계 결정 (.omc/plans/twin-platform-roadmap.md §4.4½):

- **MPI는 헤드리스 CLI 전용** — GUI(PySide6/trame) 이벤트 루프에서 mpi4py 를
  초기화하면 Qt/OpenMPI 시그널 핸들러가 충돌할 수 있으므로 금지한다.
- **mpi4py 없이도 동작** — import 실패 또는 ``mpirun`` 없이 단독 실행되면
  rank 0 / size 1 순차 실행으로 우아하게 강등된다.

Config 스키마::

    {
      "output_dir": "선택 — 결과 JSON 기본 루트",
      "jobs": [
        {
          "name": "job-a",            # 필수, 파일명-안전 문자열
          "kind": "rom" | "physics",  # 필수
          "demo": "advecting",        # 선택 (기본 advecting)
          "data_path": "...",         # 선택 — 주면 demo 대신 파일 로드
          "field": "p",               # 선택 (기본 p)
          "n_modes": 5,               # rom 전용 (기본 5)
          "epochs": 20,               # physics 전용 (기본 20)
          "output_dir": "..."         # 선택 (기본 <root>/<name>)
        }
      ]
    }

Examples:
    단독(순차) 실행::

        $ naviertwin batch-train --config jobs.json

    MPI 4-way 분산 실행::

        $ mpirun -n 4 naviertwin batch-train --config jobs.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

#: 지원하는 잡 종류 — service.build_twin / service.build_physics_ai_twin 매핑.
JOB_KINDS = ("rom", "physics")

#: 잡 이름에 허용하지 않는 문자 (출력 경로/파일명으로 그대로 쓰인다).
_UNSAFE_NAME_CHARS = ("/", "\\", "..", "\0")


def detect_mpi() -> tuple[int, int, Any]:
    """mpi4py ``COMM_WORLD`` 의 rank/size 를 감지한다.

    mpi4py 가 없거나 MPI 런타임 초기화에 실패하면 rank 0 / size 1 폴백으로
    우아하게 강등된다 — ``mpirun`` 없이 단독 실행해도 항상 동작한다.

    Returns:
        ``(rank, size, comm)``. 폴백 시 ``comm`` 은 ``None``.
    """
    try:
        from mpi4py import MPI  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 — 초기화 실패도 폴백 대상
        logger.info("mpi4py 사용 불가 (%s) — rank 0 / size 1 순차 실행으로 강등.", exc)
        return 0, 1, None
    comm = MPI.COMM_WORLD
    return int(comm.Get_rank()), int(comm.Get_size()), comm


def select_jobs(jobs: list[dict[str, Any]], rank: int, size: int) -> list[dict[str, Any]]:
    """``jobs[rank::size]`` 라운드로빈으로 이 rank 가 실행할 잡을 고른다.

    Args:
        jobs: 전체 잡 목록.
        rank: 이 프로세스의 MPI rank (0-based).
        size: 전체 프로세스 수.

    Returns:
        이 rank 에 배정된 잡 목록 (원본 순서 유지).

    Raises:
        ValueError: ``size < 1`` 이거나 ``rank`` 가 ``[0, size)`` 밖인 경우.
    """
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    if not 0 <= rank < size:
        raise ValueError(f"rank must be in [0, {size}), got {rank}")
    return jobs[rank::size]


def _validate_job(job: Any, index: int) -> dict[str, Any]:
    """잡 항목 하나를 검증하고 dict 로 돌려준다.

    Raises:
        ValueError: 필수 키 누락, 잘못된 kind/name/demo/숫자 값.
    """
    if not isinstance(job, dict):
        raise ValueError(f"jobs[{index}] must be an object, got {type(job).__name__}")

    name = job.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"jobs[{index}].name must be a non-empty string")
    if any(token in name for token in _UNSAFE_NAME_CHARS):
        raise ValueError(f"jobs[{index}].name contains unsafe path characters: {name!r}")

    kind = job.get("kind")
    if kind not in JOB_KINDS:
        raise ValueError(
            f"jobs[{index}].kind must be one of {list(JOB_KINDS)}, got {kind!r}"
        )

    if "data_path" not in job:
        from naviertwin.web.service import DEMO_TIME_SERIES_KINDS  # noqa: PLC0415

        demo = job.get("demo", "advecting")
        if demo not in DEMO_TIME_SERIES_KINDS:
            raise ValueError(
                f"jobs[{index}].demo must be one of {list(DEMO_TIME_SERIES_KINDS)}, "
                f"got {demo!r}"
            )

    for key in ("n_modes", "epochs", "nx", "ny", "n_steps", "hidden"):
        if key in job:
            value = job[key]
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"jobs[{index}].{key} must be a positive integer, got {value!r}")
    return job


def load_config(config_path: Path) -> dict[str, Any]:
    """배치 config JSON 을 읽고 스키마를 검증한다.

    Args:
        config_path: config JSON 파일 경로.

    Returns:
        검증된 config dict.

    Raises:
        FileNotFoundError: config 파일이 없는 경우.
        ValueError: JSON 파싱 실패 또는 스키마 위반.
    """
    config_path = Path(config_path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"batch config not found: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"batch config is not valid JSON: {config_path} ({exc})") from exc

    if not isinstance(config, dict):
        raise ValueError(f"batch config root must be an object, got {type(config).__name__}")
    jobs = config.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('batch config must contain a non-empty "jobs" list')
    for index, job in enumerate(jobs):
        _validate_job(job, index)
    return config


def _load_job_dataset(job: dict[str, Any]) -> Any:
    """잡 정의에서 데이터셋을 로드한다 (demo 합성 또는 파일)."""
    from naviertwin.web import service  # noqa: PLC0415

    data_path = job.get("data_path")
    if data_path:
        return service.load_dataset(Path(str(data_path)).expanduser())
    return service.make_demo_dataset(
        nx=int(job.get("nx", 32)),
        ny=int(job.get("ny", 32)),
        n_steps=int(job.get("n_steps", 8)),
        kind=str(job.get("demo", "advecting")),
    )


def _run_single_job(job: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """잡 하나를 실행하고 요약 dict 를 만든다 (예외는 요약에 흡수).

    Args:
        job: 검증된 잡 정의.
        output_dir: 이 잡의 산출물(engine.pkl) 디렉토리.

    Returns:
        ``name``/``kind``/``status``/``elapsed_s`` 와 kind 별 지표
        (rom: ``rmse``, physics: ``train_loss``·``rmse``)를 담은 요약.
    """
    import numpy as np  # noqa: PLC0415

    from naviertwin.web import service  # noqa: PLC0415

    name = str(job["name"])
    kind = str(job["kind"])
    field = str(job.get("field", "p"))
    summary: dict[str, Any] = {"name": name, "kind": kind, "field": field}
    started = time.perf_counter()
    try:
        dataset = _load_job_dataset(job)
        if kind == "rom":
            result = service.build_twin(
                dataset,
                field,
                int(job.get("n_modes", 5)),
                reducer=str(job.get("reducer", "pod")),
                surrogate=str(job.get("surrogate", "rbf")),
            )
            engine = result["engine"]
            # 학습 파라미터 전 지점 재구성 RMSE — 배치 리포트용 단일 품질 지표.
            snapshots = dataset.extract_field_snapshots(field)
            errors = [
                service.predict_twin(engine, param) - snapshots[:, step]
                for step, param in enumerate(result["params"])
            ]
            summary["rmse"] = float(np.sqrt(np.mean(np.square(np.asarray(errors)))))
            summary["n_modes"] = int(result["n_modes"])
        else:  # physics
            result = service.build_physics_ai_twin(
                dataset,
                field,
                hidden=int(job.get("hidden", 32)),
                max_epochs=int(job.get("epochs", 20)),
                max_train_points=int(job.get("max_train_points", 20_000)),
            )
            engine = result["engine"]
            losses = result.get("train_losses") or []
            if losses:
                summary["train_loss"] = float(losses[-1])
            validation = result.get("validation_metrics") or {}
            if "rmse" in validation:
                summary["rmse"] = float(validation["rmse"])

        output_dir.mkdir(parents=True, exist_ok=True)
        summary["engine_path"] = service.save_engine(engine, output_dir / "engine.pkl")
        summary["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 — 잡 하나가 배치 전체를 죽이면 안 된다
        logger.exception("배치 잡 실패: %s", name)
        summary["status"] = "error"
        summary["error"] = f"{type(exc).__name__}: {exc}"
    summary["elapsed_s"] = round(time.perf_counter() - started, 3)
    return summary


def run_batch(
    config_path: Path,
    *,
    rank: int | None = None,
    size: int | None = None,
) -> dict[str, Any]:
    """JSON config 의 트윈 학습 잡들을 이 rank 몫만큼 실행한다.

    ``rank``/``size`` 를 명시하면 (테스트/디버깅용) MPI 감지를 완전히 건너뛴다
    — mpi4py import 자체가 일어나지 않으므로 GUI-비의존 테스트에 안전하다.
    둘 다 ``None`` 이면 :func:`detect_mpi` 로 감지하고, mpi4py 가 있으면 rank 0
    이 전체 결과를 gather 해 ``batch_results.json`` 으로 통합 저장한다.

    Args:
        config_path: 배치 config JSON 경로.
        rank: 강제 rank (기본 ``None`` = MPI 자동 감지).
        size: 강제 size (기본 ``None`` = MPI 자동 감지).

    Returns:
        ``rank``/``size``/``results``(이 rank 잡 요약 목록)/``results_path``
        (rank 별 결과 JSON 경로)를 담은 dict.

    Raises:
        FileNotFoundError: config 파일이 없는 경우.
        ValueError: config 스키마 위반 또는 rank/size 값이 잘못된 경우.
    """
    config_path = Path(config_path).expanduser()
    config = load_config(config_path)

    comm: Any = None
    if rank is None and size is None:
        rank, size, comm = detect_mpi()
    elif rank is None or size is None:
        raise ValueError("rank and size must be given together (or both omitted)")

    jobs: list[dict[str, Any]] = list(config["jobs"])
    assigned = select_jobs(jobs, rank, size)
    logger.info(
        "batch-train 시작: rank %d/%d, 전체 %d개 중 %d개 잡 배정.",
        rank, size, len(jobs), len(assigned),
    )

    root = Path(str(config.get("output_dir") or config_path.parent / "batch_output")).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for job in assigned:
        job_dir = Path(str(job.get("output_dir") or root / str(job["name"]))).expanduser()
        summary = _run_single_job(job, job_dir)
        summary["output_dir"] = str(job_dir)
        logger.info(
            "잡 완료: %s (%s) status=%s elapsed=%.3fs",
            summary["name"], summary["kind"], summary["status"], summary["elapsed_s"],
        )
        results.append(summary)

    results_path = root / f"batch_results_rank{rank}.json"
    payload = {"rank": rank, "size": size, "n_jobs_total": len(jobs), "results": results}
    results_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    # mpi4py 가 실제로 있을 때만 rank 0 이 전체 결과를 통합 저장한다 (선택 기능).
    if comm is not None and size > 1:
        gathered = comm.gather(results, root=0)
        if rank == 0 and gathered is not None:
            merged = [item for chunk in gathered for item in chunk]
            merged_path = root / "batch_results.json"
            merged_path.write_text(
                json.dumps(
                    {"size": size, "n_jobs_total": len(jobs), "results": merged},
                    ensure_ascii=False, sort_keys=True, indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            logger.info("통합 결과 저장: %s (%d개 잡)", merged_path, len(merged))

    return {
        "rank": rank,
        "size": size,
        "n_jobs_total": len(jobs),
        "results": results,
        "results_path": str(results_path),
    }
