"""ROADMAP 기반 자동 고도화 루프.

미완료 체크박스(`- [ ]`)를 분석하고, 경로가 실제로 존재하는 항목은
자동으로 완료 처리(`- [x]`)할 수 있다.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from naviertwin.utils.logger import get_logger

_UNCHECKED_RE = re.compile(r"^\s*-\s\[\s\]\s+(?P<body>.+?)\s*$")
_CHECKBOX_RE = re.compile(r"(\s*-\s)\[\s\]")
_BACKTICK_RE = re.compile(r"`([^`]+)`")


@dataclass(frozen=True)
class RoadmapTask:
    """ROADMAP의 단일 미완료 태스크."""

    line_no: int
    body: str
    rel_path: str | None
    resolved_path: str | None
    path_exists: bool

    @property
    def auto_completable(self) -> bool:
        """파일 경로가 존재해 자동 완료 후보인지 여부."""
        return bool(self.rel_path and self.path_exists)


def _candidate_paths(project_root: Path, token: str) -> list[Path]:
    clean = token.strip().lstrip("./")
    candidates = [
        project_root / clean,
        project_root / "src" / clean,
        project_root / "src" / "naviertwin" / clean,
        project_root / "src" / "naviertwin" / "core" / clean,
        project_root / "src" / "naviertwin" / "core" / "operator_learning" / clean,
    ]
    uniq: list[Path] = []
    seen: set[str] = set()
    candidate_idx = 0
    while candidate_idx < len(candidates):
        c = candidates[candidate_idx]
        key = str(c)
        if key in seen:
            candidate_idx += 1
            continue
        uniq.append(c)
        seen.add(key)
        candidate_idx += 1
    return uniq


def _extract_rel_path(task_body: str, project_root: Path) -> tuple[str | None, str | None, bool]:
    """태스크 본문에서 경로를 추출하고 실제 파일 존재 여부를 계산한다."""
    tokens = _BACKTICK_RE.findall(task_body)
    token_idx = 0
    while token_idx < len(tokens):
        token = tokens[token_idx]
        if "/" not in token and not token.endswith(".py") and not token.endswith(".md"):
            token_idx += 1
            continue
        candidates = _candidate_paths(project_root, token)
        candidate_idx = 0
        while candidate_idx < len(candidates):
            cand = candidates[candidate_idx]
            if cand.exists():
                return token.strip(), str(cand.relative_to(project_root)), True
            candidate_idx += 1
        return token.strip(), None, False
    return None, None, False


def parse_unchecked_tasks(roadmap_text: str, project_root: Path) -> list[RoadmapTask]:
    """미완료 태스크를 파싱한다."""
    tasks: list[RoadmapTask] = []
    lines = roadmap_text.splitlines()
    line_idx = 0
    while line_idx < len(lines):
        idx = line_idx + 1
        line = lines[line_idx]
        m = _UNCHECKED_RE.match(line)
        if not m:
            line_idx += 1
            continue
        body = m.group("body")
        rel, resolved, exists = _extract_rel_path(body, project_root)
        tasks.append(
            RoadmapTask(
                line_no=idx,
                body=body,
                rel_path=rel,
                resolved_path=resolved,
                path_exists=exists,
            )
        )
        line_idx += 1
    return tasks


def apply_auto_completion(roadmap_text: str, tasks: list[RoadmapTask]) -> tuple[str, int]:
    """자동 완료 가능한 태스크를 `[x]`로 반영한다."""
    line_numbers: set[int] = set()
    task_idx = 0
    while task_idx < len(tasks):
        task = tasks[task_idx]
        if task.auto_completable:
            line_numbers.add(task.line_no)
        task_idx += 1
    if not line_numbers:
        return roadmap_text, 0

    updated_count = 0
    out_lines: list[str] = []
    lines = roadmap_text.splitlines()
    line_idx = 0
    while line_idx < len(lines):
        idx = line_idx + 1
        line = lines[line_idx]
        if idx in line_numbers:
            replaced, n = _CHECKBOX_RE.subn(r"\1[x]", line, count=1)
            if n > 0:
                out_lines.append(replaced)
                updated_count += 1
                line_idx += 1
                continue
        out_lines.append(line)
        line_idx += 1
    return "\n".join(out_lines) + ("\n" if roadmap_text.endswith("\n") else ""), updated_count


def _build_report(
    *,
    tasks: list[RoadmapTask],
    applied: int,
    iteration: int,
    project_root: Path,
) -> dict[str, Any]:
    pending = len(tasks)
    auto_candidates = 0
    blocked = []
    top_pending = []
    task_idx = 0
    while task_idx < len(tasks):
        task = tasks[task_idx]
        if task.auto_completable:
            auto_candidates += 1
        if task.rel_path and not task.path_exists:
            blocked.append(task)
        if task_idx < 10:
            top_pending.append(task.body)
        task_idx += 1
    blocked_examples = []
    blocked_idx = 0
    while blocked_idx < min(len(blocked), 10):
        blocked_examples.append(asdict(blocked[blocked_idx]))
        blocked_idx += 1
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "iteration": iteration,
        "project_root": str(project_root),
        "pending_count": pending,
        "auto_candidate_count": auto_candidates,
        "applied_count": applied,
        "blocked_count": len(blocked),
        "top_pending": top_pending,
        "blocked_examples": blocked_examples,
    }


def _write_report_files(report: dict[str, Any], artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    latest_json = artifact_dir / "latest.json"
    latest_md = artifact_dir / "latest.md"
    hist_json = artifact_dir / f"run_{stamp}.json"
    hist_md = artifact_dir / f"run_{stamp}.md"

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    latest_json.write_text(payload, encoding="utf-8")
    hist_json.write_text(payload, encoding="utf-8")

    lines = [
        "# AutoRefine Report",
        "",
        f"- timestamp_utc: {report['timestamp_utc']}",
        f"- iteration: {report['iteration']}",
        f"- pending_count: {report['pending_count']}",
        f"- auto_candidate_count: {report['auto_candidate_count']}",
        f"- applied_count: {report['applied_count']}",
        f"- blocked_count: {report['blocked_count']}",
        "",
        "## Top Pending",
    ]
    top_pending = report["top_pending"]
    item_idx = 0
    while item_idx < len(top_pending):
        lines.append(f"- {top_pending[item_idx]}")
        item_idx += 1
    md = "\n".join(lines) + "\n"
    latest_md.write_text(md, encoding="utf-8")
    hist_md.write_text(md, encoding="utf-8")


def run_autorefine_once(
    *,
    project_root: str | Path,
    apply: bool = True,
    artifact_dir: str | Path | None = None,
    iteration: int = 1,
) -> dict[str, Any]:
    """ROADMAP를 1회 분석하고(옵션) 자동 완료를 반영한다."""
    logger = get_logger(__name__)
    root = Path(project_root).resolve()
    roadmap_path = root / "ROADMAP.md"
    if not roadmap_path.exists():
        raise FileNotFoundError(f"ROADMAP.md를 찾을 수 없습니다: {roadmap_path}")

    text = roadmap_path.read_text(encoding="utf-8")
    tasks = parse_unchecked_tasks(text, root)
    new_text, applied_count = apply_auto_completion(text, tasks) if apply else (text, 0)

    if apply and applied_count > 0 and new_text != text:
        roadmap_path.write_text(new_text, encoding="utf-8")
        logger.info("ROADMAP 자동 완료 반영: %d건", applied_count)
    else:
        logger.info("ROADMAP 자동 완료 반영 없음")

    report = _build_report(
        tasks=parse_unchecked_tasks(new_text, root),
        applied=applied_count,
        iteration=iteration,
        project_root=root,
    )

    out_dir = Path(artifact_dir).resolve() if artifact_dir else root / "verify_artifacts" / "autorefine"
    _write_report_files(report, out_dir)
    logger.info("AutoRefine 리포트 저장: %s", out_dir)
    return report


def run_autorefine_loop(
    *,
    project_root: str | Path,
    interval_sec: int = 60,
    iterations: int = 1,
    apply: bool = True,
    artifact_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """주기적으로 ROADMAP 자동 고도화 루프를 실행한다."""
    if interval_sec < 1:
        raise ValueError("interval_sec은 1 이상이어야 합니다.")
    if iterations < 0:
        raise ValueError("iterations는 0 이상이어야 합니다. (0=무한)")

    reports: list[dict[str, Any]] = []
    i = 1
    while True:
        reports.append(
            run_autorefine_once(
                project_root=project_root,
                apply=apply,
                artifact_dir=artifact_dir,
                iteration=i,
            )
        )
        if iterations > 0 and i >= iterations:
            break
        i += 1
        time.sleep(interval_sec)
    return reports


__all__ = [
    "RoadmapTask",
    "apply_auto_completion",
    "parse_unchecked_tasks",
    "run_autorefine_loop",
    "run_autorefine_once",
]
