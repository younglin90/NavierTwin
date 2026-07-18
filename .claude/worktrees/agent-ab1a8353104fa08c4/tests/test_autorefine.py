"""Round 603 — roadmap autorefine loop."""

from __future__ import annotations


class TestAutoRefine:
    def test_parse_and_apply(self, tmp_path) -> None:
        from naviertwin.utils.workflow.autorefine import (
            apply_auto_completion,
            parse_unchecked_tasks,
        )

        src = tmp_path / "src" / "naviertwin"
        src.mkdir(parents=True)
        (src / "main.py").write_text("x=1\n", encoding="utf-8")
        op = src / "core" / "operator_learning" / "fno"
        op.mkdir(parents=True)
        (op / "adaptive_fno.py").write_text("x=1\n", encoding="utf-8")

        roadmap = (
            "- [ ] `src/naviertwin/main.py` 완료 표기 필요\n"
            "- [ ] `fno/adaptive_fno.py` 경로 단축 표기\n"
            "- [ ] `src/naviertwin/missing.py` 아직 없음\n"
            "- [ ] 일반 작업 항목\n"
        )
        tasks = parse_unchecked_tasks(roadmap, tmp_path)
        assert len(tasks) == 4
        assert tasks[0].auto_completable is True
        assert tasks[1].auto_completable is True
        assert tasks[1].resolved_path is not None
        assert tasks[1].resolved_path.replace("\\", "/") == (
            "src/naviertwin/core/operator_learning/fno/adaptive_fno.py"
        )
        assert tasks[2].auto_completable is False
        assert tasks[3].auto_completable is False

        updated, applied = apply_auto_completion(roadmap, tasks)
        assert applied == 2
        assert updated.splitlines()[0].startswith("- [x]")
        assert updated.splitlines()[1].startswith("- [x]")
        assert updated.splitlines()[2].startswith("- [ ]")

    def test_run_once_writes_report(self, tmp_path) -> None:
        from naviertwin.utils.workflow.autorefine import run_autorefine_once

        src = tmp_path / "src" / "naviertwin"
        src.mkdir(parents=True)
        (src / "main.py").write_text("x=1\n", encoding="utf-8")
        roadmap_path = tmp_path / "ROADMAP.md"
        roadmap_path.write_text("- [ ] `src/naviertwin/main.py` 확인\n", encoding="utf-8")

        report = run_autorefine_once(project_root=tmp_path, apply=True, iteration=1)
        assert report["applied_count"] == 1
        assert "latest.json" in {p.name for p in (tmp_path / "verify_artifacts" / "autorefine").iterdir()}
        assert "- [x]" in roadmap_path.read_text(encoding="utf-8")
