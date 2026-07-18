"""v5.6 P1+ — 헤드리스 MPI 배치 학습 CLI (naviertwin.cli.batch_train).

demo 데이터셋은 nx=ny=16, n_steps=5 스모크 크기로 제한해 전체 파일이 3분
이내에 끝나도록 유지한다. rank/size 는 주입값으로 검증하므로 mpi4py import
없이(순차 폴백 경로) 테스트된다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from naviertwin.cli.batch_train import load_config, run_batch, select_jobs


def _write_config(path: Path, payload: dict) -> Path:
    config_path = path / "jobs.json"
    config_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return config_path


def _smoke_job(name: str = "rom-a", kind: str = "rom", **extra) -> dict:
    job = {
        "name": name,
        "kind": kind,
        "demo": "advecting",
        "field": "p",
        "nx": 16,
        "ny": 16,
        "n_steps": 5,
    }
    if kind == "rom":
        job["n_modes"] = 3
    else:
        job.update({"epochs": 3, "hidden": 8, "max_train_points": 512})
    job.update(extra)
    return job


class TestSelectJobs:
    """(c) jobs[rank::size] 라운드로빈 분배 — 가짜 rank/size 주입."""

    def test_round_robin_partition(self) -> None:
        jobs = [{"name": f"j{i}"} for i in range(5)]

        assert [j["name"] for j in select_jobs(jobs, 0, 2)] == ["j0", "j2", "j4"]
        assert [j["name"] for j in select_jobs(jobs, 1, 2)] == ["j1", "j3"]

    def test_partition_is_complete_and_disjoint(self) -> None:
        jobs = [{"name": f"j{i}"} for i in range(7)]
        size = 3

        chunks = [select_jobs(jobs, rank, size) for rank in range(size)]
        names = [j["name"] for chunk in chunks for j in chunk]

        assert sorted(names) == sorted(j["name"] for j in jobs)
        assert len(names) == len(set(names))

    def test_single_process_gets_everything(self) -> None:
        jobs = [{"name": "a"}, {"name": "b"}]

        assert select_jobs(jobs, 0, 1) == jobs

    def test_more_ranks_than_jobs(self) -> None:
        jobs = [{"name": "only"}]

        assert select_jobs(jobs, 0, 4) == jobs
        assert select_jobs(jobs, 3, 4) == []

    def test_invalid_rank_size(self) -> None:
        with pytest.raises(ValueError, match="size"):
            select_jobs([], 0, 0)
        with pytest.raises(ValueError, match="rank"):
            select_jobs([], 2, 2)
        with pytest.raises(ValueError, match="rank"):
            select_jobs([], -1, 2)


class TestConfigValidation:
    """(b) 잘못된 config 는 명확한 ValueError."""

    def test_missing_file(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="batch config not found"):
            run_batch(tmp_path / "nope.json", rank=0, size=1)

    def test_invalid_json(self, tmp_path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")

        with pytest.raises(ValueError, match="not valid JSON"):
            load_config(bad)

    def test_root_must_be_object(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="root must be an object"):
            load_config(_write_config(tmp_path, []))  # type: ignore[arg-type]

    def test_jobs_required_and_non_empty(self, tmp_path) -> None:
        with pytest.raises(ValueError, match='"jobs"'):
            load_config(_write_config(tmp_path, {}))
        with pytest.raises(ValueError, match='"jobs"'):
            load_config(_write_config(tmp_path, {"jobs": []}))

    def test_job_requires_name_and_kind(self, tmp_path) -> None:
        with pytest.raises(ValueError, match=r"jobs\[0\]\.name"):
            load_config(_write_config(tmp_path, {"jobs": [{"kind": "rom"}]}))
        with pytest.raises(ValueError, match=r"jobs\[0\]\.kind"):
            load_config(_write_config(tmp_path, {"jobs": [{"name": "a", "kind": "lstm"}]}))

    def test_job_rejects_unknown_demo_and_bad_ints(self, tmp_path) -> None:
        with pytest.raises(ValueError, match=r"jobs\[0\]\.demo"):
            load_config(
                _write_config(tmp_path, {"jobs": [{"name": "a", "kind": "rom", "demo": "??"}]})
            )
        with pytest.raises(ValueError, match=r"jobs\[0\]\.n_modes"):
            load_config(
                _write_config(
                    tmp_path, {"jobs": [{"name": "a", "kind": "rom", "n_modes": 0}]}
                )
            )

    def test_job_name_must_be_path_safe(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="unsafe path"):
            load_config(
                _write_config(tmp_path, {"jobs": [{"name": "../evil", "kind": "rom"}]})
            )

    def test_rank_and_size_must_come_together(self, tmp_path) -> None:
        config = _write_config(tmp_path, {"jobs": [_smoke_job()]})

        with pytest.raises(ValueError, match="together"):
            run_batch(config, rank=0, size=None)


class TestRunBatch:
    """(a) demo 1잡 실행 → 결과 JSON 파일 검증 (스모크 크기)."""

    def test_single_rom_job_writes_results_json(self, tmp_path) -> None:
        outdir = tmp_path / "out"
        config = _write_config(
            tmp_path,
            {"output_dir": str(outdir), "jobs": [_smoke_job(output_dir=str(outdir / "rom-a"))]},
        )

        payload = run_batch(config, rank=0, size=1)

        assert payload["rank"] == 0
        assert payload["size"] == 1
        assert payload["n_jobs_total"] == 1

        results_path = Path(payload["results_path"])
        assert results_path == outdir / "batch_results_rank0.json"
        assert results_path.exists()

        saved = json.loads(results_path.read_text(encoding="utf-8"))
        assert saved["rank"] == 0
        (summary,) = saved["results"]
        assert summary["name"] == "rom-a"
        assert summary["kind"] == "rom"
        assert summary["status"] == "ok"
        assert summary["rmse"] >= 0.0
        assert summary["elapsed_s"] >= 0.0
        assert Path(summary["engine_path"]).exists()

    def test_physics_job_smoke(self, tmp_path) -> None:
        outdir = tmp_path / "out"
        config = _write_config(
            tmp_path,
            {
                "output_dir": str(outdir),
                "jobs": [_smoke_job(name="phys-a", kind="physics", nx=12, ny=12, n_steps=4)],
            },
        )

        payload = run_batch(config, rank=0, size=1)

        (summary,) = payload["results"]
        assert summary["status"] == "ok"
        assert summary["train_loss"] >= 0.0
        assert Path(summary["engine_path"]).exists()

    def test_injected_rank_runs_only_its_share(self, tmp_path) -> None:
        outdir = tmp_path / "out"
        jobs = [_smoke_job(name=f"rom-{i}") for i in range(3)]
        config = _write_config(tmp_path, {"output_dir": str(outdir), "jobs": jobs})

        payload = run_batch(config, rank=1, size=2)

        assert [r["name"] for r in payload["results"]] == ["rom-1"]
        assert (outdir / "batch_results_rank1.json").exists()
        assert not (outdir / "batch_results_rank0.json").exists()

    def test_failing_job_is_reported_not_raised(self, tmp_path) -> None:
        outdir = tmp_path / "out"
        # 존재하지 않는 field → 잡은 실패하지만 배치는 요약을 기록하고 계속된다.
        config = _write_config(
            tmp_path,
            {"output_dir": str(outdir), "jobs": [_smoke_job(field="no_such_field")]},
        )

        payload = run_batch(config, rank=0, size=1)

        (summary,) = payload["results"]
        assert summary["status"] == "error"
        assert "error" in summary
        assert (outdir / "batch_results_rank0.json").exists()


class TestCLIRegistration:
    """(d) argparse 서브커맨드 등록."""

    def test_batch_train_subcommand_parses(self, tmp_path) -> None:
        from naviertwin.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            ["batch-train", "--config", str(tmp_path / "jobs.json"), "--json"]
        )

        assert args.command == "batch-train"
        assert args.batch_config == str(tmp_path / "jobs.json")
        assert args.as_json is True

    def test_batch_train_requires_config(self) -> None:
        from naviertwin.main import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["batch-train"])

    def test_run_batch_train_reports_config_error(self, tmp_path, capsys) -> None:
        from naviertwin.main import _run_batch_train

        code = _run_batch_train(config=str(tmp_path / "missing.json"), as_json=False)

        assert code == 2
        assert "batch-train error" in capsys.readouterr().err
