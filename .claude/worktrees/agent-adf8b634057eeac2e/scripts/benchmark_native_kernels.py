#!/usr/bin/env python3
"""Benchmark native CFD kernels against their NumPy fallbacks.

The script is intentionally not a pytest performance gate.  It records local
latency baselines so native speed work can be compared without making normal CI
fail on noisy timing.
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from naviertwin._native import HAS_NATIVE_KERNELS, _kernels
from naviertwin.core.analysis.lambda2 import _lambda2_2d_numpy
from naviertwin.core.analysis.velocity_gradient import _field_J_2d_numpy
from naviertwin.core.flow_analysis.vortex.q_criterion import (
    _compute_lambda2_from_gradient_numpy,
    _compute_q_from_gradient_numpy,
)

GridCase = tuple[tuple[int, int], int]

SIZES: dict[str, GridCase] = {
    "small": ((32, 32), 2048),
    "medium": ((128, 128), 32768),
    "large": ((384, 384), 262144),
}

SEEDS: dict[str, int] = {
    "small": 101,
    "medium": 202,
    "large": 303,
}


def _measure(fn: Callable[[], Any], *, warmup: int, repeat: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()

    samples: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)

    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def _speedup(fallback: dict[str, Any], native: dict[str, Any] | None) -> float | None:
    if native is None or native["median_ms"] == 0.0:
        return None
    return fallback["median_ms"] / native["median_ms"]


def _benchmark_2d_case(
    name: str,
    shape: tuple[int, int],
    *,
    warmup: int,
    repeat: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEEDS[name])
    u = rng.standard_normal(shape).astype(np.float64)
    v = rng.standard_normal(shape).astype(np.float64)
    dx = 1.0 / max(shape[1] - 1, 1)
    dy = 1.0 / max(shape[0] - 1, 1)

    cases: list[tuple[str, Callable[[], Any], Callable[[], Any] | None]] = [
        (
            "field_j_2d",
            lambda: _field_J_2d_numpy(u, v, dx, dy),
            (lambda: _kernels.field_j_2d(u, v, dx, dy)) if HAS_NATIVE_KERNELS else None,
        ),
        (
            "lambda2_2d",
            lambda: _lambda2_2d_numpy(u, v, dx, dy),
            (lambda: _kernels.lambda2_2d(u, v, dx, dy)) if HAS_NATIVE_KERNELS else None,
        ),
    ]

    rows: list[dict[str, Any]] = []
    for case_name, fallback_fn, native_fn in cases:
        fallback = _measure(fallback_fn, warmup=warmup, repeat=repeat)
        native = _measure(native_fn, warmup=warmup, repeat=repeat) if native_fn else None
        rows.append(
            {
                "case": case_name,
                "size": name,
                "shape": list(shape),
                "fallback": fallback,
                "native": native,
                "native_available": native is not None,
                "speedup": _speedup(fallback, native),
            }
        )
    return rows


def _benchmark_grad_case(
    name: str,
    n_gradients: int,
    *,
    warmup: int,
    repeat: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEEDS[name] + 10_000)
    grad = rng.standard_normal((n_gradients, 3, 3)).astype(np.float64)
    cases: list[tuple[str, Callable[[], Any], Callable[[], Any] | None]] = [
        (
            "q_criterion_from_grad_3d",
            lambda: _compute_q_from_gradient_numpy(grad),
            (lambda: _kernels.q_criterion_from_grad_3d(grad)) if HAS_NATIVE_KERNELS else None,
        ),
        (
            "lambda2_from_grad_3d",
            lambda: _compute_lambda2_from_gradient_numpy(grad),
            (lambda: _kernels.lambda2_from_grad_3d(grad)) if HAS_NATIVE_KERNELS else None,
        ),
    ]

    rows: list[dict[str, Any]] = []
    for case_name, fallback_fn, native_fn in cases:
        fallback = _measure(fallback_fn, warmup=warmup, repeat=repeat)
        native = _measure(native_fn, warmup=warmup, repeat=repeat) if native_fn else None
        rows.append(
            {
                "case": case_name,
                "size": name,
                "shape": [n_gradients, 3, 3],
                "fallback": fallback,
                "native": native,
                "native_available": native is not None,
                "speedup": _speedup(fallback, native),
            }
        )
    return rows


def run_benchmarks(sizes: list[str], *, warmup: int, repeat: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name in sizes:
        shape, n_gradients = SIZES[name]
        rows.extend(_benchmark_2d_case(name, shape, warmup=warmup, repeat=repeat))
        rows.extend(_benchmark_grad_case(name, n_gradients, warmup=warmup, repeat=repeat))
    return {
        "native_available": HAS_NATIVE_KERNELS,
        "warmup": warmup,
        "repeat": repeat,
        "sizes": sizes,
        "benchmarks": rows,
    }


def _format_tsv(report: dict[str, Any]) -> str:
    lines = ["case\tsize\tshape\tfallback_median_ms\tnative_median_ms\tspeedup"]
    for row in report["benchmarks"]:
        native = row["native"]
        native_median = "" if native is None else f"{native['median_ms']:.6f}"
        speedup = "" if row["speedup"] is None else f"{row['speedup']:.3f}"
        lines.append(
            "\t".join(
                [
                    row["case"],
                    row["size"],
                    "x".join(map(str, row["shape"])),
                    f"{row['fallback']['median_ms']:.6f}",
                    native_median,
                    speedup,
                ]
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", choices=sorted(SIZES), default=sorted(SIZES))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--format", choices=["json", "tsv"], default="json")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")

    report = run_benchmarks(args.sizes, warmup=args.warmup, repeat=args.repeat)
    text = (
        json.dumps(report, indent=2, sort_keys=True)
        if args.format == "json"
        else _format_tsv(report)
    )
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
