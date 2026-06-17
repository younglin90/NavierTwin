"""Pipeline HDF5 체크포인트 — 중간 상태 저장/복원.

NavierTwinPipeline.state 의 snapshots/coeffs/mean/modes 를 HDF5 에 저장하면
재실행 없이 surrogate 재학습/예측 가능.

Examples:
    >>> from naviertwin.core.digital_twin.pipeline_checkpoint import (
    ...     save_pipeline_state, load_pipeline_state,
    ... )
    >>> # save_pipeline_state(pipe, "ckpt.h5")
    >>> # state = load_pipeline_state("ckpt.h5")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def save_pipeline_state(pipe: Any, path: str | Path) -> Path:
    """NavierTwinPipeline 의 주요 numpy 필드를 HDF5 에 덤프.

    저장 필드:
        - snapshots, coeffs
        - POD reducer: modes, singular_values, mean, energy
        - metrics (JSON serialized)
        - meta: {reducer_kind, surrogate_kind, n_modes, field_name}
    """
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py 필요") from exc

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = pipe.state

    with h5py.File(p, "w") as f:
        g = f.create_group("pipeline")
        # scalar meta
        meta = {
            "reducer_kind": getattr(pipe, "reducer_kind", ""),
            "surrogate_kind": getattr(pipe, "surrogate_kind", ""),
            "n_modes": int(getattr(pipe, "n_modes", 0)),
            "field_name": str(state.field_name),
        }
        g.attrs["meta_json"] = json.dumps(meta, ensure_ascii=False)

        # arrays
        if state.snapshots is not None:
            g.create_dataset("snapshots", data=state.snapshots)
        if state.coeffs is not None:
            g.create_dataset("coeffs", data=state.coeffs)
        if state.reducer is not None:
            r = state.reducer
            reducer_group = g.create_group("reducer")
            def write_reducer_attr(attr: str) -> None:
                v = getattr(r, attr, None)
                if v is not None:
                    reducer_group.create_dataset(
                        attr.rstrip("_"), data=np.asarray(v, dtype=np.float64)
                    )

            tuple(
                map(
                    write_reducer_attr,
                    ["modes_", "singular_values_", "mean_", "energy_ratio_"],
                )
            )
        g.attrs["metrics_json"] = json.dumps(state.metrics, ensure_ascii=False)

    logger.info("Pipeline 체크포인트 저장: %s", p)
    return p


def load_pipeline_state(path: str | Path) -> dict[str, Any]:
    """HDF5 체크포인트에서 dict 로 복원.

    Returns:
        dict:
            snapshots, coeffs, modes, singular_values, mean, energy,
            metrics, meta.
    """
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py 필요") from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    out: dict[str, Any] = {}
    with h5py.File(p, "r") as f:
        if "pipeline" not in f:
            raise ValueError("pipeline 그룹 없음")
        g = f["pipeline"]

        meta_json = g.attrs.get("meta_json", "{}")
        if isinstance(meta_json, bytes):
            meta_json = meta_json.decode()
        out["meta"] = json.loads(meta_json)

        metrics_json = g.attrs.get("metrics_json", "{}")
        if isinstance(metrics_json, bytes):
            metrics_json = metrics_json.decode()
        out["metrics"] = json.loads(metrics_json)

        def read_pipeline_array(key: str) -> None:
            if key in g:
                out[key] = g[key][...]

        tuple(map(read_pipeline_array, ("snapshots", "coeffs")))
        if "reducer" in g:
            rg = g["reducer"]
            tuple(map(lambda k: out.__setitem__(k, rg[k][...]), rg.keys()))

    logger.info("Pipeline 체크포인트 로드: %s", p)
    return out


def restore_pipeline(
    pipe: Any, ckpt: dict[str, Any]
) -> Any:
    """load_pipeline_state 결과를 기존 파이프라인 객체에 주입.

    주의: reducer 는 POD 가정 (modes/singular_values/mean). 다른 kind 은 수동 복원.
    """
    state = pipe.state
    if "snapshots" in ckpt:
        state.snapshots = ckpt["snapshots"]
    state.field_name = ckpt.get("meta", {}).get("field_name", state.field_name)
    if "coeffs" in ckpt:
        state.coeffs = ckpt["coeffs"]

    # Reducer 복원 (pod / incremental_pod / mrpod)
    if {"modes", "singular_values", "mean"}.issubset(ckpt):
        reducer_kind = str(ckpt.get("meta", {}).get("reducer_kind", "pod"))
        try:
            modes = np.asarray(ckpt["modes"])
            singular_values = np.asarray(ckpt["singular_values"])
            mean = np.asarray(ckpt["mean"])
            energy = np.asarray(ckpt.get("energy", np.ones(modes.shape[1])))

            if reducer_kind == "incremental_pod":
                from naviertwin.core.dimensionality_reduction.linear.incremental_pod import (
                    IncrementalPOD,
                )

                r = IncrementalPOD(n_modes=int(modes.shape[1]))
                r.basis = modes
                r.singular_values = singular_values
                r._mean = mean
                r.n_snapshots = int(
                    ckpt["snapshots"].shape[1] if "snapshots" in ckpt else 0
                )
                r._refresh_compat_attrs()
                state.reducer = r
            elif reducer_kind == "mrpod":
                from naviertwin.core.dimensionality_reduction.linear.mrpod import MRPOD

                r = MRPOD(n_scales=1, n_modes_per_scale=int(modes.shape[1]))
                r.scale_modes = [modes]
                r.scale_energies = [np.maximum(singular_values, 0.0) ** 2]
                r.mean_ = mean.reshape(-1, 1) if mean.ndim == 1 else mean
                r.modes_ = modes
                r.singular_values_ = singular_values
                r.energy_ratio_ = energy
                r.n_components = int(modes.shape[1])
                r.is_fitted = True
                state.reducer = r
            else:
                from naviertwin.core.dimensionality_reduction.linear.pod import (
                    SnapshotPOD,
                )

                r = SnapshotPOD(n_modes=pipe.n_modes)
                r.modes_ = modes
                r.singular_values_ = singular_values
                r.mean_ = mean
                r.energy_ratio_ = energy
                r.n_components = r.modes_.shape[1]
                r.is_fitted = True
                state.reducer = r
        except Exception as e:  # noqa: BLE001
            logger.warning("Reducer 복원 실패(%s): %s", reducer_kind, e)

    state.metrics = dict(ckpt.get("metrics", {}))
    return pipe


__all__ = ["save_pipeline_state", "load_pipeline_state", "restore_pipeline"]
