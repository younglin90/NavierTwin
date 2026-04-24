"""Lid-driven cavity 합성 스냅샷으로 POD / AE / FNO 비교 벤치마크.

실행:
    python3 examples/cavity_benchmark.py

합성 데이터 생성 → POD / Autoencoder / FNO1D 각각 학습 → 재구성 RMSE 비교.
"""

from __future__ import annotations

import time

import numpy as np


def generate_cavity_snapshots(
    n_snapshots: int = 40,
    grid: int = 32,
    n_modes_true: int = 5,
    noise: float = 0.02,
    seed: int = 0,
) -> np.ndarray:
    """2D cavity-유사 저랭크 스냅샷: (n_features, n_snapshots)."""
    rng = np.random.default_rng(seed)
    n_features = grid * grid
    U = rng.standard_normal((n_features, n_modes_true))
    V = rng.standard_normal((n_modes_true, n_snapshots))
    return (U @ V + noise * rng.standard_normal((n_features, n_snapshots))).astype(np.float64)


def main() -> None:
    X = generate_cavity_snapshots()
    n_features, n_snap = X.shape
    print(f"[Data] shape={X.shape}, ||X||={np.linalg.norm(X):.4g}")

    results: dict[str, dict[str, float]] = {}

    # 1) POD
    from naviertwin.core.dimensionality_reduction.linear.pod import SnapshotPOD

    t0 = time.time()
    pod = SnapshotPOD(n_modes=5)
    pod.fit(X)
    X_rec_pod = pod.reconstruct(X)
    t_pod = time.time() - t0
    err_pod = float(np.linalg.norm(X - X_rec_pod) / np.linalg.norm(X))
    results["POD"] = {"rel_l2": err_pod, "time_s": t_pod}
    print(f"[POD] rel_l2={err_pod:.4g}, time={t_pod:.2f}s")

    # 2) Autoencoder
    try:
        from naviertwin.core.dimensionality_reduction.nonlinear.autoencoder import (
            Autoencoder,
        )

        t0 = time.time()
        ae = Autoencoder(latent_dim=5, hidden_dims=[128, 32], max_epochs=80, lr=1e-3)
        ae.fit(X)
        X_rec_ae = ae.decode(ae.encode(X))
        t_ae = time.time() - t0
        err_ae = float(np.linalg.norm(X - X_rec_ae) / np.linalg.norm(X))
        results["Autoencoder"] = {"rel_l2": err_ae, "time_s": t_ae}
        print(f"[AE]  rel_l2={err_ae:.4g}, time={t_ae:.2f}s")
    except Exception as e:
        print(f"[AE]  skipped: {e}")

    # 3) FNO1D (각 스냅샷을 1D 시그널로 가정)
    try:
        from naviertwin.core.operator_learning.fno.fno import FNO1D

        t0 = time.time()
        # (B, N, C) = (n_snapshots, n_features, 1), 자가예측 — x → x
        Xn = X.T[:, :, None].astype(np.float32)
        fno = FNO1D(
            in_channels=1, out_channels=1, modes=8, width=16,
            n_layers=2, max_epochs=20, batch_size=8, lr=1e-3,
        )
        fno.fit({"inputs": Xn, "outputs": Xn})
        X_rec_fno = fno.predict({"x": Xn}).reshape(n_snap, n_features).T
        t_fno = time.time() - t0
        err_fno = float(np.linalg.norm(X - X_rec_fno) / np.linalg.norm(X))
        results["FNO1D"] = {"rel_l2": err_fno, "time_s": t_fno}
        print(f"[FNO] rel_l2={err_fno:.4g}, time={t_fno:.2f}s")
    except Exception as e:
        print(f"[FNO] skipped: {e}")

    # 4) Pipeline 보고서
    try:
        from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline

        pipe = NavierTwinPipeline(reducer_kind="pod", n_modes=5, surrogate_kind="kriging")
        pipe.load_snapshots(X, field_name="U")
        pipe.reduce()
        params = np.linspace(0, 1, n_snap).reshape(-1, 1)
        pipe.fit_surrogate(params)
        metrics = pipe.validate(params[-10:], pipe.state.coeffs[-10:])
        print(f"[Pipeline] surrogate metrics: {metrics}")
    except Exception as e:
        print(f"[Pipeline] skipped: {e}")

    print()
    print("=" * 50)
    print("Summary (rel. L2 ↓)")
    print("=" * 50)
    for name, r in sorted(results.items(), key=lambda x: x[1]["rel_l2"]):
        print(f"  {name:<14s} {r['rel_l2']:.6g}  ({r['time_s']:.2f}s)")


if __name__ == "__main__":
    main()
