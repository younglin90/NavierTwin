"""LBM cavity → ROM → Surrogate → Digital Twin 완전 파이프라인 예제.

실행:
    python3 examples/lbm_rom_pipeline.py

1) 내장 D2Q9 LBM 으로 다양한 u_top (Re) 에 대한 cavity 스냅샷 생성
2) 각 케이스 snapshots 를 POD 로 압축
3) 파라미터(u_top) → POD 계수 Kriging surrogate 학습
4) 새 u_top 에서 필드 예측 + 검증 메트릭
5) Jinja2 보고서 자동 생성
"""

from __future__ import annotations

import tempfile

import numpy as np


def main() -> None:
    from naviertwin.core.solver_interfaces.lbm_d2q9 import LBMD2Q9

    # 1) 다양한 u_top 에 대해 마지막 스냅샷만 수집
    u_top_values = np.linspace(0.02, 0.08, 6)
    snapshots: list[np.ndarray] = []

    print(f"[1] LBM 스냅샷 생성: {len(u_top_values)} 케이스")
    for u_top in u_top_values:
        lbm = LBMD2Q9(nx=24, ny=24, tau=0.8, u_top=float(u_top))
        snaps = lbm.run(n_steps=200, record_every=200)
        # (1, 24, 24, 3) → ux 필드만 사용 (flatten)
        ux = snaps[0, :, :, 1].ravel()
        snapshots.append(ux)

    X = np.stack(snapshots).T  # (n_features, n_snapshots)
    params = u_top_values.reshape(-1, 1)
    print(f"    스냅샷 행렬 shape: {X.shape}")

    # 2) POD
    from naviertwin.core.dimensionality_reduction.linear.pod import SnapshotPOD

    pod = SnapshotPOD(n_modes=4)
    pod.fit(X)
    coeffs = pod.encode(X)
    print(f"[2] POD 완료: 누적 에너지={pod.energy_ratio[-1]:.6f}")

    # 3) Surrogate
    from naviertwin.core.surrogate.kriging_surrogate import KrigingSurrogate

    sur = KrigingSurrogate()
    sur.fit(params, coeffs)
    print(f"[3] Surrogate 학습: {type(sur).__name__}")

    # 4) 새 u_top 에서 예측 + 검증
    u_new = np.array([[0.05]])
    c_pred = sur.predict(u_new)
    field_pred = pod.decode(c_pred).ravel()

    # Ground truth: 같은 u_top=0.05 LBM 실행
    lbm_truth = LBMD2Q9(nx=24, ny=24, tau=0.8, u_top=0.05)
    snap_truth = lbm_truth.run(n_steps=200, record_every=200)
    ux_truth = snap_truth[0, :, :, 1].ravel()

    from naviertwin.core.validation.metrics import compute_all_metrics

    metrics = compute_all_metrics(ux_truth, field_pred)
    print(f"[4] 예측 vs truth: {metrics}")

    # 5) 보고서
    try:
        from naviertwin.core.report.generator import ReportGenerator

        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/cavity_lbm_report.html"
            ReportGenerator().render_html({
                "project": "Cavity LBM Digital Twin",
                "summary": (
                    f"D2Q9 LBM cavity ({len(u_top_values)} u_top) → POD(4) → Kriging. "
                    f"예측 u_top=0.05."
                ),
                "metrics": metrics,
                "model_info": {
                    "grid": "24x24",
                    "tau": 0.8,
                    "n_modes": 4,
                    "n_snapshots": len(u_top_values),
                    "surrogate": "Kriging (SMT)",
                },
                "notes": "LBM 스냅샷은 moving-lid 상단 경계로 구동.",
            }, path)
            print(f"[5] 보고서: {path}")
    except Exception as e:
        print(f"[5] 보고서 skipped: {e}")

    print()
    print("=" * 50)
    print(f"최종 rel.L2 = {metrics['relative_l2']:.6g}, R² = {metrics['r2']:.6g}")
    print("=" * 50)


if __name__ == "__main__":
    main()
