"""Burgers 솔버 → FNO1D 학습 예제.

실행:
    python3 examples/burgers_fno.py

FNO 목표: u(x, t=0.2) → u(x, t=0.4) 매핑 (0.2s 앞서가기 연산자).
여러 개 초기조건을 무작위로 샘플링해 input/output 쌍 생성.
"""

from __future__ import annotations

import time

import numpy as np


def main() -> None:
    from naviertwin.core.solver_interfaces.pde_solvers import solve_burgers_1d

    rng = np.random.default_rng(0)
    N = 64
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)

    # 1) 32 개 랜덤 초기조건 생성 — 저주파 sin 조합
    n_samples = 32
    X_list = []
    Y_list = []

    for _ in range(n_samples):
        # 저주파 3개 계수 조합
        c = rng.standard_normal(3) * 0.5
        u0 = sum(c[k] * np.sin((k + 1) * x) for k in range(3))
        # t=0 → t=0.2 → t=0.4 두 단계 진행
        _, U = solve_burgers_1d(u0, nu=0.02, L=L, T=0.4, n_steps=200)
        u_mid = U[99]  # t ≈ 0.2
        u_end = U[-1]  # t ≈ 0.4
        X_list.append(u_mid[:, None])
        Y_list.append(u_end[:, None])

    X = np.stack(X_list).astype(np.float32)  # (n_samples, N, 1)
    Y = np.stack(Y_list).astype(np.float32)
    print(f"[1] 데이터: X={X.shape}, Y={Y.shape}")

    # 2) FNO1D 학습
    from naviertwin.core.operator_learning.fno.fno import FNO1D

    t0 = time.time()
    fno = FNO1D(
        in_channels=1, out_channels=1, modes=8, width=16,
        n_layers=3, max_epochs=50, batch_size=8, lr=1e-3,
    )
    fno.fit({"inputs": X[:24], "outputs": Y[:24]})
    t_train = time.time() - t0
    print(f"[2] FNO 학습: {t_train:.2f}s, 최종 loss={fno.train_losses_[-1]:.6g}")

    # 3) 홀드아웃 검증
    y_pred = fno.predict({"x": X[24:]})
    err = float(np.linalg.norm(Y[24:] - y_pred) / np.linalg.norm(Y[24:]))
    print(f"[3] 홀드아웃 rel.L2 = {err:.4f}")

    # 4) 연관된 초기상태에서 다중 step rollout (FNO 한 번 = 0.2s 이동)
    u0 = np.sin(x)
    _, U_ref = solve_burgers_1d(u0, nu=0.02, L=L, T=0.4, n_steps=200)
    u_start = U_ref[99][:, None].astype(np.float32)
    u_fno = fno.predict({"x": u_start[None, ...]})[0, :, 0]
    u_true = U_ref[-1]
    err_single = float(np.linalg.norm(u_fno - u_true) / np.linalg.norm(u_true))
    print(f"[4] 단일 step FNO 예측 오차 = {err_single:.4f}")

    print()
    print("=" * 50)
    print(f"FNO1D Burgers 연산자 학습 완료: 홀드아웃 rel.L2 {err:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
