"""AI 벤치마크 데이터셋 → 연산자 학습 서비스 (Qt/GL 비의존).

웹 GUI 의 "AI Bench" 패널 뒤에서 동작하는 순수 서비스 계층. 두 종류의
CFD 벤치마크 데이터 소스를 (u0 → uT) 연산자 학습 문제로 정규화한다:

1. **내장 솔버 생성** — ``core.solver_interfaces.pde_solvers`` 의 Burgers/Heat
   1D 솔버로 (초기조건, 최종상태) 쌍을 즉석 생성 (다운로드 불필요).
2. **PDEBench 포맷 HDF5** — 다운로드한 PDEBench(NeurIPS 2022) 1D 파일
   (``tensor`` shape ``(B, T, N)``)을 h5py 로 직접 읽음 (pdebench 패키지 불필요).

정규화된 데이터셋 dict 는 ``core.operator_learning.fno.FNO1D`` 가 기대하는
``inputs/outputs`` shape ``(B, N, 1)`` 을 그대로 따른다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

# 웹 UI 에 노출하는 내장 벤치마크 카탈로그.
BENCHMARKS: dict[str, dict[str, str]] = {
    "burgers": {
        "title": "Burgers 1D (u0 → u(T))",
        "description": "점성 Burgers 방정식 — 저주파 sin 초기조건에서 T 시점 상태로의 연산자",
    },
    "heat": {
        "title": "Heat 1D (u0 → u(T))",
        "description": "1D 열전도 (Crank–Nicolson, Dirichlet 0) — 확산 연산자",
    },
    "cavity2d": {
        "title": "Cavity 2D LBM (u(t) → u(t+Δt))",
        "description": "리드 드리븐 캐비티 LBM 시계열 — 2D 시간전진 연산자 (FNO2D)",
    },
}


# ──────────────────────────────────────────────────────────────────────
# 데이터셋 생성 / 로드
# ──────────────────────────────────────────────────────────────────────


def generate_operator_dataset(
    kind: str,
    n_samples: int = 64,
    n_x: int = 64,
    *,
    T: float = 0.5,
    n_steps: int = 200,
    seed: int | None = 0,
) -> dict[str, Any]:
    """내장 솔버로 (u0 → uT) 연산자 학습 데이터셋을 생성한다.

    Args:
        kind: ``"burgers"`` | ``"heat"``.
        n_samples: 샘플(궤적) 수.
        n_x: 공간 해상도.
        T: 적분 시간.
        n_steps: 시간 스텝 수.
        seed: 난수 시드.

    Returns:
        ``inputs``/``outputs`` shape ``(B, N, 1)``, ``x`` shape ``(N,)``,
        ``kind``, ``source``, ``n_samples``, ``n_x`` 를 담은 dict.

    Raises:
        ValueError: 지원하지 않는 kind.
    """
    if kind not in BENCHMARKS:
        raise ValueError(f"지원하지 않는 벤치마크: {kind}. 지원: {sorted(BENCHMARKS)}")

    n_samples = max(4, int(n_samples))
    n_x = max(16, int(n_x))
    rng = np.random.default_rng(seed)

    if kind == "cavity2d":
        return _generate_cavity2d_dataset(n_samples=n_samples, n_x=n_x, rng=rng)

    if kind == "burgers":
        from naviertwin.core.solver_interfaces.pde_solvers import solve_burgers_1d

        x = np.linspace(0.0, 2.0 * np.pi, n_x, endpoint=False)
        harmonics = np.arange(1, 4, dtype=np.float64)
        inputs = np.zeros((n_samples, n_x), dtype=np.float64)
        outputs = np.zeros((n_samples, n_x), dtype=np.float64)
        i = 0
        while i < n_samples:
            nu = float(10 ** rng.uniform(-2.5, -1.5))  # 0.003 ~ 0.03
            amps = rng.uniform(-1.0, 1.0, 3)
            u0 = (amps[:, None] * np.sin(harmonics[:, None] * x[None, :])).sum(axis=0)
            _, U = solve_burgers_1d(u0, nu=nu, L=2.0 * np.pi, T=T, n_steps=n_steps)
            inputs[i] = u0
            outputs[i] = U[-1]
            i += 1
    else:  # heat
        from naviertwin.core.solver_interfaces.pde_solvers import solve_heat_1d

        x = np.linspace(0.0, 1.0, n_x)
        inputs = np.zeros((n_samples, n_x), dtype=np.float64)
        outputs = np.zeros((n_samples, n_x), dtype=np.float64)
        i = 0
        while i < n_samples:
            alpha = float(10 ** rng.uniform(-2.5, -1.0))
            amps = rng.uniform(0.2, 1.0, 2)
            u0 = amps[0] * np.sin(np.pi * x) + amps[1] * np.sin(3.0 * np.pi * x)
            _, U = solve_heat_1d(u0, alpha=alpha, L=1.0, T=T, n_steps=n_steps)
            inputs[i] = u0
            outputs[i] = U[-1]
            i += 1

    logger.info("연산자 데이터셋 생성: kind=%s, samples=%d, n_x=%d", kind, n_samples, n_x)
    return {
        "inputs": inputs[..., None],
        "outputs": outputs[..., None],
        "x": x,
        "kind": kind,
        "source": f"builtin:{kind}",
        "n_samples": int(n_samples),
        "n_x": int(n_x),
    }


def _generate_cavity2d_dataset(
    *,
    n_samples: int,
    n_x: int,
    rng: Any,
    record_every: int = 40,
    n_trajectories: int = 4,
) -> dict[str, Any]:
    """LBM cavity 시계열에서 (u(t) → u(t+Δt)) 2D 시간전진 페어를 생성한다.

    서로 다른 lid 속도(u_top) 궤적 여러 개를 돌려 연속 스냅샷 페어를 모은다.
    채널은 [ux, uy] 2개 — shape ``(B, ny, nx, 2)``.
    """
    from naviertwin.core.solver_interfaces.lbm_d2q9 import LBMD2Q9

    n = max(16, min(int(n_x), 48))  # LBM 비용 보호 (16~48)
    n_traj = max(2, int(n_trajectories))
    pairs_per_traj = max(1, int(np.ceil(n_samples / n_traj)))
    n_records = pairs_per_traj + 1
    n_steps = record_every * n_records

    inputs_list: list[np.ndarray] = []
    outputs_list: list[np.ndarray] = []
    traj = 0
    while traj < n_traj and len(inputs_list) < n_samples:
        u_top = float(rng.uniform(0.02, 0.1))
        lbm = LBMD2Q9(nx=n, ny=n, tau=0.8, u_top=u_top)
        snaps = lbm.run(n_steps=n_steps, record_every=record_every)  # (n_rec, ny, nx, 3)
        velocity = snaps[..., 1:3]  # [ux, uy]
        step = 0
        while step < snaps.shape[0] - 1 and len(inputs_list) < n_samples:
            inputs_list.append(velocity[step])
            outputs_list.append(velocity[step + 1])
            step += 1
        traj += 1

    inputs = np.stack(inputs_list)   # (B, ny, nx, 2)
    outputs = np.stack(outputs_list)
    logger.info(
        "Cavity2D 시간전진 데이터셋: %d pairs, grid=%dx%d, traj=%d",
        inputs.shape[0], n, n, n_traj,
    )
    return {
        "inputs": inputs,
        "outputs": outputs,
        "x": np.arange(n, dtype=np.float64),
        "kind": "cavity2d",
        "source": "builtin:cavity2d",
        "n_samples": int(inputs.shape[0]),
        "n_x": int(n),
    }


def _find_pdebench_tensor(h5: Any) -> Any:
    """PDEBench HDF5 에서 주 데이터 텐서를 찾는다 ('tensor' 우선, 없으면 최대 3D float)."""
    import h5py

    if "tensor" in h5 and isinstance(h5["tensor"], h5py.Dataset):
        return h5["tensor"]

    best = None

    def _visit(_name: str, obj: Any) -> None:
        nonlocal best
        if not isinstance(obj, h5py.Dataset):
            return
        if obj.ndim != 3 or obj.dtype.kind != "f":
            return
        if best is None or obj.size > best.size:
            best = obj

    h5.visititems(_visit)
    return best


def load_pdebench_hdf5(path: str | Path, max_samples: int = 64) -> dict[str, Any]:
    """PDEBench 포맷 1D HDF5 파일을 (u0 → uT) 연산자 데이터셋으로 로드한다.

    기대 포맷: ``tensor`` dataset shape ``(B, T, N)`` (+선택 ``x-coordinate``).
    pdebench 패키지 없이 h5py 만으로 읽는다.

    Args:
        path: ``.h5``/``.hdf5`` 파일 경로.
        max_samples: 로드할 최대 샘플 수 (메모리 보호).

    Raises:
        FileNotFoundError: 파일이 없는 경우.
        ValueError: 지원 포맷이 아닌 경우.
    """
    import h5py

    target = Path(path).expanduser()
    if not target.exists():
        raise FileNotFoundError(f"경로를 찾을 수 없습니다: {target}")
    if target.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError(f"HDF5 파일(.h5/.hdf5)이 아닙니다: {target}")

    with h5py.File(target, "r") as h5:
        tensor = _find_pdebench_tensor(h5)
        if tensor is None:
            raise ValueError(
                "PDEBench 포맷 텐서를 찾지 못했습니다. "
                "shape (B, T, N) 인 3D float dataset('tensor')이 필요합니다."
            )
        n_total = int(tensor.shape[0])
        n_take = max(1, min(int(max_samples), n_total))
        data = np.asarray(tensor[:n_take], dtype=np.float64)  # (B, T, N)
        if data.ndim != 3 or data.shape[1] < 2:
            raise ValueError(f"(B, T>=2, N) 텐서가 필요합니다. 현재: {data.shape}")
        x = None
        if "x-coordinate" in h5:
            x = np.asarray(h5["x-coordinate"], dtype=np.float64).reshape(-1)

    n_x = int(data.shape[2])
    if x is None or x.size != n_x:
        x = np.linspace(0.0, 1.0, n_x)

    logger.info(
        "PDEBench HDF5 로드: %s — %d/%d samples, T=%d, N=%d",
        target.name, n_take, n_total, data.shape[1], n_x,
    )
    return {
        "inputs": data[:, 0, :][..., None],
        "outputs": data[:, -1, :][..., None],
        "x": x,
        "kind": "pdebench",
        "source": f"pdebench:{target.name}",
        "n_samples": int(n_take),
        "n_x": n_x,
    }


def dataset_summary(dataset: dict[str, Any]) -> str:
    """UI 표시용 데이터셋 요약 문자열."""
    task = "u(t) → u(t+Δt)" if dataset.get("kind") == "cavity2d" else "u0 → uT"
    return (
        f"{dataset.get('source', '?')} · {dataset.get('n_samples', 0)} samples × "
        f"N={dataset.get('n_x', 0)} ({task})"
    )


# ──────────────────────────────────────────────────────────────────────
# 연산자 학습 / 평가
# ──────────────────────────────────────────────────────────────────────


# FNO 구현 백엔드 — 같은 fit/predict 계약이라 같은 벤치에서 직접 비교된다.
#   builtin  : 이 저장소 자체 구현 (naviertwin.core.operator_learning.fno.fno)
#   neuralop : neuraloperator 레퍼런스 구현 (FNO 논문 저자 유지, PyTorch Ecosystem)
OPERATOR_BACKENDS = ("builtin", "neuralop")


def train_operator(
    dataset: dict[str, Any],
    *,
    operator: str = "auto",
    backend: str = "builtin",
    modes: int = 12,
    width: int = 32,
    epochs: int = 60,
    lr: float = 1e-3,
    batch_size: int = 16,
    test_fraction: float = 0.2,
    seed: int | None = 0,
    progress_cb: Any = None,
) -> dict[str, Any]:
    """정규화된 연산자 데이터셋으로 신경 연산자(FNO)를 학습하고 test 지표를 계산한다.

    입력 텐서 차원에 따라 자동 선택한다: ``(B,N,C)`` 3D → FNO1D,
    ``(B,H,W,C)`` 4D → FNO2D.

    Args:
        dataset: :func:`generate_operator_dataset` / :func:`load_pdebench_hdf5` 결과.
        operator: ``"auto"`` | ``"fno1d"`` | ``"fno2d"``.
        backend: FNO 구현 — ``"builtin"``(자체) | ``"neuralop"``(레퍼런스).
            둘은 같은 계약이라 동일 벤치·동일 하이퍼파라미터로 비교할 수 있다.
        modes/width/epochs/lr/batch_size: FNO 하이퍼파라미터.
        test_fraction: 홀드아웃 비율.
        seed: 셔플/모델 시드.

    Returns:
        ``model``, ``operator``, ``backend``, ``losses``(epoch별), ``final_loss``,
        ``test_rmse``, ``test_rel_l2``, ``latency_ms``(단일 예측),
        ``train_time_s``, ``n_train``/``n_test``, ``test_indices``.

    Raises:
        ValueError: 지원하지 않는 operator/backend, 차원 불일치, 샘플 부족.
        ImportError: torch 미설치, 또는 backend="neuralop" 인데 neuraloperator 미설치.
    """
    from time import perf_counter

    from naviertwin.core.validation.metrics import relative_l2_error, rmse

    if operator not in {"auto", "fno1d", "fno2d"}:
        raise ValueError(
            f"지원하지 않는 operator: {operator}. 지원: ['auto', 'fno1d', 'fno2d']"
        )
    if backend not in OPERATOR_BACKENDS:
        raise ValueError(
            f"지원하지 않는 backend: {backend}. 지원: {list(OPERATOR_BACKENDS)}"
        )

    X = np.asarray(dataset["inputs"], dtype=np.float64)
    Y = np.asarray(dataset["outputs"], dtype=np.float64)
    n_total = int(X.shape[0])
    if n_total < 4:
        raise ValueError(f"학습에는 4개 이상 샘플이 필요합니다. 현재: {n_total}")

    resolved = operator
    if resolved == "auto":
        resolved = "fno2d" if X.ndim == 4 else "fno1d"
    if resolved == "fno1d" and X.ndim != 3:
        raise ValueError(f"fno1d 는 (B,N,C) 3D 입력이 필요합니다: {X.shape}")
    if resolved == "fno2d" and X.ndim != 4:
        raise ValueError(f"fno2d 는 (B,H,W,C) 4D 입력이 필요합니다: {X.shape}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_total)
    n_test = max(1, int(round(n_total * float(test_fraction))))
    test_idx = order[:n_test]
    train_idx = order[n_test:]
    if train_idx.size < 2:
        raise ValueError("train 샘플이 부족합니다. n_samples 를 늘리세요.")

    total_epochs = max(1, int(epochs))
    epoch_cb = None
    if progress_cb is not None:
        def epoch_cb(epoch: int, loss: float) -> None:  # noqa: ANN001
            progress_cb(int(epoch), total_epochs, float(loss))

    channels = int(X.shape[-1])
    # 모드 수는 나이퀴스트 한계를 넘을 수 없다 (백엔드 공통).
    if resolved == "fno1d":
        safe_modes = min(int(modes), int(X.shape[1]) // 2)
        n_dim, safe_batch = 1, max(1, int(batch_size))
    else:
        safe_modes = min(int(modes), int(X.shape[1]) // 2, int(X.shape[2]) // 2)
        n_dim, safe_batch = 2, max(1, min(int(batch_size), 8))

    if backend == "neuralop":
        from naviertwin.core.operator_learning.fno.neuralop_fno import NeuralOpFNO

        model = NeuralOpFNO(
            in_channels=channels,
            out_channels=int(Y.shape[-1]),
            modes=safe_modes,
            width=int(width),
            n_dim=n_dim,
            max_epochs=total_epochs,
            batch_size=safe_batch,
            lr=float(lr),
            seed=seed,
            epoch_callback=epoch_cb,
        )
    elif resolved == "fno1d":
        from naviertwin.core.operator_learning.fno.fno import FNO1D

        model = FNO1D(
            in_channels=channels,
            out_channels=int(Y.shape[-1]),
            modes=safe_modes,
            width=int(width),
            max_epochs=total_epochs,
            batch_size=safe_batch,
            lr=float(lr),
            seed=seed,
            epoch_callback=epoch_cb,
        )
    else:
        from naviertwin.core.operator_learning.fno.fno import FNO2D

        model = FNO2D(
            in_channels=channels,
            out_channels=int(Y.shape[-1]),
            modes1=safe_modes,
            modes2=safe_modes,
            width=int(width),
            max_epochs=total_epochs,
            batch_size=safe_batch,
            lr=float(lr),
            seed=seed,
            epoch_callback=epoch_cb,
        )

    started = perf_counter()
    model.fit({"inputs": X[train_idx], "outputs": Y[train_idx]})
    train_time_s = perf_counter() - started

    # 홀드아웃 정확도 + 단일 예측 지연시간 (빠른 결과 검증)
    pred_test = np.asarray(model.predict({"x": X[test_idx]}), dtype=np.float64)
    test_rmse = float(rmse(Y[test_idx], pred_test))
    test_rel_l2 = float(relative_l2_error(Y[test_idx], pred_test))

    single = X[test_idx[:1]]
    model.predict({"x": single})  # warmup
    lat_started = perf_counter()
    repeat = 10
    i = 0
    while i < repeat:
        model.predict({"x": single})
        i += 1
    latency_ms = (perf_counter() - lat_started) / repeat * 1000.0

    losses = [float(v) for v in getattr(model, "train_losses_", [])]
    logger.info(
        "연산자 학습 완료: %s(%s) — final_loss=%.4g, test_rmse=%.4g, latency=%.2fms",
        resolved, backend, losses[-1] if losses else float("nan"), test_rmse, latency_ms,
    )
    return {
        "model": model,
        "operator": resolved,
        "backend": backend,
        "losses": losses,
        "final_loss": losses[-1] if losses else float("nan"),
        "test_rmse": test_rmse,
        "test_rel_l2": test_rel_l2,
        "latency_ms": float(latency_ms),
        "train_time_s": float(train_time_s),
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "test_indices": [int(v) for v in test_idx],
    }


def evaluate_sample(
    model: Any,
    dataset: dict[str, Any],
    index: int = 0,
) -> dict[str, Any]:
    """단일 샘플 예측 — 참값/예측과 지연시간을 반환한다 (차트용).

    1D 는 곡선(x/input/true/pred), 2D 는 채널 크기(magnitude) 필드
    (``true2d``/``pred2d``/``err2d``, ``is_2d=True``)를 반환한다.
    """
    from time import perf_counter

    from naviertwin.core.validation.metrics import rmse

    X = np.asarray(dataset["inputs"], dtype=np.float64)
    Y = np.asarray(dataset["outputs"], dtype=np.float64)
    idx = max(0, min(int(index), X.shape[0] - 1))

    started = perf_counter()
    pred = np.asarray(model.predict({"x": X[idx]}), dtype=np.float64)
    latency_ms = (perf_counter() - started) * 1000.0

    base = {
        "index": idx,
        "rmse": float(rmse(Y[idx].reshape(-1), pred.reshape(-1))),
        "latency_ms": float(latency_ms),
    }

    if X.ndim == 4:  # (B, H, W, C) — 2D 필드
        true_mag = np.linalg.norm(Y[idx], axis=-1)
        pred_mag = np.linalg.norm(pred, axis=-1)
        base.update(
            {
                "is_2d": True,
                "true2d": true_mag.tolist(),
                "pred2d": pred_mag.tolist(),
                "err2d": np.abs(true_mag - pred_mag).tolist(),
            }
        )
        return base

    x = np.asarray(dataset.get("x", np.arange(X.shape[1])), dtype=np.float64)
    base.update(
        {
            "is_2d": False,
            "x": x.tolist(),
            "input": X[idx].reshape(-1).tolist(),
            "true": Y[idx].reshape(-1).tolist(),
            "pred": pred.reshape(-1).tolist(),
        }
    )
    return base


__all__ = [
    "BENCHMARKS",
    "dataset_summary",
    "evaluate_sample",
    "generate_operator_dataset",
    "load_pdebench_hdf5",
    "train_operator",
]
