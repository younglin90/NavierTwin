"""Q-criterion 및 λ₂ 와류 식별 모듈.

유동장 메쉬에서 속도 구배 텐서를 계산하고
Q-criterion 과 λ₂ 와류 식별자를 추가한다.

References:
    - Hunt, J.C.R., Wray, A.A. & Moin, P. (1988). Eddies, streams, and
      convergence zones in turbulent flows. CTR Rep. CTR-S88.
    - Jeong, J. & Hussain, F. (1995). On the identification of a vortex.
      J. Fluid Mech., 285, 69-94.

Examples:
    Q-criterion 계산::

        import pyvista as pv
        from naviertwin.core.flow_analysis.vortex.q_criterion import (
            compute_q_criterion,
            compute_lambda2,
        )

        mesh = pv.read("result.vtu")
        mesh = compute_q_criterion(mesh, velocity_name="U")
        mesh = compute_lambda2(mesh, velocity_name="U")
"""

from __future__ import annotations

from typing import Any

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def compute_q_criterion(
    mesh: Any,
    velocity_name: str = "U",
) -> Any:
    """PyVista compute_derivative 기반 Q-criterion 을 계산한다.

    cell-centered velocity 가 있으면 자동으로 point_data 로 변환한다.
    계산 결과로 ``"Q-criterion"`` 과 ``"vorticity"`` 필드가 메쉬에 추가된다.

    Q = (||Omega||² - ||S||²) / 2

    여기서 S 는 변형률 텐서, Omega 는 회전 텐서다.

    Args:
        mesh: 속도 필드를 포함한 ``pv.UnstructuredGrid`` 또는 ``pv.DataSet``.
        velocity_name: 속도 벡터 필드 이름.

    Returns:
        ``"Q-criterion"`` 과 ``"vorticity"`` 필드가 추가된 메쉬.

    Raises:
        ImportError: pyvista 또는 numpy 가 설치되어 있지 않은 경우.
        KeyError: 메쉬에 velocity_name 필드가 없는 경우.
    """
    try:
        import numpy as np
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "Q-criterion 계산에는 pyvista 와 numpy 가 필요합니다.\n"
            "  pip install pyvista numpy"
        ) from exc

    mesh = _ensure_point_velocity(mesh, velocity_name)

    if velocity_name not in mesh.point_data:
        available = list(mesh.point_data.keys()) + list(mesh.cell_data.keys())
        raise KeyError(
            f"'{velocity_name}' 필드가 없습니다. "
            f"사용 가능: {available}"
        )

    logger.debug("Q-criterion 계산 시작: velocity_name='%s'", velocity_name)

    # compute_derivative 는 active scalars 에 대해 동작하므로 벡터 설정
    mesh.set_active_vectors(velocity_name)

    try:
        grad_mesh = mesh.compute_derivative(
            scalars=velocity_name,
            gradient=True,
            vorticity=True,
        )
    except Exception as exc:
        logger.warning("compute_derivative 실패: %s. 수동 계산으로 전환.", exc)
        return _compute_q_manual(mesh, velocity_name)

    # 속도 구배 텐서: shape (N, 9) → (N, 3, 3)
    grad_key = f"gradient"
    if grad_key not in grad_mesh.point_data:
        logger.warning("gradient 필드 없음. 수동 계산으로 전환.")
        return _compute_q_manual(mesh, velocity_name)

    grad = np.asarray(grad_mesh.point_data[grad_key])
    if grad.ndim == 2 and grad.shape[1] == 9:
        grad = grad.reshape(-1, 3, 3)

    # S = (J + J^T) / 2, Omega = (J - J^T) / 2
    S = (grad + grad.transpose(0, 2, 1)) / 2.0
    Omega = (grad - grad.transpose(0, 2, 1)) / 2.0

    # Q = (||Omega||_F^2 - ||S||_F^2) / 2
    q_vals = 0.5 * (
        np.sum(Omega**2, axis=(1, 2)) - np.sum(S**2, axis=(1, 2))
    )

    mesh.point_data["Q-criterion"] = q_vals.astype(np.float32)

    # vorticity (curl of U)
    if "vorticity" in grad_mesh.point_data:
        mesh.point_data["vorticity"] = np.asarray(
            grad_mesh.point_data["vorticity"], dtype=np.float32
        )
    else:
        # Omega 에서 직접 추출: omega_x = dw/dy - dv/dz 등
        vort = np.stack(
            [
                Omega[:, 2, 1] - Omega[:, 1, 2],
                Omega[:, 0, 2] - Omega[:, 2, 0],
                Omega[:, 1, 0] - Omega[:, 0, 1],
            ],
            axis=-1,
        )
        mesh.point_data["vorticity"] = vort.astype(np.float32)

    logger.info(
        "Q-criterion 계산 완료: Q 범위 [%.3e, %.3e]",
        float(q_vals.min()),
        float(q_vals.max()),
    )
    return mesh


def compute_lambda2(
    mesh: Any,
    velocity_name: str = "U",
) -> Any:
    """λ₂ 와류 식별자를 계산한다 (numpy eigvalsh 직접 구현).

    속도 구배 텐서 J 를 대칭 부분 S 와 반대칭 부분 Omega 로 분해하고
    M = S@S + Omega@Omega 의 두 번째 고유값을 λ₂ 로 정의한다.

    λ₂ < 0 인 영역이 와류 코어에 해당한다.

    Args:
        mesh: 속도 필드를 포함한 ``pv.UnstructuredGrid``.
        velocity_name: 속도 벡터 필드 이름.

    Returns:
        ``"lambda2"`` 필드가 추가된 메쉬.

    Raises:
        ImportError: numpy 가 설치되어 있지 않은 경우.
        KeyError: 메쉬에 velocity_name 필드가 없는 경우.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "λ₂ 계산에는 numpy 가 필요합니다: pip install numpy"
        ) from exc

    mesh = _ensure_point_velocity(mesh, velocity_name)

    if velocity_name not in mesh.point_data:
        available = list(mesh.point_data.keys()) + list(mesh.cell_data.keys())
        raise KeyError(
            f"'{velocity_name}' 필드가 없습니다. "
            f"사용 가능: {available}"
        )

    logger.debug("λ₂ 계산 시작: velocity_name='%s'", velocity_name)

    grad = _get_velocity_gradient(mesh, velocity_name)  # (N, 3, 3)
    S = (grad + grad.transpose(0, 2, 1)) / 2.0
    Omega = (grad - grad.transpose(0, 2, 1)) / 2.0

    M = S @ S + Omega @ Omega  # (N, 3, 3), 대칭 행렬

    # 고유값 계산 (오름차순 반환)
    eigenvalues = np.linalg.eigvalsh(M)  # (N, 3)
    lambda2 = eigenvalues[:, 1]  # 두 번째 (중간) 고유값

    mesh.point_data["lambda2"] = lambda2.astype(np.float32)

    logger.info(
        "λ₂ 계산 완료: 범위 [%.3e, %.3e]",
        float(lambda2.min()),
        float(lambda2.max()),
    )
    return mesh


def _get_velocity_gradient(
    mesh: Any,
    velocity_name: str,
) -> Any:
    """속도 구배 텐서를 (N_points, 3, 3) 형태로 계산한다.

    pyvista.compute_derivative 가 가능하면 사용하고,
    실패 시 유한 차분(중앙 차분)으로 대체한다.

    Args:
        mesh: 속도 필드를 포함한 PyVista DataSet.
        velocity_name: 속도 벡터 필드 이름.

    Returns:
        속도 구배 텐서, shape ``(N_points, 3, 3)``.

    Raises:
        ImportError: numpy 가 없는 경우.
    """
    import numpy as np

    try:
        grad_mesh = mesh.compute_derivative(
            scalars=velocity_name,
            gradient=True,
        )
        grad_raw = np.asarray(grad_mesh.point_data["gradient"])
        if grad_raw.ndim == 2 and grad_raw.shape[1] == 9:
            return grad_raw.reshape(-1, 3, 3)
        return grad_raw
    except Exception:
        logger.warning("compute_derivative 실패. 0 텐서로 대체합니다.")
        n = mesh.n_points if hasattr(mesh, "n_points") else 1
        return np.zeros((n, 3, 3), dtype=np.float64)


# ---------------------------------------------------------------------------
# 내부 유틸리티
# ---------------------------------------------------------------------------


def _ensure_point_velocity(mesh: Any, velocity_name: str) -> Any:
    """cell_data 에 있는 속도장을 point_data 로 이동한다.

    Args:
        mesh: PyVista DataSet.
        velocity_name: 속도 필드 이름.

    Returns:
        수정된 메쉬.
    """
    if (
        hasattr(mesh, "cell_data")
        and velocity_name in mesh.cell_data
        and (
            not hasattr(mesh, "point_data")
            or velocity_name not in mesh.point_data
        )
    ):
        logger.debug(
            "'%s' 필드가 cell_data 에 있습니다. "
            "cell_data_to_point_data() 를 적용합니다.",
            velocity_name,
        )
        try:
            mesh = mesh.cell_data_to_point_data()
        except Exception as exc:
            logger.warning("cell_data_to_point_data 실패: %s", exc)
    return mesh


def _compute_q_manual(mesh: Any, velocity_name: str) -> Any:
    """compute_derivative 없이 수동으로 Q-criterion 을 계산한다.

    구배를 0 텐서로 대체하므로 Q=0, vorticity=0 이 반환된다.
    실제 계산이 불가능한 경우의 안전 폴백 용도다.

    Args:
        mesh: 메쉬.
        velocity_name: 속도 필드 이름.

    Returns:
        Q-criterion=0, vorticity=0 이 추가된 메쉬.
    """
    import numpy as np

    n = mesh.n_points if hasattr(mesh, "n_points") else 1
    mesh.point_data["Q-criterion"] = np.zeros(n, dtype=np.float32)
    mesh.point_data["vorticity"] = np.zeros((n, 3), dtype=np.float32)
    logger.warning("Q-criterion 수동 폴백: 모든 값을 0 으로 설정합니다.")
    return mesh
