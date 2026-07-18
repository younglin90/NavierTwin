"""Parametric steady-state CFD test dataset generator.

동일 geometry에 대해 inlet velocity만 바뀌는 steady-state VTU 컬렉션을 만든다.
각 케이스는 cell_data로 pressure ``p``와 velocity ``U``를 갖고, GUI 학습 편의를
위해 ``Ux``, ``Uy``, ``U_mag`` scalar field도 함께 저장한다.

Usage:
    python3 scripts/make_parametric_steady_dataset.py [out_dir]

Output:
    steady_inlet_series.pvd       - inlet_u 값을 timestep으로 쓰는 VTU 컬렉션
    cases/case_0000.vtu ...       - 동일 geometry steady-state case snapshots
    params.csv                    - build-twin용 inlet parameter table
    csv_snapshots/case_0000.csv   - CLI/GUI Tools용 scalar snapshot CSV
    README.md                     - GUI 테스트 절차 요약
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def make_parametric_steady_dataset(
    out_dir: Path,
    *,
    nx: int = 32,
    ny: int = 16,
    inlet_values: np.ndarray | None = None,
) -> Path:
    """Generate steady-state VTU snapshots for several inlet velocities."""
    import pyvista as pv

    inlet_values = (
        np.linspace(0.6, 2.0, 12, dtype=float)
        if inlet_values is None
        else np.asarray(inlet_values, dtype=float)
    )
    case_dir = out_dir / "cases"
    csv_dir = out_dir / "csv_snapshots"
    case_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    grid_cls = getattr(pv, "ImageData", None) or pv.UniformGrid
    base = grid_cls(
        dimensions=(nx + 1, ny + 1, 1),
        spacing=(1.0 / nx, 1.0 / ny, 1.0),
    ).cast_to_unstructured_grid()

    centers = base.cell_centers().points
    x = centers[:, 0]
    y = centers[:, 1]
    y_profile = 4.0 * y * (1.0 - y)
    shear_layer = np.sin(np.pi * x) * np.sin(2.0 * np.pi * y)

    pvd_entries: list[tuple[float, str]] = []
    params_rows = ["case_id,inlet_u"]
    for idx, inlet_u in enumerate(inlet_values):
        mesh = base.copy(deep=True)
        u_x = inlet_u * y_profile * (1.0 - 0.18 * x)
        u_x += 0.08 * inlet_u**2 * shear_layer
        u_y = 0.06 * inlet_u * np.sin(2.0 * np.pi * x) * y * (1.0 - y)
        u_z = np.zeros_like(u_x)
        u_mag = np.sqrt(u_x**2 + u_y**2)
        pressure = 1.0 + 0.5 * inlet_u**2 * (1.0 - x)
        pressure += 0.04 * inlet_u * np.cos(np.pi * y)
        pressure += 0.03 * inlet_u**2 * np.sin(2.0 * np.pi * x) * np.sin(np.pi * y)

        mesh.cell_data["p"] = pressure.astype(np.float32)
        mesh.cell_data["U"] = np.column_stack([u_x, u_y, u_z]).astype(np.float32)
        mesh.cell_data["Ux"] = u_x.astype(np.float32)
        mesh.cell_data["Uy"] = u_y.astype(np.float32)
        mesh.cell_data["U_mag"] = u_mag.astype(np.float32)
        mesh.cell_data["inlet_u"] = np.full(mesh.n_cells, inlet_u, dtype=np.float32)

        file_name = f"case_{idx:04d}.vtu"
        mesh.save(str(case_dir / file_name))
        pvd_entries.append((float(inlet_u), f"cases/{file_name}"))
        params_rows.append(f"{idx},{inlet_u:.8f}")
        _write_case_csv(
            csv_dir / f"case_{idx:04d}.csv",
            centers=centers,
            inlet_u=float(inlet_u),
            p=pressure,
            ux=u_x,
            uy=u_y,
            u_mag=u_mag,
        )

    pvd_path = out_dir / "steady_inlet_series.pvd"
    _write_pvd(pvd_path, pvd_entries)
    (out_dir / "params.csv").write_text("\n".join(params_rows) + "\n", encoding="utf-8")
    _write_readme(out_dir, pvd_path, inlet_values)
    return pvd_path


def _write_case_csv(
    path: Path,
    *,
    centers: np.ndarray,
    inlet_u: float,
    p: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    u_mag: np.ndarray,
) -> None:
    """Write one scalar snapshot CSV with cell centers and field columns."""
    lines = ["x,y,z,inlet_u,p,Ux,Uy,U_mag"]
    for center, p_val, ux_val, uy_val, mag_val in zip(
        centers,
        p,
        ux,
        uy,
        u_mag,
        strict=True,
    ):
        lines.append(
            f"{center[0]:.8f},{center[1]:.8f},{center[2]:.8f},"
            f"{inlet_u:.8f},{p_val:.8f},{ux_val:.8f},{uy_val:.8f},{mag_val:.8f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pvd(path: Path, entries: list[tuple[float, str]]) -> None:
    """Write a ParaView PVD collection file."""
    rows = "\n".join(
        f'    <DataSet timestep="{time_value}" group="" part="0" file="{file_name}"/>'
        for time_value, file_name in entries
    )
    path.write_text(
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
        "  <Collection>\n"
        f"{rows}\n"
        "  </Collection>\n"
        "</VTKFile>\n",
        encoding="utf-8",
    )


def _write_readme(out_dir: Path, pvd_path: Path, inlet_values: np.ndarray) -> None:
    """Write a compact GUI test guide next to the generated dataset."""
    text = f"""# NavierTwin Parametric Steady-State Test Dataset

Goal: same geometry, cell-wise p/U, multiple steady-state results by inlet velocity.

Main GUI input:
- {pvd_path.name}

Fields:
- p: cell pressure scalar
- U: cell velocity vector
- Ux, Uy, U_mag: scalar components for easier scalar surrogate training
- inlet_u: per-cell copy of the inlet velocity

Inlet range:
- min={float(np.min(inlet_values)):.4f}
- max={float(np.max(inlet_values)):.4f}
- samples={len(inlet_values)}

Recommended GUI flow:
1. Import tab: load `{pvd_path}`.
2. Analyze tab: select p, U, U_mag and verify the same mesh is shown for each inlet case.
3. Reduce tab: select `p` or `U_mag`, method `Snapshot POD`, modes 4-6, run reduction.
4. Model tab: click `모델 학습`. It uses the Reduce artifact and PVD timestep values as inlet_u.
5. Twin tab: set parameter count to 1, enter an inlet velocity such as 1.35, then run prediction.
6. Export tab: save the TwinEngine/project if needed.

Notes:
- Training `U` directly currently uses vector magnitude in the generic snapshot extractor.
- To inspect components separately, train `Ux`, `Uy`, or `U_mag`.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "tmp/naviertwin_parametric_steady")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"출력 디렉토리: {out_dir}")
    try:
        pvd_path = make_parametric_steady_dataset(out_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"생성 실패: {exc}")
        return 1
    print(f"  OK {pvd_path}")
    print(f"  OK {out_dir / 'params.csv'}")
    print(f"  OK {out_dir / 'csv_snapshots'}")
    print(f"  OK {out_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
