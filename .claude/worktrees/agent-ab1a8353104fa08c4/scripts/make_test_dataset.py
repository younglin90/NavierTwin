"""GUI 검증용 합성 CFD 결과 파일 생성기.

합성 cavity flow 비슷한 데이터를 VTU + ntwin (HDF5) 형식으로 저장.
GUI에서 ① Import → 이 파일을 로드해 모든 탭 검증.

Usage:
    python3 scripts/make_test_dataset.py [out_dir]

Output (default: /tmp/naviertwin_demo/):
    cavity.vtu                         — 정상 상태 VTU (U/p/T fields)
    cavity_time.npz                    — 시간-공간 필드 행렬 (n_t=50, n_x=400)
    cavity_probe.csv                   — 단일 프로브 시계열 (CSV)
    time_series/cavity_series.pvd      — time-series VTU 컬렉션
    time_series/cavity_0000.vtu ...    — 각 timestep VTU 스냅샷
    time_series/params.csv             — build-twin 입력 파라미터 예시
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def make_cavity_vtu(out_path: Path, nx: int = 20, ny: int = 20) -> None:
    """Lid-driven cavity 비슷한 정상 상태 필드를 VTU로 저장."""
    import pyvista as pv

    grid_cls = getattr(pv, "ImageData", None) or pv.UniformGrid
    grid = grid_cls(
        dimensions=(nx, ny, 1),
        spacing=(1.0 / (nx - 1), 1.0 / (ny - 1), 1.0),
    )
    pts = grid.points
    x, y = pts[:, 0], pts[:, 1]
    # 회전 와류 비슷한 패턴
    U = np.column_stack([
        np.sin(np.pi * x) * np.cos(np.pi * y),
        -np.cos(np.pi * x) * np.sin(np.pi * y),
        np.zeros_like(x),
    ]).astype(np.float32)
    p = (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)) * 0.5
    T = 300.0 + 10.0 * np.sin(np.pi * x) * np.sin(np.pi * y)
    grid.point_data["U"] = U
    grid.point_data["p"] = p.astype(np.float32)
    grid.point_data["T"] = T.astype(np.float32)
    grid.point_data["wallShearStress"] = np.column_stack([
        0.5 * np.cos(np.pi * x), 0.0 * x, 0.0 * x,
    ]).astype(np.float32)
    ug = grid.cast_to_unstructured_grid()
    ug.save(str(out_path))


def make_time_series_npz(out_path: Path, n_t: int = 50, n_x: int = 400) -> None:
    """(n_t, n_x) 시공간 행렬 + 시간 배열."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, n_x)
    t = np.linspace(0, 5, n_t)
    # rank 5 진짜 모드 + 노이즈
    spatial = np.column_stack([
        np.sin(np.pi * x), np.sin(2 * np.pi * x),
        np.sin(3 * np.pi * x), np.sin(5 * np.pi * x),
        np.cos(np.pi * x),
    ])
    temporal = np.column_stack([
        np.sin(2 * np.pi * t), np.cos(2 * np.pi * t),
        np.sin(4 * np.pi * t), np.exp(-t / 2),
        np.sin(6 * np.pi * t),
    ])
    X = temporal @ spatial.T + 0.05 * rng.standard_normal((n_t, n_x))
    np.savez_compressed(str(out_path), times=t, X=X.astype(np.float32))


def make_cavity_time_series_vtu(
    out_dir: Path,
    nx: int = 20,
    ny: int = 20,
    n_t: int = 12,
) -> Path:
    """동일 토폴로지 VTU 스냅샷 모음과 PVD 컬렉션을 생성한다."""
    import pyvista as pv

    series_dir = out_dir / "time_series"
    series_dir.mkdir(parents=True, exist_ok=True)

    grid_cls = getattr(pv, "ImageData", None) or pv.UniformGrid
    base_grid = grid_cls(
        dimensions=(nx, ny, 1),
        spacing=(1.0 / (nx - 1), 1.0 / (ny - 1), 1.0),
    )
    pts = base_grid.points
    x, y = pts[:, 0], pts[:, 1]
    times = np.linspace(0.0, 2.0, n_t)

    entries: list[tuple[float, str]] = []
    params = ["time,lid_velocity,pressure_gradient"]
    for index, time_value in enumerate(times):
        phase = 2.0 * np.pi * time_value
        lid_velocity = 1.0 + 0.25 * np.sin(phase)
        pressure_gradient = 0.15 * np.cos(0.5 * phase)
        swirl = 1.0 + 0.18 * np.sin(1.7 * phase)

        u_x = lid_velocity * np.sin(np.pi * x) * np.cos(np.pi * y)
        u_x += 0.08 * np.sin(phase) * y * (1.0 - y)
        u_y = -swirl * np.cos(np.pi * x) * np.sin(np.pi * y)
        u_y += 0.05 * np.cos(phase) * x * (1.0 - x)
        u_z = np.zeros_like(x)
        pressure = 0.5 * (np.cos(2.0 * np.pi * x) + np.cos(2.0 * np.pi * y))
        pressure += pressure_gradient * (x - 0.5)
        temperature = 300.0 + 8.0 * np.sin(np.pi * x) * np.sin(np.pi * y)
        temperature += 2.0 * np.sin(phase) * x
        vorticity = -np.pi * (lid_velocity + swirl) * np.sin(np.pi * x)
        vorticity *= np.sin(np.pi * y)

        grid = base_grid.copy(deep=True)
        grid.point_data["U"] = np.column_stack([u_x, u_y, u_z]).astype(np.float32)
        grid.point_data["p"] = pressure.astype(np.float32)
        grid.point_data["T"] = temperature.astype(np.float32)
        grid.point_data["vorticity"] = vorticity.astype(np.float32)
        grid.point_data["wallShearStress"] = np.column_stack(
            [
                lid_velocity * np.cos(np.pi * x),
                pressure_gradient * np.ones_like(x),
                np.zeros_like(x),
            ]
        ).astype(np.float32)

        file_name = f"cavity_{index:04d}.vtu"
        grid.cast_to_unstructured_grid().save(str(series_dir / file_name))
        entries.append((float(time_value), file_name))
        params.append(f"{time_value:.8f},{lid_velocity:.8f},{pressure_gradient:.8f}")

    pvd_path = series_dir / "cavity_series.pvd"
    _write_pvd(pvd_path, entries)
    (series_dir / "params.csv").write_text("\n".join(params) + "\n", encoding="utf-8")
    return pvd_path


def make_probe_csv(out_path: Path, n: int = 2000) -> None:
    """단일 probe 시계열 (PSD/change-points/anomaly 검증용)."""
    rng = np.random.default_rng(1)
    t = np.linspace(0, 20, n)
    sig = (
        np.sin(2 * np.pi * 5 * t)
        + 0.3 * np.sin(2 * np.pi * 17 * t)
        + 0.1 * rng.standard_normal(n)
    )
    # 두 번째 절반에 평균 변화 + spike
    sig[n // 2:] += 1.5
    sig[1500] = 10.0
    out_path.write_text(
        "time,signal\n" + "\n".join(
            f"{ti:.6f},{si:.6f}" for ti, si in zip(t, sig)
        ),
        encoding="utf-8",
    )


def _write_pvd(path: Path, entries: list[tuple[float, str]]) -> None:
    """naviertwin import 없이도 실행되도록 PVD writer를 fallback 제공한다."""
    try:
        from naviertwin.core.cfd_reader.vtk_pvd_writer import write_pvd
    except ImportError:
        rows = "\n".join(
            (
                f'    <DataSet timestep="{time_value}" group="" '
                f'part="0" file="{file_name}"/>'
            )
            for time_value, file_name in entries
        )
        path.write_text(
            "<?xml version=\"1.0\"?>\n"
            "<VTKFile type=\"Collection\" version=\"0.1\" "
            "byte_order=\"LittleEndian\">\n"
            "  <Collection>\n"
            f"{rows}\n"
            "  </Collection>\n"
            "</VTKFile>\n",
            encoding="utf-8",
        )
    else:
        write_pvd(path, entries)


def main() -> int:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/naviertwin_demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"출력 디렉토리: {out_dir}")
    try:
        vtu_path = out_dir / "cavity.vtu"
        make_cavity_vtu(vtu_path)
        print(f"  ✅ {vtu_path}")
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ cavity.vtu: {e} (pyvista 필요)")

    npz_path = out_dir / "cavity_time.npz"
    make_time_series_npz(npz_path)
    print(f"  ✅ {npz_path}")

    csv_path = out_dir / "cavity_probe.csv"
    make_probe_csv(csv_path)
    print(f"  ✅ {csv_path}")

    try:
        pvd_path = make_cavity_time_series_vtu(out_dir)
        print(f"  ✅ {pvd_path}")
        print(f"  ✅ {pvd_path.parent / 'params.csv'}")
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ time_series/cavity_series.pvd: {e} (pyvista 필요)")

    print("\n사용법:")
    print("  GUI 실행: PYTHONPATH=src python3 -m naviertwin --gui")
    print(f"  ① Import 탭에서 {vtu_path} 로드")
    print(f"  ② Time-series 검증은 {out_dir / 'time_series' / 'cavity_series.pvd'} 로드")
    return 0


if __name__ == "__main__":
    sys.exit(main())
