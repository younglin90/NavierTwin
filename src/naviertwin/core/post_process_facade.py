"""Post-processor 통합 Facade — 신규 모듈에 대한 단일 API 표면.

R591-647에서 추가한 ~30개 후처리/AI/ROM 모듈의 핵심 기능을 단일 클래스로
노출. GUI나 CLI에서 파라미터 dict 하나로 모든 분석을 호출 가능.

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> u = np.sin(np.linspace(0, 4 * np.pi, 200)) + 0.1 * rng.standard_normal(200)
    >>> from naviertwin.core.post_process_facade import PostProcessFacade
    >>> facade = PostProcessFacade()
    >>> result = facade.run("psd_welch", signal=u, fs=100.0)
    >>> "frequency" in result and "psd" in result
    True
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class PostProcessFacade:
    """모든 신규 후처리 도구의 단일 호출 표면.

    각 메서드는 dict로 결과를 반환 (GUI 표시용 + JSON 직렬화 친화).

    사용법::

        facade = PostProcessFacade()
        ops = facade.list_operations()  # 사용 가능한 op 이름 list
        info = facade.describe(op_name)  # 파라미터 설명
        result = facade.run(op_name, **kwargs)
    """

    def list_operations(self) -> list[str]:
        """지원하는 모든 op 이름 반환."""
        return sorted(_OPERATIONS.keys())

    def describe(self, op_name: str) -> dict[str, Any]:
        """op의 인자 / 반환 명세를 dict로 반환."""
        if op_name not in _OPERATIONS:
            raise KeyError(f"unknown op '{op_name}'")
        spec = _OPERATIONS[op_name]
        return {
            "name": op_name,
            "category": spec["category"],
            "description": spec["description"],
            "params": spec["params"],
            "returns": spec["returns"],
        }

    def run(self, op_name: str, **kwargs: Any) -> dict[str, Any]:
        """op 실행. **kwargs는 op_spec의 params와 일치해야 한다."""
        if op_name not in _OPERATIONS:
            raise KeyError(f"unknown op '{op_name}'")
        try:
            return _OPERATIONS[op_name]["fn"](**kwargs)
        except TypeError as e:
            raise ValueError(
                f"invalid parameters for '{op_name}': {e}"
            ) from e


# ---------------------------------------------------------------------------
# Op 정의 — 신규 모듈을 dict로 등록
# ---------------------------------------------------------------------------


def _op_psd_welch(
    signal: NDArray[np.float64],
    fs: float = 1.0,
    nperseg: int | None = None,
    window: str = "hann",
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.psd import welch_psd

    f, P = welch_psd(signal, fs=fs, nperseg=nperseg, window=window)
    return {"frequency": f, "psd": P}


def _op_reynolds_stats(
    u: NDArray[np.float64],
    v: NDArray[np.float64] | None = None,
    w: NDArray[np.float64] | None = None,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.reynolds_stats import (
        mean_field,
        rms,
        turbulence_intensity,
        turbulent_kinetic_energy,
    )

    out: dict[str, Any] = {
        "mean": mean_field(u, axis=0),
        "rms": rms(u, axis=0),
    }
    if v is not None:
        out["tke"] = turbulent_kinetic_energy(u, v, w, axis=0)
        out["intensity"] = turbulence_intensity(u, v, w, axis=0)
    return out


def _op_quadrant_analysis(
    up: NDArray[np.float64],
    vp: NDArray[np.float64],
    hole: float = 0.0,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.quadrant_pdf import quadrant_split

    quads = quadrant_split(up, vp, hole=hole)
    return {"quadrants": quads}


def _op_kolmogorov_slope(
    signal: NDArray[np.float64],
    dx: float = 1.0,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.spectral_energy import (
        energy_spectrum_1d,
        kolmogorov_slope,
    )

    k, E = energy_spectrum_1d(signal, dx=dx)
    slope, r2 = kolmogorov_slope(k, E)
    return {"k": k, "E": E, "slope": slope, "r2": r2}


def _op_box_stats(
    x: NDArray[np.float64],
    whisker_factor: float = 1.5,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.quantile_stats import box_stats

    return {"box": box_stats(x, whisker_factor=whisker_factor)}


def _op_anomaly_mahalanobis(
    X: NDArray[np.float64],
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.anomaly_score import mahalanobis_score

    return {"scores": mahalanobis_score(X)}


def _op_ts_features(
    signal: NDArray[np.float64],
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.ts_features import extract_features

    return {"features": extract_features(signal)}


def _op_change_points(
    signal: NDArray[np.float64],
    n_changepoints: int = 1,
    method: str = "binary",
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.change_point import (
        binary_segmentation,
        pelt,
        segment_means,
    )

    if method == "binary":
        cps = binary_segmentation(signal, n_changepoints=n_changepoints)
    elif method == "pelt":
        cps = pelt(signal)
    else:
        raise ValueError(f"method must be 'binary' or 'pelt', got '{method}'")
    means = segment_means(signal, cps)
    return {"changepoints": cps, "segment_means": means}


def _op_denoise(
    signal: NDArray[np.float64],
    window_length: int = 11,
    polyorder: int = 3,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.denoise import savgol_filter

    return {"smoothed": savgol_filter(signal, window_length=window_length, polyorder=polyorder)}


def _op_phase_average(
    t: NDArray[np.float64],
    signal: NDArray[np.float64],
    period: float,
    n_bins: int = 36,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.phase_lock import phase_average

    phases, mean, rms = phase_average(t, signal, period=period, n_bins=n_bins)
    return {"phases": phases, "mean": mean, "rms": rms}


def _op_eof(
    X: NDArray[np.float64],
    n_modes: int = 5,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.eof_analysis import eof_decomposition

    eofs, pcs, var = eof_decomposition(X, n_modes=n_modes)
    return {"eofs": eofs, "pcs": pcs, "var_explained": var}


def _op_safe_eval(
    expression: str,
    variables: dict[str, NDArray[np.float64]],
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.expression_eval import safe_eval

    result = safe_eval(expression, variables)
    return {"result": np.asarray(result)}


def _op_two_point_acf(
    u: NDArray[np.float64],
    dx: float = 1.0,
    max_lag: int | None = None,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.two_point import (
        integral_length_scale_from_R,
        spatial_autocorrelation,
    )

    r, R = spatial_autocorrelation(u, dx=dx, max_lag=max_lag)
    L = integral_length_scale_from_R(r, R)
    return {"r": r, "R": R, "L_int": L}


def _op_running_moments(
    samples: list[NDArray[np.float64]] | NDArray[np.float64],
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.running_moments import RunningMoments

    arr = (
        np.asarray(samples, dtype=np.float64)
        if not isinstance(samples, list)
        else np.stack(samples)
    )
    rm = RunningMoments(shape=arr.shape[1:] if arr.ndim > 1 else ())
    for s in arr:
        rm.update(s)
    return {
        "mean": rm.mean,
        "std": rm.std,
        "n": rm.n,
    }


def _op_pod_truncation(
    singular_values: NDArray[np.float64],
    fraction: float = 0.99,
) -> dict[str, Any]:
    from naviertwin.core.dimensionality_reduction.truncation_criteria import (
        cumulative_energy_curve,
        truncate_by_energy,
    )

    r = truncate_by_energy(singular_values, fraction=fraction)
    curve = cumulative_energy_curve(singular_values)
    return {"n_modes": r, "cumulative_energy": curve}


def _op_quantile(
    x: NDArray[np.float64],
    q: float = 50.0,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.quantile_stats import percentile

    return {"value": percentile(x, q)}


def _op_critical_points(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    dx: float = 1.0,
    dy: float = 1.0,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.critical_points import find_critical_points

    cps = find_critical_points(u, v, dx=dx, dy=dy)
    return {"critical_points": cps, "count": len(cps)}


def _op_surface_forces(
    triangles: NDArray[np.float64],
    pressure: NDArray[np.float64],
    shear_traction: NDArray[np.float64] | None = None,
    rho: float = 1.225,
    u_inf: float = 10.0,
    area_ref: float = 1.0,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.surface_integrals import (
        force_coefficient,
        lift_drag_split,
        moment_about,
        total_force,
    )

    force = total_force(triangles, pressure, shear_traction=shear_traction)
    moment = moment_about(
        triangles,
        pressure,
        center=np.zeros(3, dtype=np.float64),
        shear_traction=shear_traction,
    )
    lift, drag = lift_drag_split(force, flow_direction=np.array([1.0, 0.0, 0.0]))
    return {
        "force": force,
        "moment": moment,
        "lift": lift,
        "drag": drag,
        "force_coefficient": force_coefficient(force, rho=rho, U_inf=u_inf, A_ref=area_ref),
    }


def _op_plane_flux(
    triangles: NDArray[np.float64],
    velocity: NDArray[np.float64],
    scalar: NDArray[np.float64],
    density: NDArray[np.float64] | float = 1.0,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.plane_flux import (
        area_average,
        kinetic_energy_flux,
        mass_flux,
        momentum_flux,
        scalar_flux,
        volumetric_flow_rate,
    )

    return {
        "mass_flux": mass_flux(triangles, velocity, density=density),
        "volumetric_flow_rate": volumetric_flow_rate(triangles, velocity),
        "momentum_flux": momentum_flux(triangles, velocity, density=density),
        "scalar_flux": scalar_flux(triangles, velocity, scalar, density=density),
        "kinetic_energy_flux": kinetic_energy_flux(triangles, velocity, density=density),
        "area_average": area_average(triangles, scalar),
    }


def _op_stat_convergence(
    signal: NDArray[np.float64],
    n_batches: int = 20,
    window: int = 100,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.stat_convergence import (
        autocorrelation_time,
        batch_means_se,
        effective_sample_size,
        geweke_diagnostic,
        plateau_detector,
    )

    mean, se = batch_means_se(signal, n_batches=n_batches)
    return {
        "mean": mean,
        "standard_error": se,
        "geweke_z": geweke_diagnostic(signal),
        "effective_sample_size": effective_sample_size(signal),
        "plateau_index": plateau_detector(signal, window=window),
        "autocorrelation_time": autocorrelation_time(signal),
    }


def _op_time_interp(
    snapshots: NDArray[np.float64],
    times: NDArray[np.float64],
    t_query: float = 0.5,
    n_uniform: int = 8,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.time_interp import (
        interp_field,
        resample_uniform,
        time_average_window,
    )

    t_uniform, uniform = resample_uniform(snapshots, times, n_uniform=n_uniform)
    return {
        "interpolated": interp_field(snapshots, times, t_query=t_query),
        "uniform_times": t_uniform,
        "uniform_snapshots": uniform,
        "window_average": time_average_window(
            snapshots,
            times,
            t_center=float(np.mean(times)),
            half_width=float((times[-1] - times[0]) / 4.0),
        ),
    }


def _op_coord_transform(
    xyz: NDArray[np.float64],
    vectors: NDArray[np.float64],
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.coord_transform import (
        axis_align_rotation,
        cart_to_cyl,
        cart_to_sph,
        vector_cart_to_cyl,
    )

    return {
        "cylindrical": cart_to_cyl(xyz),
        "spherical": cart_to_sph(xyz),
        "vector_cylindrical": vector_cart_to_cyl(vectors, xyz),
        "axis_align_rotation": axis_align_rotation(np.array([1.0, 1.0, 1.0])),
    }


def _op_line_probe(
    points: NDArray[np.float64],
    field: NDArray[np.float64],
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    n_samples: int = 16,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.slice_extract import (
        line_probe,
        polyline_arc_length,
        slice_plane_arbitrary,
    )

    line_points, sampled = line_probe(
        points,
        field,
        start=start,
        end=end,
        n_samples=n_samples,
        method="idw",
    )
    plane_points, plane_field = slice_plane_arbitrary(
        points,
        field,
        plane_origin=np.zeros(3, dtype=np.float64),
        plane_normal=np.array([0.0, 0.0, 1.0]),
        tolerance=0.2,
    )
    return {
        "line_points": line_points,
        "sampled": sampled,
        "arc_length": polyline_arc_length(line_points),
        "plane_points": plane_points,
        "plane_field": plane_field,
    }


def _op_gof_normality(
    x: NDArray[np.float64],
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.goodness_of_fit import (
        anderson_darling_normal,
        chi_square_test,
        ks_test_normal,
        shapiro_wilk_simplified,
    )

    ks_d, ks_p = ks_test_normal(x)
    ad_a2, ad_crit = anderson_darling_normal(x)
    shapiro_w, shapiro_p = shapiro_wilk_simplified(x[: min(len(x), 50)])
    observed, _ = np.histogram(x, bins=5)
    expected = np.full(5, observed.sum() / 5.0)
    chi2, dof = chi_square_test(observed, expected)
    return {
        "ks_d": ks_d,
        "ks_p": ks_p,
        "anderson_a2": ad_a2,
        "anderson_critical": ad_crit,
        "shapiro_w": shapiro_w,
        "shapiro_p": shapiro_p,
        "chi_square": chi2,
        "chi_square_dof": dof,
    }


def _op_conditional_sampling(
    signal: NDArray[np.float64],
    threshold: float = 0.0,
    half_window: int = 8,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.conditional_sampling import (
        conditional_average,
        event_duration_stats,
        find_threshold_crossings,
        trigger_average,
    )

    triggers = find_threshold_crossings(signal, threshold=threshold, direction="rising")
    avg, count = trigger_average(signal, triggers, half_window=half_window)
    cond_avg, n_cond = conditional_average(signal, signal > threshold)
    return {
        "triggers": triggers,
        "trigger_average": avg,
        "trigger_count": count,
        "conditional_average": cond_avg,
        "conditional_count": n_cond,
        "event_stats": event_duration_stats(signal > threshold, dt=1.0),
    }


def _op_grid_derivatives(
    field_2d: NDArray[np.float64],
    dx: float = 1.0,
    dy: float = 1.0,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.grid_derivatives import (
        gradient_2d,
        laplacian_2d,
    )

    gx, gy = gradient_2d(field_2d, dx=dx, dy=dy, order=2)
    return {"gradient_x": gx, "gradient_y": gy, "laplacian": laplacian_2d(field_2d, dx, dy)}


def _op_anisotropy_state(
    reynolds_stress: NDArray[np.float64],
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.anisotropy import (
        anisotropy_tensor,
        barycentric_coordinates,
        invariants_II_III,
        is_realizable,
        lumley_eta_xi,
        turbulence_state,
    )

    b = anisotropy_tensor(reynolds_stress)
    ii, iii = invariants_II_III(b)
    eta, xi = lumley_eta_xi(b)
    c1, c2, c3 = barycentric_coordinates(b)
    return {
        "anisotropy_tensor": b,
        "II": ii,
        "III": iii,
        "eta": eta,
        "xi": xi,
        "state": turbulence_state(b),
        "realizable": is_realizable(b),
        "barycentric": np.array([c1, c2, c3], dtype=np.float64),
    }


def _op_morphology_components(
    field: NDArray[np.float64],
    threshold: float = 0.0,
    min_size: int = 4,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.morphology import (
        binary_closing_2d,
        binary_opening_2d,
        component_sizes,
        connected_components_2d,
        remove_small_components,
        threshold_to_mask,
    )

    mask = threshold_to_mask(field, threshold=threshold, mode="above")
    opened = binary_opening_2d(mask)
    closed = binary_closing_2d(opened)
    labels, n_components = connected_components_2d(closed)
    return {
        "mask": mask,
        "opened": opened,
        "closed": closed,
        "labels": labels,
        "n_components": n_components,
        "component_sizes": component_sizes(labels),
        "filtered": remove_small_components(closed, min_size=min_size),
    }


def _op_cell_volume_integrals(
    vertices: NDArray[np.float64],
    connectivity: NDArray[np.int_],
    field: NDArray[np.float64],
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.cell_volume import (
        cell_centroids,
        tet_volumes_batch,
        volume_average,
        volume_integral,
        volume_weighted_variance,
    )

    volumes = tet_volumes_batch(vertices, connectivity)
    return {
        "volumes": volumes,
        "centroids": cell_centroids(vertices, connectivity),
        "integral": volume_integral(volumes, field),
        "average": volume_average(volumes, field),
        "variance": volume_weighted_variance(volumes, field),
    }


def _op_mass_search(
    query: NDArray[np.float64],
    series: NDArray[np.float64],
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.ts_similarity import mass_search

    dist = mass_search(query, series)
    return {
        "distance_profile": dist,
        "best_match_index": int(np.argmin(dist)),
        "best_match_distance": float(dist.min()),
    }


def _op_find_motifs(
    series: NDArray[np.float64],
    window: int = 30,
    k: int = 1,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.ts_similarity import find_top_k_motifs

    motifs = find_top_k_motifs(series, window=window, k=k)
    return {"motifs": motifs, "n_motifs": len(motifs)}


def _op_auto_report_probe(
    signal: NDArray[np.float64],
    fs: float = 1.0,
    period_hint: float | None = None,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.auto_report import (
        analyze_probe_signal,
        to_markdown,
    )

    report = analyze_probe_signal(signal, fs=fs, period_hint=period_hint)
    return {"report": report, "markdown": to_markdown(report)}


def _op_auto_report_field(
    X: NDArray[np.float64],
    n_modes: int = 5,
) -> dict[str, Any]:
    from naviertwin.core.flow_analysis.auto_report import (
        analyze_field_snapshots,
        to_markdown,
    )

    report = analyze_field_snapshots(X, n_modes=n_modes)
    return {"report": report, "markdown": to_markdown(report)}


_OPERATIONS: dict[str, dict[str, Any]] = {
    "mass_search": {
        "fn": _op_mass_search,
        "category": "similarity",
        "description": "MASS 시계열 유사 패턴 검색 (FFT)",
        "params": ["query", "series"],
        "returns": ["distance_profile", "best_match_index", "best_match_distance"],
    },
    "find_motifs": {
        "fn": _op_find_motifs,
        "category": "similarity",
        "description": "Top-K 시계열 모티프 검색",
        "params": ["series", "window", "k"],
        "returns": ["motifs", "n_motifs"],
    },
    "auto_report_probe": {
        "fn": _op_auto_report_probe,
        "category": "report",
        "description": "Probe 시계열 종합 자동 보고서 (markdown)",
        "params": ["signal", "fs", "period_hint"],
        "returns": ["report", "markdown"],
    },
    "auto_report_field": {
        "fn": _op_auto_report_field,
        "category": "report",
        "description": "(n_t, n_x) 스냅샷 ROM/통계 자동 보고서",
        "params": ["X", "n_modes"],
        "returns": ["report", "markdown"],
    },
    "psd_welch": {
        "fn": _op_psd_welch,
        "category": "spectral",
        "description": "Welch 파워 스펙트럼 밀도",
        "params": ["signal", "fs", "nperseg", "window"],
        "returns": ["frequency", "psd"],
    },
    "reynolds_stats": {
        "fn": _op_reynolds_stats,
        "category": "statistics",
        "description": "Reynolds 분해 + TKE + intensity",
        "params": ["u", "v?", "w?"],
        "returns": ["mean", "rms", "tke?", "intensity?"],
    },
    "quadrant_analysis": {
        "fn": _op_quadrant_analysis,
        "category": "statistics",
        "description": "u'v' 사분면 분석 (Q1-Q4)",
        "params": ["up", "vp", "hole"],
        "returns": ["quadrants"],
    },
    "kolmogorov_slope": {
        "fn": _op_kolmogorov_slope,
        "category": "spectral",
        "description": "에너지 스펙트럼 + Kolmogorov -5/3 적합",
        "params": ["signal", "dx"],
        "returns": ["k", "E", "slope", "r2"],
    },
    "box_stats": {
        "fn": _op_box_stats,
        "category": "statistics",
        "description": "Tukey 박스플롯 통계",
        "params": ["x", "whisker_factor"],
        "returns": ["box"],
    },
    "anomaly_mahalanobis": {
        "fn": _op_anomaly_mahalanobis,
        "category": "anomaly",
        "description": "Mahalanobis 다변량 이상치 점수",
        "params": ["X"],
        "returns": ["scores"],
    },
    "ts_features": {
        "fn": _op_ts_features,
        "category": "features",
        "description": "시계열 18 특성 추출",
        "params": ["signal"],
        "returns": ["features"],
    },
    "change_points": {
        "fn": _op_change_points,
        "category": "anomaly",
        "description": "변화점 검출 (binary/PELT)",
        "params": ["signal", "n_changepoints", "method"],
        "returns": ["changepoints", "segment_means"],
    },
    "denoise": {
        "fn": _op_denoise,
        "category": "preprocessing",
        "description": "Savitzky-Golay 평활",
        "params": ["signal", "window_length", "polyorder"],
        "returns": ["smoothed"],
    },
    "phase_average": {
        "fn": _op_phase_average,
        "category": "statistics",
        "description": "위상잠금 평균",
        "params": ["t", "signal", "period", "n_bins"],
        "returns": ["phases", "mean", "rms"],
    },
    "eof": {
        "fn": _op_eof,
        "category": "rom",
        "description": "Empirical Orthogonal Functions 분해",
        "params": ["X", "n_modes"],
        "returns": ["eofs", "pcs", "var_explained"],
    },
    "safe_eval": {
        "fn": _op_safe_eval,
        "category": "preprocessing",
        "description": "사용자 표현식 평가 (AST sandbox)",
        "params": ["expression", "variables"],
        "returns": ["result"],
    },
    "two_point_acf": {
        "fn": _op_two_point_acf,
        "category": "statistics",
        "description": "공간 두-점 상관 + 적분 길이",
        "params": ["u", "dx", "max_lag"],
        "returns": ["r", "R", "L_int"],
    },
    "running_moments": {
        "fn": _op_running_moments,
        "category": "statistics",
        "description": "Welford 누적 평균/분산",
        "params": ["samples"],
        "returns": ["mean", "std", "n"],
    },
    "pod_truncation": {
        "fn": _op_pod_truncation,
        "category": "rom",
        "description": "에너지 기반 POD 절단 차수",
        "params": ["singular_values", "fraction"],
        "returns": ["n_modes", "cumulative_energy"],
    },
    "quantile": {
        "fn": _op_quantile,
        "category": "statistics",
        "description": "분위수 계산",
        "params": ["x", "q"],
        "returns": ["value"],
    },
    "critical_points": {
        "fn": _op_critical_points,
        "category": "topology",
        "description": "벡터장 임계점 검출 + 분류",
        "params": ["u", "v", "dx", "dy"],
        "returns": ["critical_points", "count"],
    },
    "surface_forces": {
        "fn": _op_surface_forces,
        "category": "integrals",
        "description": "표면 압력/전단 적분으로 힘, 모멘트, lift/drag 계산",
        "params": ["triangles", "pressure", "shear_traction?", "rho", "u_inf", "area_ref"],
        "returns": ["force", "moment", "lift", "drag", "force_coefficient"],
    },
    "plane_flux": {
        "fn": _op_plane_flux,
        "category": "integrals",
        "description": "평면 통과 질량/체적/운동량/스칼라/운동에너지 플럭스",
        "params": ["triangles", "velocity", "scalar", "density"],
        "returns": [
            "mass_flux",
            "volumetric_flow_rate",
            "momentum_flux",
            "scalar_flux",
            "kinetic_energy_flux",
            "area_average",
        ],
    },
    "stat_convergence": {
        "fn": _op_stat_convergence,
        "category": "statistics",
        "description": "배치 평균, Geweke, ESS, plateau, 자기상관 시간 수렴 진단",
        "params": ["signal", "n_batches", "window"],
        "returns": [
            "mean",
            "standard_error",
            "geweke_z",
            "effective_sample_size",
            "plateau_index",
            "autocorrelation_time",
        ],
    },
    "time_interp": {
        "fn": _op_time_interp,
        "category": "preprocessing",
        "description": "스냅샷 시간 보간, 균일 재샘플링, 시간창 평균",
        "params": ["snapshots", "times", "t_query", "n_uniform"],
        "returns": ["interpolated", "uniform_times", "uniform_snapshots", "window_average"],
    },
    "coord_transform": {
        "fn": _op_coord_transform,
        "category": "geometry",
        "description": "Cartesian 좌표/벡터를 원통/구면 좌표계로 변환",
        "params": ["xyz", "vectors"],
        "returns": ["cylindrical", "spherical", "vector_cylindrical", "axis_align_rotation"],
    },
    "line_probe": {
        "fn": _op_line_probe,
        "category": "sampling",
        "description": "임의 라인 probe와 평면 slice 추출",
        "params": ["points", "field", "start", "end", "n_samples"],
        "returns": ["line_points", "sampled", "arc_length", "plane_points", "plane_field"],
    },
    "gof_normality": {
        "fn": _op_gof_normality,
        "category": "statistics",
        "description": "KS, Anderson-Darling, Shapiro-Wilk, chi-square 적합도 진단",
        "params": ["x"],
        "returns": [
            "ks_d",
            "ks_p",
            "anderson_a2",
            "anderson_critical",
            "shapiro_w",
            "shapiro_p",
            "chi_square",
            "chi_square_dof",
        ],
    },
    "conditional_sampling": {
        "fn": _op_conditional_sampling,
        "category": "sampling",
        "description": "임계값 trigger, 조건부 평균, 이벤트 지속시간 통계",
        "params": ["signal", "threshold", "half_window"],
        "returns": [
            "triggers",
            "trigger_average",
            "trigger_count",
            "conditional_average",
            "conditional_count",
            "event_stats",
        ],
    },
    "grid_derivatives": {
        "fn": _op_grid_derivatives,
        "category": "topology",
        "description": "균일 격자 2D gradient와 Laplacian 계산",
        "params": ["field_2d", "dx", "dy"],
        "returns": ["gradient_x", "gradient_y", "laplacian"],
    },
    "anisotropy_state": {
        "fn": _op_anisotropy_state,
        "category": "turbulence",
        "description": "Reynolds 응력 비등방성, Lumley, barycentric 상태 진단",
        "params": ["reynolds_stress"],
        "returns": [
            "anisotropy_tensor",
            "II",
            "III",
            "eta",
            "xi",
            "state",
            "realizable",
            "barycentric",
        ],
    },
    "morphology_components": {
        "fn": _op_morphology_components,
        "category": "topology",
        "description": "2D threshold mask, opening/closing, connected components",
        "params": ["field", "threshold", "min_size"],
        "returns": [
            "mask",
            "opened",
            "closed",
            "labels",
            "n_components",
            "component_sizes",
            "filtered",
        ],
    },
    "cell_volume_integrals": {
        "fn": _op_cell_volume_integrals,
        "category": "integrals",
        "description": "사면체 cell volume, centroid, volume integral/average/variance",
        "params": ["vertices", "connectivity", "field"],
        "returns": ["volumes", "centroids", "integral", "average", "variance"],
    },
}


__all__ = ["PostProcessFacade"]
