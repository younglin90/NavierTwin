"""자동 후처리 보고서 생성 — CFD 시계열에서 모든 핵심 분석을 한 번에.

probe 시계열, 1D/2D 필드 등을 받아 표준 후처리 분석 (통계, 스펙트럼,
이상치, regime, 모드 분해)을 일괄 실행하고 markdown 보고서를 생성한다.
GUI / batch CI 양쪽에서 사용 가능.

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> u = np.sin(np.linspace(0, 4*np.pi, 1000)) + 0.1*rng.standard_normal(1000)
    >>> from naviertwin.core.flow_analysis.auto_report import (
    ...     analyze_probe_signal, to_markdown
    ... )
    >>> report = analyze_probe_signal(u, fs=100.0, period_hint=0.5)
    >>> md = to_markdown(report)
    >>> "Statistics" in md
    True
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def analyze_probe_signal(
    signal: NDArray[np.float64],
    fs: float = 1.0,
    period_hint: float | None = None,
    detect_changes: bool = True,
) -> dict[str, Any]:
    """단일 probe 시계열 종합 분석.

    Args:
        signal: (N,) 시계열.
        fs: 샘플링 주파수.
        period_hint: 위상-잠금 평균에 사용할 주기 (옵션).
        detect_changes: 변화점 검출 수행 여부.

    Returns:
        종합 결과 dict.

    Raises:
        ValueError: signal이 1D 아님 또는 너무 짧음.
    """
    s = np.asarray(signal, dtype=np.float64).ravel()
    if len(s) < 16:
        raise ValueError(f"signal too short: {len(s)} (need >= 16)")

    from naviertwin.core.flow_analysis.psd import welch_psd
    from naviertwin.core.flow_analysis.quantile_stats import box_stats
    from naviertwin.core.flow_analysis.ts_features import extract_features

    report: dict[str, Any] = {
        "n_samples": int(len(s)),
        "fs": float(fs),
        "duration": float(len(s) / fs),
    }

    # 1. 시계열 특성 (18개)
    report["features"] = extract_features(s)

    # 2. 박스플롯 통계
    report["box_stats"] = box_stats(s)

    # 3. PSD
    nperseg = min(256, len(s) // 2)
    f, P = welch_psd(s, fs=fs, nperseg=nperseg)
    # 피크 주파수
    if len(f) > 1:
        peak_idx = int(np.argmax(P[1:]) + 1)
        report["psd"] = {
            "peak_frequency": float(f[peak_idx]),
            "peak_psd": float(P[peak_idx]),
            "n_freq_bins": int(len(f)),
            "frequency_range": (float(f[0]), float(f[-1])),
        }

    # 4. 위상 잠금 평균 (period 주어진 경우)
    if period_hint is not None and period_hint > 0:
        from naviertwin.core.flow_analysis.phase_lock import phase_average

        if len(s) / fs >= 2 * period_hint:
            t = np.arange(len(s)) / fs
            phases, mean, rms = phase_average(t, s, period=period_hint, n_bins=20)
            report["phase_lock"] = {
                "period": float(period_hint),
                "n_bins": int(len(phases)),
                "phase_avg_amplitude": float(mean.max() - mean.min()),
                "rms_max": float(rms.max()),
            }

    # 5. 변화점 검출
    if detect_changes and len(s) >= 30:
        from naviertwin.core.flow_analysis.change_point import (
            binary_segmentation,
            detection_score,
        )

        try:
            cps = binary_segmentation(s, n_changepoints=2, min_size=5)
            report["change_points"] = {
                "indices": cps,
                "score": detection_score(s, cps),
            }
        except ValueError:
            pass

    # 6. 통계 수렴 진단
    from naviertwin.core.flow_analysis.stat_convergence import (
        autocorrelation_time,
        effective_sample_size,
        geweke_diagnostic,
    )

    report["convergence"] = {
        "geweke_z": geweke_diagnostic(s),
        "ess": effective_sample_size(s),
        "autocorr_time": autocorrelation_time(s),
    }

    return report


def analyze_field_snapshots(
    X: NDArray[np.float64],
    n_modes: int = 5,
) -> dict[str, Any]:
    """(n_t, n_x) 스냅샷 행렬에서 ROM/통계 종합 분석.

    Args:
        X: (n_t, n_x) 시간-공간 데이터.
        n_modes: POD 모드 수.

    Returns:
        결과 dict.

    Raises:
        ValueError: X가 2D 아님.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")

    n_t, n_x = X.shape
    report: dict[str, Any] = {
        "n_t": int(n_t),
        "n_x": int(n_x),
    }

    # 1. POD (EOF)
    from naviertwin.core.flow_analysis.eof_analysis import eof_decomposition

    n_use = min(n_modes, n_t, n_x)
    eofs, pcs, var = eof_decomposition(X, n_modes=n_use)
    report["pod"] = {
        "n_modes": int(n_use),
        "energy_per_mode": var.tolist(),
        "cumulative_energy": float(var.sum()),
    }

    # 2. 절단 차수 추천
    from naviertwin.core.dimensionality_reduction.truncation_criteria import (
        cumulative_energy_curve,
        truncate_by_energy,
    )

    s = np.sqrt(var * float(np.sum(X ** 2)))  # 근사 특이값
    report["truncation"] = {
        "n_modes_for_99_energy": int(truncate_by_energy(s, fraction=0.99)),
        "cumulative_curve": cumulative_energy_curve(s).tolist(),
    }

    # 3. 시간 평균 + RMS (공간 점별)
    from naviertwin.core.flow_analysis.reynolds_stats import (
        mean_field,
        rms,
        skewness,
    )

    report["statistics"] = {
        "mean_min": float(mean_field(X).min()),
        "mean_max": float(mean_field(X).max()),
        "rms_mean": float(rms(X).mean()),
        "skewness_mean": float(skewness(X).mean()),
    }

    return report


def to_markdown(report: dict[str, Any]) -> str:
    """결과 dict → markdown 문자열.

    Args:
        report: analyze_probe_signal 또는 analyze_field_snapshots 출력.

    Returns:
        markdown 텍스트.
    """
    lines = ["# Auto Post-Process Report", ""]

    # 메타
    if "n_samples" in report:
        lines.append("## 메타")
        lines.append(f"- 샘플 수: {report['n_samples']}")
        lines.append(f"- 샘플링 주파수: {report['fs']:.4g} Hz")
        lines.append(f"- 시간 길이: {report['duration']:.4g} s")
        lines.append("")
    elif "n_t" in report:
        lines.append("## 메타")
        lines.append(f"- 시간 스냅샷: {report['n_t']}")
        lines.append(f"- 공간 점: {report['n_x']}")
        lines.append("")

    # 시계열 features
    if "features" in report:
        lines.append("## 시계열 특성")
        feature_items = list(report["features"].items())
        idx = 0
        while idx < len(feature_items):
            k, v = feature_items[idx]
            if isinstance(v, float):
                lines.append(f"- **{k}**: {v:.6g}")
            else:
                lines.append(f"- **{k}**: {v}")
            idx += 1
        lines.append("")

    # box stats
    if "box_stats" in report:
        lines.append("## Statistics (Box Plot)")
        box_items = list(report["box_stats"].items())
        idx = 0
        while idx < len(box_items):
            k, v = box_items[idx]
            if isinstance(v, float):
                lines.append(f"- {k}: {v:.6g}")
            else:
                lines.append(f"- {k}: {v}")
            idx += 1
        lines.append("")

    # PSD
    if "psd" in report:
        lines.append("## Power Spectral Density")
        psd = report["psd"]
        lines.append(f"- 피크 주파수: {psd['peak_frequency']:.6g} Hz")
        lines.append(f"- 피크 PSD: {psd['peak_psd']:.6g}")
        fr = psd.get("frequency_range", (0, 0))
        lines.append(f"- 주파수 범위: [{fr[0]:.4g}, {fr[1]:.4g}] Hz")
        lines.append("")

    # 위상 잠금
    if "phase_lock" in report:
        lines.append("## Phase-Locked Average")
        pl = report["phase_lock"]
        lines.append(f"- 주기: {pl['period']:.6g} s")
        lines.append(f"- 위상 평균 진폭: {pl['phase_avg_amplitude']:.6g}")
        lines.append(f"- 최대 RMS: {pl['rms_max']:.6g}")
        lines.append("")

    # 변화점
    if "change_points" in report:
        lines.append("## Change Points")
        cp = report["change_points"]
        lines.append(f"- 검출 위치: {cp['indices']}")
        lines.append(f"- 분산 감소율: {cp['score']:.4g}")
        lines.append("")

    # 수렴
    if "convergence" in report:
        lines.append("## Statistical Convergence")
        c = report["convergence"]
        lines.append(f"- Geweke z: {c['geweke_z']:.4g}")
        lines.append(f"- 유효 표본 크기: {c['ess']:.4g}")
        lines.append(f"- 자기상관 시간: {c['autocorr_time']:.4g}")
        lines.append("")

    # POD
    if "pod" in report:
        lines.append("## POD Decomposition")
        pod = report["pod"]
        lines.append(f"- 모드 수: {pod['n_modes']}")
        energy = list(map(lambda e: f"{e:.4g}", pod["energy_per_mode"]))
        lines.append(f"- 모드별 에너지: {energy}")
        lines.append(f"- 누적 에너지: {pod['cumulative_energy']:.4g}")
        lines.append("")

    # truncation
    if "truncation" in report:
        lines.append("## ROM Truncation")
        t = report["truncation"]
        lines.append(f"- 99% 에너지 모드 수: {t['n_modes_for_99_energy']}")
        lines.append("")

    # statistics
    if "statistics" in report:
        lines.append("## Spatial Statistics")
        st = report["statistics"]
        stat_items = list(st.items())
        idx = 0
        while idx < len(stat_items):
            k, v = stat_items[idx]
            lines.append(f"- {k}: {v:.6g}")
            idx += 1
        lines.append("")

    return "\n".join(lines)


__all__ = [
    "analyze_probe_signal",
    "analyze_field_snapshots",
    "to_markdown",
]
