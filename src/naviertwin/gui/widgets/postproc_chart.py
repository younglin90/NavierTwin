"""Post-Processor 결과 차트 위젯 — matplotlib 기반 자동 plot 라우터.

Facade의 결과 dict를 받아 op_name에 따라 적절한 차트 (line / bar / heatmap /
loglog / scatter) 자동 선택. 표시 불가능한 결과는 빈 캔버스 유지.

Examples:
    >>> from naviertwin.gui.widgets.postproc_chart import PostProcessChart
    >>> # 위젯 생성 후
    >>> # chart.render("psd_welch", {"frequency": f, "psd": P})
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget


class PostProcessChart(QWidget):
    """Op별 결과를 자동 차트로 시각화."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._figure = Figure(figsize=(6, 4), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        layout.addWidget(self._canvas)
        self._last_op: str | None = None

    def render(self, op_name: str, result: dict[str, Any]) -> None:
        """op 결과를 그린다. 차트가 정의된 op은 plot, 그 외는 빈 캔버스."""
        self._figure.clear()
        self._last_op = op_name
        try:
            handler = _CHART_HANDLERS.get(op_name)
            if handler is None:
                self._render_fallback(op_name, result)
            else:
                handler(self._figure, result)
        except Exception as e:
            ax = self._figure.add_subplot(111)
            ax.text(
                0.5, 0.5, f"차트 생성 실패: {e}",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_axis_off()
        self._canvas.draw()

    def clear(self) -> None:
        """캔버스 초기화."""
        self._figure.clear()
        self._canvas.draw()
        self._last_op = None

    def _render_fallback(self, op_name: str, result: dict[str, Any]) -> None:
        """차트 핸들러 없는 op는 첫 번째 ndarray를 line plot."""
        ax = self._figure.add_subplot(111)
        ax.set_title(f"{op_name} (fallback)")
        plotted = False
        for k, v in result.items():
            if isinstance(v, np.ndarray) and v.size > 1 and v.ndim == 1:
                ax.plot(v, label=k)
                plotted = True
                if len([x for x in result.values()
                         if isinstance(x, np.ndarray) and x.ndim == 1]) <= 4:
                    continue
                break
        if plotted:
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "표시 가능한 데이터가 없습니다.",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()


# ---------------------------------------------------------------------------
# Op별 차트 핸들러
# ---------------------------------------------------------------------------


def _chart_psd(fig: Figure, result: dict[str, Any]) -> None:
    f = np.asarray(result["frequency"])
    P = np.asarray(result["psd"])
    ax = fig.add_subplot(111)
    # log-log이지만 0 처리
    f_pos = np.where(f > 0, f, np.nan)
    P_pos = np.where(P > 0, P, np.nan)
    ax.loglog(f_pos, P_pos)
    ax.set_xlabel("frequency")
    ax.set_ylabel("PSD")
    ax.set_title("Welch PSD")
    ax.grid(True, which="both", alpha=0.3)


def _chart_kolmogorov(fig: Figure, result: dict[str, Any]) -> None:
    k = np.asarray(result["k"])
    E = np.asarray(result["E"])
    slope = result.get("slope", 0.0)
    r2 = result.get("r2", 0.0)
    ax = fig.add_subplot(111)
    mask = (k > 0) & (E > 0)
    ax.loglog(k[mask], E[mask], "o-", markersize=3)
    ax.set_xlabel("k (wavenumber)")
    ax.set_ylabel("E(k)")
    ax.set_title(f"Energy Spectrum  (slope={slope:.3f}, R²={r2:.3f})")
    ax.grid(True, which="both", alpha=0.3)


def _chart_eof(fig: Figure, result: dict[str, Any]) -> None:
    eofs = np.asarray(result["eofs"])
    var = np.asarray(result["var_explained"])
    n_modes = eofs.shape[1]
    n_show = min(4, n_modes)
    for i in range(n_show):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.plot(eofs[:, i])
        ax.set_title(f"EOF{i + 1} ({var[i] * 100:.1f}%)")
        ax.grid(True, alpha=0.3)


def _chart_two_point_acf(fig: Figure, result: dict[str, Any]) -> None:
    r = np.asarray(result["r"])
    R = np.asarray(result["R"])
    L = result.get("L_int", 0.0)
    ax = fig.add_subplot(111)
    ax.plot(r, R, "o-", markersize=3)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(L, color="red", lw=1, label=f"L_int={L:.4g}")
    ax.set_xlabel("r")
    ax.set_ylabel("R(r)")
    ax.set_title("Spatial Autocorrelation")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _chart_box_stats(fig: Figure, result: dict[str, Any]) -> None:
    box = result.get("box", result)
    ax = fig.add_subplot(111)
    med = box.get("median", 0.0)
    q1 = box.get("Q1", 0.0)
    q3 = box.get("Q3", 0.0)
    lo = box.get("whisker_low", q1)
    hi = box.get("whisker_high", q3)
    # 박스플롯 그리기
    ax.broken_barh([(q1, q3 - q1)], (0.4, 0.2), facecolor="tab:blue", alpha=0.5)
    ax.plot([med, med], [0.4, 0.6], "k-", lw=2)
    ax.plot([lo, q1], [0.5, 0.5], "k-")
    ax.plot([q3, hi], [0.5, 0.5], "k-")
    ax.plot([lo, lo], [0.45, 0.55], "k-")
    ax.plot([hi, hi], [0.45, 0.55], "k-")
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title(
        f"Box Plot (median={med:.4g}, IQR={box.get('iqr', 0):.4g}, "
        f"outliers={box.get('n_outliers', 0)})"
    )
    ax.grid(True, axis="x", alpha=0.3)


def _chart_phase_average(fig: Figure, result: dict[str, Any]) -> None:
    phases = np.asarray(result["phases"])
    mean = np.asarray(result["mean"])
    rms = np.asarray(result["rms"])
    ax = fig.add_subplot(111)
    if mean.ndim == 1:
        ax.fill_between(phases, mean - rms, mean + rms, alpha=0.3)
        ax.plot(phases, mean, lw=2)
    else:
        ax.plot(phases, mean[:, 0])
    ax.set_xlabel("phase (rad)")
    ax.set_ylabel("⟨signal⟩")
    ax.set_title("Phase-Locked Average")
    ax.grid(True, alpha=0.3)


def _chart_quadrant(fig: Figure, result: dict[str, Any]) -> None:
    q = result.get("quadrants", result)
    fractions = [q.get(f"Q{i + 1}", {}).get("fraction", 0.0) for i in range(4)]
    ax = fig.add_subplot(111)
    bars = ax.bar(["Q1", "Q2", "Q3", "Q4"], fractions,
                   color=["tab:blue", "tab:red", "tab:blue", "tab:red"])
    for bar, f in zip(bars, fractions):
        ax.text(bar.get_x() + bar.get_width() / 2, f, f"{f:.2f}",
                ha="center", va="bottom")
    ax.set_ylabel("fraction")
    ax.set_title("Quadrant Analysis (Q2=ejection, Q4=sweep)")
    ax.grid(True, axis="y", alpha=0.3)


def _chart_change_points(fig: Figure, result: dict[str, Any]) -> None:
    cps = result.get("changepoints", [])
    means = result.get("segment_means", [])
    ax = fig.add_subplot(111)
    # bar chart of segment means
    if means:
        ax.bar(range(len(means)), means, color="tab:blue", alpha=0.7)
        ax.set_xlabel("segment index")
        ax.set_ylabel("mean")
    ax.set_title(f"Change Points (n={len(cps)}): {cps}")
    ax.grid(True, axis="y", alpha=0.3)


def _chart_pod_truncation(fig: Figure, result: dict[str, Any]) -> None:
    curve = np.asarray(result["cumulative_energy"])
    n = result.get("n_modes", 0)
    ax = fig.add_subplot(111)
    ax.plot(np.arange(1, len(curve) + 1), curve, "o-")
    if 0 < n <= len(curve):
        ax.axhline(curve[n - 1], color="red", lw=1,
                    label=f"r={n}: {curve[n - 1] * 100:.1f}%")
        ax.axvline(n, color="red", lw=1)
        ax.legend()
    ax.set_xlabel("number of modes")
    ax.set_ylabel("cumulative energy")
    ax.set_title("ROM Truncation Curve")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)


def _chart_denoise(fig: Figure, result: dict[str, Any]) -> None:
    s = np.asarray(result["smoothed"])
    ax = fig.add_subplot(111)
    ax.plot(s)
    ax.set_title("Savitzky-Golay Smoothed")
    ax.grid(True, alpha=0.3)


def _chart_critical_points(fig: Figure, result: dict[str, Any]) -> None:
    cps = result.get("critical_points", [])
    ax = fig.add_subplot(111)
    if cps:
        types = {c.get("type", "?") for c in cps}
        for t in types:
            xs = [c["x"] for c in cps if c.get("type") == t]
            ys = [c["y"] for c in cps if c.get("type") == t]
            ax.scatter(xs, ys, label=t, s=50)
        ax.legend()
    ax.set_title(f"Vector Field Critical Points (n={len(cps)})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)


def _chart_subspace_drift(fig: Figure, result: dict[str, Any]) -> None:
    angles = np.asarray(result["angles"])
    ax = fig.add_subplot(111)
    ax.bar(range(len(angles)), angles, color="tab:purple", alpha=0.7)
    ax.set_xlabel("mode index")
    ax.set_ylabel("canonical angle (rad)")
    drift = result.get("drift_score", 0.0)
    ax.set_title(f"Subspace Angles (drift={drift:.4g})")
    ax.grid(True, axis="y", alpha=0.3)


def _chart_morris_sensitivity(fig: Figure, result: dict[str, Any]) -> None:
    mu = np.asarray(result["mu_star"])
    sig = np.asarray(result["sigma"])
    ax = fig.add_subplot(111)
    ax.scatter(mu, sig, s=80)
    for i in range(len(mu)):
        ax.annotate(f"x{i}", (mu[i], sig[i]),
                     textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("μ* (mean |EE|)")
    ax.set_ylabel("σ (std EE)")
    ax.set_title("Morris Elementary Effects")
    ax.grid(True, alpha=0.3)


def _chart_helmholtz(fig: Figure, result: dict[str, Any]) -> None:
    titles = ["Solenoidal U", "Solenoidal V", "Irrotational U", "Irrotational V"]
    keys = ["solenoidal_u", "solenoidal_v", "irrotational_u", "irrotational_v"]
    for i, (k, t) in enumerate(zip(keys, titles)):
        ax = fig.add_subplot(2, 2, i + 1)
        arr = np.asarray(result[k])
        im = ax.imshow(arr, cmap="RdBu_r", origin="lower", aspect="auto")
        ax.set_title(t)
        fig.colorbar(im, ax=ax, fraction=0.046)


def _chart_mass_search(fig: Figure, result: dict[str, Any]) -> None:
    dist = np.asarray(result["distance_profile"])
    best = result.get("best_match_index", -1)
    ax = fig.add_subplot(111)
    ax.plot(dist)
    if best >= 0:
        ax.axvline(best, color="red", label=f"best @ {best}")
        ax.legend()
    ax.set_title("MASS Distance Profile")
    ax.set_xlabel("series index")
    ax.set_ylabel("z-norm distance")
    ax.grid(True, alpha=0.3)


_CHART_HANDLERS: dict[str, Any] = {
    "psd_welch": _chart_psd,
    "kolmogorov_slope": _chart_kolmogorov,
    "eof": _chart_eof,
    "two_point_acf": _chart_two_point_acf,
    "box_stats": _chart_box_stats,
    "phase_average": _chart_phase_average,
    "quadrant_analysis": _chart_quadrant,
    "change_points": _chart_change_points,
    "pod_truncation": _chart_pod_truncation,
    "denoise": _chart_denoise,
    "critical_points": _chart_critical_points,
    "subspace_drift": _chart_subspace_drift,
    "morris_sensitivity": _chart_morris_sensitivity,
    "helmholtz_decomp": _chart_helmholtz,
    "mass_search": _chart_mass_search,
}


__all__ = ["PostProcessChart"]
