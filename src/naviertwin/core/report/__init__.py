"""Report generation, tabular export, and plotting helpers."""

from naviertwin.core.report.csv_writer import read_csv, write_csv, write_metrics_table
from naviertwin.core.report.generator import ReportGenerator
from naviertwin.core.report.html_report import HTMLReport
from naviertwin.core.report.markdown import MarkdownReport
from naviertwin.core.report.plots import (
    apply_publication_style,
    plot_compare_metrics,
    plot_field_2d,
    plot_loss_curve,
    plot_pod_energy,
)

__all__ = [
    "HTMLReport",
    "MarkdownReport",
    "ReportGenerator",
    "apply_publication_style",
    "plot_compare_metrics",
    "plot_field_2d",
    "plot_loss_curve",
    "plot_pod_energy",
    "read_csv",
    "write_csv",
    "write_metrics_table",
]
