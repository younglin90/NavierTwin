"""Benchmark dataset and validation-reference public API."""

from naviertwin.core.benchmarks.dataset_catalog import (
    generate_burgers_dataset,
    generate_cavity_dataset,
    generate_heat_dataset,
)
from naviertwin.core.benchmarks.ghia_cavity import ghia_u_centerline

__all__ = [
    "generate_burgers_dataset",
    "generate_cavity_dataset",
    "generate_heat_dataset",
    "ghia_u_centerline",
]
