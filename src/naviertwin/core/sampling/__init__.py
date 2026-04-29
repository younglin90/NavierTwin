"""Sampling and sparse-sensor selection public API."""

from naviertwin.core.sampling.monte_carlo import (
    importance_sample,
    mc_integral,
    mc_integral_antithetic,
    mc_multivariate,
)
from naviertwin.core.sampling.param_sweep import generate_sweep
from naviertwin.core.sampling.poisson_disk import poisson_disk_2d
from naviertwin.core.sampling.qmc import halton, latin_hypercube, scale_to_bounds, sobol
from naviertwin.core.sampling.sensor_dmd_pipeline import SensorDMDPipeline
from naviertwin.core.sampling.sparse_sensor import reconstruct, select_sensors

__all__ = [
    "SensorDMDPipeline",
    "generate_sweep",
    "halton",
    "importance_sample",
    "latin_hypercube",
    "mc_integral",
    "mc_integral_antithetic",
    "mc_multivariate",
    "poisson_disk_2d",
    "reconstruct",
    "scale_to_bounds",
    "select_sensors",
    "sobol",
]
