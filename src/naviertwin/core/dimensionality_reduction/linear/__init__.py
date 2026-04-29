"""선형 ROM/POD 계열 차원 축소 공개 API."""

from naviertwin.core.dimensionality_reduction.linear.bpod import BalancedPOD
from naviertwin.core.dimensionality_reduction.linear.bpod_scratch import (
    bpod_reduce,
    lyapunov_disc,
)
from naviertwin.core.dimensionality_reduction.linear.cpod import ConstrainedPOD
from naviertwin.core.dimensionality_reduction.linear.deim import deim, deim_project
from naviertwin.core.dimensionality_reduction.linear.galerkin import (
    project_field_to_modes,
    project_linear_operator,
    project_mass_matrix,
    project_quadratic,
    reconstruct_from_modes,
    rom_rhs_linear,
)
from naviertwin.core.dimensionality_reduction.linear.gnat import gnat_solve
from naviertwin.core.dimensionality_reduction.linear.ica import FastICA
from naviertwin.core.dimensionality_reduction.linear.incremental_pod import IncrementalPOD
from naviertwin.core.dimensionality_reduction.linear.incremental_svd import IncrementalSVD
from naviertwin.core.dimensionality_reduction.linear.local_rom import LocalROM
from naviertwin.core.dimensionality_reduction.linear.lspg import (
    lspg_residual_norm,
    lspg_solve,
)
from naviertwin.core.dimensionality_reduction.linear.mrpod import MRPOD
from naviertwin.core.dimensionality_reduction.linear.pod import SnapshotPOD
from naviertwin.core.dimensionality_reduction.linear.pod_galerkin import PODGalerkinROM
from naviertwin.core.dimensionality_reduction.linear.pymor_pod import (
    pymor_deim,
    pymor_gram_schmidt,
    pymor_pod,
)
from naviertwin.core.dimensionality_reduction.linear.quadratic_manifold import (
    QuadraticManifold,
)
from naviertwin.core.dimensionality_reduction.linear.randomized_svd import RandomizedPOD
from naviertwin.core.dimensionality_reduction.linear.shifted_pod import (
    estimate_shifts,
    shifted_pod,
)
from naviertwin.core.dimensionality_reduction.linear.snapshot_pod_scratch import (
    reconstruct as reconstruct_snapshot_pod,
)
from naviertwin.core.dimensionality_reduction.linear.snapshot_pod_scratch import (
    snapshot_pod,
)
from naviertwin.core.dimensionality_reduction.linear.space_time_pod import SpaceTimePOD
from naviertwin.core.dimensionality_reduction.linear.weighted_pod import (
    compressed_pod,
    weighted_pod,
)

__all__ = [
    "BalancedPOD",
    "ConstrainedPOD",
    "FastICA",
    "IncrementalPOD",
    "IncrementalSVD",
    "LocalROM",
    "MRPOD",
    "PODGalerkinROM",
    "QuadraticManifold",
    "RandomizedPOD",
    "SnapshotPOD",
    "SpaceTimePOD",
    "bpod_reduce",
    "compressed_pod",
    "deim",
    "deim_project",
    "estimate_shifts",
    "gnat_solve",
    "lspg_residual_norm",
    "lspg_solve",
    "lyapunov_disc",
    "project_field_to_modes",
    "project_linear_operator",
    "project_mass_matrix",
    "project_quadratic",
    "pymor_deim",
    "pymor_gram_schmidt",
    "pymor_pod",
    "reconstruct_from_modes",
    "reconstruct_snapshot_pod",
    "rom_rhs_linear",
    "shifted_pod",
    "snapshot_pod",
    "weighted_pod",
]
