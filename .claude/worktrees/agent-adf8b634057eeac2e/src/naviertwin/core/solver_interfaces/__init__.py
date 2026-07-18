"""외부 CFD 솔버 래퍼 모음 + 내장 간이 솔버.

- ``lbm_d2q9`` : 2D D2Q9 Lattice Boltzmann (numpy 직접 구현, optional GPU 없음)
- ``lettuce_wrapper`` : Lettuce 라이브러리 선호
- ``flowtorch_wrapper`` : flowtorch 기반 POD/DMD 파이프라인 연동
- ``jax_fluids_wrapper`` : JAX-Fluids 선호
"""

from naviertwin.core.solver_interfaces.fvm_advection import (
    fvm_musclhancock_1d,
    fvm_upwind_1d,
    minmod,
    total_mass,
)
from naviertwin.core.solver_interfaces.lbm_d2q9 import LBMD2Q9
from naviertwin.core.solver_interfaces.pde_solvers import (
    solve_burgers_1d,
    solve_heat_1d,
)
from naviertwin.core.solver_interfaces.sph import (
    cubic_spline_kernel,
    sph_density_1d,
    sph_gradient_1d,
)

__all__ = [
    "LBMD2Q9",
    "cubic_spline_kernel",
    "fvm_musclhancock_1d",
    "fvm_upwind_1d",
    "minmod",
    "solve_burgers_1d",
    "solve_heat_1d",
    "sph_density_1d",
    "sph_gradient_1d",
    "total_mass",
]
