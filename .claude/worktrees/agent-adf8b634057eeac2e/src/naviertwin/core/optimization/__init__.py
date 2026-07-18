"""Optimization algorithms used by surrogate and design workflows."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "BayesianOptimizer": "naviertwin.core.optimization.bayesian_opt",
    "NSGA2": "naviertwin.core.optimization.moo_optimizer",
    "SurrogateOptimizer": "naviertwin.core.optimization.surrogate_opt",
    "PolynomialChaos": "naviertwin.core.optimization.uq_surrogate",
    "AdaGradOpt": "naviertwin.core.optimization.gradient_opt",
    "AdamOpt": "naviertwin.core.optimization.gradient_opt",
    "MomentumOpt": "naviertwin.core.optimization.gradient_opt",
    "SGDOpt": "naviertwin.core.optimization.gradient_opt",
    "armijo_backtrack": "naviertwin.core.optimization.line_search",
    "bezier_eval": "naviertwin.core.optimization.shape_opt",
    "bfgs_minimize": "naviertwin.core.optimization.bfgs",
    "bobyqa_lite": "naviertwin.core.optimization.bobyqa",
    "check_wolfe": "naviertwin.core.optimization.line_search",
    "directional_derivative": "naviertwin.core.optimization.tangent_linear",
    "fd_sensitivity": "naviertwin.core.optimization.adjoint",
    "ga": "naviertwin.core.optimization.genetic",
    "gradient_from_jvp": "naviertwin.core.optimization.tangent_linear",
    "hypervolume_2d": "naviertwin.core.optimization.pareto",
    "jvp_fd": "naviertwin.core.optimization.tangent_linear",
    "linear_adjoint_sensitivity": "naviertwin.core.optimization.adjoint",
    "mads_minimize": "naviertwin.core.optimization.mads",
    "minimize": "naviertwin.core.optimization.gradient_opt",
    "nondominated_sort": "naviertwin.core.optimization.pareto",
    "optimize_bezier": "naviertwin.core.optimization.shape_opt",
    "pareto_mask": "naviertwin.core.optimization.pareto",
    "propagate_mc": "naviertwin.core.optimization.mc_propagation",
    "sa": "naviertwin.core.optimization.simulated_annealing",
    "simp_1d": "naviertwin.core.optimization.topo_simp",
    "simp_2d": "naviertwin.core.optimization.topology_opt",
    "sqp_eq": "naviertwin.core.optimization.sqp",
    "successive_halving": "naviertwin.core.optimization.halving",
    "trust_region_minimize": "naviertwin.core.optimization.trust_region",
    "weight_grid": "naviertwin.core.optimization.tchebycheff",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Lazily expose optimization helpers without importing optional backends."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORT_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return stable public members used by autocomplete and Sphinx."""
    return sorted([*globals(), *__all__])
