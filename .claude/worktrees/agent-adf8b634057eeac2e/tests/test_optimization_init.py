"""Optimization package root public API tests."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_optimization_root_exports_safe_algorithms() -> None:
    """Package root should expose implemented non-backend optimization helpers."""
    import naviertwin.core.optimization as optimization

    expected = {
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

    assert set(expected).issubset(set(optimization.__all__))
    for symbol, module_name in expected.items():
        source_module = import_module(module_name)
        assert getattr(optimization, symbol) is getattr(source_module, symbol)


def test_optimization_root_does_not_eagerly_import_optional_backends() -> None:
    """Root import should not load optional backend wrappers."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    code = (
        "import sys; "
        "import naviertwin.core.optimization; "
        "blocked = {'botorch', 'nlopt', 'pygmo'} & set(sys.modules); "
        "raise SystemExit(1 if blocked else 0)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        check=False,
    )
    assert completed.returncode == 0


def test_optimization_root_keeps_ambiguous_and_optional_helpers_explicit() -> None:
    """Same-name modules and optional wrappers stay module-qualified."""
    import naviertwin.core.optimization as optimization

    for symbol in [
        "BoTorchBayesianOpt",
        "cma_es_simple",
        "hyperopt",
        "inverse_design",
        "list_algorithms",
        "nelder_mead",
        "nlopt_minimize",
        "nsga2_constrained",
        "pso",
        "tchebycheff",
    ]:
        assert symbol not in optimization.__all__


def test_optimization_root_preserves_same_name_submodule_imports() -> None:
    """Lazy exports should not shadow legacy module imports."""
    from naviertwin.core.optimization import nelder_mead, pso, tchebycheff

    assert nelder_mead.__name__.endswith(".nelder_mead")
    assert pso.__name__.endswith(".pso")
    assert tchebycheff.__name__.endswith(".tchebycheff")
