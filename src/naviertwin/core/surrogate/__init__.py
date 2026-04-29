"""Surrogate models, diagnostics, and batch evaluation public API."""

from naviertwin.core.surrogate.base import BaseSurrogate
from naviertwin.core.surrogate.batch_evaluation import (
    batch_predict,
    batch_predict_safe,
    batch_predict_with_uncertainty,
    split_into_chunks,
)
from naviertwin.core.surrogate.bayesian_linear import BayesianLinearRegression
from naviertwin.core.surrogate.certification_metrics import (
    calibration_curve,
    coverage_width_criterion,
    cv_rmse,
    expected_calibration_error,
    mpiw,
    normalized_rmse,
    picp,
    r2_score,
    rmse,
)
from naviertwin.core.surrogate.ensemble import EnsembleSurrogate, MixtureOfExperts
from naviertwin.core.surrogate.gp_scratch import GPRegressor, rbf_kernel
from naviertwin.core.surrogate.gradient_sensitivity import (
    finite_difference_gradient,
    morris_elementary_effects,
    permutation_importance,
    variance_decomposition_1d,
)
from naviertwin.core.surrogate.kriging_scratch import OrdinaryKriging
from naviertwin.core.surrogate.kriging_surrogate import KrigingSurrogate
from naviertwin.core.surrogate.model_averaging import (
    bic_weights,
    cv_error_weights,
    ensemble_variance,
    equal_weight_average,
    stacking_least_squares,
    weighted_average,
)
from naviertwin.core.surrogate.multi_fidelity import LinearMultiFidelity
from naviertwin.core.surrogate.rbf_surrogate import RBFSurrogate
from naviertwin.core.surrogate.residual_analysis import (
    cooks_distance,
    durbin_watson,
    leverage_scores,
    qq_data,
    residual_autocorrelation,
    shapiro_normality_diagnostic,
)
from naviertwin.core.surrogate.rff_regression import RFFRegression
from naviertwin.core.surrogate.rsm import RSMQuadratic
from naviertwin.core.surrogate.smt_advanced import (
    full_factorial,
    gekpls_fit,
    idw_fit,
    kpls_fit,
    lhs_design,
    qp_fit,
    smt_predict,
)

__all__ = [
    "BaseSurrogate",
    "BayesianLinearRegression",
    "EnsembleSurrogate",
    "GPRegressor",
    "KrigingSurrogate",
    "LinearMultiFidelity",
    "MixtureOfExperts",
    "OrdinaryKriging",
    "RBFSurrogate",
    "RFFRegression",
    "RSMQuadratic",
    "batch_predict",
    "batch_predict_safe",
    "batch_predict_with_uncertainty",
    "bic_weights",
    "calibration_curve",
    "cooks_distance",
    "coverage_width_criterion",
    "cv_error_weights",
    "cv_rmse",
    "durbin_watson",
    "ensemble_variance",
    "equal_weight_average",
    "expected_calibration_error",
    "finite_difference_gradient",
    "full_factorial",
    "gekpls_fit",
    "idw_fit",
    "kpls_fit",
    "leverage_scores",
    "lhs_design",
    "mpiw",
    "morris_elementary_effects",
    "normalized_rmse",
    "permutation_importance",
    "picp",
    "qp_fit",
    "qq_data",
    "r2_score",
    "rbf_kernel",
    "residual_autocorrelation",
    "rmse",
    "shapiro_normality_diagnostic",
    "smt_predict",
    "split_into_chunks",
    "stacking_least_squares",
    "variance_decomposition_1d",
    "weighted_average",
]
