"""Explainability algorithms exposed by NavierTwin."""

from naviertwin.core.explainability.attention_viz import (
    extract_attention,
    topk_attention_tokens,
)
from naviertwin.core.explainability.shap_explainer import KernelSHAP
from naviertwin.core.explainability.symbolic_regression import SymbolicRegressor

__all__ = [
    "KernelSHAP",
    "SymbolicRegressor",
    "extract_attention",
    "topk_attention_tokens",
]
