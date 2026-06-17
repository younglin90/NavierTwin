"""Physics AI model adapter used by the digital twin panel.

PhysicsNeMo/PINN-style models often predict the solution field directly from
coordinates or operating parameters. They do not necessarily produce reduced
POD coefficients, so they cannot be forced through the classic
``reducer -> surrogate -> decode`` TwinEngine path without losing semantics.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class PhysicsAITwinEngine:
    """Digital-twin compatible wrapper around a fitted physics AI model."""

    def __init__(
        self,
        model: object,
        *,
        model_type: str = "physics_ai",
        input_dim: int | None = None,
        output_dim: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if not hasattr(model, "predict"):
            raise TypeError("Physics AI model must expose predict(params)")
        self.model = model
        self.surrogate = model
        self.reducer = None
        self.reducer_type = "direct_physics_ai"
        self.surrogate_type = model_type
        self.model_type = model_type
        self.input_dim = input_dim or _positive_int(
            getattr(model, "input_dim", None),
            getattr(model, "in_dim", None),
        )
        self.output_dim = output_dim or _positive_int(
            getattr(model, "output_dim", None),
            getattr(model, "out_dim", None),
        )
        self.n_modes = self.output_dim
        self.training_metadata = dict(metadata or {})
        self._is_fitted = bool(getattr(model, "is_fitted", True))

    @classmethod
    def from_fitted_model(
        cls,
        model: object,
        *,
        model_type: str = "physics_ai",
        metadata: dict[str, object] | None = None,
    ) -> "PhysicsAITwinEngine":
        """Create an engine from an already fitted physics AI model."""
        if not bool(getattr(model, "is_fitted", False)):
            raise RuntimeError("physics AI model이 fit되지 않았습니다.")
        merged_meta = dict(getattr(model, "training_metadata", {}) or {})
        if metadata:
            merged_meta.update(metadata)
        return cls(
            model,
            model_type=model_type,
            input_dim=_positive_int(merged_meta.get("n_params")),
            output_dim=_positive_int(merged_meta.get("n_outputs")),
            metadata=merged_meta,
        )

    @property
    def is_fitted(self) -> bool:
        """Return whether the wrapped model is fitted."""
        return self._is_fitted and bool(getattr(self.model, "is_fitted", True))

    def predict(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict directly with the wrapped physics AI model."""
        if not self.is_fitted:
            raise RuntimeError("PhysicsAITwinEngine.predict() 전에 fit()이 필요합니다.")
        params_arr = np.asarray(params, dtype=float)
        is_single = params_arr.ndim == 1
        if is_single:
            params_arr = params_arr.reshape(1, -1)
        result = self.model.predict(params_arr)  # type: ignore[attr-defined]
        arr = np.asarray(result, dtype=float)
        if is_single and arr.ndim > 1 and arr.shape[0] == 1:
            return arr[0]
        return arr

    def save(self, path: str | Path) -> None:
        """Persist the adapter and wrapped model with pickle."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "PhysicsAITwinEngine":
        """Load a saved physics AI twin engine."""
        with Path(path).open("rb") as f:
            engine = pickle.load(f)
        if not isinstance(engine, cls):
            raise TypeError(f"PhysicsAITwinEngine 파일이 아닙니다: {path}")
        return engine

    def get_params(self) -> dict[str, Any]:
        """Return engine metadata compatible with TwinEngine callers."""
        return {
            "reducer_type": self.reducer_type,
            "surrogate_type": self.surrogate_type,
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }


def _positive_int(*values: object) -> int:
    def _parse(value: object) -> int:
        try:
            parsed = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0
        return parsed if parsed > 0 else 0

    return next(filter(lambda parsed: parsed > 0, map(_parse, values)), 0)


__all__ = ["PhysicsAITwinEngine"]
