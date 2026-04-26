"""Round 593 — dimensionality_reduction __init__ lazy-import coverage."""

from __future__ import annotations

import pytest


class TestDimReductionInit:
    def test_base_reducer_importable(self) -> None:
        from naviertwin.core.dimensionality_reduction import BaseReducer

        assert BaseReducer is not None

    def test_lazy_autoencoder(self) -> None:
        import naviertwin.core.dimensionality_reduction as dr

        AE = dr.Autoencoder
        assert AE.__name__ == "Autoencoder"

    def test_lazy_vae(self) -> None:
        import naviertwin.core.dimensionality_reduction as dr

        V = dr.VAE
        assert V.__name__ == "VAE"

    def test_unknown_attr_raises(self) -> None:
        import naviertwin.core.dimensionality_reduction as dr

        with pytest.raises(AttributeError, match="no attribute"):
            _ = dr.DoesNotExist  # type: ignore[attr-defined]
