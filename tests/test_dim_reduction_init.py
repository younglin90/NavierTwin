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

    def test_root_all_lists_lazy_reducers(self) -> None:
        import naviertwin.core.dimensionality_reduction as dr

        exported = set(dr.__all__)
        assert {"Autoencoder", "VAE", "DiffusionMaps", "TuckerDecomposition"} <= exported
        assert dr.DiffusionMaps.__name__ == "DiffusionMaps"
        assert dr.TuckerDecomposition.__name__ == "TuckerDecomposition"

    def test_nonlinear_package_public_api(self) -> None:
        import naviertwin.core.dimensionality_reduction.nonlinear as nonlinear

        exported = set(nonlinear.__all__)
        for symbol in [
            "Autoencoder",
            "VAE",
            "CNNAE",
            "GNNAE",
            "DiffusionMaps",
            "TuckerDecomposition",
            "isomap",
            "lle",
            "opinf_fit",
            "fit_closure",
            "LatentODE",
        ]:
            assert symbol in exported
            assert hasattr(nonlinear, symbol)

    def test_unknown_attr_raises(self) -> None:
        import naviertwin.core.dimensionality_reduction as dr

        with pytest.raises(AttributeError, match="no attribute"):
            _ = dr.DoesNotExist  # type: ignore[attr-defined]

    def test_nonlinear_unknown_attr_raises(self) -> None:
        import naviertwin.core.dimensionality_reduction.nonlinear as nonlinear

        with pytest.raises(AttributeError, match="no attribute"):
            _ = nonlinear.DoesNotExist  # type: ignore[attr-defined]
