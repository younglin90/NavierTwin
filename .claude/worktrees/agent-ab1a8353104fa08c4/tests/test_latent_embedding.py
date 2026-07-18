"""Round 79 — latent space 2D embedding."""

from __future__ import annotations

import numpy as np
import pytest


class TestEmbed2D:
    def test_pca(self) -> None:
        pytest.importorskip("sklearn")
        from naviertwin.core.analysis.latent_embedding import (
            embed_2d,
            embedding_spread,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 6))
        emb = embed_2d(X, method="pca")
        assert emb.shape == (40, 2)
        sp = embedding_spread(emb)
        assert sp["range_axis0"] > 0

    def test_tsne(self) -> None:
        pytest.importorskip("sklearn")
        from naviertwin.core.analysis.latent_embedding import embed_2d

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 4))
        emb = embed_2d(X, method="tsne", perplexity=5.0)
        assert emb.shape == (30, 2)

    def test_invalid_method_raises(self) -> None:
        from naviertwin.core.analysis.latent_embedding import embed_2d

        with pytest.raises(ValueError):
            embed_2d(np.zeros((10, 3)), method="bogus")

    def test_invalid_shape_raises(self) -> None:
        from naviertwin.core.analysis.latent_embedding import embed_2d

        with pytest.raises(ValueError):
            embed_2d(np.zeros(5), method="pca")
