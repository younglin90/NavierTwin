"""잠재공간 임베딩 — POD coeffs / AE latent 등을 2D 로 내리기 위한 유틸.

sklearn PCA/t-SNE + (옵션) UMAP 지원. GUI 시각화 용도.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.latent_embedding import embed_2d
    >>> X = np.random.randn(50, 8)
    >>> emb = embed_2d(X, method="pca")
    >>> emb.shape
    (50, 2)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def embed_2d(
    X: NDArray[np.float64],
    method: str = "pca",
    *,
    perplexity: float = 10.0,
    n_neighbors: int = 15,
    seed: int | None = 0,
) -> NDArray[np.float64]:
    """(N, d) → (N, 2) 임베딩.

    Args:
        X: 입력 행렬 (N >= 2).
        method: "pca" / "tsne" / "umap".
        perplexity: t-SNE 파라미터.
        n_neighbors: UMAP 파라미터.
        seed: 재현성.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("X must be (N>=2, d)")

    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(X)

    if method == "tsne":
        from sklearn.manifold import TSNE

        perp = min(perplexity, max(2.0, X.shape[0] / 4.0))
        return TSNE(
            n_components=2, perplexity=perp, random_state=seed, init="pca",
        ).fit_transform(X)

    if method == "umap":
        try:
            import umap  # type: ignore
        except ImportError as exc:
            raise RuntimeError("umap-learn 설치 필요") from exc
        return umap.UMAP(
            n_components=2, n_neighbors=min(n_neighbors, X.shape[0] - 1),
            random_state=seed,
        ).fit_transform(X)

    raise ValueError(f"알 수 없는 method: {method}")


def embedding_spread(embedding: NDArray[np.float64]) -> dict[str, float]:
    """임베딩의 스프레드 지표 (분산/범위)."""
    e = np.asarray(embedding, dtype=np.float64)
    return {
        "var_axis0": float(e[:, 0].var()),
        "var_axis1": float(e[:, 1].var()),
        "range_axis0": float(np.ptp(e[:, 0])),
        "range_axis1": float(np.ptp(e[:, 1])),
    }


__all__ = ["embed_2d", "embedding_spread"]
