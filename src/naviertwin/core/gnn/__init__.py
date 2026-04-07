"""그래프 신경망(GNN) 모듈.

공개 API:
    - :class:`BaseGNN`: GNN 추상 기반 클래스

하위 모듈:
    - :mod:`gnn_surrogate`: GNN 기반 대리 모델 (PyTorch Geometric)
    - :mod:`meshgraphnets`: MeshGraphNets
    - :mod:`egno`: E(n)-등변 GNN (e3nn)
    - :mod:`graph_transformer`: Graph Transformer (HAMLET 등)
"""

from naviertwin.core.gnn.base import BaseGNN

__all__ = ["BaseGNN"]
