"""gRPC inference server stub — runtime opt-in (grpcio).

Examples:
    >>> from naviertwin.core.serving.grpc_server import has_grpc
    >>> isinstance(has_grpc(), bool)
    True
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def has_grpc() -> bool:
    try:
        import grpc  # noqa: F401
    except ImportError:
        return False
    return True


class InferenceStub:
    """Minimal stub: in-memory predict service (no actual gRPC channel)."""

    def __init__(self, predict_fn: Callable[[Any], Any]) -> None:
        self.predict_fn = predict_fn

    def Predict(self, request: Any) -> Any:  # noqa: N802
        return self.predict_fn(request)


def make_grpc_server(
    predict_fn: Callable[[Any], Any], *, port: int = 50051,
) -> Any:
    if not has_grpc():
        raise ImportError("grpcio not installed")
    import grpc
    server = grpc.server(__import__("concurrent.futures", fromlist=["ThreadPoolExecutor"]).ThreadPoolExecutor(max_workers=2))
    server.add_insecure_port(f"[::]:{port}")
    return server


__all__ = ["InferenceStub", "has_grpc", "make_grpc_server"]
