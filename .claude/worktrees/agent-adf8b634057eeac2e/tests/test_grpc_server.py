"""Round 356 — gRPC stub."""

from __future__ import annotations


class TestGRPC:
    def test_has_grpc(self) -> None:
        from naviertwin.core.serving.grpc_server import has_grpc

        assert isinstance(has_grpc(), bool)

    def test_stub_predict(self) -> None:
        from naviertwin.core.serving.grpc_server import InferenceStub

        stub = InferenceStub(predict_fn=lambda x: x * 2)
        assert stub.Predict(3) == 6
