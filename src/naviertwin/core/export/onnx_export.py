"""PyTorch 모델 → ONNX 내보내기.

FNO/DeepONet/U-Net 등 PyTorch 기반 신경 연산자를 ONNX 로 저장하여
C++/C#/웹 런타임에서 추론할 수 있게 한다.

Examples:
    >>> import torch, torch.nn as nn, tempfile, os
    >>> from naviertwin.core.export.onnx_export import export_to_onnx
    >>> model = nn.Linear(4, 2)
    >>> sample = torch.randn(1, 4)
    >>> with tempfile.TemporaryDirectory() as d:
    ...     out = os.path.join(d, "model.onnx")
    ...     export_to_onnx(model, sample, out)
    ...     os.path.exists(out)
    True
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def export_to_onnx(
    model: Any,
    sample_input: Any,
    path: str | Path,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    opset_version: int = 17,
) -> Path:
    """학습된 PyTorch 모델을 ONNX 파일로 저장한다.

    Args:
        model: PyTorch nn.Module (eval 모드로 전환됨).
        sample_input: 추적용 샘플 텐서 또는 튜플.
        path: 저장 경로.
        input_names / output_names: ONNX 입출력 이름.
        dynamic_axes: 동적 축 지정 (배치 가변 등).
        opset_version: ONNX opset.

    Returns:
        저장된 파일 경로 (Path).

    Raises:
        RuntimeError: torch 또는 onnx 미설치.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch 가 필요합니다") from exc

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    export_kwargs = dict(
        input_names=input_names or ["input"],
        output_names=output_names or ["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )
    # 최신 torch 는 dynamo 경로가 기본이어서 onnxscript 가 필요 — legacy 경로 강제
    try:
        torch.onnx.export(model, sample_input, str(out), dynamo=False, **export_kwargs)
    except TypeError:
        # 구 버전은 dynamo 인자 없음 — 기본 경로
        torch.onnx.export(model, sample_input, str(out), **export_kwargs)
    logger.info("ONNX 저장 완료: %s (opset=%d)", out, opset_version)
    return out


def verify_onnx(path: str | Path) -> dict[str, Any]:
    """저장된 ONNX 파일을 onnx.checker 로 검증하고 메타 정보를 반환한다."""
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("onnx 가 필요합니다: pip install onnx") from exc

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    info = {
        "ir_version": int(model.ir_version),
        "opset_imports": [
            (op.domain or "ai.onnx", int(op.version)) for op in model.opset_import
        ],
        "graph_inputs": [i.name for i in model.graph.input],
        "graph_outputs": [o.name for o in model.graph.output],
    }
    logger.info("ONNX 검증 통과: %s", info)
    return info


__all__ = ["export_to_onnx", "verify_onnx"]
