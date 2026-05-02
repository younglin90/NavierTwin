"""Post-Process 결과 export — CSV / JSON / NPZ + 일괄 실행기.

Facade 결과 dict를 디스크에 저장. CI/오프라인 보고서 생성에도 사용.

Examples:
    >>> import numpy as np
    >>> result = {"frequency": np.array([1.0, 2.0, 3.0]),
    ...           "psd": np.array([0.1, 0.2, 0.3])}
    >>> import tempfile
    >>> from naviertwin.core.post_process_export import result_to_csv_text
    >>> txt = result_to_csv_text(result)
    >>> "frequency" in txt
    True
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _is_table_like(result: dict[str, Any]) -> bool:
    """결과가 1D ndarray 키 위주면 테이블로 export 가능."""
    arr_keys = [
        k for k, v in result.items()
        if isinstance(v, np.ndarray) and v.ndim == 1
    ]
    if not arr_keys:
        return False
    lengths = {result[k].shape[0] for k in arr_keys}
    return len(lengths) == 1


def result_to_csv_text(result: dict[str, Any]) -> str:
    """결과 dict → CSV 문자열.

    1D ndarray 키만 columns로 변환. 길이 다르면 가장 긴 것 기준으로 패딩.

    Args:
        result: facade.run() 결과.

    Returns:
        CSV 문자열 (header + rows).
    """
    arr_keys = [
        k for k, v in result.items()
        if isinstance(v, np.ndarray) and v.ndim == 1 and v.size > 0
    ]
    if not arr_keys:
        # 스칼라/dict만 있는 경우
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["key", "value"])
        for k, v in result.items():
            if isinstance(v, (int, float, str, bool)):
                w.writerow([k, v])
            elif isinstance(v, np.ndarray) and v.size <= 1:
                w.writerow([k, float(v.flatten()[0]) if v.size == 1 else ""])
        return out.getvalue()

    max_len = max(result[k].shape[0] for k in arr_keys)
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(arr_keys)
    for i in range(max_len):
        row = []
        for k in arr_keys:
            arr = result[k]
            if i < arr.shape[0]:
                row.append(float(arr[i]))
            else:
                row.append("")
        w.writerow(row)
    return out.getvalue()


def save_csv(result: dict[str, Any], path: str | Path) -> Path:
    """결과를 CSV 파일로 저장.

    Args:
        result: facade.run() 결과.
        path: 저장 경로.

    Returns:
        절대 경로.
    """
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(result_to_csv_text(result), encoding="utf-8")
    logger.info("CSV 저장: %s", p)
    return p


def _to_json_compat(value: Any) -> Any:
    """ndarray/numpy scalar를 JSON-호환 타입으로."""
    if isinstance(value, np.ndarray):
        if value.size > 1000:
            return {
                "_type": "ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "summary": {
                    "min": float(value.min()),
                    "max": float(value.max()),
                    "mean": float(value.mean()),
                    "std": float(value.std()),
                },
            }
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {k: _to_json_compat(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compat(v) for v in value]
    return value


def save_json(
    result: dict[str, Any],
    path: str | Path,
    indent: int = 2,
) -> Path:
    """결과를 JSON 파일로 저장 (큰 ndarray는 요약).

    Args:
        result: facade.run() 결과.
        path: 저장 경로.
        indent: JSON indent.

    Returns:
        절대 경로.
    """
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_json_compat(result)
    p.write_text(json.dumps(payload, indent=indent, ensure_ascii=False),
                 encoding="utf-8")
    logger.info("JSON 저장: %s", p)
    return p


def save_npz(result: dict[str, Any], path: str | Path) -> Path:
    """결과 dict의 모든 ndarray + 스칼라를 NPZ로 저장.

    Args:
        result: facade.run() 결과.
        path: 저장 경로.

    Returns:
        절대 경로.

    Raises:
        ValueError: 저장 가능한 데이터가 없는 경우.
    """
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    arrs: dict[str, NDArray] = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            arrs[k] = v
        elif isinstance(v, (int, float, np.integer, np.floating)):
            arrs[k] = np.array(v)
        elif isinstance(v, (list, tuple)):
            try:
                arrs[k] = np.asarray(v)
            except (ValueError, TypeError):
                pass
    if not arrs:
        raise ValueError("저장 가능한 ndarray/scalar 없음")
    np.savez_compressed(str(p), **arrs)
    logger.info("NPZ 저장: %s (%d 키)", p, len(arrs))
    return p


def run_category(
    facade: Any,
    category: str,
    smoke_kwargs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """특정 카테고리의 모든 op을 일괄 실행.

    Args:
        facade: PostProcessFacade.
        category: 카테고리 이름.
        smoke_kwargs: op_name → kwargs dict. 없으면 op 건너뜀.

    Returns:
        {op_name: result} dict (실패 시 {"error": str}).
    """
    out: dict[str, dict[str, Any]] = {}
    for op in facade.list_operations():
        info = facade.describe(op)
        if info["category"] != category:
            continue
        kwargs = (smoke_kwargs or {}).get(op)
        if kwargs is None:
            continue
        try:
            out[op] = facade.run(op, **kwargs)
        except Exception as e:  # noqa: BLE001
            out[op] = {"_error": str(e)}
    return out


def run_all(
    facade: Any,
    smoke_kwargs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """모든 op 일괄 실행 (smoke_kwargs 정의된 op만)."""
    out: dict[str, dict[str, Any]] = {}
    for op in facade.list_operations():
        kwargs = smoke_kwargs.get(op)
        if kwargs is None:
            continue
        try:
            out[op] = facade.run(op, **kwargs)
        except Exception as e:  # noqa: BLE001
            out[op] = {"_error": str(e)}
    return out


def bulk_summary_markdown(bulk_results: dict[str, dict[str, Any]]) -> str:
    """일괄 실행 결과 → markdown 요약."""
    lines = ["# Bulk Post-Process Summary", ""]
    n_total = len(bulk_results)
    n_failed = sum(1 for r in bulk_results.values() if "_error" in r)
    lines.append(f"- 총 op: {n_total}")
    lines.append(f"- 실패: {n_failed}")
    lines.append(f"- 성공: {n_total - n_failed}")
    lines.append("")
    for op, result in sorted(bulk_results.items()):
        lines.append(f"## {op}")
        if "_error" in result:
            lines.append(f"- ❌ 실패: {result['_error']}")
        else:
            lines.append(f"- ✅ 성공 ({len(result)} 출력 키)")
            for k, v in result.items():
                if isinstance(v, np.ndarray):
                    lines.append(f"  - {k}: shape={v.shape}, "
                                  f"mean={float(v.mean()):.4g}")
                elif isinstance(v, (int, float)):
                    lines.append(f"  - {k}: {v}")
                else:
                    lines.append(f"  - {k}: {type(v).__name__}")
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "result_to_csv_text",
    "save_csv",
    "save_json",
    "save_npz",
    "run_category",
    "run_all",
    "bulk_summary_markdown",
]
