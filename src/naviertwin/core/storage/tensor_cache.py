"""ML 텐서 캐시 (Zarr) — 케이스 세트 텐서화 결과의 콘텐츠 주소 저장소.

외부 검토 §6½ #6(저장 계층)의 ML 캐시 조각. :func:`naviertwin.core.
operator_learning.fno.case_tensorizer.cases_to_grid_tensors` 는 케이스 세트를
(N, H, W, C) 텐서로 바꾸는 비싼 연산(케이스별 ``grid.sample`` + EDT)인데,
같은 입력으로 학습을 반복할 때마다 재계산됐다. 이 모듈은 그 결과를
**콘텐츠 주소(입력 내용의 해시) 키**로 Zarr 디렉토리에 저장해 두 번째
호출부터 즉시 복원한다.

설계 원칙:
    - **캐시는 절대 정확성을 해치면 안 된다.** 캐시 항목이 손상됐거나,
      버전이 다르거나, zarr 가 없거나, 어떤 이유로든 읽기에 실패하면
      **조용히 miss 로 처리**(예외 삼키고 ``None``)하고 호출부가 재계산하게
      한다. 캐시 디렉토리 전체를 지워도 결과는 같아야 한다 — 느려질 뿐이다.
    - **콘텐츠 주소**: 키는 입력 내용(케이스 메쉬의 topology/coordinate 해시
      + 파라미터 배열 바이트 + 필드명 + 해상도 + 파라미터명)에서만 나온다.
      경로/시간/순서 밖 요인이 끼지 않으므로 같은 입력이면 언제나 같은
      키다(결정적), 입력이 조금이라도 다르면 다른 키다(오염 불가).
    - **grid 는 저장하지 않는다**: pyvista ImageData 직렬화가 번거롭고
      필요도 없다 — ImageData 는 ``dims/spacing/origin`` 3개 값으로 완전히
      결정되므로 meta 에서 :func:`grid_from_meta` 로 재구성한다.

zarr 는 선택 의존성이다([full] extra). 미설치 환경에서는 모든 연산이
조용히 no-op/miss 가 된다 — 캐시 없이 항상 재계산하는 기존 동작과 동일.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["TensorCache", "grid_from_meta"]

# 캐시 스키마 버전 — 저장 항목의 배열/메타 구성이 바뀌면 올린다. 버전이
# 다른 항목은 읽지 않고 miss 처리한다 (재계산이 항상 안전한 폴백).
_CACHE_VERSION = 1

# 키 길이 — data_model/signature.py 와 같은 근거로 sha256 앞 16 hex(64 bit).
# 케이스 세트 조합은 많아야 수천 개 수준 → 우연 충돌 확률은 사실상 0.
_KEY_HEX_LEN = 16

# 캐시 항목에 반드시 있어야 하는 배열 키 — 하나라도 없으면 손상으로 본다.
_REQUIRED_ARRAYS = ("inputs", "targets", "valid_mask")


def _zarr() -> Any | None:
    """zarr 모듈을 지연 import 한다 — 없으면 None (조용한 폴백)."""
    try:
        import zarr
    except Exception:  # noqa: BLE001 — 미설치/버전 파손 모두 "캐시 없음"으로.
        return None
    return zarr


def grid_from_meta(meta: dict[str, Any]) -> Any:
    """meta(dims/spacing/origin)에서 공통 격자 pyvista ImageData 를 재구성한다.

    ``cases_to_grid_tensors`` 의 공통 격자는 균일 ImageData 라 이 3개 값으로
    완전히 결정된다 — 격자 자체를 직렬화할 필요가 없는 이유다.

    Args:
        meta: ``dims``/``spacing``/``origin`` 키를 가진 dict
            (``cases_to_grid_tensors`` 반환의 ``meta`` 와 같은 규약).

    Returns:
        pyvista ImageData.
    """
    import pyvista as pv

    return pv.ImageData(
        dimensions=tuple(int(d) for d in meta["dims"]),
        spacing=tuple(float(s) for s in meta["spacing"]),
        origin=tuple(float(o) for o in meta["origin"]),
    )


def _jsonable_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """meta dict 를 JSON-호환(리스트/기본형)으로 정리한다 (tuple → list 등)."""
    return json.loads(json.dumps(meta, default=_json_default))


def _json_default(value: Any) -> Any:
    """JSON 이 모르는 타입의 최소 변환 — numpy 스칼라/배열, tuple 대응."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"meta 에 JSON 직렬화 불가 타입이 있습니다: {type(value)!r}")


def _restore_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """JSON 왕복으로 list 가 된 meta 필드를 원래 tuple 규약으로 되돌린다."""
    out = dict(meta)
    for key in ("dims", "spacing", "origin", "hw"):
        if key in out and isinstance(out[key], list):
            out[key] = tuple(out[key])
    return out


class TensorCache:
    """``cases_to_grid_tensors`` 결과의 콘텐츠 주소(해시) 기반 Zarr 캐시.

    사용 규약::

        cache = TensorCache()                       # ~/.naviertwin/tensor_cache/
        key = TensorCache.key_for(datasets, params, fields, resolution, names)
        tensors = cache.get(key)
        if tensors is None:
            tensors = cases_to_grid_tensors(...)    # miss → 계산
            cache.put(key, tensors)                 # 다음 호출부터 hit

    Attributes:
        cache_dir: 캐시 루트 디렉토리. 항목 하나 = ``cache_dir/<key>`` Zarr
            그룹 하나.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        """캐시를 초기화한다 (디렉토리는 put 시점에 생성).

        Args:
            cache_dir: 캐시 루트. None 이면 ``~/.naviertwin/tensor_cache/``.
        """
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path.home() / ".naviertwin" / "tensor_cache"
        )

    # ── 키 ──────────────────────────────────────────────────────────

    @staticmethod
    def key_for(
        datasets: Sequence[Any],
        params: Any,
        field_names: Sequence[str],
        resolution: int,
        param_names: Sequence[str] | None = None,
    ) -> str:
        """텐서화 입력 내용에서 콘텐츠 주소 캐시 키(16 hex)를 만든다.

        **케이스 메쉬·파라미터·필드·해상도(·파라미터명) 중 하나라도 바뀌면
        키가 바뀐다** — 케이스별 메쉬는 :func:`naviertwin.core.data_model.
        signature.compute_signature` 의 topology/coordinate 해시로, 파라미터는
        float64 배열 바이트 그대로(반올림 금지 — signature.py 와 같은 원칙)
        섞는다. 같은 입력은 언제나 같은 키를 준다(결정적).

        Args:
            datasets: 케이스 목록 (CFDDataset 또는 pyvista 메쉬 —
                ``cases_to_grid_tensors`` 의 ``datasets`` 와 동일 규약).
            params: 케이스별 파라미터, (N, k) 또는 (N,).
            field_names: 타깃 필드 이름 목록.
            resolution: 공통 격자 최장 축 분할 수.
            param_names: 파라미터 채널 이름 (None 허용 — 키에 그대로 반영).

        Returns:
            sha256 앞 16자리 hex 문자열.
        """
        from naviertwin.core.data_model.signature import compute_signature

        digest = hashlib.sha256()
        digest.update(f"tensor-cache-v{_CACHE_VERSION}".encode())
        for ds in datasets:
            sig = compute_signature(ds)
            digest.update(sig.topology_hash.encode())
            digest.update(b":")
            digest.update(sig.coordinate_hash.encode())
            digest.update(b";")
        mu = np.ascontiguousarray(np.asarray(params, dtype=np.float64))
        digest.update(str(mu.shape).encode())  # (6,)·(3,2) 바이트 동형 구분
        digest.update(mu.tobytes())
        digest.update("|".join(str(f) for f in field_names).encode())
        digest.update(b"#")
        digest.update(str(int(resolution)).encode())
        digest.update(b"#")
        digest.update(
            "|".join(str(n) for n in param_names).encode()
            if param_names is not None
            else b"<none>"
        )
        return digest.hexdigest()[:_KEY_HEX_LEN]

    # ── 읽기/쓰기 ────────────────────────────────────────────────────

    def get(self, key: str) -> dict[str, Any] | None:
        """캐시 항목을 복원한다 — 실패는 전부 조용히 miss(None).

        복원 dict 는 ``cases_to_grid_tensors`` 반환과 같은 키 구성이다:
        ``inputs``/``targets``/``valid_mask``/``grid``/``channel_names``/
        ``meta``. ``grid`` 는 저장본이 아니라 meta 의 dims/spacing/origin
        에서 :func:`grid_from_meta` 로 재구성한 것이다.

        Args:
            key: :meth:`key_for` 가 만든 키.

        Returns:
            복원된 텐서 dict, 또는 miss/손상/버전 불일치/zarr 부재 시 None.
        """
        zarr = _zarr()
        if zarr is None:
            return None
        entry = self.cache_dir / key
        if not entry.exists():
            return None
        # 캐시는 절대 정확성을 해치면 안 된다 — 어떤 예외든(손상 항목,
        # 스키마 불일치, zarr 내부 오류) miss 로 강등하고 재계산에 맡긴다.
        try:
            group = zarr.open_group(str(entry), mode="r")
            attrs = dict(group.attrs)
            if int(attrs.get("cache_version", -1)) != _CACHE_VERSION:
                return None
            meta = _restore_meta(dict(attrs["meta"]))
            channel_names = [str(c) for c in attrs["channel_names"]]
            arrays = {
                name: np.asarray(group[name][:]) for name in _REQUIRED_ARRAYS
            }
        except Exception:  # noqa: BLE001 — 원칙: 캐시 실패 = 조용한 miss.
            logger.debug("텐서 캐시 항목 %s 읽기 실패 → miss 처리", key, exc_info=True)
            return None
        return {
            "inputs": arrays["inputs"],
            "targets": arrays["targets"],
            "valid_mask": arrays["valid_mask"],
            "grid": grid_from_meta(meta),
            "channel_names": channel_names,
            "meta": meta,
        }

    def put(self, key: str, tensors: dict[str, Any]) -> None:
        """텐서화 결과를 캐시에 저장한다 — 실패는 조용히 무시(no-op).

        ``cases_to_grid_tensors`` 반환 dict 중 배열 3개(inputs/targets/
        valid_mask)와 meta·channel_names 만 저장한다. ``grid`` 는 저장하지
        않는다 — meta 의 dims/spacing/origin 으로 재구성 가능하기 때문.

        임시 디렉토리에 쓴 뒤 최종 이름으로 바꿔치기해, 도중에 중단돼도
        반쯤 쓰인 항목이 유효한 키로 남지 않게 한다.

        Args:
            key: :meth:`key_for` 가 만든 키.
            tensors: ``cases_to_grid_tensors`` 반환 dict.
        """
        zarr = _zarr()
        if zarr is None:
            return
        entry = self.cache_dir / key
        tmp = self.cache_dir / f".{key}.tmp"
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if tmp.exists():
                shutil.rmtree(tmp)
            group = zarr.open_group(str(tmp), mode="w")
            for name in _REQUIRED_ARRAYS:
                data = np.asarray(tensors[name])
                try:
                    # zarr 3.x
                    group.create_array(name, data=data)
                except (AttributeError, TypeError):
                    # zarr 2.x
                    group.create_dataset(name, data=data)
            group.attrs["cache_version"] = _CACHE_VERSION
            group.attrs["meta"] = _jsonable_meta(dict(tensors["meta"]))
            group.attrs["channel_names"] = [
                str(c) for c in tensors.get("channel_names", [])
            ]
            if entry.exists():
                shutil.rmtree(entry)
            tmp.replace(entry)
            logger.info("텐서 캐시 저장: %s (%s)", key, entry)
        except Exception:  # noqa: BLE001 — 캐시 쓰기 실패가 학습을 막으면 안 된다.
            logger.debug("텐서 캐시 항목 %s 쓰기 실패 → 무시", key, exc_info=True)
            shutil.rmtree(tmp, ignore_errors=True)

    # ── 관리 ─────────────────────────────────────────────────────────

    def clear(self) -> None:
        """캐시 디렉토리의 모든 항목을 지운다 (디렉토리 자체는 유지)."""
        if not self.cache_dir.exists():
            return
        for child in self.cache_dir.iterdir():
            shutil.rmtree(child, ignore_errors=True)

    def stats(self) -> dict[str, Any]:
        """캐시 현황 요약.

        Returns:
            ``n_entries`` (완결 항목 수 — 쓰다 만 ``.tmp`` 제외) 와
            ``total_bytes`` (캐시 디렉토리 전체 파일 크기 합) 를 담은 dict.
        """
        n_entries = 0
        total_bytes = 0
        if self.cache_dir.exists():
            for child in self.cache_dir.iterdir():
                if child.is_dir() and not child.name.startswith("."):
                    n_entries += 1
                for f in child.rglob("*") if child.is_dir() else [child]:
                    if f.is_file():
                        total_bytes += f.stat().st_size
        return {"n_entries": n_entries, "total_bytes": total_bytes}
