"""Canonical `.ntwin` 저장 계층 — 원본 CFD 파일 재파싱 회피 캐시.

외부 검토 §6½ #6(4계층 저장 아키텍처)이 요구한 계층 2("canonical 중간
포맷")의 배선. `.ntwin`(:mod:`naviertwin.core.export.ntwin_format`)은 이미
ParaView 호환 VTKHDF 구조를 쓰는 canonical 포맷이므로 이 모듈은 **새 포맷을
만들지 않는다** — 대신 임의의 원본 리더(OpenFOAM/CGNS/VTK/Fluent/SU2/Gmsh
등, :class:`~naviertwin.core.cfd_reader.reader_factory.ReaderFactory` 가 아는
모든 확장자)로 읽은 결과를 표준 캐시 경로 하나의 `.ntwin` 파일로 저장해두고,
다음에 같은 원본을 다시 읽을 때 리더 파싱을 건너뛰고 `.ntwin`에서 바로
복원하는 얇은 조회/변환 계층이다.

캐시 키: 원본 파일 경로 + mtime(ns) + size 를 해시한다. **파일 내용을 통째로
해시하지 않는다** — 대용량 CFD 결과 파일을 매번 전체 읽어 해시하면, 캐시가
없애려는 바로 그 비용(원본 재파싱)을 해시 계산으로 다시 지불하게 되어 캐시의
존재 이유가 사라진다. mtime/size 는 `stat()` 호출 하나로 얻으므로 사실상
공짜다. 트레이드오프: 파일 내용이 바뀌었는데 mtime/size 가 우연히 같게
보존되는 극히 드문 경우(다른 파일을 같은 이름/타임스탬프로 덮어쓰는 등)는
탐지하지 못한다 — tensor_cache.py/signature.py 와 마찬가지로, 이 캐시는
정확성보다 속도를 위한 것이고 항상 miss(=재변환) 로 안전하게 되돌아갈 수
있으므로 이 트레이드오프를 받아들인다.

설계 원칙(``core/storage/tensor_cache.py`` 와 동일):
    - **캐시는 절대 정확성을 해치면 안 된다.** 캐시 항목이 손상됐거나,
      존재하지 않거나, 읽기에 실패하면 예외를 삼키고 조용히 miss 로
      처리한다 — 호출부는 항상 ``reader_fn`` 재호출로 원본을 다시 읽을 수
      있다. 캐시 디렉토리를 통째로 지워도 결과는 같다(느려질 뿐).
    - **원자적 쓰기**: 임시 파일에 쓴 뒤 최종 이름으로 rename 해, 쓰다 만
      항목이 유효한 키로 남아 다음 조회에서 손상 파일로 읽히지 않게 한다.
    - **원본 무변경**: 이 모듈은 원본 파일을 절대 열어서 쓰지 않는다(읽기
      전용 ``reader_fn`` 호출만) — 계층 1(원본 불변 보존)을 침범하지 않는다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["CanonicalCache"]

# 캐시 스키마 버전 — 키 계산 방식이 바뀌면 올린다(예전 키와 절대 충돌하지
# 않게 하기 위함, 실제로는 .ntwin 파일 자체의 버전은 ntwin_format 이 관리).
_CACHE_VERSION = 1

# 키 길이 — tensor_cache.py/signature.py 와 같은 근거(64 bit, 우연 충돌
# 확률 사실상 0 — 실제 캐시 항목 수는 많아야 수천~수만 개).
_KEY_HEX_LEN = 16


class CanonicalCache:
    """원본 CFD 파일 → `.ntwin` 콘텐츠(경로+mtime+size) 주소 캐시.

    사용 규약::

        cache = CanonicalCache()               # ~/.naviertwin/canonical_cache/
        dataset = cache.get_or_convert(path, lambda p: ReaderFactory().create_and_read(p))

    첫 호출(miss)은 ``reader_fn(path)`` 를 호출해 얻은 데이터셋을 그대로
    반환하면서 `.ntwin` 으로 저장해 둔다. 원본이 바뀌지 않은 채 다시
    호출하면(hit) `.ntwin` 에서 즉시 복원하고 ``reader_fn`` 은 전혀 호출하지
    않는다 — 대용량 원본 재파싱(OpenFOAM 케이스 디렉토리 스캔, CGNS 트리
    순회 등)을 건너뛴다.

    Attributes:
        cache_dir: 캐시 루트 디렉토리. 항목 하나 = ``cache_dir/<key>.ntwin``.
    """

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        """캐시를 초기화한다 (디렉토리는 최초 저장 시점에 생성).

        Args:
            cache_dir: 캐시 루트. None 이면 ``~/.naviertwin/canonical_cache/``.
        """
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path.home() / ".naviertwin" / "canonical_cache"
        )

    # ── 키 ──────────────────────────────────────────────────────────

    @staticmethod
    def key_for(source_path: Path | str) -> str:
        """원본 파일의 경로+mtime+size 에서 콘텐츠 주소 캐시 키(16 hex)를 만든다.

        Args:
            source_path: 원본 CFD 파일(또는 디렉토리, 예: OpenFOAM 케이스) 경로.

        Returns:
            sha256 앞 16자리 hex 문자열.

        Raises:
            OSError: ``source_path`` 가 존재하지 않거나 stat 할 수 없는 경우.
        """
        resolved = Path(source_path).expanduser().resolve()
        stat = resolved.stat()

        digest = hashlib.sha256()
        digest.update(f"canonical-cache-v{_CACHE_VERSION}".encode())
        digest.update(b"|")
        digest.update(str(resolved).encode("utf-8", errors="surrogateescape"))
        digest.update(b"|")
        digest.update(str(stat.st_size).encode())
        digest.update(b"|")
        digest.update(str(stat.st_mtime_ns).encode())
        return digest.hexdigest()[:_KEY_HEX_LEN]

    # ── 조회/변환 ────────────────────────────────────────────────────

    def get_or_convert(
        self, source_path: Path | str, reader_fn: Callable[[Path], Any]
    ) -> Any:
        """캐시 히트면 `.ntwin` 에서 바로 로드하고, 미스면 ``reader_fn`` 을 불러 변환·저장한다.

        Args:
            source_path: 원본 CFD 파일 경로.
            reader_fn: 캐시 미스 시 호출할 원본 리더 함수 — ``Path`` 하나를
                받아 :class:`~naviertwin.core.cfd_reader.base.CFDDataset` 를
                돌려줘야 한다 (예: ``ReaderFactory().create_and_read``).

        Returns:
            로드된(또는 갓 변환된) ``CFDDataset``.
        """
        from naviertwin.core.export import ntwin_format

        path = Path(source_path)

        try:
            key = self.key_for(path)
        except OSError:
            # stat 실패(파일 사라짐 등) — 캐시 없이 리더에 그대로 맡긴다.
            logger.debug("canonical 캐시 키 계산 실패 → 캐시 우회: %s", path, exc_info=True)
            return reader_fn(path)

        entry = self.cache_dir / f"{key}.ntwin"

        if entry.exists():
            # 캐시는 절대 정확성을 해치면 안 된다 — 읽기 실패(손상 항목,
            # h5py 부재, 스키마 불일치)는 전부 조용히 miss 로 강등한다.
            try:
                dataset = ntwin_format.load_dataset(entry)
                logger.info("canonical 캐시 히트: %s → %s (재파싱 생략)", path, entry)
                return dataset
            except Exception:  # noqa: BLE001 — 원칙: 캐시 실패 = 조용한 miss.
                logger.debug(
                    "canonical 캐시 항목 읽기 실패 → miss 처리: %s", entry, exc_info=True
                )

        dataset = reader_fn(path)

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_dir / f".{key}.tmp.ntwin"
            if tmp.exists():
                tmp.unlink()
            ntwin_format.save_dataset(dataset, tmp)
            tmp.replace(entry)
            logger.info("canonical 캐시 저장: %s → %s", path, entry)
        except Exception:  # noqa: BLE001 — 캐시 쓰기 실패가 로드를 막으면 안 된다.
            logger.debug("canonical 캐시 쓰기 실패 → 무시: %s", path, exc_info=True)

        return dataset

    # ── 관리 ─────────────────────────────────────────────────────────

    def clear(self) -> None:
        """캐시 디렉토리의 모든 항목을 지운다 (디렉토리 자체는 유지)."""
        if not self.cache_dir.exists():
            return
        for child in self.cache_dir.iterdir():
            try:
                if child.is_dir():
                    import shutil

                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink()
            except OSError:
                pass

    def stats(self) -> dict[str, Any]:
        """캐시 현황 요약.

        Returns:
            ``n_entries``(완결 `.ntwin` 항목 수 — 쓰다 만 ``.tmp`` 제외)와
            ``total_bytes``(캐시 디렉토리 전체 파일 크기 합)를 담은 dict.
        """
        n_entries = 0
        total_bytes = 0
        if self.cache_dir.exists():
            for child in self.cache_dir.iterdir():
                if child.is_file():
                    total_bytes += child.stat().st_size
                    if child.suffix == ".ntwin" and not child.name.startswith("."):
                        n_entries += 1
        return {"n_entries": n_entries, "total_bytes": total_bytes}
