"""Gmsh .msh 파일 리더 모듈.

Gmsh 메쉬 파일(.msh v2.2, v4.0, v4.1)을 읽는다.
NASTRAN 등 다른 도구의 .msh 포맷은 meshio 폴백에서 자동 감지 실패 시 오류 출력.

폴백 체인:
    1. gmsh Python API (probe — 설치 확인 + meshio 위임)
    2. meshio (primary path)

v1.1.0 범위:
    gmsh API 를 사용한 직접 변환(getNodes/getElements → pv.UnstructuredGrid)은
    v1.2.0 으로 이연. v1.1.0 에서 gmsh API 는 probe 용도로만 사용한다.

Examples:
    직접 사용::

        from pathlib import Path
        from naviertwin.core.cfd_reader.gmsh_reader import GmshReader

        reader = GmshReader()
        dataset = reader.read(Path("mesh.msh"))
"""

from __future__ import annotations

from pathlib import Path

from naviertwin.core.cfd_reader._mesh_utils import meshio_to_cfd_dataset
from naviertwin.core.cfd_reader.base import BaseReader, CFDDataset
from naviertwin.core.cfd_reader.reader_factory import ReaderFactory
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


@ReaderFactory.register
class GmshReader(BaseReader):
    """Gmsh .msh 파일 리더.

    폴백 체인: gmsh API (probe) → meshio.
    gmsh v2.x (msh v1 포맷)은 미지원.

    Attributes:
        supported_extensions: ``.msh`` 확장자를 지원한다.
    """

    supported_extensions: frozenset[str] = frozenset({".msh"})

    def read(self, path: Path) -> CFDDataset:
        """Gmsh .msh 파일을 읽어 CFDDataset 을 반환한다.

        Args:
            path: .msh 파일 경로.

        Returns:
            파싱된 CFDDataset.

        Raises:
            FileNotFoundError: 경로가 존재하지 않는 경우.
            ValueError: 모든 파서가 실패한 경우.
        """
        if not path.exists():
            raise FileNotFoundError(f"파일이 존재하지 않습니다: {path}")

        logger.info("Gmsh .msh 읽기 시작: %s", path)

        # gmsh API probe: 설치 여부만 확인, 실제 변환은 meshio 에 위임
        _gmsh_probe(path)

        meshio_err: Exception | None = None

        try:
            return self._read_with_meshio(path)
        except ImportError:
            logger.warning(
                "meshio 미설치. 설치: pip install 'naviertwin[full]'"
            )
        except SystemExit as e:
            meshio_err = ValueError(f"meshio sys.exit({e.code}): Gmsh 파싱 실패 ({path})")
            logger.debug("meshio GmshReader sys.exit: %s", path)
        except Exception as e:
            meshio_err = e
            logger.debug("meshio GmshReader 실패: %s", e)

        raise ValueError(
            f"[GmshReader] 모든 파서 실패 (gmsh API/meshio): {path}\n"
            f"  meshio: {meshio_err}\n"
            "  참고: 다른 .msh 포맷(NASTRAN 등)은 미지원. "
            "Gmsh 2.x 구버전(.msh v1)도 미지원."
        )

    def _read_with_meshio(self, path: Path) -> CFDDataset:
        try:
            import meshio
        except ImportError as exc:
            raise ImportError("meshio 미설치") from exc

        logger.debug("meshio 로 Gmsh .msh 읽기: %s", path)
        mesh = meshio.read(str(path), file_format="gmsh")
        return meshio_to_cfd_dataset(mesh, str(path), "meshio/Gmsh")


# ---------------------------------------------------------------------------
# gmsh API probe
# ---------------------------------------------------------------------------


def _gmsh_probe(path: Path) -> None:
    """gmsh Python API 설치 여부를 확인한다 (probe only).

    gmsh 가 설치됐으면 initialize → finalize 만 실행한다.
    실제 변환은 meshio 에 위임한다 (v1.1.0).

    gmsh 가 미설치면 logger.info 만 출력하고 정상 반환한다
    (meshio 폴백이 처리).

    Args:
        path: .msh 파일 경로 (로그용).
    """
    try:
        import gmsh
    except ImportError:
        logger.info(
            "gmsh 미설치 — meshio 로 읽기. "
            "gmsh API 지원: pip install 'naviertwin[full]'"
        )
        return

    try:
        gmsh.initialize()
    except Exception as e:
        logger.debug("gmsh.initialize() 실패: %s", e)
        return

    try:
        pass  # v1.1.0: probe only. v1.2.0 에서 getNodes/getElements 구현.
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass

    logger.debug("gmsh API probe 완료: %s", path)
