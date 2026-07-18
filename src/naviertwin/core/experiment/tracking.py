"""MLflow 기반 학습 실행(run) 추적기 — 로컬 파일 백엔드, 서버 불필요.

외부 검토 §6½ #6(저장 계층 — 실험 관리)의 구현. ``build_geometry_fno_twin``/
``build_mesh_gnn_twin_from_cases``/``compare_models``/``build_twin``(ROM) 등
여러 학습 경로가 각자 흩어져 있어, 어떤 트윈을 어떤 설정(resolution/epochs/
modes 등)으로 학습했고 결과(train_loss, remap_floor, held-out rel_l2 등)가
어땠는지 기록이 남지 않았다 — 매번 새로 학습해야만 비교할 수 있었다. 이
모듈은 학습 실행마다 params/metrics 를 MLflow run 하나로 가볍게 기록한다.

설계 원칙:
    - **실험 추적은 학습을 절대 막으면 안 된다.** mlflow 가 설치돼 있지
      않거나, tracking 디렉토리를 만들 수 없거나, 기록 중 어떤 이유로든
      실패하면 **조용히 no-op** 으로 폴백한다(예외를 삼키고 실패를 나타내는
      값 — ``None``/빈 리스트 — 을 돌려준다). 호출부는 이 반환값을 무시해도
      안전하다.
    - mlflow 는 선택 의존성이다([full] extra). 미설치 환경에서는 모든
      메서드가 아무것도 하지 않고 성공(빈 결과)을 돌려준다.
    - 서버가 필요 없는 tracking-only 사용 — ``file:<tracking_dir>`` 로컬
      파일 백엔드만 쓴다(원격 tracking server 미지원, 필요 없음).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["ExperimentTracker"]

# run 을 전략(strategy)별로 구분하기 위한 태그 키. mlflow experiment 는
# strategy 마다 새로 만들지 않고(naviertwin 실험 하나에 몰아넣고) 이 태그로
# 필터링한다 — 전략을 추가/변경해도 experiment 스키마를 안 건드려도 된다.
_STRATEGY_TAG = "naviertwin.strategy"


def _mlflow() -> Any | None:
    """mlflow 모듈을 지연 import 한다 — 없으면 None (조용한 폴백)."""
    try:
        import mlflow
    except Exception:  # noqa: BLE001 — 미설치/버전 파손 모두 "추적 없음"으로.
        return None
    return mlflow


class ExperimentTracker:
    """학습 실행(run)을 MLflow 로컬 파일 백엔드에 기록/조회한다.

    사용 규약::

        tracker = ExperimentTracker()               # ~/.naviertwin/mlruns
        run_id = tracker.log_run(
            "geometry_fno",
            params={"resolution": 48, "epochs": 200},
            metrics={"remap_floor_rel_l2": 0.03},
        )
        recent = tracker.list_runs("geometry_fno")   # 최신순 dict 목록

    mlflow 가 없거나 기록/조회가 실패해도 예외를 던지지 않는다 — 학습
    파이프라인이 이 클래스 호출 여부와 무관하게 항상 성공해야 하기 때문
    이다(:class:`~naviertwin.core.storage.tensor_cache.TensorCache` 와
    동일한 "캐시/부가 기능은 핵심 흐름을 막지 않는다" 원칙).

    Attributes:
        tracking_dir: MLflow 로컬 파일 백엔드 루트 디렉토리.
        experiment_name: 모든 run 을 몰아넣는 단일 MLflow experiment 이름.
    """

    def __init__(
        self,
        tracking_dir: Path | None = None,
        experiment_name: str = "naviertwin",
    ) -> None:
        """추적기를 초기화한다 (디렉토리/experiment 는 최초 기록 시점에 생성).

        Args:
            tracking_dir: MLflow tracking 디렉토리. None 이면
                ``~/.naviertwin/mlruns``.
            experiment_name: MLflow experiment 이름. 기본 ``"naviertwin"``.
        """
        self.tracking_dir = (
            Path(tracking_dir)
            if tracking_dir is not None
            else Path.home() / ".naviertwin" / "mlruns"
        )
        self.experiment_name = experiment_name

    def _prepare(self) -> Any | None:
        """mlflow 모듈을 준비한다(tracking_uri/experiment 설정) — 실패 시 None.

        Returns:
            준비된 ``mlflow`` 모듈, 또는 mlflow 미설치/준비 실패 시 None.
        """
        mlflow = _mlflow()
        if mlflow is None:
            return None
        try:
            self.tracking_dir.mkdir(parents=True, exist_ok=True)
            # MLflow ≥3 는 로컬 파일 백엔드를 "유지보수 모드"로 표시하고 기본
            # 거부한다(DB 백엔드 권장). 이 프로젝트는 서버 없는 tracking-only
            # 사용이 목적이라 파일 백엔드를 그대로 허용한다 — opt-out 플래그.
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            mlflow.set_tracking_uri(f"file:{self.tracking_dir}")
            mlflow.set_experiment(self.experiment_name)
        except Exception:  # noqa: BLE001 — 추적 준비 실패는 조용히 폴백.
            logger.debug("MLflow 준비 실패 → 추적 없이 진행", exc_info=True)
            return None
        return mlflow

    def log_run(
        self,
        strategy: str,
        params: dict[str, Any],
        metrics: dict[str, Any],
        *,
        tags: dict[str, Any] | None = None,
    ) -> str | None:
        """학습 실행 하나를 MLflow run 으로 기록한다.

        Args:
            strategy: 학습 전략 식별자(예: ``"geometry_fno"``, ``"mesh_gnn"``,
                ``"rom"``). :meth:`list_runs` 필터와 MLflow 태그로 쓰인다.
            params: 기록할 하이퍼파라미터(resolution/epochs/modes 등).
                값은 문자열로 변환해 기록한다.
            metrics: 기록할 지표(train_loss/remap_floor_rel_l2/rel_l2 등).
                float 로 변환 불가능한 값은 조용히 건너뛴다.
            tags: 추가로 붙일 MLflow 태그(선택).

        Returns:
            성공하면 MLflow run_id, mlflow 미설치나 기록 실패 시 None.
        """
        mlflow = self._prepare()
        if mlflow is None:
            return None
        try:
            with mlflow.start_run(run_name=strategy) as run:
                mlflow.set_tag(_STRATEGY_TAG, strategy)
                for key, value in (tags or {}).items():
                    mlflow.set_tag(str(key), value)
                for key, value in params.items():
                    mlflow.log_param(str(key), value)
                for key, value in metrics.items():
                    try:
                        mlflow.log_metric(str(key), float(value))
                    except (TypeError, ValueError):
                        # 지표가 아닌 값(문자열 등)은 metric 으로 못 남기니
                        # 건너뛴다 — 기록 전체를 실패시키지 않는다.
                        continue
                return str(run.info.run_id)
        except Exception:  # noqa: BLE001 — run 기록 실패는 학습을 막지 않는다.
            logger.debug("MLflow run 기록 실패 → 무시", exc_info=True)
            return None

    def list_runs(self, strategy: str | None = None) -> list[dict[str, Any]]:
        """기록된 run 들을 최신순으로 나열한다.

        Args:
            strategy: 지정하면 해당 전략의 run 만 필터링한다. None 이면 전체.

        Returns:
            ``run_id``/``strategy``/``params``/``metrics``/``start_time`` 을
            담은 dict 목록(최신 시작순). mlflow 미설치나 조회 실패 시 빈 리스트.
        """
        mlflow = self._prepare()
        if mlflow is None:
            return []
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return []
            filter_string = (
                f"tags.`{_STRATEGY_TAG}` = '{strategy}'" if strategy else ""
            )
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filter_string,
                order_by=["start_time DESC"],
                output_format="list",
            )
        except Exception:  # noqa: BLE001 — 조회 실패는 빈 목록으로 폴백.
            logger.debug("MLflow run 목록 조회 실패 → 빈 목록", exc_info=True)
            return []
        result: list[dict[str, Any]] = []
        for run in runs:
            data = run.data
            result.append(
                {
                    "run_id": run.info.run_id,
                    "strategy": dict(data.tags).get(_STRATEGY_TAG, ""),
                    "params": dict(data.params),
                    "metrics": dict(data.metrics),
                    "start_time": run.info.start_time,
                }
            )
        return result
