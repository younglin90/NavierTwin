"""실험 관리 (외부 검토 §6½ #6, 저장 계층) — 학습 실행(run) 기록.

지금까지는 어떤 트윈을 어떤 설정(resolution/epochs/modes 등)으로 학습했고
결과가 어땠는지(train_loss, remap_floor, held-out rel_l2 등)가 남지 않아
매번 새로 학습해야 비교할 수 있었다. 이 모듈은 MLflow(로컬 파일 백엔드,
서버 불필요)로 학습 실행마다 params/metrics 를 가볍게 기록한다.

설계 원칙: **실험 추적은 학습을 절대 막으면 안 된다.** mlflow 가 설치돼
있지 않거나 기록 중 어떤 이유로든 실패하면 조용히 no-op 으로 폴백한다
(:class:`~naviertwin.core.storage.tensor_cache.TensorCache` 와 같은 원칙).
"""

from naviertwin.core.experiment.tracking import ExperimentTracker

__all__ = ["ExperimentTracker"]
