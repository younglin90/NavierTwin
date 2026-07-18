"""DMD 동역학 트윈 어댑터 (PyDMD 기반).

DMD/Koopman 계열은 상태의 **시간 전이 규칙**을 학습하므로, 다른 계열과 달리
학습 구간 **너머로 외삽(forecast)** 할 수 있다 — 이것이 이 계열의 존재 이유다
(POD+보간이나 신경장은 학습 파라미터 범위 안 내삽이 전제).

:class:`naviertwin.core.flow_analysis.modal.dmd.DMDAnalyzer` (PyDMD 래퍼)는
``reconstruct(t)`` 를 노출하므로, 여기서 ``predict(params)`` 계약으로 감싸
``TwinEngine``/``PhysicsAITwinEngine`` 과 동일한 덕타이핑을 갖게 한다 — 웹/GUI
의 예측·저장 경로를 수정 없이 재사용하기 위함이다.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class DMDTwinEngine:
    """학습된 DMD 분석기를 트윈 엔진 계약으로 감싼다.

    Attributes:
        analyzer: fit 완료된 ``DMDAnalyzer``.
        reducer_type: 계열 식별자 ``"dmd_dynamics"`` (복원 시 분기용).
        surrogate_type: DMD 변형 이름 (``dmd``/``fbdmd``/``hodmd``/``spdmd``).
        training_metadata: 학습 범위·필드 등 (웹 GUI 가 슬라이더 구성에 사용).
    """

    def __init__(
        self,
        analyzer: Any,
        *,
        method: str = "fbdmd",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not hasattr(analyzer, "reconstruct"):
            raise TypeError("DMD analyzer must expose reconstruct(t)")
        self.analyzer = analyzer
        self.model = analyzer  # 공통 접근자 (predict_to_mesh 등이 model 을 본다)
        self.reducer = None
        self.surrogate = analyzer
        self.reducer_type = "dmd_dynamics"
        self.surrogate_type = str(method)
        self.model_type = f"dmd_{method}"
        self.training_metadata: dict[str, Any] = dict(metadata or {})

    @property
    def is_fitted(self) -> bool:
        """감싼 분석기의 학습 완료 여부."""
        return bool(getattr(self.analyzer, "is_fitted", True))

    def predict(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        """시간에서 유동장을 재구성한다 (학습 구간 밖이면 외삽).

        Args:
            params: 시간값. 스칼라 배열 ``(1,)`` 또는 ``(n_times, 1)``.

        Returns:
            단일 시간이면 ``(n_features,)``, 다중 시간이면
            ``(n_features, n_times)`` — ``TwinEngine.predict`` 와 같은 레이아웃.
        """
        times = np.asarray(params, dtype=np.float64).reshape(-1)
        field = np.asarray(self.analyzer.reconstruct(times), dtype=np.float64)
        if times.size == 1:
            return field[:, 0]
        return field

    def save(self, path: str | Path) -> None:
        """엔진을 pickle 로 저장한다."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "DMDTwinEngine":
        """저장된 DMD 트윈 엔진을 로드한다."""
        with Path(path).open("rb") as f:
            engine = pickle.load(f)
        if not isinstance(engine, cls):
            raise TypeError(f"DMDTwinEngine 파일이 아닙니다: {path}")
        return engine

    def get_params(self) -> dict[str, Any]:
        """TwinEngine 호출자와 호환되는 메타데이터."""
        return {
            "reducer_type": self.reducer_type,
            "surrogate_type": self.surrogate_type,
            "model_type": self.model_type,
        }


__all__ = ["DMDTwinEngine"]
