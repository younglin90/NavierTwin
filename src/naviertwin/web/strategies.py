"""웹 호환 shim — 전략 레지스트리는 core 로 이동했다 (v5.2, gui/web 공유).

실체는 :mod:`naviertwin.core.digital_twin.strategies`. 웹 코드의 기존 import
경로(``naviertwin.web.strategies``)를 깨지 않기 위한 재수출이다.
"""

from naviertwin.core.digital_twin.strategies import (
    STRATEGIES,
    DataProfile,
    StrategySpec,
    profile_data,
    recommend,
    strategy_report,
)

__all__ = [
    "STRATEGIES",
    "DataProfile",
    "StrategySpec",
    "profile_data",
    "recommend",
    "strategy_report",
]
