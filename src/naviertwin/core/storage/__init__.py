"""저장 계층 (외부 검토 §6½ #6) — ML 텐서 캐시 등 파생 산출물 저장소.

검토가 요구한 4계층 저장(원본 불변 보존 / 정규화 스토어 / **ML 캐시** /
메타데이터) 중 ML 캐시 조각을 담는 모듈이다. 원칙: 캐시는 어디까지나
파생물이므로 **언제 지워져도 정확성이 깨지면 안 된다** — 손상/부재는 항상
"다시 계산"으로 폴백한다.
"""

from naviertwin.core.storage.tensor_cache import TensorCache

__all__ = ["TensorCache"]
