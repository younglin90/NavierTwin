"""NavierTwin 헤드리스 CLI 패키지.

GUI(PySide6/trame) 이벤트 루프와 완전히 분리된 배치 실행 진입점을 모은다.
MPI(mpi4py) 초기화는 반드시 이 패키지의 헤드리스 경로에서만 일어나야 하며,
GUI 프로세스에서는 절대 import 하지 않는다 (설계 결정: v5.6 P1+).
"""

from __future__ import annotations

__all__ = ["batch_train"]
