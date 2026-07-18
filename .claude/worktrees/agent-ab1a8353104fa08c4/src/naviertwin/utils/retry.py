"""실패 시 지수 백오프 재시도 데코레이터.

네트워크 / 파일 I/O / 외부 프로세스 호출 등 일시 실패 가능한 작업에 사용.

Examples:
    >>> from naviertwin.utils.retry import retry
    >>> calls = {"n": 0}
    >>> @retry(max_attempts=3, delay=0.01, exceptions=(ValueError,))
    ... def flaky():
    ...     calls["n"] += 1
    ...     should_fail = calls["n"] < 3
    ...     if should_fail:
    ...         raise ValueError("fail")
    ...     return "ok"
    >>> flaky()
    'ok'
    >>> calls["n"]
    3
"""

from __future__ import annotations

import time
from functools import wraps
from typing import Callable, TypeVar

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def retry(
    max_attempts: int = 3,
    delay: float = 0.1,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    jitter: float = 0.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """지수 백오프 재시도.

    Args:
        max_attempts: 총 시도 횟수 (>=1).
        delay: 첫 대기(sec).
        backoff: 배수.
        exceptions: 잡을 예외 타입.
        jitter: [0, jitter] 랜덤 가산.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts >= 1")

    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            import random

            d = float(delay)
            last: BaseException | None = None
            attempt = 1
            while attempt <= max_attempts:
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last = exc
                    if attempt == max_attempts:
                        logger.warning(
                            "%s: 최종 실패 (%d회): %s", fn.__name__, attempt, exc
                        )
                        raise
                    sleep_delay = d + (random.random() * jitter if jitter > 0 else 0.0)
                    logger.info(
                        "%s: 시도 %d/%d 실패 (%s), %.3fs 대기",
                        fn.__name__, attempt, max_attempts, exc, sleep_delay,
                    )
                    time.sleep(sleep_delay)
                    d *= backoff
                    attempt += 1
            assert last is not None
            raise last

        return wrapper

    return deco


__all__ = ["retry"]
