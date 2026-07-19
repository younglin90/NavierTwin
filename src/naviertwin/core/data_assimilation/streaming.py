"""Timestamp-aware sensor buffering and online assimilation scheduling."""

from __future__ import annotations

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from threading import RLock
from typing import Iterable, Literal


class SensorQuality(str, Enum):
    """Quality state supplied by sensor gateways."""

    GOOD = "good"
    SUSPECT = "suspect"
    BAD = "bad"


@dataclass(frozen=True, slots=True)
class SensorSample:
    """One immutable, timestamped sensor observation."""

    sensor_id: str
    timestamp: float
    values: tuple[float, ...]
    quality: SensorQuality = SensorQuality.GOOD
    sequence: int | None = None

    def __post_init__(self) -> None:
        sensor_id = self.sensor_id.strip()
        values = tuple(float(value) for value in self.values)
        if not sensor_id:
            raise ValueError("sensor_id must not be empty")
        if not isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        if not values or not all(map(isfinite, values)):
            raise ValueError("values must contain finite numbers")
        if self.sequence is not None and self.sequence < 0:
            raise ValueError("sequence must be >= 0")
        object.__setattr__(self, "sensor_id", sensor_id)
        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "quality", SensorQuality(self.quality))


class SensorBuffer:
    """Thread-safe, bounded, timestamp-ordered buffers grouped by sensor."""

    def __init__(
        self,
        max_samples_per_sensor: int = 1024,
        *,
        out_of_order: Literal["insert", "reject"] = "insert",
    ) -> None:
        if max_samples_per_sensor < 2:
            raise ValueError("max_samples_per_sensor must be >= 2")
        if out_of_order not in {"insert", "reject"}:
            raise ValueError("out_of_order must be 'insert' or 'reject'")
        self.max_samples_per_sensor = int(max_samples_per_sensor)
        self.out_of_order = out_of_order
        self._samples: dict[str, deque[SensorSample]] = {}
        self._lock = RLock()

    def append(self, sample: SensorSample) -> bool:
        """Insert a sample; return ``False`` for an older duplicate sequence."""
        with self._lock:
            samples = self._samples.setdefault(sample.sensor_id, deque())
            if samples and sample.timestamp < samples[-1].timestamp:
                if self.out_of_order == "reject":
                    raise ValueError(
                        f"out-of-order sample for {sample.sensor_id}: "
                        f"{sample.timestamp} < {samples[-1].timestamp}"
                    )
            items = list(samples)
            index = bisect_left([item.timestamp for item in items], sample.timestamp)
            if index < len(items) and items[index].timestamp == sample.timestamp:
                current = items[index]
                if (
                    current.sequence is not None
                    and sample.sequence is not None
                    and sample.sequence < current.sequence
                ):
                    return False
                items[index] = sample
            else:
                items.insert(index, sample)
            if len(items) > self.max_samples_per_sensor:
                items = items[-self.max_samples_per_sensor :]
            self._samples[sample.sensor_id] = deque(items)
            return True

    def samples(
        self,
        sensor_id: str,
        *,
        start: float | None = None,
        end: float | None = None,
        include_bad: bool = False,
    ) -> tuple[SensorSample, ...]:
        """Return an immutable time window."""
        with self._lock:
            source = tuple(self._samples.get(sensor_id, ()))
        return tuple(
            sample
            for sample in source
            if (start is None or sample.timestamp >= start)
            and (end is None or sample.timestamp <= end)
            and (include_bad or sample.quality is not SensorQuality.BAD)
        )

    def latest(self, sensor_id: str) -> SensorSample | None:
        """Return latest usable sample."""
        samples = self.samples(sensor_id)
        return samples[-1] if samples else None

    def sensor_ids(self) -> tuple[str, ...]:
        """Return stable sensor identifiers."""
        with self._lock:
            return tuple(sorted(self._samples))


@dataclass(frozen=True, slots=True)
class TimeAlignmentPolicy:
    """Limits used by nearest/linear sensor time alignment."""

    tolerance: float = 0.1
    max_interpolation_gap: float = 1.0
    max_age: float = 5.0

    def __post_init__(self) -> None:
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be >= 0")
        if self.max_interpolation_gap <= 0.0:
            raise ValueError("max_interpolation_gap must be > 0")
        if self.max_age < 0.0:
            raise ValueError("max_age must be >= 0")


@dataclass(frozen=True, slots=True)
class AlignedObservation:
    """Sensor values aligned to one model time."""

    timestamp: float
    sensor_ids: tuple[str, ...]
    values: tuple[tuple[float, ...] | None, ...]
    ages: tuple[float | None, ...]
    missing_sensor_ids: tuple[str, ...]
    stale_sensor_ids: tuple[str, ...]

    @property
    def complete(self) -> bool:
        """Whether every requested sensor has a usable value."""
        return not self.missing_sensor_ids and not self.stale_sensor_ids

    def flatten(self) -> tuple[float, ...]:
        """Flatten aligned values in requested sensor order."""
        if not self.complete:
            raise ValueError("cannot flatten incomplete observation")
        return tuple(value for values in self.values if values for value in values)


def align_observations(
    buffer: SensorBuffer,
    sensor_ids: Iterable[str],
    timestamp: float,
    *,
    policy: TimeAlignmentPolicy | None = None,
    method: Literal["nearest", "linear"] = "linear",
) -> AlignedObservation:
    """Align multiple sensor streams without extrapolating across large gaps."""
    if not isfinite(timestamp):
        raise ValueError("timestamp must be finite")
    if method not in {"nearest", "linear"}:
        raise ValueError("method must be 'nearest' or 'linear'")
    active_policy = policy or TimeAlignmentPolicy()
    requested = tuple(sensor_id.strip() for sensor_id in sensor_ids)
    if not requested or any(not sensor_id for sensor_id in requested):
        raise ValueError("sensor_ids must not be empty")

    aligned_values: list[tuple[float, ...] | None] = []
    ages: list[float | None] = []
    missing: list[str] = []
    stale: list[str] = []
    for sensor_id in requested:
        samples = buffer.samples(sensor_id)
        value, age, had_candidate = _align_one(
            samples,
            float(timestamp),
            policy=active_policy,
            method=method,
        )
        if value is None:
            (stale if had_candidate else missing).append(sensor_id)
        aligned_values.append(value)
        ages.append(age)
    return AlignedObservation(
        timestamp=float(timestamp),
        sensor_ids=requested,
        values=tuple(aligned_values),
        ages=tuple(ages),
        missing_sensor_ids=tuple(missing),
        stale_sensor_ids=tuple(stale),
    )


def _align_one(
    samples: tuple[SensorSample, ...],
    timestamp: float,
    *,
    policy: TimeAlignmentPolicy,
    method: Literal["nearest", "linear"],
) -> tuple[tuple[float, ...] | None, float | None, bool]:
    if not samples:
        return None, None, False
    times = [sample.timestamp for sample in samples]
    right = bisect_left(times, timestamp)
    candidates = []
    if right < len(samples):
        candidates.append(samples[right])
    if right > 0:
        candidates.append(samples[right - 1])
    nearest = min(candidates, key=lambda sample: abs(sample.timestamp - timestamp))
    nearest_age = abs(nearest.timestamp - timestamp)

    if nearest_age <= policy.tolerance:
        if nearest_age > policy.max_age:
            return None, nearest_age, True
        return nearest.values, nearest_age, True

    if method == "linear" and 0 < right < len(samples):
        before = samples[right - 1]
        after = samples[right]
        gap = after.timestamp - before.timestamp
        age = max(timestamp - before.timestamp, after.timestamp - timestamp)
        if (
            gap <= policy.max_interpolation_gap
            and age <= policy.max_age
            and len(before.values) == len(after.values)
        ):
            weight = (timestamp - before.timestamp) / gap
            values = tuple(
                left + weight * (right_value - left)
                for left, right_value in zip(before.values, after.values, strict=True)
            )
            return values, age, True

    return None, nearest_age, True


class OnlineAssimilationScheduler:
    """Thread-safe gate preventing duplicate or overly frequent updates."""

    def __init__(self, interval: float = 0.0) -> None:
        if interval < 0.0:
            raise ValueError("interval must be >= 0")
        self.interval = float(interval)
        self._last_timestamp: float | None = None
        self._lock = RLock()

    @property
    def last_timestamp(self) -> float | None:
        with self._lock:
            return self._last_timestamp

    def due(self, timestamp: float) -> bool:
        with self._lock:
            return self._is_due(timestamp)

    def claim(self, timestamp: float) -> bool:
        """Atomically reserve an assimilation time when due."""
        if not isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        with self._lock:
            if not self._is_due(timestamp):
                return False
            self._last_timestamp = float(timestamp)
            return True

    def _is_due(self, timestamp: float) -> bool:
        if not isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self._last_timestamp is None:
            return True
        if self.interval == 0.0:
            return timestamp > self._last_timestamp
        return timestamp >= self._last_timestamp + self.interval


__all__ = [
    "AlignedObservation",
    "OnlineAssimilationScheduler",
    "SensorBuffer",
    "SensorQuality",
    "SensorSample",
    "TimeAlignmentPolicy",
    "align_observations",
]
