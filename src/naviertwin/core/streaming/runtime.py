"""Durable sensor runtime with recovery and operational metrics."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from time import time
from typing import Iterable, Literal

from naviertwin.core.data_assimilation.streaming import (
    AlignedObservation,
    SensorBuffer,
    SensorSample,
    TimeAlignmentPolicy,
    align_observations,
)
from naviertwin.core.streaming.persistence import SQLiteSensorStore


@dataclass(frozen=True, slots=True)
class StreamMetricsSnapshot:
    accepted_samples: int
    rejected_samples: int
    recovered_samples: int
    alignment_requests: int
    complete_alignments: int
    connector_errors: int
    latest_lag_seconds: float
    maximum_lag_seconds: float


class StreamMetrics:
    """Dependency-free metrics registry for API and Prometheus bridges."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._values: dict[str, int | float] = {
            "accepted_samples": 0,
            "rejected_samples": 0,
            "recovered_samples": 0,
            "alignment_requests": 0,
            "complete_alignments": 0,
            "connector_errors": 0,
            "latest_lag_seconds": 0.0,
            "maximum_lag_seconds": 0.0,
        }

    def record_sample(
        self, sample: SensorSample, *, accepted: bool, now: float | None = None
    ) -> None:
        with self._lock:
            key = "accepted_samples" if accepted else "rejected_samples"
            self._values[key] = int(self._values[key]) + 1
            if accepted and now is not None:
                lag = max(0.0, float(now) - sample.timestamp)
                self._values["latest_lag_seconds"] = lag
                self._values["maximum_lag_seconds"] = max(
                    float(self._values["maximum_lag_seconds"]), lag
                )

    def record_recovery(self, count: int) -> None:
        with self._lock:
            self._values["recovered_samples"] = int(
                self._values["recovered_samples"]
            ) + int(count)

    def record_alignment(self, *, complete: bool) -> None:
        with self._lock:
            self._values["alignment_requests"] = int(
                self._values["alignment_requests"]
            ) + 1
            if complete:
                self._values["complete_alignments"] = int(
                    self._values["complete_alignments"]
                ) + 1

    def record_connector_error(self) -> None:
        with self._lock:
            self._values["connector_errors"] = int(
                self._values["connector_errors"]
            ) + 1

    def snapshot(self) -> StreamMetricsSnapshot:
        with self._lock:
            values = dict(self._values)
        return StreamMetricsSnapshot(**values)  # type: ignore[arg-type]


class DurableSensorRuntime:
    """Combine buffer, optional SQLite durability, recovery, and metrics."""

    def __init__(
        self,
        *,
        max_samples_per_sensor: int = 1024,
        store: SQLiteSensorStore | None = None,
        recover: bool = True,
    ) -> None:
        self.buffer = SensorBuffer(max_samples_per_sensor=max_samples_per_sensor)
        self.store = store
        self.metrics = StreamMetrics()
        if store is not None and recover:
            recovered = store.load_recent(max_samples_per_sensor)
            for sample in recovered:
                self.buffer.append(sample)
            self.metrics.record_recovery(len(recovered))

    def ingest(self, sample: SensorSample, *, received_at: float | None = None) -> bool:
        """Persist then expose a sample; reject stale duplicate sequences."""
        accepted = self.store.append(sample) if self.store is not None else True
        if accepted:
            accepted = self.buffer.append(sample)
        self.metrics.record_sample(
            sample,
            accepted=accepted,
            now=time() if received_at is None else received_at,
        )
        return accepted

    def align(
        self,
        sensor_ids: Iterable[str],
        timestamp: float,
        *,
        policy: TimeAlignmentPolicy | None = None,
        method: Literal["nearest", "linear"] = "linear",
    ) -> AlignedObservation:
        aligned = align_observations(
            self.buffer, sensor_ids, timestamp, policy=policy, method=method
        )
        self.metrics.record_alignment(complete=aligned.complete)
        return aligned

    def close(self) -> None:
        if self.store is not None:
            self.store.close()


__all__ = [
    "DurableSensorRuntime",
    "StreamMetrics",
    "StreamMetricsSnapshot",
]
