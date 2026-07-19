"""Operational sensor buffering and alignment tests."""

from __future__ import annotations

import pytest

from naviertwin.core.data_assimilation.streaming import (
    OnlineAssimilationScheduler,
    SensorBuffer,
    SensorQuality,
    SensorSample,
    TimeAlignmentPolicy,
    align_observations,
)


def _sample(sensor_id: str, timestamp: float, value: float, **kwargs) -> SensorSample:
    return SensorSample(sensor_id, timestamp, (value,), **kwargs)


def test_buffer_sorts_late_samples_and_replaces_duplicate() -> None:
    buffer = SensorBuffer(max_samples_per_sensor=3)
    assert buffer.append(_sample("p1", 2.0, 20.0, sequence=2))
    assert buffer.append(_sample("p1", 1.0, 10.0, sequence=1))
    assert buffer.append(_sample("p1", 2.0, 21.0, sequence=3))
    assert not buffer.append(_sample("p1", 2.0, 19.0, sequence=2))
    assert [sample.timestamp for sample in buffer.samples("p1")] == [1.0, 2.0]
    assert buffer.latest("p1").values == (21.0,)


def test_buffer_reject_mode_and_bounded_history() -> None:
    buffer = SensorBuffer(max_samples_per_sensor=2, out_of_order="reject")
    buffer.append(_sample("p1", 1.0, 1.0))
    buffer.append(_sample("p1", 2.0, 2.0))
    buffer.append(_sample("p1", 3.0, 3.0))
    assert [sample.timestamp for sample in buffer.samples("p1")] == [2.0, 3.0]
    with pytest.raises(ValueError, match="out-of-order"):
        buffer.append(_sample("p1", 2.5, 2.5))


def test_bad_quality_is_excluded() -> None:
    buffer = SensorBuffer()
    buffer.append(_sample("p1", 1.0, 1.0, quality=SensorQuality.BAD))
    assert buffer.latest("p1") is None
    assert len(buffer.samples("p1", include_bad=True)) == 1


def test_linear_alignment_and_flatten() -> None:
    buffer = SensorBuffer()
    for sensor_id, scale in (("p1", 1.0), ("u1", 2.0)):
        buffer.append(_sample(sensor_id, 0.0, 0.0))
        buffer.append(_sample(sensor_id, 2.0, 2.0 * scale))
    aligned = align_observations(
        buffer,
        ["p1", "u1"],
        1.0,
        policy=TimeAlignmentPolicy(
            tolerance=0.01, max_interpolation_gap=2.0, max_age=2.0
        ),
    )
    assert aligned.complete
    assert aligned.flatten() == pytest.approx((1.0, 2.0))


def test_missing_stale_and_large_gap_are_reported() -> None:
    buffer = SensorBuffer()
    buffer.append(_sample("old", 0.0, 1.0))
    buffer.append(_sample("gap", 0.0, 1.0))
    buffer.append(_sample("gap", 10.0, 2.0))
    aligned = align_observations(
        buffer,
        ["old", "gap", "absent"],
        5.0,
        policy=TimeAlignmentPolicy(
            tolerance=0.1, max_interpolation_gap=2.0, max_age=1.0
        ),
    )
    assert aligned.missing_sensor_ids == ("absent",)
    assert aligned.stale_sensor_ids == ("old", "gap")
    with pytest.raises(ValueError, match="incomplete"):
        aligned.flatten()


def test_scheduler_claim_is_monotonic_and_interval_limited() -> None:
    scheduler = OnlineAssimilationScheduler(interval=1.0)
    assert scheduler.claim(10.0)
    assert not scheduler.claim(10.0)
    assert not scheduler.claim(10.9)
    assert scheduler.claim(11.0)
    assert not scheduler.claim(9.0)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: SensorSample("", 0.0, (1.0,)), "sensor_id"),
        (lambda: SensorSample("p", float("nan"), (1.0,)), "timestamp"),
        (lambda: SensorSample("p", 0.0, (float("inf"),)), "values"),
        (lambda: TimeAlignmentPolicy(tolerance=-1.0), "tolerance"),
    ],
)
def test_validation(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()
