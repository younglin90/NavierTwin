"""Durable operational streaming tests without external brokers."""

from __future__ import annotations

import json

import pytest

from naviertwin.core.data_assimilation.streaming import SensorSample
from naviertwin.core.streaming import (
    DurableSensorRuntime,
    KafkaConfig,
    MQTTConfig,
    OPCUAConfig,
    SQLiteSensorStore,
    decode_sensor_payload,
)


def test_json_and_csv_payloads_share_sensor_contract() -> None:
    sample = decode_sensor_payload(
        json.dumps(
            {
                "sensor_id": "p1",
                "timestamp": 10.0,
                "values": [101325.0, 300.0],
                "sequence": 4,
            }
        )
    )
    assert sample.sensor_id == "p1"
    assert sample.values == (101325.0, 300.0)
    assert sample.sequence == 4
    csv_sample = decode_sensor_payload("10.0,1.5,-2.0", default_sensor_id="u1")
    assert csv_sample == SensorSample("u1", 10.0, (1.5, -2.0))


def test_payload_validation() -> None:
    with pytest.raises(ValueError, match="empty"):
        decode_sensor_payload("")
    with pytest.raises(ValueError, match="default_sensor_id"):
        decode_sensor_payload("1.0,2.0")


def test_sqlite_store_rejects_older_duplicate_sequence(tmp_path) -> None:
    path = tmp_path / "sensors.sqlite3"
    with SQLiteSensorStore(path) as store:
        assert store.append(SensorSample("p", 1.0, (10.0,), sequence=2))
        assert not store.append(SensorSample("p", 1.0, (9.0,), sequence=1))
        assert store.append(SensorSample("p", 1.0, (11.0,), sequence=3))
        assert store.count() == 1
        assert store.load_recent()[0].values == (11.0,)


def test_runtime_recovers_bounded_history_and_metrics(tmp_path) -> None:
    path = tmp_path / "recovery.sqlite3"
    store = SQLiteSensorStore(path)
    runtime = DurableSensorRuntime(max_samples_per_sensor=2, store=store)
    for index in range(3):
        assert runtime.ingest(
            SensorSample("p", float(index), (float(index),), sequence=index),
            received_at=float(index) + 0.25,
        )
    assert len(runtime.buffer.samples("p")) == 2
    assert runtime.metrics.snapshot().accepted_samples == 3
    assert runtime.metrics.snapshot().latest_lag_seconds == pytest.approx(0.25)
    runtime.close()

    recovered_store = SQLiteSensorStore(path)
    recovered = DurableSensorRuntime(
        max_samples_per_sensor=2, store=recovered_store, recover=True
    )
    assert [sample.timestamp for sample in recovered.buffer.samples("p")] == [1.0, 2.0]
    assert recovered.metrics.snapshot().recovered_samples == 2
    aligned = recovered.align(["p"], 2.0)
    assert aligned.complete
    assert recovered.metrics.snapshot().complete_alignments == 1
    recovered.close()


def test_store_retention_prunes_old_rows() -> None:
    with SQLiteSensorStore(":memory:") as store:
        for timestamp in (1.0, 2.0, 3.0):
            store.append(SensorSample("p", timestamp, (timestamp,)))
        assert store.prune_before(3.0) == 2
        assert store.count() == 1


def test_connector_configs_validate_and_hide_passwords() -> None:
    mqtt = MQTTConfig("broker", "plant/#", password="secret")
    opcua = OPCUAConfig(
        "opc.tcp://localhost:4840", {"p": "ns=2;s=p"}, password="secret"
    )
    kafka = KafkaConfig("localhost:9093", "sensors")
    assert "secret" not in repr(mqtt)
    assert "secret" not in repr(opcua)
    assert kafka.security_protocol == "SASL_SSL"
    with pytest.raises(ValueError):
        MQTTConfig("", "topic")
