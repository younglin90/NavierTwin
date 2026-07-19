"""Optional MQTT, Kafka, and OPC UA sensor transport adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from threading import Event, Thread
from time import sleep, time
from typing import Callable, Protocol

from naviertwin.core.data_assimilation.streaming import SensorSample

SensorHandler = Callable[[SensorSample], None]


class SensorConnector(Protocol):
    """Lifecycle contract shared by transport adapters."""

    def start(self, handler: SensorHandler) -> SensorConnector: ...
    def stop(self) -> None: ...


def decode_sensor_payload(
    payload: bytes | str,
    *,
    default_sensor_id: str | None = None,
    default_timestamp: float | None = None,
) -> SensorSample:
    """Decode canonical JSON or ``timestamp,value...`` CSV payloads."""
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    text = text.strip()
    if not text:
        raise ValueError("sensor payload must not be empty")
    if text.startswith("{"):
        data = json.loads(text)
        sensor_id = data.get("sensor_id", default_sensor_id)
        timestamp = data.get("timestamp", default_timestamp)
        raw_values = data.get("values")
        values = raw_values if isinstance(raw_values, list) else [raw_values]
        return SensorSample(
            sensor_id=str(sensor_id or ""),
            timestamp=float(time() if timestamp is None else timestamp),
            values=tuple(values),
            quality=data.get("quality", "good"),
            sequence=data.get("sequence"),
        )
    if default_sensor_id is None:
        raise ValueError("CSV sensor payload requires default_sensor_id")
    tokens = [token.strip() for token in text.split(",")]
    if len(tokens) < 2:
        raise ValueError("CSV sensor payload requires timestamp and values")
    return SensorSample(
        sensor_id=default_sensor_id,
        timestamp=float(tokens[0]),
        values=tuple(float(token) for token in tokens[1:]),
    )


@dataclass(frozen=True, slots=True)
class MQTTConfig:
    host: str
    topic: str
    port: int = 8883
    qos: int = 1
    client_id: str = "naviertwin"
    username: str | None = None
    password: str | None = field(default=None, repr=False)
    tls: bool = True

    def __post_init__(self) -> None:
        if not self.host.strip() or not self.topic.strip():
            raise ValueError("MQTT host and topic must not be empty")
        if not 1 <= self.port <= 65535 or self.qos not in {0, 1, 2}:
            raise ValueError("invalid MQTT port or qos")


class MQTTConnector:
    """paho-mqtt adapter; dependency imported only when started."""

    def __init__(self, config: MQTTConfig) -> None:
        self.config = config
        self._client: object | None = None

    def start(self, handler: SensorHandler) -> MQTTConnector:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise ImportError("MQTT connector requires paho-mqtt") from exc
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, self.config.client_id)
        if self.config.username:
            client.username_pw_set(self.config.username, self.config.password)
        if self.config.tls:
            client.tls_set()

        def on_connect(active_client, _userdata, _flags, reason_code, _properties):
            if int(reason_code) != 0:
                raise ConnectionError(f"MQTT connection failed: {reason_code}")
            active_client.subscribe(self.config.topic, qos=self.config.qos)

        def on_message(_client, _userdata, message):
            handler(
                decode_sensor_payload(
                    message.payload, default_sensor_id=str(message.topic)
                )
            )

        client.on_connect = on_connect
        client.on_message = on_message
        client.connect_async(self.config.host, self.config.port)
        client.loop_start()
        self._client = client
        return self

    def stop(self) -> None:
        if self._client is not None:
            self._client.loop_stop()  # type: ignore[attr-defined]
            self._client.disconnect()  # type: ignore[attr-defined]
            self._client = None


@dataclass(frozen=True, slots=True)
class KafkaConfig:
    bootstrap_servers: str
    topic: str
    group_id: str = "naviertwin"
    security_protocol: str = "SASL_SSL"
    poll_seconds: float = 0.25

    def __post_init__(self) -> None:
        if not self.bootstrap_servers.strip() or not self.topic.strip():
            raise ValueError("Kafka bootstrap_servers and topic must not be empty")
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")


class KafkaConnector:
    """confluent-kafka polling adapter running in a bounded worker thread."""

    def __init__(self, config: KafkaConfig) -> None:
        self.config = config
        self._consumer: object | None = None
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self, handler: SensorHandler) -> KafkaConnector:
        try:
            from confluent_kafka import Consumer
        except ImportError as exc:
            raise ImportError("Kafka connector requires confluent-kafka") from exc
        consumer = Consumer(
            {
                "bootstrap.servers": self.config.bootstrap_servers,
                "group.id": self.config.group_id,
                "auto.offset.reset": "latest",
                "security.protocol": self.config.security_protocol,
                "enable.auto.commit": True,
            }
        )
        consumer.subscribe([self.config.topic])
        self._consumer = consumer
        self._stop.clear()

        def run() -> None:
            while not self._stop.is_set():
                message = consumer.poll(self.config.poll_seconds)
                if message is None:
                    continue
                if message.error():
                    continue
                handler(
                    decode_sensor_payload(
                        message.value(), default_sensor_id=self.config.topic
                    )
                )

        self._thread = Thread(target=run, name="naviertwin-kafka", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2 * self.config.poll_seconds))
        if self._consumer is not None:
            self._consumer.close()  # type: ignore[attr-defined]
        self._thread = None
        self._consumer = None


@dataclass(frozen=True, slots=True)
class OPCUAConfig:
    endpoint: str
    nodes: dict[str, str]
    poll_seconds: float = 0.25
    username: str | None = None
    password: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.endpoint.strip() or not self.nodes:
            raise ValueError("OPC UA endpoint and nodes must not be empty")
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")


class OPCUAConnector:
    """asyncua synchronous-client polling adapter."""

    def __init__(self, config: OPCUAConfig) -> None:
        self.config = config
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self, handler: SensorHandler) -> OPCUAConnector:
        try:
            from asyncua.sync import Client
        except ImportError as exc:
            raise ImportError("OPC UA connector requires asyncua") from exc
        self._stop.clear()

        def run() -> None:
            with Client(self.config.endpoint) as client:
                if self.config.username:
                    client.set_user(self.config.username)
                    client.set_password(self.config.password or "")
                nodes = {
                    sensor_id: client.get_node(node_id)
                    for sensor_id, node_id in self.config.nodes.items()
                }
                while not self._stop.is_set():
                    timestamp = time()
                    for sensor_id, node in nodes.items():
                        value = node.read_value()
                        values = value if isinstance(value, (list, tuple)) else (value,)
                        handler(SensorSample(sensor_id, timestamp, tuple(values)))
                    sleep(self.config.poll_seconds)

        self._thread = Thread(target=run, name="naviertwin-opcua", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2 * self.config.poll_seconds))
        self._thread = None


__all__ = [
    "KafkaConfig",
    "KafkaConnector",
    "MQTTConfig",
    "MQTTConnector",
    "OPCUAConfig",
    "OPCUAConnector",
    "SensorConnector",
    "decode_sensor_payload",
]
