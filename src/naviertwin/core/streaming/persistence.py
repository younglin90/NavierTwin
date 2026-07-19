"""Durable SQLite storage for timestamped sensor observations."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import RLock

from naviertwin.core.data_assimilation.streaming import SensorQuality, SensorSample


class SQLiteSensorStore:
    """Thread-safe WAL-backed sensor store with deterministic duplicate handling."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sensor_samples (
                sensor_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                values_json TEXT NOT NULL,
                quality TEXT NOT NULL,
                sequence INTEGER,
                ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (sensor_id, timestamp)
            )
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_sensor_time "
            "ON sensor_samples(sensor_id, timestamp DESC)"
        )
        self._connection.commit()
        self._lock = RLock()

    def append(self, sample: SensorSample) -> bool:
        """Insert or replace a timestamp when sequence is not older."""
        payload = json.dumps(sample.values, separators=(",", ":"), allow_nan=False)
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO sensor_samples
                    (sensor_id, timestamp, values_json, quality, sequence)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(sensor_id, timestamp) DO UPDATE SET
                    values_json=excluded.values_json,
                    quality=excluded.quality,
                    sequence=excluded.sequence,
                    ingested_at=CURRENT_TIMESTAMP
                WHERE excluded.sequence IS NULL
                   OR sensor_samples.sequence IS NULL
                   OR excluded.sequence >= sensor_samples.sequence
                """,
                (
                    sample.sensor_id,
                    sample.timestamp,
                    payload,
                    sample.quality.value,
                    sample.sequence,
                ),
            )
        return cursor.rowcount > 0

    def load_recent(self, limit_per_sensor: int = 1024) -> tuple[SensorSample, ...]:
        """Load recent rows per sensor in global timestamp order."""
        if limit_per_sensor < 1:
            raise ValueError("limit_per_sensor must be positive")
        with self._lock:
            sensor_rows = self._connection.execute(
                "SELECT DISTINCT sensor_id FROM sensor_samples ORDER BY sensor_id"
            ).fetchall()
            rows = []
            for (sensor_id,) in sensor_rows:
                rows.extend(
                    self._connection.execute(
                        """
                        SELECT sensor_id, timestamp, values_json, quality, sequence
                        FROM sensor_samples
                        WHERE sensor_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (sensor_id, limit_per_sensor),
                    ).fetchall()
                )
        samples = [
            SensorSample(
                sensor_id=row[0],
                timestamp=row[1],
                values=tuple(json.loads(row[2])),
                quality=SensorQuality(row[3]),
                sequence=row[4],
            )
            for row in rows
        ]
        return tuple(sorted(samples, key=lambda sample: (sample.timestamp, sample.sensor_id)))

    def prune_before(self, timestamp: float) -> int:
        """Delete retained samples older than a timestamp."""
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "DELETE FROM sensor_samples WHERE timestamp < ?", (float(timestamp),)
            )
        return max(0, cursor.rowcount)

    def count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) FROM sensor_samples"
            ).fetchone()
        return int(row[0])

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SQLiteSensorStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = ["SQLiteSensorStore"]
