"""
Standalone test of the DB + cleaning logic from main.py, without requiring
fastapi/paho-mqtt to be installed. Run this to verify the core logic before
the team installs the full backend dependencies.
"""

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from typing import Optional

DB_PATH = os.path.join(tempfile.gettempdir(), "test_edgeguard.db")

VALID_RANGES = {
    "temperature": (-20.0, 150.0),
    "vibration": (0.0, 10.0),
    "oil_pressure": (0.0, 15.0),
    "hydraulic_pressure": (0.0, 700.0),
    "suspension_pressure": (0.0, 20.0),
    "battery_voltage": (0.0, 30.0),
}


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            truck_id TEXT NOT NULL,
            sensor_type TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT,
            device_ts INTEGER,
            received_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def insert_reading(truck_id, sensor_type, value, unit, device_ts) -> bool:
    try:
        conn = get_connection()
        conn.execute(
            """
            INSERT INTO sensor_readings (truck_id, sensor_type, value, unit, device_ts, received_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (truck_id, sensor_type, value, unit, device_ts, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"[db] write failed: {e}")
        return False


def clean_and_validate(payload: dict) -> Optional[dict]:
    required_keys = {"truck_id", "sensor", "value", "unit", "ts"}
    if not required_keys.issubset(payload.keys()):
        print(f"[clean] dropping message, missing keys: {payload}")
        return None

    sensor = payload["sensor"]
    try:
        value = float(payload["value"])
    except (TypeError, ValueError):
        print(f"[clean] dropping message, non-numeric value: {payload}")
        return None

    if sensor in VALID_RANGES:
        lo, hi = VALID_RANGES[sensor]
        if value < lo or value > hi:
            print(f"[clean] clipping out-of-range {sensor}={value} to [{lo}, {hi}]")
            value = max(lo, min(hi, value))

    return {
        "truck_id": payload["truck_id"],
        "sensor_type": sensor,
        "value": value,
        "unit": payload["unit"],
        "device_ts": int(payload["ts"]),
    }


def run_tests():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    init_db()

    print("=== Test 1: valid reading (real Wokwi payload shape) ===")
    raw = json.loads(
        '{"truck_id":"truck1","sensor":"temperature","value":94.30,"unit":"C","ts":125905}'
    )
    cleaned = clean_and_validate(raw)
    assert cleaned is not None, "FAIL: valid reading was dropped"
    assert cleaned["sensor_type"] == "temperature"
    assert cleaned["value"] == 94.30
    ok = insert_reading(**cleaned)
    assert ok, "FAIL: valid insert failed"
    print("PASS")

    print("=== Test 2: malformed JSON (missing keys) ===")
    raw2 = {"truck_id": "truck1", "sensor": "vibration"}  # missing value/unit/ts
    cleaned2 = clean_and_validate(raw2)
    assert cleaned2 is None, "FAIL: malformed message was not dropped"
    print("PASS")

    print("=== Test 3: non-numeric value ===")
    raw3 = {"truck_id": "truck1", "sensor": "vibration", "value": "garbage", "unit": "g", "ts": 1000}
    cleaned3 = clean_and_validate(raw3)
    assert cleaned3 is None, "FAIL: non-numeric value was not dropped"
    print("PASS")

    print("=== Test 4: out-of-range spike gets clipped, not dropped ===")
    raw4 = {"truck_id": "truck1", "sensor": "temperature", "value": 9999, "unit": "C", "ts": 2000}
    cleaned4 = clean_and_validate(raw4)
    assert cleaned4 is not None, "FAIL: spike was dropped instead of clipped"
    assert cleaned4["value"] == 150.0, f"FAIL: expected clip to 150.0, got {cleaned4['value']}"
    print("PASS")

    print("=== Test 5: query back what we inserted ===")
    conn = get_connection()
    rows = conn.execute("SELECT * FROM sensor_readings WHERE truck_id = ?", ("truck1",)).fetchall()
    conn.close()
    assert len(rows) == 1, f"FAIL: expected 1 row, got {len(rows)}"
    assert rows[0]["sensor_type"] == "temperature"
    assert rows[0]["value"] == 94.30
    print("PASS")

    print("\nAll tests passed.")


if __name__ == "__main__":
    run_tests()
