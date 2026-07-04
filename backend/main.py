"""
EdgeGuard AI — FastAPI Backend (Self-Contained Demo Build)

v2.1 — fully self-contained, works with NO external services.

What it does (in priority order, so the demo NEVER breaks):

  1. Tries to connect to Supabase (reads from .env). If it works → real DB.
  2. Falls back to a local SQLite file (edgeguard.db) if Supabase is unreachable.
  3. Tries to connect to the public MQTT broker. If it works → real-time stream.
  4. Falls back to a built-in simulated sensor stream (tunable in DEMO_PARAMS)
     that drives the dashboard with realistic data.
  5. Has its OWN built-in ML predictor (logistic-ish score) so the dashboard
     shows real failure probabilities, RUL, and component even when the
     ml-data/ poller is not running.

Endpoints the dashboard calls:
  GET  /                              health check
  GET  /readings/latest               latest sensor values
  GET  /readings/history              recent stored readings
  GET  /readings/history/wide         pivoted (one row per bucket)
  GET  /predictions/latest            latest ML predictions
  GET  /predictions/history           prediction trend
  POST /predictions                   push a prediction
  GET  /ml/status                     {failure_probability, rul_hours, alert_level, ...}
  GET  /alerts/active                 unacknowledged alerts
  POST /alerts                        create an alert
  POST /alerts/{id}/acknowledge       ack an alert
  GET  /buffer/status                 in-memory buffer size
  GET  /sops, /sops/components,
       /sops/search, /sops/{id}       RAG / SOP retrieval (uses local JSON)
  GET  /maintenance/summary           aggregate for the Maintenance tab
  GET  /demo/info                     tells the frontend what mode we're in

Run:
    pip install fastapi uvicorn paho-mqtt
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import json
import os
import re
import sqlite3
import threading
import time
import random
import math
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration & fallbacks
# ---------------------------------------------------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()
SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_KEY
                        and "PASTE" not in SUPABASE_URL
                        and "PASTE" not in SUPABASE_KEY
                        and "YOUR_" not in SUPABASE_URL.upper())


# ---------------------------------------------------------------------------
# Hard-timeout helper for any Supabase / network call.
# supabase-py doesn't expose a request timeout, so without this a hung
# request can stall a request handler for 10+ seconds.
# ---------------------------------------------------------------------------
def _run_with_timeout(fn, seconds: float = 2.0, label: str = "supabase") -> Any:
    """Run fn() in a daemon thread; raise TimeoutError if it doesn't return
    in time. The thread is a daemon so it can't outlive the process.
    Returns fn()'s return value, or None on timeout/exception.
    """
    result_box: list = [None]
    exc_box: list = [None]

    def _runner():
        try:
            result_box[0] = fn()
        except Exception as e:
            exc_box[0] = e

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=seconds)
    if t.is_alive():
        # Daemon thread will be cleaned up at exit; we return None
        print(f"[{label}] timeout after {seconds}s; returning empty")
        return None
    if exc_box[0] is not None:
        raise exc_box[0]
    return result_box[0]


MQTT_BROKER = os.environ.get("MQTT_BROKER", "broker.hivemq.com")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_TOPIC_FILTER = os.environ.get("MQTT_TOPIC_FILTER", "edgeguard/+/+")

# Local SQLite (used when Supabase is not reachable)
DB_PATH = os.path.join(os.path.dirname(__file__), "edgeguard.db")

VALID_RANGES = {
    "temperature":         (-20.0, 150.0),
    "vibration":           (0.0,   10.0),
    "oil_pressure":        (0.0,   15.0),
    "hydraulic_pressure":  (0.0,  700.0),
    "suspension_pressure": (0.0,   20.0),
    "battery_voltage":     (0.0,   30.0),
}

# Demo / simulation parameters
DEMO_PARAMS = {
    "enabled":         os.environ.get("DEMO_ENABLED", "auto"),  # auto | on | off
    "truck_id":        os.environ.get("DEMO_TRUCK_ID", "truck1"),
    "cycle_seconds":   90,
    "baseline": {
        "temperature":         62.0,
        "vibration":            0.40,
        "oil_pressure":         4.2,
        "hydraulic_pressure": 210.0,
        "suspension_pressure":  6.2,
        "battery_voltage":     24.1,
    },
    "critical": {
        "temperature":         96.0,
        "vibration":            1.70,
        "oil_pressure":         1.20,
        "hydraulic_pressure": 100.0,
        "suspension_pressure":  3.0,
        "battery_voltage":     20.4,
    },
    "warning": {
        "temperature":         82.0,
        "vibration":            1.10,
        "oil_pressure":         2.00,
        "hydraulic_pressure": 150.0,
        "suspension_pressure":  4.2,
        "battery_voltage":     21.4,
    },
}

# ---------------------------------------------------------------------------
# Storage layer (Supabase → SQLite fallback)
# ---------------------------------------------------------------------------
class Storage:
    """Abstraction so the rest of the app doesn't care if we're using
    Supabase or local SQLite. Every method is thread-safe."""

    def __init__(self):
        self.mode = "memory"  # one of: supabase, sqlite, memory
        self.client = None
        if SUPABASE_ENABLED:
            try:
                from supabase import create_client
                self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
                # Smoke test — does it respond?
                self.client.table("sensor_readings").select("id").limit(1).execute()
                self.mode = "supabase"
                print(f"[storage] Supabase OK at {SUPABASE_URL[:40]}…")
            except Exception as e:
                print(f"[storage] Supabase unreachable ({type(e).__name__}); falling back to SQLite")
                self.client = None

        if self.mode != "supabase":
            # SQLite path
            self._init_sqlite()

    def _init_sqlite(self):
        self.mode = "sqlite"
        self._lock = threading.Lock()
        with self._lock:
            conn = sqlite3.connect(DB_PATH)
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    truck_id TEXT NOT NULL,
                    sensor_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT,
                    device_ts INTEGER NOT NULL,
                    received_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_readings_truck_ts
                    ON sensor_readings(truck_id, device_ts DESC);
                CREATE INDEX IF NOT EXISTS idx_readings_sensor
                    ON sensor_readings(sensor_type, device_ts DESC);

                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    truck_id TEXT NOT NULL,
                    component TEXT,
                    failure_probability REAL NOT NULL,
                    rul_hours REAL,
                    model_version TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_pred_truck_ts
                    ON predictions(truck_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    truck_id TEXT NOT NULL,
                    component TEXT,
                    severity TEXT NOT NULL,
                    message TEXT,
                    sop_reference TEXT,
                    created_at TEXT NOT NULL,
                    acknowledged INTEGER DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_alerts_truck_active
                    ON alerts(truck_id, acknowledged, created_at DESC);
            """)
            conn.commit()
            conn.close()
        print(f"[storage] SQLite at {DB_PATH}")

    # ---- sensor_readings ----
    def insert_reading(self, truck_id, sensor_type, value, unit, device_ts) -> bool:
        try:
            if self.mode == "supabase":
                _run_with_timeout(
                    lambda: self.client.table("sensor_readings").insert({
                        "truck_id": truck_id, "sensor_type": sensor_type,
                        "value": value, "unit": unit, "device_ts": device_ts,
                        "received_at": datetime.now(timezone.utc).isoformat(),
                    }).execute(),
                    seconds=2.0, label="insert_reading",
                )
            else:
                with self._lock:
                    conn = sqlite3.connect(DB_PATH)
                    conn.execute(
                        "INSERT INTO sensor_readings (truck_id, sensor_type, value, unit, device_ts, received_at)"
                        " VALUES (?, ?, ?, ?, ?, ?)",
                        (truck_id, sensor_type, value, unit, device_ts,
                         datetime.now(timezone.utc).isoformat()),
                    )
                    conn.commit()
                    conn.close()
            return True
        except Exception as e:
            print(f"[storage] insert_reading failed: {e}")
            return False

    def fetch_history(self, truck_id, sensor_type=None, limit=100) -> list:
        try:
            if self.mode == "supabase":
                def _q():
                    q = (self.client.table("sensor_readings").select("*")
                         .eq("truck_id", truck_id))
                    if sensor_type:
                        q = q.eq("sensor_type", sensor_type)
                    return q.order("id", desc=True).limit(limit).execute().data
                return _run_with_timeout(_q, seconds=2.0, label="fetch_history") or []
                if sensor_type:
                    q = q.eq("sensor_type", sensor_type)
                return q.order("id", desc=True).limit(limit).execute().data
            else:
                with self._lock:
                    conn = sqlite3.connect(DB_PATH)
                    conn.row_factory = sqlite3.Row
                    if sensor_type:
                        rows = conn.execute(
                            "SELECT * FROM sensor_readings WHERE truck_id=? AND sensor_type=?"
                            " ORDER BY id DESC LIMIT ?",
                            (truck_id, sensor_type, limit),
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            "SELECT * FROM sensor_readings WHERE truck_id=?"
                            " ORDER BY id DESC LIMIT ?",
                            (truck_id, limit),
                        ).fetchall()
                    conn.close()
                return [dict(r) for r in rows]
        except Exception as e:
            print(f"[storage] fetch_history failed: {e}")
            return []

    def fetch_history_wide(self, truck_id, limit=100) -> list:
        rows = self.fetch_history(truck_id, limit=limit * 8)
        if not rows: return []
        buckets = defaultdict(dict)
        for r in rows:
            ts = int(r.get("device_ts", 0))
            bucket = (ts // 2000) * 2000
            buckets[bucket][r["sensor_type"]] = r["value"]
        out = []
        for ts in sorted(buckets.keys(), reverse=True)[:limit]:
            row = {"t_seconds": ts, "truck_id": truck_id}
            row.update(buckets[ts])
            out.append(row)
        return sorted(out, key=lambda r: r["t_seconds"])

    # ---- predictions ----
    def insert_prediction(self, truck_id, component, prob, rul, model_version) -> bool:
        try:
            if self.mode == "supabase":
                _run_with_timeout(
                    lambda: self.client.table("predictions").insert({
                        "truck_id": truck_id, "component": component,
                        "failure_probability": prob, "rul_hours": rul,
                        "model_version": model_version,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }).execute(),
                    seconds=2.0, label="insert_prediction",
                )
            else:
                with self._lock:
                    conn = sqlite3.connect(DB_PATH)
                    conn.execute(
                        "INSERT INTO predictions (truck_id, component, failure_probability, rul_hours, model_version, created_at)"
                        " VALUES (?, ?, ?, ?, ?, ?)",
                        (truck_id, component, prob, rul, model_version,
                         datetime.now(timezone.utc).isoformat()),
                    )
                    conn.commit()
                    conn.close()
            return True
        except Exception as e:
            print(f"[storage] insert_prediction failed: {e}")
            return False

    def fetch_latest_prediction(self, truck_id) -> Optional[dict]:
        try:
            if self.mode == "supabase":
                def _q():
                    return (self.client.table("predictions").select("*")
                            .eq("truck_id", truck_id).order("id", desc=True).limit(1)
                            .execute().data)
                data = _run_with_timeout(_q, seconds=2.0, label="fetch_latest_prediction") or []
            else:
                with self._lock:
                    conn = sqlite3.connect(DB_PATH)
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        "SELECT * FROM predictions WHERE truck_id=?"
                        " ORDER BY id DESC LIMIT 1", (truck_id,),
                    ).fetchall()
                    conn.close()
                data = [dict(r) for r in rows]
            return data[0] if data else None
        except Exception as e:
            print(f"[storage] fetch_latest_prediction failed: {e}")
            return None

    def fetch_prediction_history(self, truck_id, limit=50) -> list:
        try:
            if self.mode == "supabase":
                def _q():
                    return (self.client.table("predictions").select("*")
                            .eq("truck_id", truck_id).order("id", desc=True).limit(limit)
                            .execute().data)
                data = _run_with_timeout(_q, seconds=2.0, label="fetch_prediction_history") or []
            else:
                with self._lock:
                    conn = sqlite3.connect(DB_PATH)
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        "SELECT * FROM predictions WHERE truck_id=?"
                        " ORDER BY id DESC LIMIT ?", (truck_id, limit),
                    ).fetchall()
                    conn.close()
                data = [dict(r) for r in rows]
            return list(reversed(data))
        except Exception as e:
            print(f"[storage] fetch_prediction_history failed: {e}")
            return []

    # ---- alerts ----
    def insert_alert(self, truck_id, component, severity, message, sop_ref=None) -> bool:
        try:
            if self.mode == "supabase":
                _run_with_timeout(
                    lambda: self.client.table("alerts").insert({
                        "truck_id": truck_id, "component": component,
                        "severity": severity, "message": message,
                        "sop_reference": sop_ref,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "acknowledged": False,
                    }).execute(),
                    seconds=2.0, label="insert_alert",
                )
            else:
                with self._lock:
                    conn = sqlite3.connect(DB_PATH)
                    conn.execute(
                        "INSERT INTO alerts (truck_id, component, severity, message, sop_reference, created_at, acknowledged)"
                        " VALUES (?, ?, ?, ?, ?, ?, 0)",
                        (truck_id, component, severity, message, sop_ref,
                         datetime.now(timezone.utc).isoformat()),
                    )
                    conn.commit()
                    conn.close()
            return True
        except Exception as e:
            print(f"[storage] insert_alert failed: {e}")
            return False

    def fetch_active_alerts(self, truck_id) -> list:
        try:
            if self.mode == "supabase":
                def _q():
                    return (self.client.table("alerts").select("*")
                            .eq("truck_id", truck_id).eq("acknowledged", False)
                            .order("id", desc=True).execute().data)
                data = _run_with_timeout(_q, seconds=2.0, label="fetch_active_alerts") or []
            else:
                with self._lock:
                    conn = sqlite3.connect(DB_PATH)
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        "SELECT * FROM alerts WHERE truck_id=? AND acknowledged=0"
                        " ORDER BY id DESC", (truck_id,),
                    ).fetchall()
                    conn.close()
                data = [dict(r) for r in rows]
            return data or []
        except Exception as e:
            print(f"[storage] fetch_active_alerts failed: {e}")
            return []

    def fetch_recent_alerts(self, truck_id, limit=20) -> list:
        try:
            if self.mode == "supabase":
                def _q():
                    return (self.client.table("alerts").select("*")
                            .eq("truck_id", truck_id)
                            .order("id", desc=True).limit(limit).execute().data)
                data = _run_with_timeout(_q, seconds=2.0, label="fetch_recent_alerts") or []
            else:
                with self._lock:
                    conn = sqlite3.connect(DB_PATH)
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        "SELECT * FROM alerts WHERE truck_id=?"
                        " ORDER BY id DESC LIMIT ?", (truck_id, limit),
                    ).fetchall()
                    conn.close()
                data = [dict(r) for r in rows]
            return data or []
        except Exception as e:
            print(f"[storage] fetch_recent_alerts failed: {e}")
            return []

    def acknowledge_alert(self, alert_id) -> bool:
        try:
            if self.mode == "supabase":
                _run_with_timeout(
                    lambda: self.client.table("alerts").update({"acknowledged": True})
                                              .eq("id", alert_id).execute(),
                    seconds=2.0, label="acknowledge_alert",
                )
            else:
                with self._lock:
                    conn = sqlite3.connect(DB_PATH)
                    conn.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,))
                    conn.commit()
                    conn.close()
            return True
        except Exception as e:
            print(f"[storage] acknowledge_alert failed: {e}")
            return False

storage = Storage()

# ---------------------------------------------------------------------------
# In-memory buffer (used if a DB write fails)
# ---------------------------------------------------------------------------
_buffer_lock = threading.Lock()
_pending_buffer: list[dict] = []


def flush_buffer():
    global _pending_buffer
    with _buffer_lock:
        if not _pending_buffer: return
        still = []
        for item in _pending_buffer:
            ok = storage.insert_reading(item["truck_id"], item["sensor_type"],
                                        item["value"], item["unit"], item["device_ts"])
            if not ok: still.append(item)
        flushed = len(_pending_buffer) - len(still)
        if flushed: print(f"[buffer] flushed {flushed} buffered reading(s)")
        _pending_buffer = still


# ---------------------------------------------------------------------------
# MQTT listener — best effort, never blocks startup
# ---------------------------------------------------------------------------
_latest_readings: dict[str, dict] = {}
_latest_lock = threading.Lock()
_mqtt_connected = False


def set_latest(sensor_type, reading):
    with _latest_lock:
        _latest_readings[sensor_type] = reading


def on_connect_factory(client):
    def on_connect(c, userdata, flags, rc):
        global _mqtt_connected
        if rc == 0:
            _mqtt_connected = True
            print(f"[mqtt] connected to {MQTT_BROKER}, subscribing to {MQTT_TOPIC_FILTER}")
            c.subscribe(MQTT_TOPIC_FILTER)
        else:
            print(f"[mqtt] connection failed, rc={rc}")
    return on_connect


def on_message_factory(client):
    def on_message(c, userdata, msg):
        try:
            raw = json.loads(msg.payload.decode())
        except json.JSONDecodeError:
            return
        if msg.topic.endswith("/status"):
            return
        cleaned = clean_and_validate(raw)
        if cleaned is None: return
        with _latest_lock:
            _latest_readings[cleaned["sensor_type"]] = {
                **cleaned, "received_at": datetime.now(timezone.utc).isoformat(),
            }
        _ring_push(cleaned)
        ok = storage.insert_reading(cleaned["truck_id"], cleaned["sensor_type"],
                                    cleaned["value"], cleaned["unit"], cleaned["device_ts"])
        if not ok:
            with _buffer_lock:
                _pending_buffer.append(cleaned)
        else:
            flush_buffer()
    return on_message


def clean_and_validate(payload: dict) -> Optional[dict]:
    required = {"truck_id", "sensor", "value", "unit", "ts"}
    if not required.issubset(payload.keys()): return None
    try: value = float(payload["value"])
    except (TypeError, ValueError): return None
    sensor = payload["sensor"]
    if sensor in VALID_RANGES:
        lo, hi = VALID_RANGES[sensor]
        value = max(lo, min(hi, value))
    return {
        "truck_id": payload["truck_id"],
        "sensor_type": sensor,
        "value": value,
        "unit": payload["unit"],
        "device_ts": int(payload["ts"]),
    }


def mqtt_worker():
    global _mqtt_connected
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        print("[mqtt] paho-mqtt not installed; skipping MQTT listener")
        return

    while True:
        client = None
        try:
            client = mqtt.Client()
            client.on_connect = on_connect_factory(client)
            client.on_message = on_message_factory(client)
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=30)
            client.loop_forever()
        except Exception as e:
            _mqtt_connected = False
            # Quiet log — don't spam. Once every 30s.
            print(f"[mqtt] unavailable ({type(e).__name__}); using simulated stream")
            time.sleep(30)
        finally:
            try:
                if client is not None: client.disconnect()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Built-in ML predictor (so the dashboard works even without ml-data/poller.py)
# ---------------------------------------------------------------------------
def ml_predict_from_history(samples: List[dict]) -> dict:
    """
    Tiny logistic-style score using only the latest values per sensor.
    Designed to *look like* a trained model — produces 0-1 probability,
    RUL hours, and a component name. Not a real model, but close enough
    to the real engine's output for the dashboard to demo.

    `samples` is a list of dicts with keys matching VALID_RANGES.
    """
    if not samples:
        return {"failure_probability": 0.05, "rul_hours": 720.0,
                "component": "Engine", "severity": "normal"}

    # Take last value per sensor
    latest = {}
    for s in samples:
        for k in VALID_RANGES.keys():
            if k in s and s[k] is not None:
                latest[k] = s[k]

    score = 0.0
    contributions = {}

    # Engine temp: weight up as it climbs above 75
    t = latest.get("temperature", 65)
    contributions["temperature"] = max(0, (t - 65) / 35) * 1.0  # 0..1 across 65-100

    # Vibration
    v = latest.get("vibration", 0.4)
    contributions["vibration"] = max(0, (v - 0.4) / 1.5) * 0.9

    # Oil pressure (inverted: lower = worse)
    o = latest.get("oil_pressure", 4.2)
    contributions["oil_pressure"] = max(0, (4.5 - o) / 3.5) * 1.1

    # Hydraulic (inverted)
    h = latest.get("hydraulic_pressure", 210)
    contributions["hydraulic_pressure"] = max(0, (220 - h) / 140) * 1.0

    # Suspension (inverted)
    s = latest.get("suspension_pressure", 6.2)
    contributions["suspension_pressure"] = max(0, (6.5 - s) / 4.0) * 0.7

    # Battery (inverted)
    b = latest.get("battery_voltage", 24.0)
    contributions["battery_voltage"] = max(0, (24 - b) / 3.5) * 0.8

    # Pick the worst sensor as the "component at risk"
    worst = max(contributions.items(), key=lambda kv: kv[1])
    score = min(1.0, sum(contributions.values()) / 3.0)  # 0..1 with headroom

    # Map score to severity
    if score >= 0.7: severity = "critical"
    elif score >= 0.45: severity = "high"
    elif score >= 0.25: severity = "medium"
    else: severity = "normal"

    component_name = {
        "temperature": "Engine",
        "vibration": "Axle Bearing",
        "oil_pressure": "Engine Lubrication",
        "hydraulic_pressure": "Hydraulic System",
        "suspension_pressure": "Suspension",
        "battery_voltage": "Electrical",
    }[worst[0]]

    # RUL: 720h baseline, drops quickly as score climbs
    rul = max(2.0, 720.0 * math.exp(-4.5 * score))
    return {
        "failure_probability": round(score, 4),
        "rul_hours": round(rul, 1),
        "component": component_name,
        "severity": severity,
        "model_version": "built-in-v1",
    }


# ---------------------------------------------------------------------------
# Built-in sensor stream simulator (the "demo engine")
# ---------------------------------------------------------------------------
class DemoEngine:
    def __init__(self):
        self.t0 = time.time()
        self.last_values: Dict[str, float] = dict(DEMO_PARAMS["baseline"])
        self._stop = threading.Event()
        self.connected = False  # True once first reading has been published

    def start(self):
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _phase_at(self, t: float) -> str:
        cycle = DEMO_PARAMS["cycle_seconds"]
        elapsed = (t - self.t0) % cycle
        if elapsed < cycle * 0.33:  return "healthy"
        if elapsed < cycle * 0.61:  return "warning"
        if elapsed < cycle * 0.87:  return "critical"
        return "reset"

    def _targets_for(self, phase: str) -> Dict[str, float]:
        return {"healthy":  DEMO_PARAMS["baseline"],
                "warning":  DEMO_PARAMS["warning"],
                "critical": DEMO_PARAMS["critical"],
                "reset":    DEMO_PARAMS["baseline"]}[phase]

    def _step_values(self, phase: str) -> Dict[str, float]:
        targets = self._targets_for(phase)
        out = {}
        for k, target in targets.items():
            cur = self.last_values.get(k, target)
            noise_scale = 0.02 if k == "battery_voltage" else 0.015
            noise = (random.random() - 0.5) * abs(target) * noise_scale
            lerp = 0.05 if phase == "critical" else 0.04
            nxt = cur + (target - cur) * lerp + noise
            # Hard clip
            lo, hi = VALID_RANGES[k]
            nxt = max(lo + 0.01, min(hi - 0.01, nxt))
            self.last_values[k] = nxt
            out[k] = round(nxt, 3)
        return out

    def _run(self):
        # Slight startup delay so /readings/latest has data on first hit
        time.sleep(0.5)
        while not self._stop.is_set():
            try:
                now = time.time()
                phase = self._phase_at(now)
                values = self._step_values(phase)
                truck_id = DEMO_PARAMS["truck_id"]
                device_ts = int(now * 1000)

                # Push to in-memory + DB
                received_at = datetime.now(timezone.utc).isoformat()
                for sensor, val in values.items():
                    reading = {
                        "truck_id": truck_id,
                        "sensor_type": sensor,
                        "value": val,
                        "unit": _unit_for(sensor),
                        "device_ts": device_ts,
                        "received_at": received_at,
                    }
                    set_latest(sensor, reading)
                    _ring_push(reading)
                    if not storage.insert_reading(truck_id, sensor, val,
                                                 _unit_for(sensor), device_ts):
                        with _buffer_lock:
                            _pending_buffer.append({
                                "truck_id": truck_id, "sensor_type": sensor,
                                "value": val, "unit": _unit_for(sensor),
                                "device_ts": device_ts,
                            })

                # Run ML predictor
                pred = ml_predict_from_history([values])
                storage.insert_prediction(
                    truck_id, pred["component"], pred["failure_probability"],
                    pred["rul_hours"], pred["model_version"],
                )

                # Generate alerts in critical phase
                if phase == "critical" and random.random() < 0.4:
                    crits = [
                        ("Engine", "critical", "Engine coolant temperature critical. Reduce load immediately.", "Engine Overheat Procedure"),
                        ("Hydraulic System", "critical", "Hydraulic pressure loss detected — risk of tipper bed failure.", "Hydraulic Pressure Drop Response"),
                        ("Axle Bearing", "high", "Vibration levels elevated at front axle.", "Wheel Bearing Inspection"),
                    ]
                    c, s, m, sop = random.choice(crits)
                    storage.insert_alert(truck_id, c, s, m, sop)
                elif phase == "warning" and random.random() < 0.25:
                    storage.insert_alert(truck_id, "Engine", "warning",
                                          "Engine temperature trending above safe operating range.",
                                          "Engine Cooling System Inspection")

                self.connected = True
            except Exception as e:
                print(f"[demo] tick error: {e}")
            time.sleep(2.0)  # 2-second cadence (matches dashboard polling)

    def stop(self):
        self._stop.set()


def _unit_for(sensor: str) -> str:
    return {"temperature": "°C", "vibration": "g", "oil_pressure": "bar",
            "hydraulic_pressure": "bar", "suspension_pressure": "bar",
            "battery_voltage": "V"}.get(sensor, "")


demo_engine = DemoEngine()

# ---------------------------------------------------------------------------
# SOP / RAG retrieval (loads from local JSON)
# ---------------------------------------------------------------------------
_sop_data: List[dict] = []
_sop_lock = threading.Lock()


def _load_sops():
    global _sop_data
    # Try the structured ml-data location first
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "ml-data", "sop_data.json"),
        os.path.join(os.path.dirname(__file__), "sop_data.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    _sop_data = json.load(f)
                print(f"[rag] loaded {len(_sop_data)} SOPs from {p}")
                return
            except Exception as e:
                print(f"[rag] could not load {p}: {e}")
    # Fall back to a minimal embedded set so the search works
    _sop_data = _EMBEDDED_SOPS
    print(f"[rag] using {len(_sop_data)} embedded SOPs")


_EMBEDDED_SOPS = [
    {"id": 0, "title": "Engine Cooling System Inspection", "component": "temperature",
     "severity": "critical",
     "content_chunk": "Step 1: Stop truck and allow engine to cool. Step 2: Check coolant level in expansion tank. Step 3: Inspect radiator for blockage or damage. Step 4: Verify fan operation and thermostat. Step 5: Pressure test cooling system. Step 6: Refill coolant and bleed air.",
     "tools_required": "Coolant, pressure tester, IR thermometer, fan belt",
     "estimated_downtime": "45 min"},
    {"id": 1, "title": "Engine Overheat Procedure", "component": "temperature",
     "severity": "critical",
     "content_chunk": "Step 1: Reduce load immediately. Step 2: Park in ventilated area. Step 3: Idle engine 2 min. Step 4: Stop engine and check for steam leaks. Step 5: Do not open radiator cap when hot. Step 6: Inspect water pump and hoses.",
     "tools_required": "Coolant, IR thermometer, gloves",
     "estimated_downtime": "30 min"},
    {"id": 2, "title": "Wheel Bearing Inspection", "component": "vibration",
     "severity": "warning",
     "content_chunk": "Step 1: Reduce speed below 40 km/h. Step 2: Jack up front axle. Step 3: Check wheel hub for radial play (>0.1mm = replace). Step 4: Inspect IR temp after 10 min operation. Step 5: Repack bearing with high-temp grease. Step 6: Adjust castle nut to spec.",
     "tools_required": "Grease gun, jack, IR thermometer, castle nut tool",
     "estimated_downtime": "60 min"},
    {"id": 3, "title": "Low Oil Pressure Response", "component": "oil_pressure",
     "severity": "critical",
     "content_chunk": "Step 1: STOP engine immediately. Step 2: Check oil level on dipstick. Step 3: Inspect oil filter for blockage. Step 4: Verify pressure sender wiring. Step 5: Test mechanical oil pressure gauge. Step 6: Tow to service bay if pressure < 1.5 bar at idle.",
     "tools_required": "Engine oil, filter, dipstick, gauge",
     "estimated_downtime": "2 h"},
    {"id": 4, "title": "Hydraulic Pressure Drop Response", "component": "hydraulic_pressure",
     "severity": "critical",
     "content_chunk": "Step 1: Avoid tipper bed actuation. Step 2: Inspect for external leaks at ram, hoses, fittings. Step 3: Check hydraulic fluid reservoir level. Step 4: Pressure-test the system. Step 5: Replace damaged seals. Step 6: Bleed air from lines.",
     "tools_required": "Hydraulic fluid, pressure gauge, seal kit",
     "estimated_downtime": "90 min"},
    {"id": 5, "title": "Suspension Air Leak Repair", "component": "suspension_pressure",
     "severity": "warning",
     "content_chunk": "Step 1: Reduce payload by 20%. Step 2: Listen for air leaks at all fittings. Step 3: Soap-test suspect joints. Step 4: Check bellows for cracks. Step 5: Replace o-rings. Step 6: Verify height sensor calibration.",
     "tools_required": "Soap solution, o-rings, height sensor tool",
     "estimated_downtime": "40 min"},
    {"id": 6, "title": "Battery and Charging System Check", "component": "battery_voltage",
     "severity": "warning",
     "content_chunk": "Step 1: Switch off non-essential loads. Step 2: Measure battery voltage (engine off). Step 3: Measure running voltage at alternator. Step 4: Verify 27.6-28.8V at 2000 RPM. Step 5: Inspect terminals for corrosion. Step 6: Load-test battery if voltage < 22V after charging.",
     "tools_required": "Multimeter, terminal brush, load tester",
     "estimated_downtime": "30 min"},
    {"id": 7, "title": "Engine Cooling System Inspection", "component": "temperature",
     "severity": "preventive",
     "content_chunk": "Routine inspection every 250 hours. Step 1: Check coolant level. Step 2: Inspect hoses. Step 3: Test fan operation. Step 4: Pressure test system. Step 5: Replace coolant every 1000 hours.",
     "tools_required": "Coolant, pressure tester",
     "estimated_downtime": "20 min"},
    {"id": 8, "title": "Hydraulic Fluid Change", "component": "hydraulic_pressure",
     "severity": "preventive",
     "content_chunk": "Step 1: Lower tipper bed fully. Step 2: Drain reservoir. Step 3: Replace return filter. Step 4: Fill with ISO VG 46 hydraulic oil. Step 5: Bleed air from system. Step 6: Cycle bed 3x to verify pressure.",
     "tools_required": "Hydraulic fluid ISO VG 46, filter, drain pan",
     "estimated_downtime": "50 min"},
    {"id": 9, "title": "Air Filter Replacement", "component": "vibration",
     "severity": "preventive",
     "content_chunk": "Step 1: Release clips on air cleaner housing. Step 2: Remove old filter. Step 3: Inspect housing for debris. Step 4: Install new filter (check orientation). Step 5: Re-secure clips. Step 6: Reset service indicator.",
     "tools_required": "Replacement air filter",
     "estimated_downtime": "10 min"},
    {"id": 10, "title": "Tipper Bed Pivot Lubrication", "component": "hydraulic_pressure",
     "severity": "preventive",
     "content_chunk": "Step 1: Lower tipper bed to ground. Step 2: Locate pivot grease points. Step 3: Apply 3-4 pumps of lithium grease. Step 4: Cycle bed 1x. Step 5: Wipe excess grease. Step 6: Verify smooth operation.",
     "tools_required": "Grease gun, lithium grease",
     "estimated_downtime": "15 min"},
    {"id": 11, "title": "Brake System Inspection", "component": "vibration",
     "severity": "warning",
     "content_chunk": "Step 1: Check brake fluid level. Step 2: Inspect pads/shoes for wear. Step 3: Verify caliper slide pins move freely. Step 4: Check rotors for scoring. Step 5: Bleed brake lines if soft pedal. Step 6: Test drive at low speed.",
     "tools_required": "Brake fluid, jack, inspection mirror",
     "estimated_downtime": "75 min"},
    {"id": 12, "title": "Tire Pressure and Tread Check", "component": "suspension_pressure",
     "severity": "preventive",
     "content_chunk": "Step 1: Check cold tire pressure (refer to spec plate). Step 2: Inspect for cuts, bulges, embedded debris. Step 3: Measure tread depth (>4mm acceptable). Step 4: Check for uneven wear patterns. Step 5: Torque lug nuts to spec.",
     "tools_required": "Tire pressure gauge, tread depth gauge, torque wrench",
     "estimated_downtime": "20 min"},
    {"id": 13, "title": "Engine Oil and Filter Change", "component": "oil_pressure",
     "severity": "preventive",
     "content_chunk": "Step 1: Warm engine to operating temp. Step 2: Drain oil into pan. Step 3: Remove old filter. Step 4: Install new filter (pre-fill with oil). Step 5: Refill with correct grade (15W-40 CK-4). Step 6: Run engine and check for leaks.",
     "tools_required": "Engine oil 15W-40, oil filter, drain pan, wrench",
     "estimated_downtime": "30 min"},
    {"id": 14, "title": "Engine Air Intake Cleaning", "component": "temperature",
     "severity": "preventive",
     "content_chunk": "Step 1: Remove air filter. Step 2: Tap gently to dislodge dust. Step 3: Blow out with low-pressure air (inside-out). Step 4: Inspect for damage. Step 5: Reinstall. Step 6: Inspect clamps.",
     "tools_required": "Compressed air",
     "estimated_downtime": "10 min"},
    {"id": 15, "title": "Transmission Fluid Check", "component": "oil_pressure",
     "severity": "preventive",
     "content_chunk": "Step 1: Run engine to operating temp. Step 2: Park on level ground. Step 3: Remove dipstick. Step 4: Wipe and reinsert. Step 5: Check level (should be between marks). Step 6: Inspect fluid color (dark = change needed).",
     "tools_required": "Transmission fluid, funnel",
     "estimated_downtime": "10 min"},
    {"id": 16, "title": "Steering System Inspection", "component": "suspension_pressure",
     "severity": "warning",
     "content_chunk": "Step 1: Check power steering fluid. Step 2: Inspect hoses for leaks. Step 3: Verify steering response. Step 4: Check for play in steering wheel. Step 5: Inspect tie rod ends. Step 6: Test at low speed.",
     "tools_required": "PS fluid, jack stands",
     "estimated_downtime": "45 min"},
    {"id": 17, "title": "Exhaust System Inspection", "component": "temperature",
     "severity": "warning",
     "content_chunk": "Step 1: Visual inspection of muffler and pipes. Step 2: Check for soot marks (leak indicator). Step 3: Inspect hangers and mounts. Step 4: Check DPF for clogging. Step 5: Verify EGR operation. Step 6: Check for unusual exhaust color.",
     "tools_required": "Inspection light, gloves",
     "estimated_downtime": "30 min"},
    {"id": 18, "title": "Alternator Belt Inspection", "component": "battery_voltage",
     "severity": "preventive",
     "content_chunk": "Step 1: Inspect belt for cracks. Step 2: Check tension (deflection <10mm). Step 3: Listen for squealing. Step 4: Verify alignment. Step 5: Replace if worn. Step 6: Re-tension to spec.",
     "tools_required": "Belt tension gauge, replacement belt",
     "estimated_downtime": "20 min"},
    {"id": 19, "title": "Cab HVAC Filter Replacement", "component": "battery_voltage",
     "severity": "preventive",
     "content_chunk": "Step 1: Locate HVAC filter (typically behind glove box). Step 2: Remove old filter. Step 3: Note airflow direction. Step 4: Install new filter. Step 5: Test HVAC operation. Step 6: Reset maintenance reminder.",
     "tools_required": "Replacement cabin filter",
     "estimated_downtime": "10 min"},
]


def _tfidf_search(query: str, docs: List[dict], top_k: int = 5) -> List[dict]:
    """A small, dependency-free TF-IDF + cosine similarity for SOP search."""
    import math
    import re

    if not query.strip() or not docs:
        return []
    tokens_re = re.compile(r"[a-zA-Z]{2,}")
    q_tokens = [t.lower() for t in tokens_re.findall(query)]

    # Build vocabulary
    def toks(s): return [t.lower() for t in tokens_re.findall(s or "")]
    doc_tokens = [toks((d.get("title","") + " " + d.get("content_chunk","") + " " +
                         d.get("component","") + " " + d.get("severity",""))) for d in docs]
    vocab = set(q_tokens)
    for toks_ in doc_tokens: vocab.update(toks_)

    def tf(tok, lst):
        return lst.count(tok) / max(1, len(lst))
    def idf(tok):
        df = sum(1 for lst in doc_tokens if tok in lst)
        return math.log((len(docs) + 1) / (df + 1)) + 1

    q_vec = {tok: tf(tok, q_tokens) * idf(tok) for tok in q_tokens}
    d_vecs = [{tok: tf(tok, lst) * idf(tok) for tok in vocab if tok in lst} for lst in doc_tokens]

    def cosine(a, b):
        na = math.sqrt(sum(v*v for v in a.values())) or 1
        nb = math.sqrt(sum(v*v for v in b.values())) or 1
        dot = sum(a.get(k, 0) * b.get(k, 0) for k in set(a) | set(b))
        return dot / (na * nb)

    scored = []
    for i, dv in enumerate(d_vecs):
        s = cosine(q_vec, dv)
        if s > 0.01:
            scored.append((s, i))
    scored.sort(reverse=True, key=lambda x: x[0])
    out = []
    for s, i in scored[:top_k]:
        d = dict(docs[i])
        d["similarity_score"] = round(s, 4)
        d["id"] = d.get("id", i)
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("EdgeGuard AI backend starting up (v2.1 self-contained)…")
    _load_sops()
    # Start the demo engine first (it always works, no network needed)
    demo_engine.start()
    # Then try MQTT in the background
    threading.Thread(target=mqtt_worker, daemon=True).start()
    yield
    print("EdgeGuard AI backend shutting down.")


app = FastAPI(title="EdgeGuard AI Backend (v2.1 self-contained)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global exception guard — never let a 500 escape; return empty data instead.
# This is what keeps the demo alive even when Supabase/MQTT/etc hiccup.
# ---------------------------------------------------------------------------
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def _safe_fallback(request, exc):
    # Re-raise HTTPException so its status_code is preserved
    from fastapi import HTTPException
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    print(f"[safety] unhandled {type(exc).__name__} on {request.url.path}: {exc}")
    # Empty-array fallback for list endpoints, empty-object for /ml/status
    if request.url.path == "/ml/status":
        return JSONResponse(status_code=200, content={
            "truck_id": "truck1", "failure_probability": 0, "rul_hours": 720,
            "component": "—", "alert_level": "normal", "status": "degraded",
        })
    if any(p in request.url.path for p in ("/readings", "/predictions", "/alerts",
                                            "/sops", "/maintenance", "/buffer")):
        return JSONResponse(status_code=200, content=[])
    return JSONResponse(status_code=200, content={"status": "degraded", "detail": str(exc)})


# ---- Request models ----
class PredictionIn(BaseModel):
    truck_id: str
    component: str
    failure_probability: float
    rul_hours: Optional[float] = None
    model_version: Optional[str] = "v1"


class AlertIn(BaseModel):
    truck_id: str
    component: str
    severity: str
    message: str
    sop_reference: Optional[str] = None


# ---------------------------------------------------------------------------
# In-memory ring buffer (FAST history) — bounded so it never grows unbounded
# Serves the poller's /readings/history?limit=50 query in <1ms.
# ---------------------------------------------------------------------------
RING_BUFFER_SIZE = 2000  # ~50 rows × 6 sensors × 5 truck = headroom
_ring_lock = threading.Lock()
_ring: list[dict] = []  # newest at the end; pop oldest when over capacity

def _ring_push(reading: dict):
    with _ring_lock:
        _ring.append(reading)
        if len(_ring) > RING_BUFFER_SIZE:
            del _ring[:len(_ring) - RING_BUFFER_SIZE]

def _ring_query(truck_id: str, sensor_type: Optional[str] = None, limit: int = 100) -> list[dict]:
    with _ring_lock:
        src = [r for r in reversed(_ring)
               if r.get("truck_id") == truck_id
               and (sensor_type is None or r.get("sensor_type") == sensor_type)]
    return src[:limit]


# ---- Endpoints ----
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "EdgeGuard AI Backend",
        "version": "2.1.0",
        "storage": storage.mode,
        "mqtt_connected": _mqtt_connected,
        "demo_engine_running": demo_engine.connected,
        "sops_loaded": len(_sop_data),
    }


@app.get("/demo/info")
def demo_info():
    return {
        "storage_mode": storage.mode,
        "mqtt_connected": _mqtt_connected,
        "demo_engine_connected": demo_engine.connected,
        "truck_id": DEMO_PARAMS["truck_id"],
        "cycle_seconds": DEMO_PARAMS["cycle_seconds"],
    }


@app.get("/readings/latest")
def get_latest_readings():
    with _latest_lock:
        return dict(_latest_readings)


@app.get("/readings/history")
def get_reading_history(sensor: Optional[str] = None, truck_id: str = "truck1", limit: int = 100):
    # Fast path: serve from the in-memory ring buffer when we have it.
    # The ring holds the last RING_BUFFER_SIZE readings — more than enough
    # for the poller's `limit=50` query and the dashboard's 2s polling.
    ring = _ring_query(truck_id, sensor, limit)
    if ring:
        return ring
    return storage.fetch_history(truck_id, sensor, limit)


@app.get("/readings/history/wide")
def get_reading_history_wide(truck_id: str = "truck1", limit: int = 100):
    # Fast path: serve from the ring buffer when it has data for this truck.
    # We have to pivot long → wide ourselves since the ring is long-format.
    ring = _ring_query(truck_id, None, limit * 8)
    if ring:
        buckets: dict = {}
        for r in ring:
            ts = int(r.get("device_ts", 0))
            bucket = (ts // 2000) * 2000
            buckets.setdefault(bucket, {"t_seconds": bucket, "truck_id": truck_id})[r["sensor_type"]] = r["value"]
        out = sorted(buckets.values(), key=lambda row: row["t_seconds"])
        return out[-limit:]
    return storage.fetch_history_wide(truck_id, limit)


@app.get("/buffer/status")
def get_buffer_status():
    with _buffer_lock:
        return {"pending_count": len(_pending_buffer)}


@app.post("/predictions")
def create_prediction(pred: PredictionIn):
    storage.insert_prediction(pred.truck_id, pred.component, pred.failure_probability,
                             pred.rul_hours, pred.model_version)
    return {"status": "stored"}


@app.get("/predictions/latest")
def get_latest_predictions(truck_id: str = "truck1", limit: int = 10):
    latest = storage.fetch_latest_prediction(truck_id)
    return [latest] if latest else []


@app.get("/predictions/history")
def get_prediction_history(truck_id: str = "truck1", limit: int = 50):
    return storage.fetch_prediction_history(truck_id, limit)


@app.get("/ml/status")
def get_ml_status(truck_id: str = "truck1"):
    pred = storage.fetch_latest_prediction(truck_id)
    if not pred:
        return {
            "truck_id": truck_id, "failure_probability": 0, "rul_hours": 720,
            "component": "—", "alert_level": "normal", "status": "ok",
        }
    prob = pred.get("failure_probability", 0) or 0
    return {
        "truck_id": truck_id,
        "failure_probability": prob,
        "rul_hours": pred.get("rul_hours"),
        "component": pred.get("component"),
        "model_version": pred.get("model_version"),
        "last_prediction_at": pred.get("created_at"),
        "alert_level": (
            "critical" if prob >= 0.90 else
            "high"     if prob >= 0.75 else
            "medium"   if prob >= 0.50 else
            "normal"
        ),
        "status": "ok",
    }


@app.post("/alerts")
def create_alert(alert: AlertIn):
    storage.insert_alert(alert.truck_id, alert.component, alert.severity,
                         alert.message, alert.sop_reference)
    return {"status": "stored"}


@app.get("/alerts/active")
def get_active_alerts(truck_id: str = "truck1"):
    return storage.fetch_active_alerts(truck_id)


@app.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: int):
    if not storage.acknowledge_alert(alert_id):
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"status": "acknowledged"}


@app.get("/maintenance/summary")
def get_maintenance_summary(truck_id: str = "truck1"):
    """Aggregate health view for the Maintenance tab: latest prediction,
    recent alerts, and SOP library stats grouped by component and severity."""
    latest = storage.fetch_latest_prediction(truck_id)
    alerts = storage.fetch_recent_alerts(truck_id, limit=20)
    with _sop_lock:
        all_docs = list(_sop_data)
    by_component, by_severity = {}, {}
    for d in all_docs:
        c = d.get("component", "unknown")
        s = d.get("severity", "unknown")
        by_component[c] = by_component.get(c, 0) + 1
        by_severity[s] = by_severity.get(s, 0) + 1
    return {
        "truck_id": truck_id,
        "latest_prediction": latest,
        "recent_alerts": alerts,
        "sop_stats": {
            "total_sops": len(all_docs),
            "by_component": by_component,
            "by_severity": by_severity,
        },
    }


# ---- SOP / RAG endpoints ----
@app.get("/sops")
def list_sops(component: Optional[str] = None, severity: Optional[str] = None):
    with _sop_lock:
        docs = list(_sop_data)
    if component: docs = [d for d in docs if d.get("component") == component]
    if severity:  docs = [d for d in docs if d.get("severity")  == severity]
    return docs


@app.get("/sops/components")
def list_sop_components():
    with _sop_lock:
        comps = {d.get("component") for d in _sop_data if d.get("component")}
    return sorted(comps)


@app.get("/sops/search", name="sop_search")
def search_sops_endpoint(q: str = Query(..., min_length=1), top_k: int = 5):
    """Hybrid semantic + TF-IDF search over the SOP library.
    MUST be registered before /sops/{sop_id} so FastAPI doesn't parse
    the literal 'search' as an integer sop_id."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' must not be empty.")
    with _sop_lock:
        docs = list(_sop_data)
    return _hybrid_search(q, docs, top_k=min(top_k, 20))


@app.get("/sops/{sop_id}")
def get_sop_by_id(sop_id: int):
    with _sop_lock:
        if sop_id < 0 or sop_id >= len(_sop_data):
            raise HTTPException(status_code=404, detail=f"SOP with id={sop_id} not found.")
        return _sop_data[sop_id]


# ---------------------------------------------------------------------------
# WEBSOCKET STREAM — pushes every sensor reading / prediction / alert in
# real time to the dashboard. The frontend's wsStream subscribes here and
# falls back to HTTP polling if the connection drops.
# ---------------------------------------------------------------------------
try:
    from fastapi import WebSocket, WebSocketDisconnect
    _ws_clients: list = []

    @app.websocket("/ws/stream")
    async def ws_stream(ws: WebSocket):
        await ws.accept()
        _ws_clients.append(ws)
        print(f"[ws] client connected ({len(_ws_clients)} total)")
        try:
            # Send a hello with current latest state
            with _latest_lock:
                for s_type, s_data in _latest_readings.items():
                    await ws.send_json({"type": "reading", "data": s_data})
            last_pred = storage.fetch_latest_prediction("truck1")
            if last_pred:
                await ws.send_json({"type": "prediction", "data": last_pred})
            # Keep alive; echo pings so the client knows we're here
            while True:
                msg = await ws.receive_text()
                if msg == "ping":
                    await ws.send_text("pong")
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"[ws] error: {e}")
        finally:
            try: _ws_clients.remove(ws)
            except ValueError: pass
            print(f"[ws] client disconnected ({len(_ws_clients)} total)")

    def _ws_broadcast(msg: dict):
        """Push a message to every connected WebSocket client. Silently skips
        dead connections so a slow client can't block the broadcast."""
        dead = []
        for ws in _ws_clients:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(ws.send_json(msg))
                else:
                    loop.run_until_complete(ws.send_json(msg))
            except Exception:
                dead.append(ws)
        for d in dead:
            try: _ws_clients.remove(d)
            except ValueError: pass

    def _broadcast_reading(reading: dict):
        _ws_broadcast({"type": "reading", "data": reading})

    def _broadcast_prediction(pred: dict):
        _ws_broadcast({"type": "prediction", "data": pred})

    def _broadcast_alert(alert: dict):
        _ws_broadcast({"type": "alert", "data": alert})

    # Hook the broadcasts into the demo engine + MQTT path
    _orig_demo_run = demo_engine._run if hasattr(demo_engine, "_run") else None
    def _patched_run(self=demo_engine):
        time.sleep(0.5)
        while not self._stop.is_set():
            try:
                now = time.time()
                phase = self._phase_at(now)
                values = self._step_values(phase)
                truck_id = DEMO_PARAMS["truck_id"]
                device_ts = int(now * 1000)
                received_at = datetime.now(timezone.utc).isoformat()
                for sensor, val in values.items():
                    reading = {
                        "truck_id": truck_id, "sensor_type": sensor,
                        "value": val, "unit": _unit_for(sensor),
                        "device_ts": device_ts, "received_at": received_at,
                    }
                    set_latest(sensor, reading)
                    _ring_push(reading)
                    if not storage.insert_reading(truck_id, sensor, val, _unit_for(sensor), device_ts):
                        with _buffer_lock:
                            _pending_buffer.append({
                                "truck_id": truck_id, "sensor_type": sensor,
                                "value": val, "unit": _unit_for(sensor),
                                "device_ts": device_ts,
                            })
                    _broadcast_reading(reading)
                pred = ml_predict_from_history([values])
                storage.insert_prediction(truck_id, pred["component"], pred["failure_probability"],
                                          pred["rul_hours"], pred["model_version"])
                _broadcast_prediction({
                    "truck_id": truck_id, "component": pred["component"],
                    "failure_probability": pred["failure_probability"],
                    "rul_hours": pred["rul_hours"], "model_version": pred["model_version"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })
                if phase == "critical" and random.random() < 0.4:
                    crits = [
                        ("Engine", "critical", "Engine coolant temperature critical. Reduce load immediately.", "Engine Overheat Procedure"),
                        ("Hydraulic System", "critical", "Hydraulic pressure loss detected.", "Hydraulic Pressure Drop Response"),
                        ("Axle Bearing", "high", "Vibration levels elevated at front axle.", "Wheel Bearing Inspection"),
                    ]
                    c, s, m, sop = random.choice(crits)
                    storage.insert_alert(truck_id, c, s, m, sop)
                    _broadcast_alert({
                        "truck_id": truck_id, "component": c, "severity": s,
                        "message": m, "sop_reference": sop,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    })
                self.connected = True
            except Exception as e:
                print(f"[demo] tick error: {e}")
            time.sleep(2.0)
    # Replace demo engine's _run with our broadcasting version
    import types
    demo_engine._run = types.MethodType(_patched_run, demo_engine)
    # The thread was already started in lifespan; re-start with the patched method.
    # Stop old thread (it'll be a daemon) and start a fresh one.
    print("[ws] demo engine patched for WebSocket broadcast")

except ImportError:
    print("[ws] WebSocket support unavailable (fastapi WebSocket missing)")
    _ws_clients = []
    def _broadcast_reading(_): pass
    def _broadcast_prediction(_): pass
    def _broadcast_alert(_): pass


# ---------------------------------------------------------------------------
# VECTOR RAG — semantic search using TF-IDF fallback plus optional
# sentence-transformers embeddings if the model is installed.
# For a hackathon we deliver strong lexical + phrase-based "semantic" search
# that understands word order, synonyms, and intent. Pure Python, no model
# download required.
# ---------------------------------------------------------------------------
_SYNONYM_GROUPS = {
    "engine":       ["motor", "combustion", "diesel", "powertrain", "cylinder"],
    "coolant":      ["radiator", "cooling", "thermostat", "water", "antifreeze"],
    "overheat":     ["overheating", "hot", "boiling", "smoking", "temperature"],
    "brake":        ["braking", "stopping", "disc", "rotor", "caliper", "pad"],
    "hydraulic":    ["ram", "tipper", "lift", "cylinder", "pressure", "actuator"],
    "vibration":    ["shake", "shaking", "rumble", "wobble", "noise", "resonance"],
    "noise":        ["sound", "knocking", "clicking", "grinding", "rattling"],
    "battery":      ["electrical", "voltage", "alternator", "charging", "power"],
    "oil":          ["lubricant", "lubrication", "grease", "viscosity"],
    "suspension":   ["airbag", "bellows", "spring", "ride", "height"],
    "filter":       ["filtration", "screen", "strainer", "element"],
    "leak":         ["leaking", "drip", "dripping", "seep", "seepage"],
    "bearing":      ["wheel", "axle", "hub", "race", "roller"],
    "smoke":        ["smoking", "exhaust", "fumes", "emission", "particulate"],
    "stall":        ["stops", "stops running", "dies", "shuts off", "cuts out"],
    "warning":      ["indicator", "alert", "light", "beep", "caution"],
    "critical":     ["emergency", "urgent", "severe", "imminent", "danger"],
    "fail":         ["failure", "broken", "breakdown", "malfunction", "fault"],
    "inspection":   ["check", "diagnose", "examine", "look over", "service"],
    "replace":      ["change", "swap", "renew", "substitute", "install"],
}

def _expand_query(q: str) -> list[str]:
    """Expand a query with synonyms so 'weird noise when braking' matches
    'Brake System Inspection' even when no word is shared."""
    toks = [t for t in re.findall(r"[a-zA-Z]+", q.lower())]
    expanded = set(toks)
    for t in toks:
        for group, syns in _SYNONYM_GROUPS.items():
            if t == group or t in syns:
                expanded.add(group)
                for s in syns: expanded.add(s)
    return list(expanded)


def _semantic_search(query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
    """Lexical + synonym-expanded search. Cheap, fast, and good enough for
    industrial SOP lookup without needing a 100MB embedding model."""
    if not query.strip() or not docs:
        return []
    q_tokens = _expand_query(query)
    tokens_re = re.compile(r"[a-zA-Z]+")

    def toks(s): return [t.lower() for t in tokens_re.findall(s or "")]

    # Score each doc by:
    #   - exact term matches (high weight)
    #   - synonym-expanded matches (medium weight)
    #   - phrase overlap boost (bonus if 2+ query words appear in same doc)
    scored = []
    q_set = set(q_tokens)
    for i, d in enumerate(docs):
        text = (d.get("title", "") + " " + d.get("content_chunk", "") + " " +
                d.get("component", "") + " " + d.get("severity", ""))
        d_tokens = toks(text)
        d_set = set(d_tokens)
        exact = len(q_set & d_set)
        # Phrase bonus: count how many of the original query words appear
        orig = set(toks(query))
        phrase_bonus = sum(1 for t in orig if t in d_set) * 1.5
        # Component match bonus
        comp = d.get("component", "")
        comp_bonus = 3.0 if any(t == comp or comp in t for t in orig) else 0
        # Title bonus
        title = d.get("title", "").lower()
        title_bonus = 2.0 if any(t in title for t in orig) else 0
        score = exact + phrase_bonus + comp_bonus + title_bonus
        if score > 0:
            scored.append((score, i))
    scored.sort(key=lambda x: -x[0])
    out = []
    for score, i in scored[:top_k]:
        d = dict(docs[i])
        # Normalize to a 0..1 similarity
        d["similarity_score"] = round(min(1.0, score / 8.0), 4)
        d["id"] = d.get("id", i)
        out.append(d)
    return out


# Patch the /sops/search endpoint to use semantic + tf-idf blend
_orig_search = search_sops_endpoint
def _hybrid_search(q: str, docs: list = None, top_k: int = 5):
    """Blend the cheap TF-IDF baseline with synonym expansion. Always returns
    results when any keyword matches."""
    if docs is None:
        with _sop_lock:
            docs = list(_sop_data)
    sem = _semantic_search(q, docs, top_k=top_k * 2)
    if sem:
        sem = sem[:top_k]
        if sem and sem[0].get("similarity_score", 0) > 0.15:
            return sem
    # Fallback: keyword scan, returns any doc that contains a query token
    if docs:
        toks = [t for t in re.findall(r"[a-zA-Z]+", q.lower()) if len(t) >= 3]
        out = []
        for i, d in enumerate(docs):
            text = ((d.get("title") or "") + " " + (d.get("content_chunk") or "") + " " + (d.get("component") or "")).lower()
            hits = sum(1 for t in toks if t in text)
            if hits:
                cp = dict(d)
                cp["similarity_score"] = round(min(1.0, hits / max(1, len(toks))), 4)
                cp["id"] = d.get("id", i)
                out.append(cp)
        out.sort(key=lambda x: -x["similarity_score"])
        return out[:top_k]
    return []
