"""
EdgeGuard AI - Live ML Poller (Member 4)

The bridge that makes the ML pipeline operational end-to-end:

    Supabase sensor_readings
         ↓  (via GET /readings/history)
    FastAPI backend
         ↓
    inference.py → failure_probability + rul_hours + explanation
         ↓
    POST /predictions  (always)
    POST /alerts       (if failure_probability > ALERT_THRESHOLD)
         ↓
    Supabase predictions + alerts tables

Runs as a standalone loop, polling every POLL_INTERVAL_SECONDS seconds.
All predictions are also appended to predictions_log.jsonl for offline
review and debugging.

Configuration (via environment variables or ml-data/.env):
    BACKEND_URL            http://localhost:8000  (default)
    TRUCK_ID               truck1                 (default)
    ALERT_THRESHOLD        0.70                   (default)
    POLL_INTERVAL_SECONDS  10                     (default)
    HISTORY_LIMIT          50                     (default — number of readings to pull)
    COMPONENT_NAME         hydraulic_system       (default — sent to /predictions)
    MODEL_VERSION          v2                     (default)
    LOG_PATH               predictions_log.jsonl  (default)

Run:
    cd ml-data
    python poller.py

Or keep it running as a background service:
    python poller.py > poller.log 2>&1 &
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND_URL           = os.environ.get("BACKEND_URL", "http://localhost:8000")
TRUCK_ID              = os.environ.get("TRUCK_ID", "truck1")
ALERT_THRESHOLD       = float(os.environ.get("ALERT_THRESHOLD", "0.70"))
POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL_SECONDS", "10"))
HISTORY_LIMIT         = int(os.environ.get("HISTORY_LIMIT", "50"))
COMPONENT_NAME        = os.environ.get("COMPONENT_NAME", "hydraulic_system")
MODEL_VERSION         = os.environ.get("MODEL_VERSION", "v2")
LOG_PATH              = Path(os.environ.get("LOG_PATH",
                             str(Path(__file__).parent / "predictions_log.jsonl")))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("edgeguard.poller")

# ---------------------------------------------------------------------------
# Lazy-load the inference service (heavy — only load once at startup)
# ---------------------------------------------------------------------------

def _load_prediction_service():
    from inference import PredictionService
    return PredictionService()


# ---------------------------------------------------------------------------
# Backend API calls
# ---------------------------------------------------------------------------

def fetch_readings(session: requests.Session) -> list[dict]:
    """Pulls the last HISTORY_LIMIT sensor readings from the backend."""
    url = f"{BACKEND_URL}/readings/history"
    params = {"truck_id": TRUCK_ID, "limit": HISTORY_LIMIT}
    resp = session.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def post_prediction(session: requests.Session, result: dict) -> bool:
    """Pushes a prediction to the backend /predictions endpoint."""
    payload = {
        "truck_id":           TRUCK_ID,
        "component":          COMPONENT_NAME,
        "failure_probability": result["failure_probability"],
        "rul_hours":          result["rul_hours"],
        "model_version":      MODEL_VERSION,
    }
    try:
        resp = session.post(f"{BACKEND_URL}/predictions", json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        log.warning(f"POST /predictions failed: {e}")
        return False


def post_alert(session: requests.Session, result: dict, severity: str) -> bool:
    """Posts an alert to the backend /alerts endpoint."""
    payload = {
        "truck_id":      TRUCK_ID,
        "component":     COMPONENT_NAME,
        "severity":      severity,
        "message":       result["explanation"],
        "sop_reference": _sop_reference(result["failure_probability"]),
    }
    try:
        resp = session.post(f"{BACKEND_URL}/alerts", json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        log.warning(f"POST /alerts failed: {e}")
        return False


def _sop_reference(prob: float) -> str:
    """Maps failure probability to a standard operating procedure reference."""
    if prob >= 0.90:
        return "SOP-MAINT-001: Immediate shutdown and inspection"
    elif prob >= 0.75:
        return "SOP-MAINT-002: Schedule maintenance within 4 hours"
    else:
        return "SOP-MAINT-003: Monitor and inspect at next service interval"


def _alert_severity(prob: float) -> str:
    if prob >= 0.90:
        return "critical"
    elif prob >= 0.75:
        return "high"
    else:
        return "medium"


# ---------------------------------------------------------------------------
# Prediction log
# ---------------------------------------------------------------------------

def append_log(result: dict, readings_count: int, alert_fired: bool):
    entry = {
        "ts":                 datetime.now(timezone.utc).isoformat(),
        "truck_id":           TRUCK_ID,
        "failure_probability": result["failure_probability"],
        "failure_class":      result.get("failure_class"),
        "rul_hours":          result["rul_hours"],
        "alert_fired":        alert_fired,
        "readings_in_buffer": readings_count,
        "explanation":        result.get("explanation", ""),
        "top_features":       result.get("top_features", [])[:3],
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Main poll loop
# ---------------------------------------------------------------------------

def run_poll_loop(svc):
    session = requests.Session()
    log.info(f"EdgeGuard AI poller started.")
    log.info(f"  Backend:        {BACKEND_URL}")
    log.info(f"  Truck:          {TRUCK_ID}")
    log.info(f"  Alert threshold:{ALERT_THRESHOLD}")
    log.info(f"  Poll interval:  {POLL_INTERVAL_SECONDS}s")
    log.info(f"  Prediction log: {LOG_PATH}")

    consecutive_failures = 0

    while True:
        cycle_start = time.monotonic()

        try:
            # ── Pull readings ────────────────────────────────────────────────
            readings = fetch_readings(session)
            if not readings:
                log.warning("No readings returned — waiting for sensor data.")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # ── Convert long → wide ──────────────────────────────────────────
            from inference import readings_to_wide_df
            wide_df = readings_to_wide_df(readings)

            if len(wide_df) < 3:
                log.warning(f"Only {len(wide_df)} wide rows — need at least 3. Waiting...")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # ── Run inference ────────────────────────────────────────────────
            result = svc.predict(wide_df)
            prob   = result["failure_probability"]
            rul    = result["rul_hours"]

            # ── Push prediction ──────────────────────────────────────────────
            post_prediction(session, result)

            # ── Maybe fire alert ─────────────────────────────────────────────
            alert_fired = False
            if prob >= ALERT_THRESHOLD:
                severity    = _alert_severity(prob)
                alert_fired = post_alert(session, result, severity)

            # ── Log locally ──────────────────────────────────────────────────
            append_log(result, len(readings), alert_fired)

            # ── Console output ───────────────────────────────────────────────
            prob_bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            alert_tag = f" 🚨 ALERT ({_alert_severity(prob).upper()})" if alert_fired else ""
            log.info(
                f"[{TRUCK_ID}] prob={prob:.3f} [{prob_bar}] "
                f"RUL={rul:.1f}h | buf={len(readings)} readings{alert_tag}"
            )
            if prob >= 0.5:
                log.info(f"           → {result.get('explanation', '')}")

            consecutive_failures = 0

        except KeyboardInterrupt:
            log.info("Poller stopped by user.")
            sys.exit(0)

        except requests.exceptions.ConnectionError:
            consecutive_failures += 1
            if consecutive_failures == 1:
                log.error(
                    f"Cannot reach backend at {BACKEND_URL}. "
                    "Is `uvicorn main:app --port 8000` running?"
                )
            elif consecutive_failures % 10 == 0:
                log.error(f"Still unable to reach backend ({consecutive_failures} attempts).")

        except Exception as e:
            log.exception(f"Unexpected error in poll cycle: {e}")
            consecutive_failures += 1

        # ── Sleep for the remainder of the interval ──────────────────────────
        elapsed = time.monotonic() - cycle_start
        sleep_for = max(0.0, POLL_INTERVAL_SECONDS - elapsed)
        time.sleep(sleep_for)


def main():
    log.info("Loading ML models...")
    try:
        svc = _load_prediction_service()
        log.info("Models loaded successfully.")
    except FileNotFoundError as e:
        log.error(
            f"Model files not found: {e}\n"
            "Run 'python train.py' first to train and save the models."
        )
        sys.exit(1)
    except Exception as e:
        log.error(f"Failed to load models: {e}")
        sys.exit(1)

    run_poll_loop(svc)


if __name__ == "__main__":
    main()
