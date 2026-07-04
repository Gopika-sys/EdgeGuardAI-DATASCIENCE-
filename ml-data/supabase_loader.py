"""
EdgeGuard AI - Supabase Training Data Loader (Member 4)

Pulls training data from Supabase so that train.py can retrain on real
live sensor data as it accumulates, rather than always using the synthetic
CSV.

Two pull strategies are supported:

  1. training_data table (wide format) — if the team has populated a
     dedicated `training_data` table with the same column layout as the
     CSV (truck_id, cycle_id, t_seconds, temperature, vibration, ...,
     failure_probability, rul_hours, label_failure_within_1hr).
     This is the preferred path.

  2. sensor_readings table (long format) + pivot — falls back to the live
     `sensor_readings` table, pivots it to wide format using the same 2s
     bucket logic as inference.py. The labels (failure_probability,
     rul_hours, label_failure_within_1hr) will NOT be available from
     this table (they come from the simulator's ground truth), so this
     path is only useful for EDA / drift detection, not for supervised
     retraining.

Usage:
    from supabase_loader import load_training_data
    df = load_training_data()  # wide-format DataFrame, same schema as CSV
"""

import os
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Supabase client (lazy-loaded so the module can be imported without crashing
# if supabase-py is not installed)
# ---------------------------------------------------------------------------

def _get_client():
    from dotenv import load_dotenv
    from supabase import create_client
    load_dotenv(Path(__file__).parent / ".env")
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        raise EnvironmentError(
            "SUPABASE_URL and SUPABASE_KEY must be set in ml-data/.env "
            "(copy .env.example and fill in the values)"
        )
    return create_client(url, key)


# ---------------------------------------------------------------------------
# Wide-format training_data table pull
# ---------------------------------------------------------------------------

REQUIRED_WIDE_COLS = [
    "truck_id", "cycle_id", "t_seconds",
    "temperature", "vibration", "oil_pressure",
    "hydraulic_pressure", "suspension_pressure", "battery_voltage",
    "failure_probability", "rul_hours", "label_failure_within_1hr",
]


def pull_training_table(sb, page_size: int = 1000) -> pd.DataFrame:
    """
    Pulls all rows from the `training_data` table in Supabase.
    Uses pagination to handle large tables beyond the default 1000-row limit.
    """
    rows = []
    offset = 0
    while True:
        resp = (
            sb.table("training_data")
            .select("*")
            .order("cycle_id")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = resp.data
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("training_data table is empty.")
    return df


# ---------------------------------------------------------------------------
# Long-format sensor_readings pivot (fallback / EDA only)
# ---------------------------------------------------------------------------

def pull_sensor_readings_wide(sb, truck_id: str = "truck1", page_size: int = 1000) -> pd.DataFrame:
    """
    Pulls sensor_readings (long format) for a truck and pivots to wide format.
    NOTE: no label columns — useful for drift detection, not supervised training.
    """
    rows = []
    offset = 0
    while True:
        resp = (
            sb.table("sensor_readings")
            .select("device_ts, sensor_type, value")
            .eq("truck_id", truck_id)
            .order("device_ts")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = resp.data
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    if not rows:
        raise ValueError(f"No sensor_readings found for truck_id='{truck_id}'.")

    df = pd.DataFrame(rows)
    df["time_bucket"] = (df["device_ts"] // 2000) * 2000
    wide = df.pivot_table(
        index="time_bucket", columns="sensor_type", values="value", aggfunc="max"
    ).reset_index().rename(columns={"time_bucket": "t_seconds"})
    wide["truck_id"]  = truck_id
    wide["cycle_id"]  = "live"
    return wide


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_training_data(
    prefer: str = "training_table",
    truck_id: str = "truck1",
    csv_fallback: Path = None,
) -> pd.DataFrame:
    """
    Master loader. Tries Supabase first, then falls back to CSV.

    prefer:
        "training_table"   — pull the wide training_data table (has labels)
        "sensor_readings"  — pull + pivot sensor_readings (no labels; EDA only)

    csv_fallback:
        Path to the local CSV to use if Supabase is unreachable.
        Defaults to ml-data/edgeguard_training_data.csv.
    """
    if csv_fallback is None:
        csv_fallback = Path(__file__).parent / "edgeguard_training_data.csv"

    try:
        sb = _get_client()
        if prefer == "training_table":
            df = pull_training_table(sb)
            print(f"[supabase_loader] Loaded {len(df):,} rows from training_data table.")
        else:
            df = pull_sensor_readings_wide(sb, truck_id=truck_id)
            print(f"[supabase_loader] Loaded {len(df):,} pivoted rows from sensor_readings.")
        return df

    except ImportError:
        print("[supabase_loader] supabase-py not installed — using CSV fallback.")
    except EnvironmentError as e:
        print(f"[supabase_loader] Env error ({e}) — using CSV fallback.")
    except Exception as e:
        print(f"[supabase_loader] Supabase error ({e}) — using CSV fallback.")

    print(f"[supabase_loader] Reading {csv_fallback}")
    return pd.read_csv(csv_fallback)


# ---------------------------------------------------------------------------
# Quick health check
# ---------------------------------------------------------------------------

def check_supabase_connection() -> dict:
    """
    Returns a status dict so the poller / dashboard can surface DB health.
    """
    try:
        sb   = _get_client()
        resp = sb.table("sensor_readings").select("id").limit(1).execute()
        return {"status": "ok", "sensor_readings_reachable": True}
    except Exception as e:
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    status = check_supabase_connection()
    print(f"Supabase connection: {status}")

    df = load_training_data()
    print(f"Loaded {len(df):,} rows with columns: {list(df.columns)}")
    print(df.head(3).to_string(index=False))
