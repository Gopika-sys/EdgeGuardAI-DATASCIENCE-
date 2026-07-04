"""
EdgeGuard AI — Feature Engineering (v2, enterprise schema)

Module 03 (Feature Engineering) of the ml work.txt specification.

Generates windowed + cross-sensor + time-aware features from raw
sensor readings. Works with the 45-column enterprise schema
(see generate_training_data.py).

FEATURE FAMILIES
----------------
1. Raw sensor values
2. Rolling statistics (mean, std, min, max) over 5 / 10 / 20 sample windows
3. Rate of change (1-sample, 5-sample, 10-sample)
4. Lag features (5, 10, 20 samples back)
5. Exponential moving average (alpha=0.3)
6. Cross-sensor interaction features (cascade patterns)
7. Time-since-feature deltas (cumulative-in-cycle health gradient)
8. Truck-level encoded state (one-hot material, zone, operator)

Used in two places with identical code path:
  - train.py on the full historical CSV (grouped by truck_id + cycle_id)
  - inference.py on a short recent buffer from /readings/history

DESIGN NOTE
-----------
`groupby().rolling()` returns a Series whose index is the original row index
grouped. We assign back to the DataFrame using `.values` (positional alignment
is guaranteed because we sorted by the group keys before iterating).
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Sensor columns from the 45-column enterprise dataset
# ---------------------------------------------------------------------------
SENSOR_COLS = [
    # Pressure
    "oil_pressure", "hydraulic_pressure", "fuel_pressure",
    "brake_pressure", "suspension_pressure",
    # Electrical
    "battery_voltage", "alternator_voltage", "current_draw",
    # Vibration / mechanical
    "vibration_x", "vibration_y", "vibration_z", "vibration_rms",
    "bearing_temp", "transmission_temp",
    # Temperature
    "engine_temp", "coolant_temp", "brake_temp",
    # Engine / load
    "rpm", "engine_load_pct", "fuel_consumption_rate",
    # Environment
    "ambient_temp", "humidity", "road_gradient", "vehicle_speed",
    # Payload / health
    "payload_weight", "carryback_pct", "health_score",
]

CATEGORICAL_COLS = ["material_type", "gps_zone", "operator_id", "op_state"]

WINDOWS = [5, 10, 20]
LAGS = [5, 10, 20]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ema(series: pd.Series, alpha: float = 0.3) -> pd.Series:
    return series.ewm(alpha=alpha, adjust=False).mean()


def _per_group_rolling(df: pd.DataFrame, group_cols: list, col: str,
                       window: int, stat: str) -> np.ndarray:
    """Compute per-group rolling stat, return as numpy array aligned to df rows."""
    arr = np.full(len(df), np.nan, dtype=np.float64)
    for _, idx in df.groupby(group_cols, sort=False).groups.items():
        s = df[col].iloc[idx]
        roll = s.rolling(window=window, min_periods=1)
        if   stat == "mean": arr[idx] = roll.mean().values
        elif stat == "std":  arr[idx] = roll.std().fillna(0.0).values
        elif stat == "min":  arr[idx] = roll.min().values
        elif stat == "max":  arr[idx] = roll.max().values
    return arr


def _per_group_diff(df: pd.DataFrame, group_cols: list, col: str,
                    periods: int) -> np.ndarray:
    arr = np.zeros(len(df), dtype=np.float64)
    for _, idx in df.groupby(group_cols, sort=False).groups.items():
        s = df[col].iloc[idx]
        long_d  = s.diff(periods=periods)
        short_d = s.diff(periods=1)
        arr[idx] = long_d.fillna(short_d).fillna(0.0).values
    return arr


def _per_group_lag(df: pd.DataFrame, group_cols: list, col: str,
                   lag: int) -> np.ndarray:
    arr = np.zeros(len(df), dtype=np.float64)
    for _, idx in df.groupby(group_cols, sort=False).groups.items():
        s = df[col].iloc[idx]
        shifted = s.shift(lag)
        first_val = s.iloc[0] if len(s) > 0 else 0.0
        arr[idx] = shifted.fillna(first_val).values
    return arr


def _per_group_ema(df: pd.DataFrame, group_cols: list, col: str,
                   alpha: float = 0.3) -> np.ndarray:
    arr = np.zeros(len(df), dtype=np.float64)
    for _, idx in df.groupby(group_cols, sort=False).groups.items():
        s = df[col].iloc[idx]
        arr[idx] = _ema(s, alpha=alpha).values
    return arr


def _one_hot_encode(df: pd.DataFrame, cols: list, max_cardinality: int = 12) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        top = out[col].value_counts().head(max_cardinality).index.tolist()
        out[col] = out[col].where(out[col].isin(top), other="__other__")
        dummies = pd.get_dummies(out[col], prefix=col, dummy_na=False).astype(np.int8)
        out = pd.concat([out, dummies], axis=1)
        out = out.drop(columns=[col])
    return out


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------
def compute_features(df: pd.DataFrame,
                     group_cols: tuple = ("truck_id", "cycle_id"),
                     windows: list = WINDOWS,
                     lags: list = LAGS) -> pd.DataFrame:
    sort_cols = list(group_cols) + (["t_seconds"] if "t_seconds" in df.columns else [])
    df = df.sort_values(sort_cols).reset_index(drop=True)
    out = df.copy()
    gc = list(group_cols)
    new_cols: dict = {}  # collect all new columns then concat at once (perf)

    # --- 1. Rolling stats + rate of change + lag + EMA per sensor ----------
    for col in SENSOR_COLS:
        if col not in df.columns:
            continue
        for w in windows:
            new_cols[f"{col}_rmean_{w}"] = _per_group_rolling(df, gc, col, w, "mean")
            new_cols[f"{col}_rstd_{w}"]  = _per_group_rolling(df, gc, col, w, "std")
            new_cols[f"{col}_rmin_{w}"]  = _per_group_rolling(df, gc, col, w, "min")
            new_cols[f"{col}_rmax_{w}"]  = _per_group_rolling(df, gc, col, w, "max")
        for lag in lags:
            new_cols[f"{col}_roc_{lag}"] = _per_group_diff(df, gc, col, lag)
            new_cols[f"{col}_lag{lag}"]  = _per_group_lag(df, gc, col, lag)
        new_cols[f"{col}_ema"] = _per_group_ema(df, gc, col)

    # --- 2. Cross-sensor interaction features -------------------------------
    if "engine_temp" in df.columns and "vibration_rms" in df.columns:
        tmean = df["engine_temp"].rolling(10, min_periods=1).mean()
        vmean = df["vibration_rms"].rolling(10, min_periods=1).mean()
        new_cols["temp_vib_cascade"] = (df["engine_temp"] - tmean) * (df["vibration_rms"] - vmean)

    if "oil_pressure" in df.columns and "hydraulic_pressure" in df.columns:
        new_cols["oil_hyd_ratio"] = df["oil_pressure"] / df["hydraulic_pressure"].clip(lower=1.0)

    if "engine_temp" in df.columns and "oil_pressure" in df.columns:
        new_cols["temp_oil_stress"] = df["engine_temp"] / df["oil_pressure"].clip(lower=0.1)

    if "vibration_rms" in df.columns and "suspension_pressure" in df.columns:
        new_cols["susp_vib_cross"] = df["vibration_rms"] / df["suspension_pressure"].clip(lower=0.1)

    if "bearing_temp" in df.columns and "vibration_rms" in df.columns:
        new_cols["bearing_vib_product"] = df["bearing_temp"] * df["vibration_rms"]

    if "rpm" in df.columns and "engine_load_pct" in df.columns:
        new_cols["power_proxy"] = df["rpm"] * df["engine_load_pct"]

    if "vehicle_speed" in df.columns and "engine_load_pct" in df.columns:
        new_cols["speed_load_ratio"] = df["vehicle_speed"] / df["engine_load_pct"].clip(lower=1.0)

    if "payload_weight" in df.columns and "engine_temp" in df.columns:
        new_cols["load_temp_stress"] = df["payload_weight"] * df["engine_temp"] / 1000.0

    # --- 3. Time-since-feature deltas (cumulative-in-cycle) ----------------
    if "health_score" in df.columns and "t_seconds" in df.columns:
        new_cols["health_drop_per_hour"] = _per_group_apply(
            df, gc, "health_score",
            lambda s: (s.iloc[0] - s) / max(len(s), 1) * 1800
        )

    # --- 4. Categorical encoding (one-hot) ---------------------------------
    out = _one_hot_encode(out, CATEGORICAL_COLS)

    # Bulk-attach the engineered numeric columns
    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)

    # --- 5. Fill residual NaN ----------------------------------------------
    num_cols = [c for c in out.columns
                if c not in gc + ["t_seconds", "failure_mode"]
                and out[c].dtype.kind in "biufc"]
    for col in num_cols:
        if out[col].isna().any():
            out[col] = out.groupby(gc)[col].transform(
                lambda s: s.ffill().bfill()
            ).fillna(0.0)

    return out


def _per_group_apply(df, group_cols, col, fn):
    arr = np.zeros(len(df), dtype=np.float64)
    for _, idx in df.groupby(group_cols, sort=False).groups.items():
        s = df[col].iloc[idx]
        arr[idx] = fn(s).values
    return arr


# Feature columns used by the classifier / regressors (excludes target + ids)
EXCLUDE_FROM_FEATURES = {
    "truck_id", "cycle_id", "t_seconds", "failure_mode",
    "failure_probability", "rul_hours", "label_failure_within_1hr",
    "operator_id",
}


def get_feature_columns(df: pd.DataFrame | None = None) -> list:
    if df is not None:
        return [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]
    return []
