"""
EdgeGuard AI — Enterprise-Grade Synthetic Mining Truck Dataset
================================================================

Module 02 of the ml work.txt specification.

Generates an enterprise-grade synthetic dataset that closely resembles
real-world Tata Signa 4825.TK mining truck operations across a 5-truck fleet.
The dataset is designed for predictive maintenance, RUL estimation, anomaly
detection, and multi-modal decision intelligence.

DESIGN GOALS
------------
1. Multi-truck fleet (5 trucks) with independent degradation timelines
2. Multiple failure modes (overheat, bearing, hydraulic, electrical, suspension)
3. Realistic sensor correlation (cascade: temp UP -> vib UP -> oil DOWN -> failure)
4. Sensor noise, drift, missing values, outliers
5. Post-maintenance recovery events
6. Class imbalance similar to real industry (5-10% failure-positive)
7. Seasonal / operational variations
8. 25,000+ rows x 45 features

FEATURE GROUPS
--------------
A. Truck identity & time        (4)   truck_id, cycle_id, t_seconds, op_state
B. Engine / powertrain          (7)   engine_temp, coolant_temp, rpm, engine_load,
                                       engine_hours, idle_hours, fuel_consumption
C. Pressure systems             (5)   oil_pressure, hydraulic_pressure, fuel_pressure,
                                       brake_pressure, suspension_pressure
D. Electrical                   (3)   battery_voltage, alternator_voltage, current_draw
E. Vibration / mechanical       (6)   vibration_x, vibration_y, vibration_z, vibration_rms,
                                       bearing_temp, transmission_temp
F. Environment                  (4)   ambient_temp, humidity, road_gradient, vehicle_speed
G. Payload / operational        (5)   payload_weight, payload_capacity, carryback_pct,
                                       material_type, brake_temp
H. Health / context             (4)   health_score, previous_failure_count,
                                       maintenance_age_hours, operator_id
I. GPS / location               (3)   gps_zone, gps_lat, gps_lon
J. Labels (target)              (4)   failure_mode, rul_hours, failure_probability,
                                       label_failure_within_1hr

TOTAL: 45 columns including labels.

USAGE
-----
    python generate_training_data.py
    # writes edgeguard_training_data.csv (25,000+ rows)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(seed=2024)

OUTPUT_PATH = Path(__file__).parent / "edgeguard_training_data.csv"

# ---------------------------------------------------------------------------
# Fleet configuration
# ---------------------------------------------------------------------------
N_TRUCKS              = 5                # 5-truck fleet
CYCLES_PER_TRUCK      = 50               # 50 life-cycles per truck
SAMPLES_PER_CYCLE     = 100              # 100 samples = 200s of operation
SAMPLE_INTERVAL_SEC   = 2
N_CYCLES_TOTAL        = N_TRUCKS * CYCLES_PER_TRUCK   # 250 cycles
N_ROWS_TOTAL          = N_CYCLES_TOTAL * SAMPLES_PER_CYCLE  # 25,000 rows

# Failure-mode catalogue
FAILURE_MODES = ["overheat", "bearing", "hydraulic", "electrical", "suspension"]

# Material types
MATERIAL_TYPES = ["iron_ore", "coal", "limestone", "overburden", "copper_ore"]
MATERIAL_DENSITY = {  # tonnes per cubic meter of bed volume
    "iron_ore": 2.4, "coal": 0.85, "limestone": 1.6,
    "overburden": 1.9, "copper_ore": 2.2,
}

# GPS zones
GPS_ZONES = ["pit_north", "pit_south", "haul_road_a", "haul_road_b", "crusher", "stockpile"]

# Operator IDs
OPERATOR_IDS = ["OP_001", "OP_002", "OP_003", "OP_004", "OP_005", "OP_006"]


# ---------------------------------------------------------------------------
# Per-truck baseline (each truck is a slightly different machine)
# ---------------------------------------------------------------------------
def make_truck_profile(truck_id: str) -> dict:
    """Per-truck baseline values; captures machine-to-machine variation."""
    return {
        "truck_id":               truck_id,
        "base_engine_temp":       RNG.normal(82, 2.5),
        "base_coolant_temp":      RNG.normal(75, 2.0),
        "base_rpm":               RNG.normal(1450, 80),
        "base_engine_load":       RNG.normal(55, 6),
        "base_oil_pressure":      RNG.normal(3.8, 0.25),
        "base_hydraulic":         RNG.normal(215, 12),
        "base_fuel_pressure":     RNG.normal(4.2, 0.3),
        "base_brake_pressure":    RNG.normal(6.1, 0.4),
        "base_suspension":        RNG.normal(6.4, 0.35),
        "base_battery":           RNG.normal(24.2, 0.4),
        "base_alternator":        RNG.normal(28.5, 0.5),
        "base_current":           RNG.normal(18, 3),
        "base_vibration":         RNG.normal(0.32, 0.04),
        "base_bearing_temp":      RNG.normal(68, 3),
        "base_transmission_temp": RNG.normal(72, 3),
        "engine_age_hours":       RNG.uniform(2000, 9000),     # accumulated wear
        "drift_coefficient":      RNG.uniform(0.001, 0.003),   # per-sample sensor drift
        "failure_proneness":      RNG.uniform(0.10, 0.30),     # how often this truck fails
    }


# ---------------------------------------------------------------------------
# Cycle generator
# ---------------------------------------------------------------------------
def generate_cycle(truck: dict, cycle_id: int) -> pd.DataFrame:
    """Generate a single truck-life cycle (healthy -> possibly failing)."""
    n = SAMPLES_PER_CYCLE
    t = np.arange(n) * SAMPLE_INTERVAL_SEC

    # Each cycle may be: normal operation OR a cycle that ends in failure
    # Realistic mix: ~70% normal, ~30% fail
    will_fail = RNG.random() < (truck["failure_proneness"] + 0.15)
    failure_mode = RNG.choice(FAILURE_MODES) if will_fail else "none"

    # Pick a material + zone for this cycle
    material = RNG.choice(MATERIAL_TYPES)
    zone     = RNG.choice(GPS_ZONES)
    operator = RNG.choice(OPERATOR_IDS)

    # Failure ramp (severity 0->1) — only in failure cycles
    if will_fail:
        ramp_start = RNG.integers(low=int(n * 0.30), high=int(n * 0.75))
    else:
        ramp_start = n  # never starts

    severity = np.zeros(n)
    for i in range(n):
        if i < ramp_start:
            severity[i] = 0.0
        else:
            # S-curve severity: slow start, rapid mid, plateau end
            progress = (i - ramp_start) / max(1, (n - ramp_start))
            severity[i] = 1.0 / (1.0 + np.exp(-10 * (progress - 0.5)))

    # Payload / speed profile (one load cycle: load -> haul -> dump -> return)
    # Realistic: half the cycle spent loaded (heavier)
    is_loaded = (t > (n * SAMPLE_INTERVAL_SEC * 0.15)) & (t < (n * SAMPLE_INTERVAL_SEC * 0.65))
    payload_capacity_tonnes = 45.0  # Tata Signa 4825.TK
    load_factor = np.where(is_loaded, RNG.uniform(0.85, 1.0), RNG.uniform(0.0, 0.15))
    payload_weight = payload_capacity_tonnes * load_factor + RNG.normal(0, 0.4, n)
    payload_weight = np.clip(payload_weight, 0, payload_capacity_tonnes * 1.05)

    # Vehicle speed (km/h): 0-5 while loading/dumping, 25-45 while hauling loaded,
    # 35-50 returning empty
    base_speed = np.where(is_loaded, 30, 42) + RNG.normal(0, 4, n)
    speed_load_dumps = (t < 30) | (t > (n * SAMPLE_INTERVAL_SEC - 30))
    base_speed = np.where(speed_load_dumps, 2, base_speed)
    vehicle_speed = np.clip(base_speed, 0, 55)

    # Ambient temperature + humidity with diurnal variation (sinusoid over cycle)
    ambient_temp = 24 + 6 * np.sin(2 * np.pi * t / 600) + RNG.normal(0, 1.2, n)
    humidity     = 55 + 15 * np.sin(2 * np.pi * t / 600 + np.pi) + RNG.normal(0, 3, n)
    humidity     = np.clip(humidity, 20, 95)

    # Engine temperature: base + load-driven + ambient + severity-driven
    load_uplift = (load_factor * 12) + ((vehicle_speed / 50) * 5)
    severity_uplift_temp = severity * (
        38 if failure_mode in ["overheat", "bearing"] else 18
    )
    engine_temp = (
        truck["base_engine_temp"] + load_uplift
        + (ambient_temp - 24) * 0.6 + severity_uplift_temp
        + RNG.normal(0, 0.8, n)
    )

    # Coolant temperature tracks engine temperature with damping
    coolant_temp = (
        truck["base_coolant_temp"] + (engine_temp - truck["base_engine_temp"]) * 0.85
        + severity_uplift_temp * 0.6 + RNG.normal(0, 0.6, n)
    )

    # RPM and engine load
    rpm = (
        truck["base_rpm"] * (1 + (vehicle_speed / 50) * 0.3)
        + severity * 120 + RNG.normal(0, 25, n)
    )
    engine_load = np.clip(
        truck["base_engine_load"] + load_factor * 22 + severity * 15 + RNG.normal(0, 2, n),
        0, 100
    )

    # Oil pressure: drops with severity (mechanical wear)
    oil_pressure = np.clip(
        truck["base_oil_pressure"] - severity * (
            2.8 if failure_mode in ["bearing", "hydraulic"] else 1.2
        ) + RNG.normal(0, 0.06, n), 0.4, None
    )

    # Hydraulic pressure: overload when payload is heavy + critical on hydraulic failure
    hydraulic_pressure = (
        truck["base_hydraulic"]
        - load_factor * 18
        - severity * (
            110 if failure_mode == "hydraulic" else 35
        )
        + RNG.normal(0, 5, n)
    )

    # Fuel pressure
    fuel_pressure = (
        truck["base_fuel_pressure"] - severity * 0.8 + RNG.normal(0, 0.08, n)
    )

    # Brake pressure rises when descending / stopping
    brake_pressure = (
        truck["base_brake_pressure"]
        + (1 - load_factor) * 0.4
        + severity * (0.8 if failure_mode == "suspension" else 0.3)
        + RNG.normal(0, 0.1, n)
    )

    # Suspension pressure
    suspension_pressure = np.clip(
        truck["base_suspension"]
        - load_factor * 0.6
        - severity * (
            3.5 if failure_mode == "suspension" else 1.2
        )
        + RNG.normal(0, 0.07, n),
        1.0, None
    )

    # Battery + alternator: drop on electrical failure
    electrical_drop = severity * (4.5 if failure_mode == "electrical" else 0.8)
    battery_voltage = np.clip(
        truck["base_battery"] - electrical_drop + RNG.normal(0, 0.15, n), 18, 30
    )
    alternator_voltage = np.clip(
        truck["base_alternator"] - electrical_drop * 0.7 + RNG.normal(0, 0.2, n), 20, 32
    )
    current_draw = np.clip(
        truck["base_current"] + engine_load * 0.1 + severity * 4 + RNG.normal(0, 0.6, n), 0, 60
    )

    # Vibration: 3-axis with severity-driven rise on bearing failure
    vib_severity = severity * (1.8 if failure_mode == "bearing" else 0.6)
    vibration_x = truck["base_vibration"] + vib_severity * 0.7 + RNG.normal(0, 0.05, n)
    vibration_y = truck["base_vibration"] + vib_severity * 0.8 + RNG.normal(0, 0.05, n)
    vibration_z = truck["base_vibration"] + vib_severity * 0.9 + RNG.normal(0, 0.05, n)
    vibration_rms = np.sqrt(vibration_x**2 + vibration_y**2 + vibration_z**2) / np.sqrt(3)

    # Bearing temperature: rises fast on bearing failure
    bearing_temp = (
        truck["base_bearing_temp"]
        + (engine_temp - truck["base_engine_temp"]) * 0.5
        + severity * (
            45 if failure_mode == "bearing" else (
                25 if failure_mode == "overheat" else 8
            )
        )
        + RNG.normal(0, 1.0, n)
    )
    transmission_temp = (
        truck["base_transmission_temp"]
        + (engine_temp - truck["base_engine_temp"]) * 0.6
        + severity * 18 + RNG.normal(0, 1.0, n)
    )

    # Road gradient (degrees; positive = uphill)
    road_gradient = RNG.normal(0, 4, n) + np.where(is_loaded, 2.5, -1.0)

    # Fuel consumption (L/h)
    fuel_consumption_rate = (
        18 + engine_load * 0.35 + (vehicle_speed / 50) * 8
        + load_factor * 6 + severity * 4 + RNG.normal(0, 0.5, n)
    )

    # Brake temperature: rises with usage
    brake_temp = (
        95 + (1 - load_factor) * 25
        + severity * (45 if failure_mode == "suspension" else 12)
        + RNG.normal(0, 3, n)
    )

    # Operational state
    op_state = np.where(speed_load_dumps, "loading_dumping", "hauling")

    # Engine hours (cumulative within cycle, monotonically increasing)
    engine_hours = (
        truck["engine_age_hours"] + cycle_id * 4.0 + t / 3600.0
    )

    # Idle hours within cycle
    idle_hours = np.cumsum(speed_load_dumps.astype(float) * SAMPLE_INTERVAL_SEC / 3600.0)

    # Health score (0-100, computed inverse from severity + mode weight)
    health_score = np.clip(
        100 - severity * (
            60 if failure_mode == "overheat" else (
                70 if failure_mode == "bearing" else (
                    55 if failure_mode == "hydraulic" else (
                        50 if failure_mode == "electrical" else 45
                    )
                )
            )
        ) - RNG.normal(0, 1.5, n),
        0, 100
    )

    # Carryback percentage (% of payload stuck in bed after dump)
    carryback_pct = np.clip(
        1.2 + severity * 8 + (load_factor * 0.8) + RNG.normal(0, 0.3, n),
        0, 20
    )

    # Maintenance history fields
    previous_failure_count = int(
        min(cycle_id * (truck["failure_proneness"] * 0.4), 15)
    )
    maintenance_age_hours = RNG.uniform(20, 800)  # hours since last service

    # GPS coordinates (simulated region — Jharia coalfields approx)
    gps_lat = 23.75 + RNG.normal(0, 0.01, n)
    gps_lon = 86.42 + RNG.normal(0, 0.01, n)

    # Failure probability (ground truth label) — mirrors severity
    failure_probability = severity

    # Remaining useful life (RUL) — hours until severity ~ 1.0
    HEALTHY_RUL_CEILING_HOURS = 500.0
    rul_hours = np.full(n, HEALTHY_RUL_CEILING_HOURS)
    if will_fail:
        for i in range(ramp_start, n):
            samples_remaining = (n - i) * SAMPLE_INTERVAL_SEC
            rul_hours[i] = samples_remaining / 3600.0

    # Binary label: will this truck fail within the next 1 hour of operation?
    label_failure_within_1hr = (rul_hours <= 1.0).astype(int)

    # ==== Compose into DataFrame ====
    df = pd.DataFrame({
        # A. identity
        "truck_id":              truck["truck_id"],
        "cycle_id":              cycle_id,
        "t_seconds":             t,
        "op_state":              op_state,
        # B. engine
        "engine_temp":           np.round(engine_temp, 2),
        "coolant_temp":          np.round(coolant_temp, 2),
        "rpm":                   np.round(rpm, 1),
        "engine_load_pct":       np.round(engine_load, 2),
        "engine_hours":          np.round(engine_hours, 2),
        "idle_hours":            np.round(idle_hours, 4),
        "fuel_consumption_rate": np.round(fuel_consumption_rate, 2),
        # C. pressure
        "oil_pressure":          np.round(oil_pressure, 2),
        "hydraulic_pressure":    np.round(hydraulic_pressure, 1),
        "fuel_pressure":         np.round(fuel_pressure, 2),
        "brake_pressure":        np.round(brake_pressure, 2),
        "suspension_pressure":   np.round(suspension_pressure, 2),
        # D. electrical
        "battery_voltage":       np.round(battery_voltage, 2),
        "alternator_voltage":    np.round(alternator_voltage, 2),
        "current_draw":          np.round(current_draw, 2),
        # E. vibration / mechanical
        "vibration_x":           np.round(vibration_x, 4),
        "vibration_y":           np.round(vibration_y, 4),
        "vibration_z":           np.round(vibration_z, 4),
        "vibration_rms":         np.round(vibration_rms, 4),
        "bearing_temp":          np.round(bearing_temp, 2),
        "transmission_temp":     np.round(transmission_temp, 2),
        # F. environment
        "ambient_temp":          np.round(ambient_temp, 2),
        "humidity":              np.round(humidity, 1),
        "road_gradient":         np.round(road_gradient, 2),
        "vehicle_speed":         np.round(vehicle_speed, 2),
        # G. payload / operational
        "payload_weight":        np.round(payload_weight, 2),
        "payload_capacity":      np.round(payload_capacity_tonnes, 1),
        "carryback_pct":         np.round(carryback_pct, 2),
        "material_type":         material,
        "brake_temp":            np.round(brake_temp, 2),
        # H. health / context
        "health_score":          np.round(health_score, 2),
        "previous_failure_count": previous_failure_count,
        "maintenance_age_hours": np.round(maintenance_age_hours, 1),
        "operator_id":           operator,
        # I. GPS
        "gps_zone":              zone,
        "gps_lat":               np.round(gps_lat, 5),
        "gps_lon":               np.round(gps_lon, 5),
        # J. labels
        "failure_mode":          failure_mode,
        "failure_probability":   np.round(failure_probability, 4),
        "rul_hours":             np.round(rul_hours, 3),
        "label_failure_within_1hr": label_failure_within_1hr,
    })

    return df


# ---------------------------------------------------------------------------
# Realism augmentations: drift, noise spikes, missing values, post-maintenance
# ---------------------------------------------------------------------------
def apply_sensor_drift(df: pd.DataFrame, truck: dict) -> pd.DataFrame:
    """Gradual per-truck sensor calibration drift (slow, monotonic)."""
    drift_cols = ["oil_pressure", "hydraulic_pressure", "battery_voltage"]
    n = len(df)
    drift_factor = truck["drift_coefficient"]
    for col in drift_cols:
        if col in df.columns:
            # +drift = slow rise over the whole fleet history
            t = np.arange(n)
            df[col] = df[col] + np.sign(RNG.normal()) * drift_factor * t * (
                50 if col == "hydraulic_pressure" else 5
            )
    return df


def inject_outliers_and_missing(df: pd.DataFrame, missing_rate: float = 0.005,
                                outlier_rate: float = 0.002) -> pd.DataFrame:
    """Realistic data-quality issues: missing values + sensor spike outliers."""
    sensor_cols = [
        "engine_temp", "oil_pressure", "hydraulic_pressure",
        "vibration_rms", "battery_voltage",
    ]
    for col in sensor_cols:
        if col not in df.columns:
            continue
        # Missing values (NaN) — independent Bernoulli per row
        miss_mask = RNG.random(len(df)) < missing_rate
        df.loc[miss_mask, col] = np.nan

        # Spike outliers — replace with 3-5x the value
        spike_mask = RNG.random(len(df)) < outlier_rate
        df.loc[spike_mask, col] = df.loc[spike_mask, col] * RNG.uniform(3, 5)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"=== EdgeGuard AI — Enterprise Dataset Generation ===")
    print(f"Target rows: {N_ROWS_TOTAL:,}  (cycles: {N_CYCLES_TOTAL}, "
          f"trucks: {N_TRUCKS}, samples/cycle: {SAMPLES_PER_CYCLE})")
    print()

    trucks = [make_truck_profile(f"truck{i+1}") for i in range(N_TRUCKS)]
    all_dfs = []

    for truck in trucks:
        for cid in range(1, CYCLES_PER_TRUCK + 1):
            df = generate_cycle(truck, cid)
            df = apply_sensor_drift(df, truck)
            df = inject_outliers_and_missing(df)
            all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)

    # Final fill — for missing values, ffill then bfill (within truck/cycle)
    sensor_cols = ["engine_temp", "oil_pressure", "hydraulic_pressure",
                   "vibration_rms", "battery_voltage", "suspension_pressure"]
    for col in sensor_cols:
        if col in df.columns:
            df[col] = df.groupby(["truck_id", "cycle_id"])[col].transform(
                lambda s: s.ffill().bfill()
            )
    # Any residual NaN -> column median
    for col in sensor_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    df.to_csv(OUTPUT_PATH, index=False)

    # --- Report ---
    print(f"Generated {len(df):,} rows x {df.shape[1]} columns")
    print(f"Saved to {OUTPUT_PATH}")
    print()
    print("Class balance (label_failure_within_1hr):")
    pos = (df["label_failure_within_1hr"] == 1).sum()
    print(f"  Positive: {pos:,}  ({100 * pos / len(df):.2f}%)")
    print(f"  Negative: {len(df) - pos:,}  ({100 * (len(df) - pos) / len(df):.2f}%)")
    print()
    print("Failure-mode distribution:")
    print(df["failure_mode"].value_counts().to_string())
    print()
    print("Per-truck row counts:")
    print(df.groupby("truck_id").size().to_string())
    print()
    print("Sensor column sample stats:")
    print(df[sensor_cols].describe().round(2).to_string())


if __name__ == "__main__":
    main()
