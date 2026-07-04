-- EdgeGuard AI — Member 3 (Database) Starter Queries
-- Run these one at a time in Supabase SQL Editor, against your live data.
-- Each section builds toward what Member 4 (ML) will need for training.

-- =============================================================
-- 1. SANITY CHECKS — confirm what's actually in the table
-- =============================================================

-- How many readings total, and the time range covered
SELECT
    COUNT(*) AS total_readings,
    MIN(received_at) AS earliest,
    MAX(received_at) AS latest
FROM sensor_readings;

-- Row count per sensor type — should be roughly even across all 6
SELECT sensor_type, COUNT(*) AS row_count
FROM sensor_readings
GROUP BY sensor_type
ORDER BY sensor_type;

-- Latest reading per sensor (mirrors what /readings/latest returns)
SELECT DISTINCT ON (sensor_type)
    sensor_type, value, unit, device_ts, received_at
FROM sensor_readings
WHERE truck_id = 'truck1'
ORDER BY sensor_type, id DESC;


-- =============================================================
-- 2. RAW TIME SERIES — what one sensor looks like over time
-- =============================================================

-- Full vibration history, oldest to newest (good for sanity-plotting)
SELECT device_ts, value, received_at
FROM sensor_readings
WHERE truck_id = 'truck1' AND sensor_type = 'vibration'
ORDER BY device_ts ASC;


-- =============================================================
-- 3. ROLLING WINDOW FEATURES — what Member 4's model actually needs
-- =============================================================
-- These use Postgres window functions to compute rolling mean / stddev /
-- rate of change over the last N readings, per sensor. This is the SQL
-- equivalent of what the working doc's ML section describes as
-- "windowed features — rolling mean, rolling standard deviation, rate of
-- change, min/max over a trailing window."

-- Rolling mean + stddev over the last 10 readings, per sensor
SELECT
    sensor_type,
    device_ts,
    value,
    AVG(value) OVER (
        PARTITION BY sensor_type
        ORDER BY device_ts
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ) AS rolling_mean_10,
    STDDEV(value) OVER (
        PARTITION BY sensor_type
        ORDER BY device_ts
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ) AS rolling_stddev_10
FROM sensor_readings
WHERE truck_id = 'truck1'
ORDER BY sensor_type, device_ts;

-- Rate of change: difference from the previous reading, per sensor
SELECT
    sensor_type,
    device_ts,
    value,
    value - LAG(value) OVER (
        PARTITION BY sensor_type ORDER BY device_ts
    ) AS delta_from_previous
FROM sensor_readings
WHERE truck_id = 'truck1'
ORDER BY sensor_type, device_ts;

-- Rolling min/max over the last 10 readings, per sensor
SELECT
    sensor_type,
    device_ts,
    value,
    MIN(value) OVER (
        PARTITION BY sensor_type
        ORDER BY device_ts
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ) AS rolling_min_10,
    MAX(value) OVER (
        PARTITION BY sensor_type
        ORDER BY device_ts
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ) AS rolling_max_10
FROM sensor_readings
WHERE truck_id = 'truck1'
ORDER BY sensor_type, device_ts;


-- =============================================================
-- 4. ONE-ROW-PER-TIMESTAMP VIEW — wide format, ML-ready shape
-- =============================================================
-- The queries above are "long" format (one row per sensor reading).
-- Member 4's model will likely want "wide" format: one row per moment in
-- time, with all 6 sensors as columns. This pivots the data that way,
-- bucketing readings into 2-second windows (matches the firmware's
-- publish interval) so all 6 sensors line up into the same row.

SELECT
    (device_ts / 2000) * 2000 AS time_bucket_ms,
    truck_id,
    MAX(value) FILTER (WHERE sensor_type = 'temperature') AS temperature,
    MAX(value) FILTER (WHERE sensor_type = 'vibration') AS vibration,
    MAX(value) FILTER (WHERE sensor_type = 'oil_pressure') AS oil_pressure,
    MAX(value) FILTER (WHERE sensor_type = 'hydraulic_pressure') AS hydraulic_pressure,
    MAX(value) FILTER (WHERE sensor_type = 'suspension_pressure') AS suspension_pressure,
    MAX(value) FILTER (WHERE sensor_type = 'battery_voltage') AS battery_voltage
FROM sensor_readings
WHERE truck_id = 'truck1'
GROUP BY time_bucket_ms, truck_id
ORDER BY time_bucket_ms;


-- =============================================================
-- 5. EXPORT-READY: a clean view Member 4 can just SELECT * FROM
-- =============================================================
-- Optional: turn query #4 into a reusable view, so Member 4 (or their
-- Python script) can just run `SELECT * FROM sensor_readings_wide` instead
-- of pasting the pivot query every time.

CREATE OR REPLACE VIEW sensor_readings_wide AS
SELECT
    date_trunc('second', received_at) AS time_bucket,
    truck_id,
    MAX(value) FILTER (WHERE sensor_type = 'temperature') AS temperature,
    MAX(value) FILTER (WHERE sensor_type = 'vibration') AS vibration,
    MAX(value) FILTER (WHERE sensor_type = 'oil_pressure') AS oil_pressure,
    MAX(value) FILTER (WHERE sensor_type = 'hydraulic_pressure') AS hydraulic_pressure,
    MAX(value) FILTER (WHERE sensor_type = 'suspension_pressure') AS suspension_pressure,
    MAX(value) FILTER (WHERE sensor_type = 'battery_voltage') AS battery_voltage
FROM sensor_readings
GROUP BY time_bucket, truck_id
ORDER BY time_bucket;

-- After creating the view, test it:
SELECT * FROM sensor_readings_wide LIMIT 20;
