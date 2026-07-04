-- EdgeGuard AI - Supabase schema
-- Run this once in the Supabase dashboard: Project > SQL Editor > New Query
-- Paste this whole file, click Run. Creates all 3 tables the backend needs.

CREATE TABLE IF NOT EXISTS sensor_readings (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    truck_id TEXT NOT NULL,
    sensor_type TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT,
    device_ts BIGINT,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS predictions (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    truck_id TEXT NOT NULL,
    component TEXT NOT NULL,
    failure_probability REAL NOT NULL,
    rul_hours REAL,
    model_version TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS alerts (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    truck_id TEXT NOT NULL,
    component TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    sop_reference TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    acknowledged BOOLEAN NOT NULL DEFAULT false
);

CREATE TABLE IF NOT EXISTS sop_documents (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title TEXT NOT NULL,
    component TEXT NOT NULL,
    content_chunk TEXT NOT NULL,
    embedding_ref TEXT
);

-- Helpful indexes for the queries main.py actually runs (filter by
-- truck_id, order by id descending, sometimes filter by sensor_type too).
-- The poller hammers `WHERE truck_id=… ORDER BY id DESC LIMIT 50` every 10s,
-- so a composite index on (truck_id, id DESC) keeps that sub-millisecond
-- even when the table has 100k+ rows.
CREATE INDEX IF NOT EXISTS idx_sensor_readings_truck_id ON sensor_readings (truck_id);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_truck_id_desc
    ON sensor_readings (truck_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_truck_sensor_desc
    ON sensor_readings (truck_id, sensor_type, id DESC);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_sensor_type ON sensor_readings (sensor_type);
CREATE INDEX IF NOT EXISTS idx_predictions_truck_id_desc
    ON predictions (truck_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_truck_id_ack
    ON alerts (truck_id, acknowledged, id DESC);
CREATE INDEX IF NOT EXISTS idx_sop_component ON sop_documents (component);

-- IMPORTANT: Disable Row Level Security for the hackathon so the backend's
-- secret key can read/write freely without extra policy setup. Supabase
-- enables RLS by default on new tables in some project configurations.
-- Re-enabling RLS with real policies is a Phase 2/3 concern, not needed
-- for a 5-day build using the secret key server-side.
ALTER TABLE sensor_readings DISABLE ROW LEVEL SECURITY;
ALTER TABLE predictions DISABLE ROW LEVEL SECURITY;
ALTER TABLE alerts DISABLE ROW LEVEL SECURITY;
ALTER TABLE sop_documents DISABLE ROW LEVEL SECURITY;
