# EdgeGuard AI — Backend (Member 2)

FastAPI service with a background MQTT listener. Subscribes to the same
broker the Wokwi simulation publishes to, cleans incoming readings, stores
them in **Supabase (hosted Postgres)**, and exposes REST endpoints for the
dashboard (Member 1) and ML service (Member 4).

## Why Supabase instead of SQLite

SQLite is a single file on one laptop — only that laptop can see the data.
Supabase is a hosted Postgres database, so all 5 team members can read and
write the same live data from their own machines at the same time. This
matters once more than one person needs to query sensor data (dashboard
dev, ML dev, DB dev all working in parallel).

## One-time setup (do this first, before anyone runs the backend)

1. Go to https://supabase.com and sign up (free).
2. Create a new project. Wait ~2 minutes for it to spin up.
3. Open the **SQL Editor** tab, paste the entire contents of
   `supabase_schema.sql`, and click Run. This creates the 3 tables
   (`sensor_readings`, `predictions`, `alerts`) the backend needs.
4. Go to **Project Settings > API Keys**. Copy:
   - Your **Project URL** (looks like `https://xxxxx.supabase.co`)
   - Your **secret key** (starts with `sb_secret_`) — NOT the old
     `service_role` key; Supabase is moving to this new naming.
5. Share these two values with the whole team (e.g. in your team chat) —
   everyone's backend instance points at the same Supabase project.

## Setup (each team member, on their own laptop)

```bash
pip install fastapi uvicorn paho-mqtt supabase
```

Set your Supabase credentials as environment variables before running:

**Windows (Git Bash):**
```bash
export SUPABASE_URL="https://xxxxx.supabase.co"
export SUPABASE_KEY="sb_secret_xxxxxxxxxxxx"
```

**Windows (Command Prompt):**
```cmd
set SUPABASE_URL=https://xxxxx.supabase.co
set SUPABASE_KEY=sb_secret_xxxxxxxxxxxx
```

Alternatively, just paste your URL/key directly into the `SUPABASE_URL` /
`SUPABASE_KEY` constants near the top of `main.py` — simplest for a 5-day
hackathon, just don't push real keys to a public GitHub repo.

## Run

```bash
uvicorn main:app --reload --port 8000
```

Then open http://localhost:8000/docs for interactive Swagger API docs.

Make sure the Wokwi simulation is running first (or at the same time) —
this service connects out to `broker.hivemq.com:1883` and subscribes to
`edgeguard/+/+`.

**Network note:** if `[mqtt] connection error: timed out` appears, your
network is likely blocking port 1883. Switch to a mobile hotspot — this
was confirmed to fix it during development. Supabase itself runs over
HTTPS (port 443), so it should work fine even on restrictive networks.

## Verify it's working

1. Start this service.
2. Start/keep running the Wokwi simulation.
3. Watch the terminal for `[mqtt] connected to broker.hivemq.com`.
4. Hit `GET http://localhost:8000/readings/latest` — you should see all 6
   sensors with their most recent value.
5. Open your Supabase project's **Table Editor** tab in the browser and
   look at the `sensor_readings` table directly — you should see rows
   appearing in real time. This is also a great way for Member 3
   (Database) to confirm data without needing to run any code at all.

## Testing the cleaning logic without the full stack

`test_logic.py` exercises the cleaning logic directly (still using a local
SQLite file purely for this isolated test — it does NOT touch Supabase).
Useful for a fast sanity check of the validation/clipping rules without
needing any network access:

```bash
python3 test_logic.py
```

## Key endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/readings/latest` | Latest value per sensor — for live dashboard view |
| GET | `/readings/history?sensor=X&limit=N` | Recent stored readings |
| GET | `/buffer/status` | How many readings are buffered due to a DB write failure (demo connectivity-drop resilience) |
| POST | `/predictions` | ML service pushes a failure probability + RUL here |
| GET | `/predictions/latest` | Recent predictions |
| POST | `/alerts` | Create an alert (e.g. once failure probability crosses threshold) |
| GET | `/alerts/active` | Unacknowledged alerts — dashboard polls this |
| POST | `/alerts/{id}/acknowledge` | Mark an alert as handled |

## Notes for Member 3 (Database) and Member 4 (ML)

- Member 3 can browse/query the `sensor_readings`, `predictions`, and
  `alerts` tables directly in the Supabase Table Editor — no Python
  required to just look at the data.
- Member 4's inference service should call `GET /readings/history` to pull
  recent windows for feature engineering, then `POST /predictions` with the
  result. Member 4 can also query Supabase directly with the same
  `supabase-py` client if they'd rather not go through the REST API.
- If failure_probability crosses your chosen threshold, also `POST /alerts`
  so the dashboard can surface it immediately.

## Honest status of what's been tested

- Cleaning/validation logic: tested directly, passes against real Wokwi
  payload shapes.
- MQTT connection + full Wokwi → backend → SQLite pipeline: confirmed
  working end-to-end on a real laptop (see project history).
- Supabase storage layer specifically: written carefully against current
  Supabase Python client docs, but not yet run end-to-end at the time of
  writing — run through the "Verify it's working" steps above and report
  back if anything errors, the same way the MQTT port issue was found and
  fixed.
