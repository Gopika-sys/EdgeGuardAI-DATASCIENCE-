# EdgeGuard AI — Bring-Up Guide

End-to-end startup sequence for the live demo. 5 services total, ~3 minutes
from a clean clone to a populated dashboard.

> **TL;DR** — start the **backend** first, then the **poller**, then the
> **frontend**, then the **Wokwi sim** (browser). The Wokwi sim can be started
> any time after the backend.

---

## 0. Prerequisites (one-time)

| Tool | Version | Install |
|---|---|---|
| Python | 3.10+ | https://python.org |
| pip packages | — | see below |
| Browser | any | for Wokwi sim + dashboard |
| Supabase project | free tier | https://supabase.com |

```bash
# from repo root
pip install -r backend/requirements.txt
pip install -r ml-data/requirements.txt

# copy env files (already filled in with the team Supabase creds)
cp backend/.env.example backend/.env
cp ml-data/.env.example  ml-data/.env
```

> ⚠️ The `.env` files contain shared team credentials. **Do not commit them.**
> `.env` is already in `.gitignore`.

---

## 1. Database (one-time, per Supabase project)

Open the Supabase dashboard → Project → SQL Editor → New Query.
Paste and run the contents of `backend/supabase_schema.sql`.

This creates 4 tables: `sensor_readings`, `predictions`, `alerts`, `sop_documents`
+ indexes + disables RLS (hackathon-grade; tighten for production).

---

## 2. Train the ML models (one-time, or after data refresh)

```bash
cd ml-data

# Option A: train from the bundled synthetic CSV (fast, ~30s)
python train.py

# Option B: retrain from live Supabase sensor_readings (slower, more realistic)
python train.py --source supabase
```

Output goes to `ml-data/models/`:
- `classifier.json` — XGBoost binary failure gate (F1 ≈ 0.99)
- `regressor.json` — RUL in hours (MAE ≈ 12h on near-failure)
- `prob_regressor.json` — smooth 0–1 failure probability (MAE ≈ 0.018)
- `feature_columns.json` — exact feature order (must match at inference time)
- `metrics.json` — confusion matrix, R², top SHAP features

After training, rebuild the RAG index if you edited `sop_data.json`:

```bash
python build_rag_index.py   # → models/rag_index.pkl
```

Generate a shareable HTML evaluation report (optional, for the pitch deck):

```bash
python evaluate.py
# opens model_report.html
```

---

## 2b. Computer Vision engine (one-time, or after data refresh)

AI Engine 1 is a YOLOv8n pipeline that detects 4 classes on dump-truck
frames: `truck_bed`, `payload`, `carryback`, `hydraulic_ram`. Output is
a 7-field JSON consumed by the Multi-Modal Decision engine (engine3).

```bash
cd ml-data/engine1_vision

# 1) Generate 500 synthetic frames + YOLO labels (~10s)
python dataset.py --n 500

# 2) Fine-tune YOLOv8n (~3-4 min on CPU, 8 epochs)
python train.py --epochs 8 --batch 16

# 3) Evaluate on the test split → vision_report.html
python evaluate.py

# 4) Smoke-test the inference API
python infer.py synthetic_frames/frame_0000.jpg
```

Outputs:
- `engine1_vision/models/best.pt` — YOLOv8n weights (6.2 MB)
- `engine1_vision/models/metrics.json` — mAP50, P, R, F1, per-class
- `engine1_vision/reports/vision_report.html` — dark-mode self-contained report

Expected metrics on the synthetic dataset: **mAP50 ≈ 0.91, F1 ≈ 0.91**.

The full pipeline (CV → engine3 fusion → business impact) is verified
by running `python engine3_multimodal/decision_engine.py` — the
self-test at the bottom of that file renders a fresh frame, runs
YOLOv8n, and shows the fused decision with the real YOLO output in
the `reasoning` field.

---

## 3. Backend (start first)

```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --log-level info
```

What happens on startup:
1. Connects to Supabase (uses `SUPABASE_URL` + `SUPABASE_KEY` from `.env`).
2. Connects to the public MQTT broker `broker.hivemq.com:1883` and subscribes
   to `edgeguard/+/+`.
3. Lazy-loads the RAG index on first `/sops/*` request.

Health check:
```bash
curl http://localhost:8000/                  # → {"status":"ok",...}
curl http://localhost:8000/readings/latest    # latest cleaned reading per sensor
curl http://localhost:8000/buffer/status      # pending offline-buffered readings
```

Interactive API docs: <http://localhost:8000/docs>

---

## 4. ML live poller (start second, ~10s after backend)

```bash
cd ml-data
python poller.py
```

Output every 10 s:
```
13:14:42 [INFO] EdgeGuard AI poller started.
13:14:52 [INFO] [truck1] prob=0.39 [███████░░░░░░░░░░░░░] RUL=0.0h | buf=50
13:15:02 [INFO] [truck1] prob=0.41 [████████░░░░░░░░░░░] RUL=0.0h | buf=50
```

Behavior:
- Pulls last 50 readings from `GET /readings/history`.
- Runs the 3-model inference pipeline.
- **Always** `POST /predictions`.
- **Conditionally** `POST /alerts` if `failure_probability > ALERT_THRESHOLD`
  (default 0.70, configurable in `ml-data/.env`).
- Appends a row to `predictions_log.jsonl` for offline review.

Tunable env vars (`ml-data/.env`):
| Var | Default | Meaning |
|---|---|---|
| `BACKEND_URL` | `http://localhost:8000` | Where to read from + write to |
| `TRUCK_ID` | `truck1` | Which truck to monitor |
| `ALERT_THRESHOLD` | `0.70` | Min prob to fire an alert |
| `POLL_INTERVAL_SECONDS` | `10` | Seconds between cycles |
| `HISTORY_LIMIT` | `50` | Rows pulled per cycle |
| `COMPONENT_NAME` | `hydraulic_system` | Component tag for predictions |
| `MODEL_VERSION` | `v2` | Version string for predictions |

---

## 5. Frontend dashboard (start third)

```bash
cd frontend
python -m http.server 3000 --bind 0.0.0.0
```

Open <http://localhost:3000>.

**Six views (one command center, left-rail navigation):**
- **Command Center** — live sensor sparklines, ML gauge + RUL + component,
  failure-probability trend chart (last 50 predictions), AI Copilot
  explanations, active alerts feed, fleet-health summary card
- **Digital Twin** — animated SVG tipper truck with 6 sensor nodes,
  cascade arrows showing degradation spreading, tipper bed raises at
  critical, exhaust particles when engine is hot, click any node to
  inspect history + SOP
- **Fleet View** — 3 trucks monitored in parallel, click a truck to
  switch active truck, fleet aggregates (avg health, alerts, at-risk)
- **ROI / Business** — big animated **X-RI** number, ₹-amounts (Revenue
  protected, Maint saved, Fuel saved), 6 KPI cards, cost-savings bar
  chart, full assumptions table — all from Engine 4
- **Maintenance** — RAG search + 20 SOPs + alert → SOP timeline (kept
  from v1, plus the AI engine status strip in the sidebar)
- **Settings** — backend URL, poll interval, alert threshold, demo-mode
  toggle, business assumptions, system info — all persisted to
  `localStorage`

The dashboard polls three backend endpoints every 2–5 s:
- `GET /readings/latest` (live sparklines)
- `GET /ml/status` (gauge + RUL)
- `GET /alerts/active` (alerts list)
- `GET /predictions/history` (failure-probability trend)
- `GET /readings/history/wide` (on first load — fills charts with history)

**Standalone Digital Twin:** `frontend/digital_twin.html` is also
served at <http://localhost:3000/digital_twin.html> — a deep-link to
just the twin view without the surrounding app shell. Useful for
embedding in slide decks / live demos.

The Maintenance tab lazy-loads on first switch:
- `GET /sops` (all 20 SOPs)
- `GET /sops/components` (filter pill labels)
- `GET /sops/search?q=…` (semantic search via TF-IDF cosine similarity)
- `GET /maintenance/summary` (timeline + stats)

---

## 6. Wokwi IoT simulation (start last, in the browser)

1. Go to <https://wokwi.com/projects/new/esp32>.
2. Open the `wokwi-sim/` folder from this repo. You need three files:
   - `diagram.json` — schematic (paste into the JSON tab)
   - `sketch.ino` — Arduino firmware (paste into the `sketch.ino` tab)
   - `libraries.txt` — install these via Wokwi's library manager
3. Hit play. The sim connects to `broker.hivemq.com` and starts publishing
   6 sensor topics under `edgeguard/truck1/`.
4. **Failure scenario**: at T+60 s the sim starts a 90 s auto-ramp that
   pushes the readings into a critical state (temperature up, vibration up,
   oil pressure down).

You should see readings start appearing in the backend within 2–5 s.

> 📡 **WiFi gotchas:** some campus/office networks block MQTT's port 1883.
> If you see `[mqtt] connection error: timed out` in `backend.log`, switch
> the laptop to a mobile hotspot.

> 🎬 **For the demo**: instead of running Wokwi in a browser, a second
> laptop can run `wokwi-sim/test_subscriber.py` to verify the broker is
> receiving data. For the **real** vehicle, replace the Wokwi firmware
> with the actual ESP32 + sensors sketch — the topic schema is identical.

---

## 7. Quick verification

After everything is up, run this one-liner to confirm the full data path:

```bash
echo "=== Sensor → Backend → Poller → Alerts ==="
curl -s http://localhost:8000/readings/latest
echo ""
curl -s http://localhost:8000/ml/status
echo ""
curl -s http://localhost:8000/alerts/active | head -c 400
echo ""
echo "=== RAG retrieval ==="
curl -s "http://localhost:8000/sops/search?q=hydraulic+overload" | head -c 400
```

Expected: live readings from the Wokwi sim, non-null ML gauge, an empty
alerts list (or one if the Wokwi sim is past T+90 s), and a relevant SOP
matching the query.

---

## 8. Operational notes

### Buffering / connectivity drop

If `insert_reading` fails (Supabase is briefly unreachable), the reading
goes into an in-memory buffer. On the next successful write, the buffer
flushes automatically. Inspect it:

```bash
curl http://localhost:8000/buffer/status
# → {"pending_count": 0}
```

This is how the system simulates the "vehicle drives through a tunnel"
resilience scenario for the demo — kill network for 30 s, restore, and
the buffer will drain.

### Restart order

If you change code, restart in this order:
1. Backend (`Ctrl+C` + re-launch)
2. Poller (`Ctrl+C` + re-launch — picks up new model files automatically)
3. Frontend: just refresh the browser tab (no restart needed)

### Ports used

| Port | Service | Notes |
|---|---|---|
| `3000` | Frontend (static) | `python -m http.server` |
| `8000` | Backend API | FastAPI + uvicorn |
| `1883` | MQTT (outbound to `broker.hivemq.com`) | No local listener |

### Logs

| Service | Log file | Where |
|---|---|---|
| Backend | `backend/backend.log` | repo |
| Poller | `ml-data/poller.log` + `poller.err.log` | repo |
| Predictions history | `ml-data/predictions_log.jsonl` | append-only |
| Frontend | `frontend.log` | repo |

### Common errors

| Symptom | Cause | Fix |
|---|---|---|
| `supabase: Not Found` on every insert | Wrong project URL or key in `.env` | Re-paste from Supabase dashboard → Settings → API |
| `RAG index not found` warning on `/sops` | `models/rag_index.pkl` missing | `cd ml-data && python build_rag_index.py` |
| Poller log shows `No readings returned` | Backend not running, or Wokwi not playing | Check backend is up; check Wokwi sim is hitting play |
| MQTT timeout on Wokwi | Campus WiFi blocks port 1883 | Use mobile hotspot |

---

## 9. Process map (for the demo)

```
┌──────────────┐  MQTT    ┌──────────────┐   HTTPS    ┌──────────────┐
│  Wokwi sim   │ ───────► │  Backend     │ ────────── │  Poller      │
│  (browser)   │ port 1883│  :8000       │            │  (python)    │
└──────────────┘          └──────┬───────┘            └──────┬───────┘
                                │                            │
                                │  REST (readings,           │  POST
                                │  /ml/status, /sops/*)      │  /predictions
                                │                            │  /alerts
                                ▼                            ▼
                          ┌──────────────┐            ┌──────────────┐
                          │  Frontend    │            │  Supabase    │
                          │  :3000       │            │  (Postgres)  │
                          └──────────────┘            └──────────────┘
```
