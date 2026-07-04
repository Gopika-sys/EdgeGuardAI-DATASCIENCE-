# EdgeGuard AI

Predictive maintenance system for Tata Signa 4825.TK tipper trucks — 5-day hackathon build.

## Team

| Member | Role |
|---|---|
| Member 1 | Frontend Developer |
| Member 2 | Full-Stack Developer (backend, MQTT, Phase 1 simulation support) |
| Member 3 | Database Handler |
| Member 4 | Machine Learning Engineer |
| Member 5 | Sensor Handler |

## Repo structure

```
EdgeGuardAi/
├── wokwi-sim/          Member 5 + Member 2 — IoT simulation (Wokwi ESP32 project)
│   ├── diagram.json
│   ├── sketch.ino
│   ├── libraries.txt
│   └── test_subscriber.py
├── backend/            Member 2 — FastAPI + MQTT listener + Supabase storage
│   ├── main.py
│   ├── requirements.txt
│   ├── .env.example
│   ├── supabase_schema.sql
│   ├── member3_queries.sql
│   ├── test_logic.py
│   └── README.md
├── ml-data/            Member 3 + Member 4 — ML intelligence layer
│   ├── train.py                 # AI Engine 2: XGBoost + RUL + IsolationForest
│   ├── inference.py             # AI Engine 2: prediction service
│   ├── evaluate.py              # AI Engine 2: HTML report generator
│   ├── features.py              # AI Engine 2: feature engineering
│   ├── poller.py                # bridges backend → ML → predictions
│   ├── rag.py                   # Knowledge layer: TF-IDF over SOPs
│   ├── build_rag_index.py
│   ├── sop_data.json            # 20 industrial mining SOPs
│   ├── generate_training_data.py
│   ├── edgeguard_training_data.csv  # 25K-row synthetic dataset
│   ├── README.md
│   ├── engine1_vision/          # AI Engine 1: YOLOv8n CV pipeline
│   │   ├── synth_frames.py      # synthetic frame renderer (geometric GT)
│   │   ├── dataset.py           # YOLO dataset builder
│   │   ├── train.py             # YOLOv8n fine-tune
│   │   ├── infer.py             # VisionIntelligence public API
│   │   ├── evaluate.py          # vision_report.html generator
│   │   ├── data.yaml            # Ultralytics dataset config
│   │   ├── models/best.pt       # trained weights (6.2 MB)
│   │   ├── models/metrics.json
│   │   ├── reports/vision_report.html
│   │   └── synthetic_frames/    # 3 sample frames
│   ├── engine3_multimodal/      # AI Engine 3: fuse AI1 + AI2 + history
│   │   └── decision_engine.py   # fuse_and_decide() → DecisionOutput
│   ├── engine4_business/        # AI Engine 4: INR/ROI
│   │   └── business_impact.py   # compute_business_impact()
│   ├── copilot/                 # AI Engine 5: Gemini LLM
│   │   └── gemini_copilot.py    # explain_prediction(), recommend_repair()
│   ├── models/                  # Engine 2 trained artefacts
│   └── reports/
├── frontend/           Member 1 — 5-tab command center (static HTML/JS, served by `python -m http.server`)
│   ├── index.html          # Main entry point — 5 views
│   ├── app.js              # ~1200 LOC, 5 AI engine integrations
│   ├── index.css           # Premium dark theme
│   └── digital_twin.html   # Standalone twin page (deep-link to the twin)
├── docs/               Working documents, source PDFs
│   ├── ml_work.txt                # original mentor spec
│   └── EdgeGuard_AI_Working_Document.pdf
├── RUN.md             End-to-end bring-up guide (read this first!)
└── README.md          (this file)
```

## First-time setup (everyone, do this once)

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/EdgeGuardAi.git
   cd EdgeGuardAi
   ```
2. Get the real Supabase credentials from Abi (team chat, NOT git) and create your own `.env`:
   ```bash
   cd backend
   cp .env.example .env
   # then open .env and paste the real SUPABASE_URL / SUPABASE_KEY values
   ```
3. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running things

- **Wokwi simulation:** open wokwi.com, paste in `wokwi-sim/diagram.json` and `wokwi-sim/sketch.ino`, install libraries from `wokwi-sim/libraries.txt`, hit play. See `backend/README.md` for network notes (some campus WiFi blocks MQTT's port — use a mobile hotspot if `[mqtt] connection error: timed out` shows up).
- **Backend:** `cd backend && python -m uvicorn main:app --reload --port 8000`
- **Check live data:** `http://localhost:8000/readings/latest` or browse the `sensor_readings` table directly in the Supabase dashboard.

## Git workflow (keep this simple for a 5-day hackathon)

- `main` branch is always the working version — don't push broken code straight to `main`.
- Each person works in their own folder (`wokwi-sim/`, `backend/`, `ml-data/`, etc.) most of the time, which keeps merge conflicts rare.
- For anything touching shared files (like `backend/main.py`), pull before you start, and push small commits often rather than one giant commit at the end.
- Quick workflow:
  ```bash
  git pull
  # ... make changes ...
  git add .
  git commit -m "short description of what changed"
  git push
  ```

## Status

- ✅ Wokwi simulation — working, verified, auto-ramp failure scenario confirmed
- ✅ Backend (FastAPI + MQTT + Supabase) — working, verified end-to-end
- ✅ Database schema + synthetic training data — live in Supabase (`training_data` table, 25,000 rows)
- ✅ ML training (XGBoost classifier + RUL regressor + Isolation Forest) — see `ml-data/models/`
- ✅ Frontend — 5-tab command center: Command Center / Digital Twin / Fleet / ROI / Maintenance / Settings
- ✅ RAG / SOP retrieval layer — `ml-data/rag.py` + `/sops/*` endpoints
- ✅ Multi-modal decision engine (engine3) — fuses AI1 + AI2 + history
- ✅ Business impact engine (engine4) — INR/ROI calculation, **live in the ROI tab**
- ✅ Gemini Copilot (engine5) — natural-language explanations, **live in Command Center**
- ✅ Computer vision (engine1) — YOLOv8n, 4-class detection, 91% mAP50 on test set
- ✅ Demo mode — 90-second simulated degradation cycle when backend is offline, so the dashboard never breaks in front of judges

## Hackathon pitch

See **`DEMO_PITCH.md`** for the 30-second pitch, the ROI story, and a
90-second demo script.
