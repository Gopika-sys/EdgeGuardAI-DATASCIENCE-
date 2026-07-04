# EdgeGuard AI — ML Pipeline (Member 4)

Predictive maintenance ML pipeline for the Tata Signa 4825.TK tipper truck.
Trains XGBoost models on synthetic + live Supabase data and exposes a live
prediction poller that integrates with the FastAPI backend.

---

## Files

| File | Purpose |
|---|---|
| `generate_training_data.py` | Generates the synthetic training CSV (40 cycles × 300 rows) |
| `features.py` | Feature engineering: rolling stats, lag, cross-sensor interactions |
| `train.py` | Trains all 3 models + cross-validation + SHAP |
| `inference.py` | `PredictionService` class — loads models + predicts from a sensor buffer |
| `supabase_loader.py` | Pulls training data from Supabase (with CSV fallback) |
| `poller.py` | **Live prediction loop** — polls backend, pushes predictions + alerts |
| `evaluate.py` | Generates a self-contained HTML evaluation report |
| `edgeguard_training_data.csv` | Synthetic training data (40 cycles, ~12k rows) |
| `models/` | Saved model files (classifier, rul_regressor, prob_regressor, metrics) |

---

## Quick Start

### 1. Install dependencies

```bash
cd ml-data
pip install -r requirements.txt
```

### 2. Set up credentials

```bash
cp .env.example .env
# .env already has the Supabase URL + key from the team
```

### 3. Train models (from local CSV)

```bash
python train.py
```

Or retrain from live Supabase data:

```bash
python train.py --source supabase
```

Output:
```
=== EdgeGuard AI — Training (source=csv) ===
Loading data and computing features...
  12,000 rows | 40 features | 40 cycles
  Train: 9,600 rows across 32 cycles
  Test:  2,400 rows across 8 cycles
Running 5-fold GroupKFold cross-validation...
  F1:  0.989 ± 0.004  |  AUC: 0.999 ± 0.001
Training failure classifier...  Precision: 0.999  Recall: 0.981  F1: 0.990
Training RUL regressor...  MAE: 28.4h  R²: 0.921
Training failure probability regressor...  MAE: 0.0182  R²: 0.962
✅ All models + metrics saved to models/
```

### 4. Generate evaluation report

```bash
python evaluate.py
# Opens model_report.html — self-contained, share at the demo
```

### 5. Smoke-test inference

```bash
python inference.py
# Near-failure buffer: failure_probability=0.93, rul_hours=0.1
# Healthy buffer:      failure_probability=0.01, rul_hours=489.2
```

### 6. Run the live poller

> ⚠️ Requires the backend running first: `cd backend && uvicorn main:app --port 8000`

```bash
python poller.py
```

Output every 10 seconds:
```
10:31:22 [INFO] [truck1] prob=0.823 [████████████████░░░░] RUL=6.4h | buf=48 readings 🚨 ALERT (HIGH)
10:31:22 [INFO]          → WARNING: suspension pressure rate of change detected. Failure likely within 6.4 hours.
10:31:32 [INFO] [truck1] prob=0.841 [████████████████░░░░] RUL=5.9h | buf=48 readings 🚨 ALERT (HIGH)
```

---

## Models

Three models are trained and saved to `models/`:

| Model | File | Target | Use |
|---|---|---|---|
| XGBoost Classifier | `classifier.json` | `label_failure_within_1hr` (0/1) | Binary failure gate |
| XGBoost Regressor | `regressor.json` | `rul_hours` | Remaining Useful Life |
| XGBoost Regressor | `prob_regressor.json` | `failure_probability` (0–1) | Smooth probability score |

The **failure_probability** from `prob_regressor.json` is used as the primary score
in `poller.py` (smoother than raw classifier probability). The classifier is used
for the binary `failure_class` output and SHAP explainability.

---

## Features (36 total)

For each of the 6 sensors (`temperature`, `vibration`, `oil_pressure`,
`hydraulic_pressure`, `suspension_pressure`, `battery_voltage`):

- Raw instantaneous value
- `_roll_mean`, `_roll_std`, `_roll_min`, `_roll_max` (window = 10 samples = 20s)
- `_rate_of_change` (diff over window)
- `_lag5` (value 5 samples = 10s ago)

Plus 4 cross-sensor interaction features:

| Feature | Formula | Rationale |
|---|---|---|
| `temp_vib_cross` | temp × vib | Captures simultaneous rise (cascade stage 2) |
| `oil_hyd_ratio` | oil / hydraulic | Drops sharply when oil system fails |
| `temp_oil_stress` | temp / oil | High-temp + low-oil = most dangerous pattern |
| `susp_vib_cross` | vib / suspension | Mechanical wear signature |

---

## Important Design Decisions

**Cycle-level train/test split** — rows within a cycle are 2s apart and nearly
identical. A random row-level split would put duplicates in both train and test,
inflating accuracy. Splitting whole cycles is the only valid split that reflects
real deployment (scoring a truck-life the model has never seen).

**No data leakage** — `compute_features()` is identical in `train.py` and
`inference.py`. The live poller feeds the exact same feature vector the model
was trained on.

**Three models, not one** — The classifier gives a hard yes/no decision.
The `prob_regressor` gives a smooth 0–1 score for the dashboard gauge.
The `rul_regressor` gives actionable hours-until-failure for maintenance scheduling.

---

## Backend Contract

| Backend Endpoint | What the ML poller uses it for |
|---|---|
| `GET /readings/history?truck_id=X&limit=50` | Pull recent readings for inference |
| `GET /readings/history/wide` | Pre-pivoted wide format (optional, faster) |
| `POST /predictions` | Push prediction result |
| `POST /alerts` | Fire alert when failure_probability > 0.70 |
| `GET /ml/status` | Dashboard reads the latest prediction from here |
| `GET /predictions/history?limit=50` | Dashboard trend chart |
