# AI Engine 1 — Computer Vision (YOLOv8)

Detects 4 classes on dump-truck frames: **truck_bed**, **payload**, **carryback**, **hydraulic_ram**.
Emits a 7-field operational-intelligence JSON consumed by `engine3_multimodal.decision_engine`.

## Files

| File | Purpose |
|---|---|
| `synth_frames.py` | Geometric renderer — produces a 640×480 frame + YOLO labels with perfect ground truth |
| `dataset.py` | CLI: build the YOLO dataset (default 500 frames, 80/15/5 split) |
| `data.yaml` | Ultralytics dataset config — keeps `CLASS_MAP` in sync with `synth_frames.py` |
| `train.py` | YOLOv8n fine-tune. Saves `models/best.pt` + `models/metrics.json` |
| `infer.py` | `VisionIntelligence` class — the public API used by `engine3_multimodal` |
| `evaluate.py` | Test-set eval → `reports/vision_report.html` (dark-mode self-contained) |
| `synthetic_frames/` | 3 sample frames for smoke-testing |
| `models/best.pt` | Trained YOLOv8n weights (6.2 MB) |
| `models/metrics.json` | mAP50, mAP50-95, P, R, F1, per-class breakdown |
| `reports/vision_report.html` | Dark-mode evaluation report (open in browser) |

## JSON contract (consumed by engine3)

```json
{
  "carryback_pct":              7.5,    // 0-20
  "payload_efficiency":         0.78,   // 0-1
  "material_occupancy":         0.79,   // 0-1
  "loading_quality":            0.62,   // 0-1
  "hydraulic_stress_indicator": 0.55,   // 0-1
  "material_distribution":      "uniform",  // "uniform" | "left_heavy" | "right_heavy"
  "confidence_score":           0.66    // 0-1, mean of detection confidences
}
```

## How to run (from `ml-data/`)

```bash
# 1. Generate the dataset (one-time, ~10s)
python engine1_vision/dataset.py --n 500

# 2. Fine-tune (CPU, ~3-4 min for 8 epochs at imgsz=320)
python engine1_vision/train.py --epochs 8 --batch 16 --imgsz 320

# 3. Evaluate on the test split
python engine1_vision/evaluate.py

# 4. Smoke-test inference on a single image
python engine1_vision/infer.py engine1_vision/synthetic_frames/frame_0000.jpg
```

## Expected metrics on the synthetic dataset

| Metric | Value |
|---|---|
| mAP50 | ≈ 0.91 |
| mAP50-95 | ≈ 0.70 |
| Precision | ≈ 0.93 |
| Recall | ≈ 0.92 |
| F1 | ≈ 0.91 |

## End-to-end smoke test

Run `python engine3_multimodal/decision_engine.py` — the self-test at
the bottom renders a fresh frame, runs YOLOv8n on it, and shows the
multi-modal decision with the real YOLO output in the `reasoning` field.

## Re-creating the dataset

The 500 generated frames live in `dataset/images/{train,val,test}/`
and `dataset/labels/{train,val,test}/`. They are **not** checked into
git (re-creatable in ~10s with `dataset.py --n 500`). Each
regeneration uses a different RNG seed so subsequent runs train on
slightly different data.
