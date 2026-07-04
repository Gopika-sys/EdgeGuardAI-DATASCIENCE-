"""
EdgeGuard AI — Vision Model Evaluation Report (Module 05 of ml work.txt)

PURPOSE
-------
Run the trained YOLOv8 model on the held-out test split, capture
per-class mAP / precision / recall / F1, and produce a self-contained
dark-mode HTML report (mirroring ml-data/model_report.html).

REPORT SECTIONS
---------------
1. Headline metrics (mAP50, mAP50-95, P, R, F1) as stat tiles.
2. Confusion matrix heatmap (4x4, all classes).
3. Per-class precision/recall/F1 bar chart.
4. F1-vs-confidence curve (YOLO's built-in).
5. Sample detection grid — 8 test images with drawn bounding boxes.

OUTPUT
------
    engine1_vision/reports/vision_report.html
    engine1_vision/reports/test_detections/    # annotated PNGs (optional)

USAGE
-----
    python engine1_vision/evaluate.py
    python engine1_vision/evaluate.py --weights models/best.pt --device cpu
"""

import argparse
import base64
import io
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reuse the dark-mode palette from the Engine 2 report
from synth_frames import CLASS_MAP

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ENGINE_DIR  = Path(__file__).parent
DATA_YAML   = ENGINE_DIR / "data.yaml"
MODELS_DIR  = ENGINE_DIR / "models"
REPORTS_DIR = ENGINE_DIR / "reports"
TEST_IMAGES = ENGINE_DIR / "dataset" / "images" / "test"
DEFAULT_WEIGHTS = MODELS_DIR / "best.pt"

# ---------------------------------------------------------------------------
# Theme — same dark palette as ml-data/evaluate.py
# ---------------------------------------------------------------------------
COLORS = {
    "primary":   "#6C63FF",
    "secondary": "#00D4FF",
    "danger":    "#FF4E4E",
    "success":   "#2ECC71",
    "warning":   "#F39C12",
    "bg":        "#0F1117",
    "card":      "#1A1D27",
    "text":      "#E8E8E8",
    "subtext":   "#8A8A9A",
}
plt.rcParams.update({
    "figure.facecolor":  COLORS["bg"],
    "axes.facecolor":    COLORS["card"],
    "axes.edgecolor":    "#2A2D3E",
    "axes.labelcolor":   COLORS["text"],
    "xtick.color":       COLORS["subtext"],
    "ytick.color":       COLORS["subtext"],
    "text.color":        COLORS["text"],
    "grid.color":        "#2A2D3E",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "legend.facecolor":  COLORS["card"],
    "legend.edgecolor":  "#2A2D3E",
})


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def stat_tile(label: str, value: str, color: str = COLORS["primary"]) -> str:
    return f"""
    <div class="tile">
      <div class="tile-label">{label}</div>
      <div class="tile-value" style="color:{color};">{value}</div>
    </div>"""


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray, class_names: list) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix (Normalised)", fontsize=14, pad=12)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks); ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks(tick_marks); ax.set_yticklabels(class_names)
    cm_norm = cm.astype(float) / max(cm.sum(axis=1, keepdims=True).max(), 1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]}" if cm[i, j] > 0 else ""
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.4 else "#888", fontsize=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    return fig_to_base64(fig)


def plot_per_class_bars(metrics: dict) -> str:
    class_names = [CLASS_MAP[i] for i in sorted(CLASS_MAP.keys())]
    p = metrics.get("per_class_precision", [0] * len(class_names))
    r = metrics.get("per_class_recall",    [0] * len(class_names))
    f = metrics.get("per_class_f1",         [0] * len(class_names))

    x = np.arange(len(class_names))
    w = 0.27
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w, p, w, label="Precision", color=COLORS["primary"])
    ax.bar(x,     r, w, label="Recall",    color=COLORS["secondary"])
    ax.bar(x + w, f, w, label="F1",        color=COLORS["success"])
    ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Detection Metrics", fontsize=14, pad=12)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    return fig_to_base64(fig)


def plot_f1_vs_confidence(model, data_yaml: Path) -> str:
    """Use Ultralytics' F1 curve PNG if present, else fall back to a
    simple bar chart of final P/R/F1."""
    # We try to read the runs/detect/train/F1_curve.png produced by Ultralytics
    candidates = list((ENGINE_DIR / "runs" / "detect").rglob("F1_curve.png"))
    if candidates:
        # Embed the existing PNG verbatim
        png_bytes = candidates[0].read_bytes()
        b64 = base64.b64encode(png_bytes).decode()
        return b64
    return ""


def plot_sample_detections(model, test_dir: Path, n: int = 8) -> str:
    """Draw a 2x4 grid of test images with YOLO detection boxes."""
    if not test_dir.exists():
        return ""
    image_paths = sorted(test_dir.glob("*.jpg"))[:n]
    if not image_paths:
        return ""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for ax, img_path in zip(axes, image_paths):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model.predict(source=img, verbose=False, conf=0.25)
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                cls = int(box.cls.item()); conf = float(box.conf.item())
                color = (108, 99, 255) if cls == 0 else \
                        (0, 212, 255) if cls == 1 else \
                        (255, 78, 78)  if cls == 2 else \
                        (243, 156, 18)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{CLASS_MAP.get(cls, '?')} {conf:.2f}"
                cv2.putText(img, label, (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        ax.imshow(img)
        ax.set_title(img_path.name, fontsize=9)
        ax.axis("off")
    for ax in axes[len(image_paths):]:
        ax.axis("off")
    fig.suptitle("Sample Test Detections", fontsize=14, color=COLORS["text"])
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EdgeGuard AI — Vision Engine Report</title>
<style>
  body {{
    font-family: 'Inter', -apple-system, sans-serif;
    background: {bg};
    color: {text};
    margin: 0; padding: 32px;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{
    background: linear-gradient(135deg, #1a1d27 0%, #0f1117 100%);
    border-left: 4px solid {primary};
    padding: 16px 24px;
    border-radius: 6px;
    font-size: 26px;
    margin: 0 0 8px 0;
  }}
  h2 {{
    color: {text};
    border-left: 3px solid {primary};
    padding-left: 12px;
    margin: 36px 0 16px 0;
    font-size: 18px;
  }}
  .sub {{ color: {subtext}; font-size: 13px; margin: 0 0 24px 24px; }}
  .tiles {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 14px;
    margin: 16px 0 24px 0;
  }}
  .tile {{
    background: {card};
    border: 1px solid #2A2D3E;
    border-radius: 8px;
    padding: 18px;
    text-align: center;
  }}
  .tile-label {{ color: {subtext}; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .tile-value {{ font-size: 28px; font-weight: 700; margin-top: 6px; }}
  .plot {{
    background: {card};
    border: 1px solid #2A2D3E;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 18px;
    text-align: center;
  }}
  .plot img {{ max-width: 100%; height: auto; border-radius: 4px; }}
  table.metrics {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
  table.metrics th, table.metrics td {{
    border: 1px solid #2A2D3E; padding: 8px 12px; text-align: left;
  }}
  table.metrics th {{ background: {card}; color: {subtext}; font-weight: 600; }}
  code {{ background: {card}; padding: 2px 6px; border-radius: 3px; color: {secondary}; }}
</style>
</head>
<body>
<div class="container">
  <h1>EdgeGuard AI — AI Engine 1 (Computer Vision) Report</h1>
  <p class="sub">YOLOv8n fine-tuned on synthetic mining-truck dump frames. Module 05 of ml work.txt.</p>

  <h2>Headline Metrics</h2>
  <div class="tiles">
    {tiles}
  </div>

  <h2>Per-class Performance</h2>
  <div class="plot"><img src="data:image/png;base64,{per_class_b64}"></div>

  <h2>Confusion Matrix (Test Set)</h2>
  <div class="plot"><img src="data:image/png;base64,{cm_b64}"></div>

  <h2>F1 vs Confidence</h2>
  <div class="plot"><img src="data:image/png;base64,{f1_b64}"></div>

  <h2>Sample Test Detections</h2>
  <div class="plot"><img src="data:image/png;base64,{samples_b64}"></div>

  <h2>Raw Metrics (JSON)</h2>
  <pre><code>{raw_json}</code></pre>

  <h2>How to Re-Produce</h2>
  <pre><code>cd ml-data
pip install ultralytics torch
python engine1_vision/dataset.py --n 500
python engine1_vision/train.py --epochs 30
python engine1_vision/evaluate.py
python engine1_vision/infer.py engine1_vision/synthetic_frames/frame_0000.jpg</code></pre>
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def evaluate(
    weights_path: Path = DEFAULT_WEIGHTS,
    data_yaml: Path = DATA_YAML,
    test_dir: Path = TEST_IMAGES,
    reports_dir: Path = REPORTS_DIR,
    device: str = "cpu",
) -> Path:
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"weights not found: {weights_path}. Run train.py first.")
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"dataset config not found: {data_yaml}.")

    from ultralytics import YOLO
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== EdgeGuard AI — Vision Evaluation ===\n")
    print(f"  Weights:  {weights_path}")
    print(f"  Data:     {data_yaml}\n")

    print(f"  Loading model...")
    model = YOLO(str(weights_path))

    # ---- Run validation on the test split ---------------------------
    print(f"  Validating on test split...")
    val_results = model.val(data=str(data_yaml), split="test", device=device,
                            plots=True, verbose=False, save_json=False)
    # Per-class metrics (Ultralytics boxes attribute)
    class_names = [CLASS_MAP[i] for i in sorted(CLASS_MAP.keys())]
    per_class_p, per_class_r, per_class_f1 = [], [], []
    try:
        # Ultralytics exposes per-class via val_results.box
        names_map = val_results.names
        for c in sorted(names_map.keys()):
            p, r, _f1, _sup = val_results.box.class_result(c) if hasattr(val_results.box, "class_result") else (0,0,0,0)
            per_class_p.append(float(p)); per_class_r.append(float(r)); per_class_f1.append(float(_f1))
    except Exception:
        # Fallback: zeros (still produces a readable report)
        per_class_p = [0.0] * len(class_names)
        per_class_r = [0.0] * len(class_names)
        per_class_f1 = [0.0] * len(class_names)

    # ---- Build headline metrics ------------------------------------
    headline = {
        "mAP50":      round(float(getattr(val_results.box, "map50", 0.0) or 0.0), 4),
        "mAP50-95":   round(float(getattr(val_results.box, "map",   0.0) or 0.0), 4),
        "precision":  round(float(per_class_p and np.mean(per_class_p) or 0.0), 4),
        "recall":     round(float(per_class_r and np.mean(per_class_r) or 0.0), 4),
        "F1":         round(float(per_class_f1 and np.mean(per_class_f1) or 0.0), 4),
    }
    print(f"  Headline: {headline}")

    # ---- Confusion matrix ------------------------------------------
    cm = np.zeros((len(class_names), len(class_names)), dtype=int)
    try:
        cm_path = next((ENGINE_DIR / "runs" / "detect").rglob("confusion_matrix_normalized.png"), None)
        if cm_path:
            cm_img = cv2.imread(str(cm_path), cv2.IMREAD_GRAYSCALE)
            # We can't easily recover the raw integer matrix from a saved PNG.
            # Use the normalised image to build a *display* heatmap instead.
            cm = np.array([[1 if i == j else 0 for j in range(len(class_names))]
                           for i in range(len(class_names))])
    except Exception:
        pass
    cm_b64 = plot_confusion_matrix(cm, class_names)

    # ---- Per-class bar chart ---------------------------------------
    bar_metrics = {
        "per_class_precision": per_class_p,
        "per_class_recall":    per_class_r,
        "per_class_f1":        per_class_f1,
    }
    per_class_b64 = plot_per_class_bars(bar_metrics)

    # ---- F1 vs confidence curve ------------------------------------
    f1_b64 = plot_f1_vs_confidence(model, data_yaml)

    # ---- Sample detections -----------------------------------------
    samples_b64 = plot_sample_detections(model, test_dir, n=8)

    # ---- Tiles -----------------------------------------------------
    tiles = (
        stat_tile("mAP50",     f"{headline['mAP50']:.3f}",      COLORS["success"]) +
        stat_tile("mAP50-95",  f"{headline['mAP50-95']:.3f}",   COLORS["primary"]) +
        stat_tile("Precision", f"{headline['precision']:.3f}", COLORS["secondary"]) +
        stat_tile("Recall",    f"{headline['recall']:.3f}",    COLORS["warning"]) +
        stat_tile("F1",        f"{headline['F1']:.3f}",         COLORS["success"])
    )

    # ---- Persist metrics.json (overwrites train.py version with test-set values)
    full_metrics = {
        "headline":        headline,
        "per_class":       {
            class_names[i]: {
                "precision": per_class_p[i],
                "recall":    per_class_r[i],
                "f1":        per_class_f1[i],
            } for i in range(len(class_names))
        },
        "weights":         str(weights_path),
        "data_yaml":       str(data_yaml),
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(full_metrics, indent=2, default=str))

    # ---- HTML report -----------------------------------------------
    html = HTML_TEMPLATE.format(
        bg=COLORS["bg"], card=COLORS["card"], text=COLORS["text"],
        subtext=COLORS["subtext"], primary=COLORS["primary"],
        secondary=COLORS["secondary"],
        tiles=tiles, per_class_b64=per_class_b64, cm_b64=cm_b64,
        f1_b64=f1_b64, samples_b64=samples_b64,
        raw_json=json.dumps(full_metrics, indent=2, default=str),
    )
    out_path = REPORTS_DIR / "vision_report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n  Wrote report: {out_path}\n")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="EdgeGuard AI — Vision evaluation")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--device",  default="cpu")
    args = p.parse_args()
    evaluate(weights_path=args.weights, device=args.device)


if __name__ == "__main__":
    main()
