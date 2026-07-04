"""
EdgeGuard AI - Model Evaluation Report Generator (Member 4)

Loads the trained models + test data and produces a self-contained HTML
report with:

  Classifier:
    - ROC curve (AUC)
    - Precision-Recall curve
    - Confusion matrix heatmap
    - Feature importance bar chart (XGBoost gain)
    - SHAP summary bar chart (if shap is installed)

  RUL Regressor:
    - Actual vs Predicted scatter plot
    - Residual distribution histogram
    - Error by RUL range (near-failure vs healthy)

  Failure-Probability Regressor:
    - Actual vs Predicted scatter plot
    - Calibration curve

All plots are embedded as base64 in a single HTML file — no external
dependencies needed to view the report.

Usage:
    cd ml-data
    python evaluate.py
    # Opens model_report.html — share this at the demo!
"""

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    mean_absolute_error, r2_score,
)
import xgboost as xgb

from features import compute_features, get_feature_columns
from train import load_and_engineer, cycle_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODELS_DIR   = Path(__file__).parent / "models"
REPORT_PATH  = Path(__file__).parent / "model_report.html"

# Colour palette matching the team's dark-mode dashboard aesthetic
COLORS = {
    "primary":   "#6C63FF",  # purple
    "secondary": "#00D4FF",  # cyan
    "danger":    "#FF4E4E",  # red
    "success":   "#2ECC71",  # green
    "warning":   "#F39C12",  # orange
    "bg":        "#0F1117",  # near-black
    "card":      "#1A1D27",  # dark card
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


def plot_roc(y_true, y_proba) -> str:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc     = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color=COLORS["primary"], lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color=COLORS["subtext"], lw=1, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS["primary"])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Failure Classifier", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(True)
    return fig_to_base64(fig)


def plot_precision_recall(y_true, y_proba) -> str:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recall, precision, color=COLORS["secondary"], lw=2, label=f"AP = {pr_auc:.3f}")
    ax.fill_between(recall, precision, alpha=0.15, color=COLORS["secondary"])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(True)
    return fig_to_base64(fig)


def plot_confusion_matrix(cm: list) -> str:
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="RdYlGn")
    plt.colorbar(im, ax=ax)
    classes = ["No Failure", "Failure"]
    ticks   = np.arange(len(classes))
    ax.set(xticks=ticks, yticks=ticks, xticklabels=classes, yticklabels=classes,
           xlabel="Predicted", ylabel="Actual",
           title="Confusion Matrix")
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                    color="black", fontsize=14, fontweight="bold")
    return fig_to_base64(fig)


def plot_feature_importance(feature_cols: list, importances: list) -> str:
    pairs = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)[:15]
    names, scores = zip(*pairs)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(range(len(names)), scores, color=COLORS["primary"], alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("XGBoost Gain Importance")
    ax.set_title("Top 15 Feature Importances (Classifier)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.4)
    return fig_to_base64(fig)


def plot_shap_bar(shap_top: list) -> str:
    names  = [n.replace("_", " ") for n, _ in shap_top[:15]]
    values = [v for _, v in shap_top[:15]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(names)), values, color=COLORS["secondary"], alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance (Mean |SHAP|)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.4)
    return fig_to_base64(fig)


def plot_actual_vs_pred(y_true, y_pred, title: str, color: str) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(y_true, y_pred, alpha=0.25, s=10, color=color)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], color=COLORS["danger"], lw=1.5, linestyle="--", label="Perfect")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(True)
    return fig_to_base64(fig)


def plot_residuals(y_true, y_pred, title: str) -> str:
    residuals = y_pred - y_true
    fig, ax   = plt.subplots(figsize=(5.5, 4.5))
    ax.hist(residuals, bins=60, color=COLORS["warning"], alpha=0.80, edgecolor="none")
    ax.axvline(0, color=COLORS["danger"], lw=2, linestyle="--")
    ax.set_xlabel("Residual (Predicted − Actual)")
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.4)
    return fig_to_base64(fig)


def plot_calibration(y_true, y_pred, title: str) -> str:
    bins   = np.linspace(0, 1, 11)
    bin_idx = np.digitize(y_pred, bins) - 1
    means_pred, means_true = [], []
    for i in range(len(bins) - 1):
        mask = bin_idx == i
        if mask.sum() > 0:
            means_pred.append(y_pred[mask].mean())
            means_true.append(y_true[mask].mean())
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color=COLORS["subtext"], label="Perfect calibration")
    ax.plot(means_pred, means_true, "o-", color=COLORS["primary"], lw=2, label="Model")
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Mean Actual Probability")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(True)
    return fig_to_base64(fig)


def plot_rul_by_range(y_true, y_pred) -> str:
    """MAE breakdown by RUL range — highlights how good the model is near failure."""
    ranges = [(0, 1, "0-1h"), (1, 6, "1-6h"), (6, 24, "6-24h"),
              (24, 100, "24-100h"), (100, 600, ">100h")]
    labels, maes, counts = [], [], []
    for lo, hi, label in ranges:
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() > 0:
            labels.append(f"{label}\n(n={mask.sum()})")
            maes.append(mean_absolute_error(y_true[mask], y_pred[mask]))
            counts.append(mask.sum())
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(range(len(labels)), maes, color=COLORS["primary"], alpha=0.85)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("MAE (hours)")
    ax.set_title("RUL Regressor MAE by RUL Range", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.4)
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{mae:.1f}h", ha="center", fontsize=8, color=COLORS["text"])
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EdgeGuard AI — Model Evaluation Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Inter', sans-serif; background: {bg}; color: {text}; line-height: 1.6; }}
  .header {{ background: linear-gradient(135deg, #1a1d27 0%, #0f1117 100%);
             border-bottom: 1px solid #2a2d3e; padding: 2.5rem 3rem; }}
  .header h1 {{ font-size: 2rem; font-weight: 700;
                background: linear-gradient(90deg, {primary}, {secondary});
                -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .header p  {{ color: {subtext}; margin-top: 0.4rem; font-size: 0.95rem; }}
  .container {{ max-width: 1300px; margin: 0 auto; padding: 2rem 3rem; }}
  .section-title {{ font-size: 1.35rem; font-weight: 600; color: {text};
                    border-left: 4px solid {primary}; padding-left: 1rem;
                    margin: 2.5rem 0 1.5rem; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                   gap: 1rem; margin-bottom: 2rem; }}
  .metric-card {{ background: {card}; border: 1px solid #2a2d3e; border-radius: 12px;
                  padding: 1.2rem 1.5rem; text-align: center; }}
  .metric-card .label {{ font-size: 0.75rem; color: {subtext}; text-transform: uppercase;
                          letter-spacing: 0.08em; margin-bottom: 0.4rem; }}
  .metric-card .value {{ font-size: 1.8rem; font-weight: 700; color: {primary}; }}
  .metric-card .value.good  {{ color: {success}; }}
  .metric-card .value.warn  {{ color: {warning}; }}
  .metric-card .value.bad   {{ color: {danger};  }}
  .plots-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
                 gap: 1.5rem; }}
  .plot-card {{ background: {card}; border: 1px solid #2a2d3e; border-radius: 12px;
                padding: 1.2rem; }}
  .plot-card h3 {{ font-size: 0.9rem; font-weight: 600; color: {subtext};
                   margin-bottom: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; }}
  .plot-card img {{ width: 100%; border-radius: 8px; }}
  .shap-section {{ background: linear-gradient(135deg, #1a1d27, #0f1117);
                   border: 1px solid {primary}44; border-radius: 12px;
                   padding: 1.5rem; margin-bottom: 2rem; }}
  .shap-section h3 {{ color: {primary}; font-size: 1rem; margin-bottom: 0.5rem; }}
  .shap-section p  {{ color: {subtext}; font-size: 0.9rem; }}
  .feature-chips {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.8rem; }}
  .chip {{ background: {primary}22; border: 1px solid {primary}55; border-radius: 20px;
           padding: 0.25rem 0.85rem; font-size: 0.8rem; color: {primary}; }}
  .footer {{ text-align: center; padding: 2rem; color: {subtext}; font-size: 0.85rem;
             border-top: 1px solid #2a2d3e; margin-top: 3rem; }}
  .badge {{ display: inline-block; padding: 0.2rem 0.6rem; border-radius: 6px;
            font-size: 0.75rem; font-weight: 600; margin-left: 0.5rem; }}
  .badge-pass {{ background: {success}22; color: {success}; border: 1px solid {success}44; }}
  .badge-warn {{ background: {warning}22; color: {warning}; border: 1px solid {warning}44; }}
</style>
</head>
<body>
<div class="header">
  <h1>EdgeGuard AI — Model Evaluation Report</h1>
  <p>XGBoost Classifier + RUL Regressor + Failure Probability Regressor &nbsp;|&nbsp;
     Generated: {timestamp} &nbsp;|&nbsp; Member 4 (ML Engineer)</p>
</div>

<div class="container">

  <!-- ── Classifier ────────────────────────────────────────────────── -->
  <div class="section-title">🎯 Failure Classifier (label_failure_within_1hr)</div>
  <div class="metrics-grid">
    {clf_metric_cards}
  </div>
  <div class="plots-grid">
    <div class="plot-card"><h3>ROC Curve</h3><img src="data:image/png;base64,{roc_img}"></div>
    <div class="plot-card"><h3>Precision-Recall Curve</h3><img src="data:image/png;base64,{pr_img}"></div>
    <div class="plot-card"><h3>Confusion Matrix</h3><img src="data:image/png;base64,{cm_img}"></div>
    <div class="plot-card"><h3>Feature Importance</h3><img src="data:image/png;base64,{fi_img}"></div>
  </div>

  <!-- ── SHAP ────────────────────────────────────────────────────── -->
  {shap_section}

  <!-- ── RUL Regressor ──────────────────────────────────────────── -->
  <div class="section-title">⏱️ RUL Regressor (rul_hours)</div>
  <div class="metrics-grid">
    {rul_metric_cards}
  </div>
  <div class="plots-grid">
    <div class="plot-card"><h3>Actual vs Predicted</h3><img src="data:image/png;base64,{rul_avp_img}"></div>
    <div class="plot-card"><h3>Residual Distribution</h3><img src="data:image/png;base64,{rul_res_img}"></div>
    <div class="plot-card"><h3>MAE by RUL Range</h3><img src="data:image/png;base64,{rul_range_img}"></div>
  </div>

  <!-- ── Probability Regressor ──────────────────────────────────── -->
  <div class="section-title">📊 Failure Probability Regressor (failure_probability)</div>
  <div class="metrics-grid">
    {prob_metric_cards}
  </div>
  <div class="plots-grid">
    <div class="plot-card"><h3>Actual vs Predicted</h3><img src="data:image/png;base64,{prob_avp_img}"></div>
    <div class="plot-card"><h3>Calibration Curve</h3><img src="data:image/png;base64,{prob_cal_img}"></div>
  </div>

  <!-- ── Cross-Validation ───────────────────────────────────────── -->
  <div class="section-title">🔄 Cross-Validation (GroupKFold, {n_folds} folds)</div>
  <div class="metrics-grid">
    {cv_metric_cards}
  </div>

</div>
<div class="footer">
  EdgeGuard AI — Predictive Maintenance for Tata Signa 4825.TK Tipper Trucks<br>
  Models trained with XGBoost on {n_train:,} rows / {n_test:,} test rows — cycle-level split (no data leakage)
</div>
</body></html>
"""


def metric_card(label: str, value: str, rating: str = "") -> str:
    css_class = f"value {rating}" if rating else "value"
    return f"""<div class="metric-card">
  <div class="label">{label}</div>
  <div class="{css_class}">{value}</div>
</div>"""


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def main():
    print("=== EdgeGuard AI — Model Evaluation Report ===\n")

    # ── Load models ──────────────────────────────────────────────────────────
    print("Loading models from models/...")
    clf = xgb.XGBClassifier(); clf.load_model(str(MODELS_DIR / "classifier.json"))
    rul_reg = xgb.XGBRegressor(); rul_reg.load_model(str(MODELS_DIR / "regressor.json"))

    prob_reg = None
    prob_reg_path = MODELS_DIR / "prob_regressor.json"
    if prob_reg_path.exists():
        prob_reg = xgb.XGBRegressor(); prob_reg.load_model(str(prob_reg_path))

    with open(MODELS_DIR / "feature_columns.json") as f:
        feature_cols = json.load(f)
    with open(MODELS_DIR / "metrics.json") as f:
        saved_metrics = json.load(f)

    # ── Prepare test data ────────────────────────────────────────────────────
    print("Preparing test dataset...")
    df = load_and_engineer(source="csv")
    _, test_df, _, test_cycles = cycle_split(df)

    # Align feature columns — some cross-sensor features may have been added
    available = [c for c in feature_cols if c in test_df.columns]
    X_test    = test_df[available].reindex(columns=feature_cols, fill_value=0.0)

    y_clf  = test_df["label_failure_within_1hr"].values
    y_rul  = test_df["rul_hours"].values
    y_prob = test_df["failure_probability"].values

    print(f"  Test rows: {len(test_df):,} across cycles {test_cycles}")

    # ── Classifier predictions ───────────────────────────────────────────────
    clf_pred  = clf.predict(X_test)
    clf_proba = clf.predict_proba(X_test)[:, 1]
    cm        = confusion_matrix(y_clf, clf_pred).tolist()

    # ── RUL predictions ──────────────────────────────────────────────────────
    rul_pred = rul_reg.predict(X_test)

    # ── Prob predictions ─────────────────────────────────────────────────────
    if prob_reg is not None:
        prob_pred = np.clip(prob_reg.predict(X_test), 0.0, 1.0)
    else:
        prob_pred = clf_proba  # fallback

    # ── Generate plots ───────────────────────────────────────────────────────
    print("Generating plots...")
    roc_img      = plot_roc(y_clf, clf_proba)
    pr_img       = plot_precision_recall(y_clf, clf_proba)
    cm_img       = plot_confusion_matrix(cm)
    fi_img       = plot_feature_importance(feature_cols, clf.feature_importances_.tolist())
    rul_avp_img  = plot_actual_vs_pred(y_rul, rul_pred, "RUL: Actual vs Predicted", COLORS["primary"])
    rul_res_img  = plot_residuals(y_rul, rul_pred, "RUL Residual Distribution")
    rul_range_img = plot_rul_by_range(y_rul, rul_pred)
    prob_avp_img = plot_actual_vs_pred(y_prob, prob_pred, "Failure Prob: Actual vs Predicted", COLORS["secondary"])
    prob_cal_img = plot_calibration(y_prob, prob_pred, "Failure Probability Calibration")

    # ── SHAP section ─────────────────────────────────────────────────────────
    shap_data = saved_metrics.get("shap", {})
    if shap_data.get("computed"):
        top_shap = shap_data["top_shap_features"][:15]
        shap_img = plot_shap_bar(top_shap)
        chip_html = "".join(
            f'<span class="chip">{n.replace("_", " ")}</span>'
            for n, _ in top_shap[:8]
        )
        shap_section = f"""
<div class="shap-section">
  <h3>🔬 SHAP Explainability — Top Driving Features</h3>
  <p>SHAP (SHapley Additive exPlanations) measures each feature's actual contribution
     to the model's output — more reliable than XGBoost's gain importance alone.</p>
  <div class="feature-chips">{chip_html}</div>
</div>
<div class="plots-grid" style="margin-bottom:2rem">
  <div class="plot-card"><h3>SHAP Feature Importance</h3><img src="data:image/png;base64,{shap_img}"></div>
</div>"""
    else:
        shap_section = """<div class="shap-section">
  <h3>🔬 SHAP Explainability</h3>
  <p>SHAP not computed. Run <code>pip install shap</code> then retrain to enable.</p>
</div>"""

    # ── Metric cards ─────────────────────────────────────────────────────────
    clf_m = saved_metrics.get("classifier", {})
    rul_m = saved_metrics.get("rul_regressor", {})
    prob_m = saved_metrics.get("prob_regressor", {})
    cv_m   = saved_metrics.get("classifier_cv", {})

    clf_cards = "".join([
        metric_card("Precision",    f"{clf_m.get('precision',0):.3f}", "good" if clf_m.get('precision',0)>0.9 else "warn"),
        metric_card("Recall",       f"{clf_m.get('recall',0):.3f}",    "good" if clf_m.get('recall',0)>0.9 else "warn"),
        metric_card("F1 Score",     f"{clf_m.get('f1',0):.3f}",        "good" if clf_m.get('f1',0)>0.9 else "warn"),
        metric_card("ROC-AUC",      f"{clf_m.get('roc_auc',0):.3f}",   "good" if clf_m.get('roc_auc',0)>0.95 else "warn"),
    ])

    rul_cards = "".join([
        metric_card("MAE (all)", f"{rul_m.get('mae_hours',0):.1f}h"),
        metric_card("R²",        f"{rul_m.get('r2',0):.3f}", "good" if rul_m.get('r2',0)>0.85 else "warn"),
        metric_card("MAE (≤24h)", f"{rul_m.get('mae_near_failure_hours', rul_m.get('mae_hours_near_failure_only',0)):.1f}h"),
    ])

    prob_cards = "".join([
        metric_card("MAE",  f"{prob_m.get('mae',0):.4f}"),
        metric_card("R²",   f"{prob_m.get('r2',0):.3f}", "good" if prob_m.get('r2',0)>0.85 else "warn"),
    ])

    f1_mean = cv_m.get("f1_mean", 0); f1_std = cv_m.get("f1_std", 0)
    auc_mean = cv_m.get("auc_mean"); auc_std = cv_m.get("auc_std", 0)
    cv_cards = "".join([
        metric_card(f"CV F1 (±σ)", f"{f1_mean:.3f}", "good" if f1_mean > 0.9 else "warn"),
        metric_card("CV F1 Std",   f"±{f1_std:.3f}"),
        metric_card("CV AUC (±σ)", f"{auc_mean:.3f}" if auc_mean else "N/A"),
        metric_card("Folds",       str(cv_m.get("n_folds", 5))),
    ])

    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html = HTML_TEMPLATE.format(
        **COLORS,
        timestamp=ts,
        clf_metric_cards=clf_cards,
        roc_img=roc_img, pr_img=pr_img, cm_img=cm_img, fi_img=fi_img,
        shap_section=shap_section,
        rul_metric_cards=rul_cards,
        rul_avp_img=rul_avp_img, rul_res_img=rul_res_img, rul_range_img=rul_range_img,
        prob_metric_cards=prob_cards,
        prob_avp_img=prob_avp_img, prob_cal_img=prob_cal_img,
        cv_metric_cards=cv_cards,
        n_folds=cv_m.get("n_folds", 5),
        n_train=saved_metrics.get("n_train_rows", 0),
        n_test=saved_metrics.get("n_test_rows", 0),
    )

    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"\n✅ Report saved: {REPORT_PATH}")
    print("   Open model_report.html in your browser to view the full evaluation.")


if __name__ == "__main__":
    main()
