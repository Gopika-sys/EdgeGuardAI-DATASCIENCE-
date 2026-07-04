"""
EdgeGuard AI — Inference Service (v2)

Loads the trained classifier, RUL regressor (any of 4 algorithms),
failure-probability regressor, and Isolation Forest detector. Turns a
short buffer of recent sensor readings into a unified prediction
enriched with the RAG knowledge layer and the Gemini LLM Copilot.

OUTPUT JSON
-----------
{
    "failure_probability": 0.83,    # smooth 0-1 score (prob_regressor)
    "failure_class":       1,        # binary (classifier)
    "rul_hours":           4.2,      # RUL (best RUL regressor)
    "anomaly_score":      -0.12,     # IsolationForest decision_function
    "is_anomaly":          true,     # flagged by Isolation Forest
    "top_features":        [...],    # XGBoost feature importance
    "explanation":         "...",    # rule-based or Gemini-generated
    "sop_reference":       "..."     # RAG-retrieved SOP title
}
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from features import compute_features, get_feature_columns, SENSOR_COLS
from rag import RAGRetriever

MODELS_DIR = Path(__file__).parent / "models"

# ---------------------------------------------------------------------------
# Long -> wide format converter (for /readings/history)
# ---------------------------------------------------------------------------
def readings_to_wide_df(readings: list[dict]) -> pd.DataFrame:
    """Convert long-format sensor readings to wide DataFrame.

    Live backend data has a small set of sensors; missing sensors are
    left as NaN and f-filled by compute_features.
    """
    if not readings:
        return pd.DataFrame()
    df = pd.DataFrame(readings)
    if "device_ts" not in df.columns:
        return pd.DataFrame()
    df["time_bucket"] = (df["device_ts"] // 2000) * 2000
    wide = df.pivot_table(
        index="time_bucket", columns="sensor_type",
        values="value", aggfunc="max"
    ).reset_index().rename(columns={"time_bucket": "t_seconds"})
    wide["cycle_id"] = "live_buffer"
    wide["truck_id"] = "live"
    # Map any backend sensor_type to enterprise schema
    rename_map = {
        "temperature": "engine_temp", "vibration": "vibration_rms",
        "oil_pressure": "oil_pressure", "hydraulic_pressure": "hydraulic_pressure",
        "suspension_pressure": "suspension_pressure",
        "battery_voltage": "battery_voltage",
    }
    for old, new in rename_map.items():
        if old in wide.columns and new not in wide.columns:
            wide = wide.rename(columns={old: new})
    return wide


def _load_rul_model(models_dir: Path):
    """Load the winning RUL model (any of 4 algorithms)."""
    metrics_path = models_dir / "metrics.json"
    name = "xgboost"
    if metrics_path.exists():
        with open(metrics_path) as f:
            name = json.load(f).get("rul_regressor_best_model", "xgboost")

    path = models_dir / "regressor.json"
    if name == "xgboost":
        m = xgb.XGBRegressor()
        m.load_model(str(path))
        return m, name
    if name == "lightgbm":
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(path))
        return booster, name
    if name == "catboost":
        from catboost import CatBoostRegressor
        m = CatBoostRegressor()
        m.load_model(str(path))
        return m, name
    if name == "random_forest":
        return joblib.load(path.with_suffix(".joblib")), name
    raise FileNotFoundError(f"Unknown RUL model flavor: {name}")


def _rul_predict(model, name: str, X: pd.DataFrame) -> np.ndarray:
    if name == "lightgbm":
        return model.predict(X)
    return model.predict(X)


# ---------------------------------------------------------------------------
# PredictionService
# ---------------------------------------------------------------------------
class PredictionService:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.classifier = xgb.XGBClassifier()
        self.classifier.load_model(str(models_dir / "classifier.json"))

        self.prob_regressor = xgb.XGBRegressor()
        self.prob_regressor.load_model(str(models_dir / "prob_regressor.json"))

        self.rul_regressor, self.rul_model_name = _load_rul_model(models_dir)

        iso_path = models_dir / "anomaly_detector.joblib"
        self.anomaly_detector = joblib.load(iso_path) if iso_path.exists() else None

        with open(models_dir / "feature_columns.json") as f:
            self.feature_cols = json.load(f)

        # RAG
        self.rag = RAGRetriever()
        try:
            self.rag.load_index()
            self.rag_enabled = True
        except FileNotFoundError:
            self.rag_enabled = False

        # SHAP top features
        metrics_path = models_dir / "metrics.json"
        self._shap_top = []
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            shap = m.get("shap", {})
            if shap.get("computed"):
                self._shap_top = [n for n, _ in shap.get("top_shap_features", [])[:3]]

    def predict(self, wide_df: pd.DataFrame) -> dict:
        if len(wide_df) < 3:
            raise ValueError(f"Buffer too small ({len(wide_df)} rows); need >=3.")
        # Ensure all enterprise columns exist (live data may be sparse)
        for c in SENSOR_COLS:
            if c not in wide_df.columns:
                wide_df[c] = np.nan

        featured = compute_features(wide_df, group_cols=("truck_id", "cycle_id"))
        # Re-index to full feature set
        latest = featured.iloc[[-1]].reindex(columns=self.feature_cols, fill_value=0.0)

        # Fill residual NaN with column medians (saved at train time would be
        # better, but median(0) is a safe fallback for an isolated live row)
        latest = latest.fillna(0.0)

        # --- Models ---
        failure_probability = float(np.clip(
            self.prob_regressor.predict(latest)[0], 0.0, 1.0
        ))
        failure_class = int(self.classifier.predict(latest)[0])
        rul_hours = float(max(0.0, _rul_predict(self.rul_regressor, self.rul_model_name, latest)[0]))

        anomaly_score, is_anomaly = 0.0, False
        if self.anomaly_detector is not None:
            try:
                anomaly_score = float(self.anomaly_detector.decision_function(latest)[0])
                is_anomaly = bool(self.anomaly_detector.predict(latest)[0] == -1)
            except Exception:
                pass

        # Top features
        importances = dict(zip(self.feature_cols, self.classifier.feature_importances_.tolist()))
        top_features = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:5]

        explanation, sop_ref = self._build_explanation(
            top_features, failure_probability, rul_hours, is_anomaly, anomaly_score
        )

        return {
            "failure_probability": round(failure_probability, 4),
            "failure_class":       failure_class,
            "rul_hours":           round(rul_hours, 2),
            "anomaly_score":       round(anomaly_score, 4),
            "is_anomaly":          is_anomaly,
            "top_features":        top_features,
            "explanation":         explanation,
            "sop_reference":       sop_ref,
            "model_version":       f"v2-{self.rul_model_name}",
        }

    # -----------------------------------------------------------------------
    def _build_explanation(self, top_features, prob, rul, is_anom, anom_score):
        top_feature, _ = top_features[0]
        feature_human = top_feature.replace("_", " ").title()
        sop_ref = None

        if prob < 0.3:
            explanation = "All monitored parameters are within normal operating range."
        elif prob < 0.6:
            explanation = (f"Elevated {feature_human} detected. "
                           f"Recommend monitoring — estimated {rul:.1f} hours before maintenance required.")
        elif prob < 0.85:
            explanation = (f"WARNING: Abnormal {feature_human}. "
                           f"Failure likely within {rul:.1f} hours. Schedule inspection soon.")
        else:
            explanation = (f"CRITICAL: {feature_human} indicates imminent failure. "
                           f"Estimated RUL: {rul:.1f} hours. Immediate action required.")

        if is_anom:
            explanation += (f" [Isolation Forest flagged this reading as an outlier "
                            f"(score={anom_score:.3f}).]")

        # RAG enrichment for high-risk states
        if self.rag_enabled and prob >= 0.70:
            try:
                docs = self.rag.retrieve(query=top_feature, top_k=1)
                if docs:
                    sop_ref = docs[0].get("title")
                    explanation += (f"\n\nCopilot RAG Retrieval — {docs[0]['title']}:\n"
                                    f"{docs[0].get('content_chunk', '')[:500]}")
            except Exception:
                pass

        return explanation, sop_ref


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    csv_path = Path(__file__).parent / "edgeguard_training_data.csv"
    if not csv_path.exists():
        print("Run generate_training_data.py + train.py first.")
        raise SystemExit(1)
    df = pd.read_csv(csv_path)
    svc = PredictionService()

    # Find a truck/cycle that has a failure
    fail_cycles = df[df["failure_mode"] != "none"]["cycle_id"].unique()
    if len(fail_cycles) > 0:
        cid = fail_cycles[0]
        near = df[df["cycle_id"] == cid].tail(20).copy()
        result = svc.predict(near)
        print(f"Buffer from cycle {cid} (failure mode = "
              f"{df[df['cycle_id']==cid]['failure_mode'].iloc[0]}):")
        for k, v in result.items():
            print(f"  {k}: {v}")
