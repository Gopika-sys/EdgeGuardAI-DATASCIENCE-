"""
EdgeGuard AI — Model Training (v2 — enterprise schema)

Trains the full AI Engine 2 stack on the 45-column / 25K-row enterprise
dataset. Produces:

  1. Failure Classifier       (XGBoost binary)        - classifier.json
  2. Failure Probability Reg. (XGBoost continuous)    - prob_regressor.json
  3. RUL Regressor            (best of 4 algorithms)  - rul_regressor.json
  4. Anomaly Detector         (Isolation Forest)      - anomaly_detector.joblib
  5. SHAP summary                                       - shap_summary.json
  6. Full metrics                                     - metrics.json
  7. Model-comparison report                           - model_comparison.json

GROUP-LEVEL SPLIT: cycles are grouped by (truck_id, cycle_id). The test set
contains entire cycles the model has never seen. Random row-level splits
would inflate metrics due to within-cycle autocorrelation.

Run:
    python train.py
    python train.py --source supabase
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import GroupKFold
import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning)

from features import compute_features

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
DATA_PATH  = Path(__file__).parent / "edgeguard_training_data.csv"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
N_CV_FOLDS  = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_from_csv() -> pd.DataFrame:
    print(f"  Loading from CSV: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def load_from_supabase() -> pd.DataFrame:
    try:
        import os
        from dotenv import load_dotenv
        from supabase import create_client
        load_dotenv(Path(__file__).parent / ".env")
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_KEY"]
        sb = create_client(url, key)
        print("  Pulling training_data from Supabase...")
        resp = sb.table("training_data").select("*").execute()
        if resp.data:
            df = pd.DataFrame(resp.data)
            print(f"  Got {len(df)} rows from Supabase training_data table.")
            return df
    except Exception as e:
        print(f"  Supabase load failed ({e}) — falling back to CSV.")
    return load_from_csv()


def load_and_engineer(source: str = "csv") -> tuple:
    """Returns (df_featured, feature_cols)."""
    df = load_from_supabase() if source == "supabase" else load_from_csv()
    print(f"  Loaded {len(df):,} raw rows x {df.shape[1]} cols")
    df = compute_features(df, group_cols=("truck_id", "cycle_id"))
    feature_cols = [c for c in df.columns
                    if c not in {
                        "truck_id", "cycle_id", "t_seconds",
                        "failure_mode", "failure_probability",
                        "rul_hours", "label_failure_within_1hr",
                    }]
    print(f"  Engineered: {len(df):,} rows x {df.shape[1]} cols "
          f"({len(feature_cols)} features)")
    return df, feature_cols


# ---------------------------------------------------------------------------
# Group-aware split
# ---------------------------------------------------------------------------
def cycle_split(df: pd.DataFrame, test_fraction: float = 0.2,
                seed: int = RANDOM_SEED):
    """Split such that no (truck, cycle) appears in both train and test."""
    group_keys = df[["truck_id", "cycle_id"]].astype(str).agg("|".join, axis=1)
    unique_groups = group_keys.unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_groups)
    n_test = max(1, int(len(shuffled) * test_fraction))
    test_groups  = set(shuffled[:n_test])
    train_groups = set(shuffled[n_test:])
    train_df = df[group_keys.isin(train_groups)].reset_index(drop=True)
    test_df  = df[group_keys.isin(test_groups)].reset_index(drop=True)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Model trainers
# ---------------------------------------------------------------------------
def train_classifier(train_df, test_df, feature_cols):
    X_tr, y_tr = train_df[feature_cols], train_df["label_failure_within_1hr"]
    X_te, y_te = test_df[feature_cols],  test_df["label_failure_within_1hr"]
    pos = max((y_tr == 1).sum(), 1)
    neg = max((y_tr == 0).sum(), 1)
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        eval_metric="logloss", random_state=RANDOM_SEED,
        tree_method="hist", verbosity=0,
    )
    clf.fit(X_tr, y_tr)
    y_pred  = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)[:, 1]
    return clf, {
        "precision":  float(precision_score(y_te, y_pred, zero_division=0)),
        "recall":     float(recall_score(y_te, y_pred, zero_division=0)),
        "f1":         float(f1_score(y_te, y_pred, zero_division=0)),
        "roc_auc":    float(roc_auc_score(y_te, y_proba)),
        "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
    }


def train_prob_regressor(train_df, test_df, feature_cols):
    X_tr, y_tr = train_df[feature_cols], train_df["failure_probability"]
    X_te, y_te = test_df[feature_cols],  test_df["failure_probability"]
    reg = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_SEED, tree_method="hist", verbosity=0,
    )
    reg.fit(X_tr, y_tr)
    y_pred = np.clip(reg.predict(X_te), 0.0, 1.0)
    return reg, {
        "mae":  float(mean_absolute_error(y_te, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_te, y_pred))),
        "r2":   float(r2_score(y_te, y_pred)),
    }


def _train_rul_candidate(name, model_factory, train_df, test_df, feature_cols):
    """Train one RUL candidate and return (name, model, metrics)."""
    X_tr, y_tr = train_df[feature_cols], train_df["rul_hours"]
    X_te, y_te = test_df[feature_cols],  test_df["rul_hours"]
    model = model_factory()
    model.fit(X_tr, y_tr)
    y_pred = np.clip(model.predict(X_te), 0.0, None)
    metrics = {
        "mae_hours":      float(mean_absolute_error(y_te, y_pred)),
        "rmse_hours":     float(np.sqrt(mean_squared_error(y_te, y_pred))),
        "r2":             float(r2_score(y_te, y_pred)),
    }
    near = y_te <= 24.0
    if near.sum() > 0:
        metrics["mae_near_failure_hours"] = float(mean_absolute_error(y_te[near], y_pred[near]))
        metrics["near_failure_test_rows"] = int(near.sum())
    return name, model, metrics


def _save_rul_model(model, name: str, dest_path: Path):
    """Persist any of the 4 RUL model flavors."""
    import joblib
    if name == "xgboost":
        model.save_model(str(dest_path))
    elif name == "lightgbm":
        model.booster_.save_model(str(dest_path))
    elif name == "catboost":
        model.save_model(str(dest_path))
    else:  # random_forest
        joblib.dump(model, dest_path.with_suffix(".joblib"))


def train_rul_ensemble(train_df, test_df, feature_cols) -> dict:
    """Train all 4 RUL candidates, pick best by MAE, return chosen + comparison."""
    print(f"    Training 4 RUL candidates: XGBoost, RandomForest, LightGBM, CatBoost ...")

    # Filter to rows with valid RUL (rul_hours < HEALTHY_CEILING means the cycle is failing)
    train_fail = train_df[train_df["rul_hours"] < 400].copy()
    test_fail  = test_df[test_df["rul_hours"] < 400].copy()
    if len(train_fail) < 50:
        # Fall back to all data if too few failing rows
        train_fail, test_fail = train_df, test_df

    candidates = []
    models_dict = {}  # name -> fitted model (for later saving)

    # XGBoost
    name, model, metrics = _train_rul_candidate(
        "xgboost",
        lambda: xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_SEED, tree_method="hist", verbosity=0,
        ),
        train_fail, test_fail, feature_cols,
    )
    candidates.append((name, model, metrics))
    models_dict[name] = model

    # Random Forest
    name, model, metrics = _train_rul_candidate(
        "random_forest",
        lambda: RandomForestRegressor(
            n_estimators=300, max_depth=18, min_samples_leaf=3,
            n_jobs=-1, random_state=RANDOM_SEED,
        ),
        train_fail, test_fail, feature_cols,
    )
    candidates.append((name, model, metrics))
    models_dict[name] = model

    # LightGBM
    try:
        import lightgbm as lgb
        name, model, metrics = _train_rul_candidate(
            "lightgbm",
            lambda: lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.04, num_leaves=63,
                max_depth=-1, subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_SEED, verbosity=-1,
            ),
            train_fail, test_fail, feature_cols,
        )
        candidates.append((name, model, metrics))
        models_dict[name] = model
    except ImportError:
        print("    [lightgbm] not installed — skipping")

    # CatBoost
    try:
        from catboost import CatBoostRegressor
        name, model, metrics = _train_rul_candidate(
            "catboost",
            lambda: CatBoostRegressor(
                iterations=500, learning_rate=0.04, depth=6,
                loss_function="RMSE", random_seed=RANDOM_SEED, verbose=False,
            ),
            train_fail, test_fail, feature_cols,
        )
        candidates.append((name, model, metrics))
        models_dict[name] = model
    except ImportError:
        print("    [catboost] not installed — skipping")

    # Rank by MAE on near-failure samples (operational metric)
    def score(c):
        return c[2].get("mae_near_failure_hours", c[2]["mae_hours"])
    candidates.sort(key=score)
    best_name, best_model, best_metrics = candidates[0]

    comparison = {name: metrics for name, _, metrics in candidates}
    return best_name, best_model, best_metrics, comparison, models_dict


def train_isolation_forest(train_df, feature_cols, contamination: float = 0.10):
    """Train an Isolation Forest for unsupervised anomaly detection.

    Trained on NORMAL data only (label_failure_within_1hr == 0) so that
    it learns the 'healthy' manifold and flags out-of-distribution samples.
    """
    normal = train_df[train_df["label_failure_within_1hr"] == 0]
    X = normal[feature_cols].values
    print(f"    Training IsolationForest on {len(X):,} normal samples ...")
    iso = IsolationForest(
        n_estimators=200,
        max_samples=min(10_000, len(X)),
        contamination=contamination,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    iso.fit(X)

    # Quick sanity check: score the full test set, report anomaly rate
    test_X = train_df[feature_cols].values  # using train tail for quick check
    preds = iso.predict(test_X)
    anomaly_rate = (preds == -1).mean()
    return iso, {
        "n_estimators": 200,
        "contamination": contamination,
        "train_samples": int(len(X)),
        "anomaly_rate": float(anomaly_rate),
    }


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def compute_shap_summary(model, X_sample: pd.DataFrame, feature_cols: list, top_n: int = 20):
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sample = X_sample.sample(min(500, len(X_sample)), random_state=RANDOM_SEED)
        sv = explainer.shap_values(sample)
        if isinstance(sv, list):
            sv = sv[1]  # binary classifier returns list
        mean_abs = np.abs(sv).mean(axis=0)
        summary = dict(zip(feature_cols, mean_abs.tolist()))
        top = sorted(summary.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return {"top_shap_features": top, "computed": True}
    except ImportError:
        return {"computed": False, "reason": "shap not installed"}
    except Exception as e:
        return {"computed": False, "reason": str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="EdgeGuard AI model training v2")
    parser.add_argument("--source", choices=["csv", "supabase"], default="csv")
    args = parser.parse_args()

    print(f"=== EdgeGuard AI — Training (source={args.source}) ===\n")
    df, feature_cols = load_and_engineer(args.source)

    train_df, test_df = cycle_split(df)
    print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

    # Cross-validation on classifier
    print(f"\nRunning {N_CV_FOLDS}-fold GroupKFold cross-validation (classifier)...")
    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    groups = (train_df["truck_id"].astype(str) + "|" + train_df["cycle_id"].astype(str)).values
    f1s, aucs = [], []
    X_full = train_df[feature_cols]
    y_full = train_df["label_failure_within_1hr"]
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_full, y_full, groups)):
        clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=max((y_full.iloc[tr_idx] == 0).sum() / max((y_full.iloc[tr_idx] == 1).sum(), 1), 1),
            eval_metric="logloss", random_state=RANDOM_SEED,
            tree_method="hist", verbosity=0,
        )
        clf.fit(X_full.iloc[tr_idx], y_full.iloc[tr_idx])
        p = clf.predict(X_full.iloc[te_idx])
        proba = clf.predict_proba(X_full.iloc[te_idx])[:, 1]
        f1s.append(f1_score(y_full.iloc[te_idx], p, zero_division=0))
        try:
            aucs.append(roc_auc_score(y_full.iloc[te_idx], proba))
        except Exception:
            pass
    cv = {
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "auc_mean": float(np.mean(aucs)) if aucs else None,
        "auc_std":  float(np.std(aucs)) if aucs else None,
        "n_folds": N_CV_FOLDS,
    }
    print(f"  F1: {cv['f1_mean']:.3f} ± {cv['f1_std']:.3f}"
          + (f" | AUC: {cv['auc_mean']:.3f} ± {cv['auc_std']:.3f}" if cv['auc_mean'] else ""))

    # Classifier
    print("\nTraining failure classifier ...")
    clf, clf_metrics = train_classifier(train_df, test_df, feature_cols)
    print(f"  Precision: {clf_metrics['precision']:.3f} | "
          f"Recall: {clf_metrics['recall']:.3f} | "
          f"F1: {clf_metrics['f1']:.3f} | "
          f"AUC: {clf_metrics['roc_auc']:.3f}")

    # Probability regressor
    print("\nTraining failure-probability regressor ...")
    prob_reg, prob_metrics = train_prob_regressor(train_df, test_df, feature_cols)
    print(f"  MAE: {prob_metrics['mae']:.4f} | RMSE: {prob_metrics['rmse']:.4f} | "
          f"R²: {prob_metrics['r2']:.3f}")

    # RUL — 4-model comparison
    print("\nTraining RUL regressor (4-model comparison) ...")
    best_name, rul_reg, rul_metrics, rul_comparison, rul_models = train_rul_ensemble(
        train_df, test_df, feature_cols
    )
    print(f"  Best model: {best_name}")
    for k, v in rul_metrics.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    print(f"  All candidates:")
    for name, m in rul_comparison.items():
        score = m.get("mae_near_failure_hours", m["mae_hours"])
        print(f"    {name:14s} MAE: {m['mae_hours']:.2f}h | R²: {m['r2']:.3f} | "
              f"MAE-near-fail: {m.get('mae_near_failure_hours', 'n/a')}")

    # Isolation Forest
    print("\nTraining Isolation Forest anomaly detector ...")
    iso, iso_metrics = train_isolation_forest(train_df, feature_cols)
    print(f"  Anomaly rate (sanity): {iso_metrics['anomaly_rate']:.3%}")

    # Feature importance + SHAP
    importances = dict(zip(feature_cols, clf.feature_importances_.tolist()))
    top_features = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:15]
    print(f"\nTop 10 XGBoost features:")
    for name, score in top_features[:10]:
        print(f"  {name}: {score:.4f}")

    print("\nComputing SHAP values ...")
    shap_result = compute_shap_summary(clf, train_df[feature_cols], feature_cols)
    if shap_result["computed"]:
        print("  Top SHAP features:")
        for name, val in shap_result["top_shap_features"][:5]:
            print(f"    {name}: {val:.4f}")

    # Save everything
    print("\nSaving models and metrics ...")
    clf.save_model(str(MODELS_DIR / "classifier.json"))
    prob_reg.save_model(str(MODELS_DIR / "prob_regressor.json"))
    # Persist the winning RUL model in a format that's easy to load
    rul_dest = MODELS_DIR / "regressor.json"
    _save_rul_model(rul_reg, best_name, rul_dest)
    # Also dump the runner-up models for comparison / fallback
    for n, m in rul_models.items():
        if n == best_name:
            continue
        _save_rul_model(m, n, MODELS_DIR / f"rul_{n}.json")
    joblib.dump(iso, MODELS_DIR / "anomaly_detector.joblib")
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(MODELS_DIR / "model_comparison.json", "w") as f:
        json.dump(rul_comparison, f, indent=2)

    full_metrics = {
        "classifier":     clf_metrics,
        "classifier_cv":  cv,
        "prob_regressor": prob_metrics,
        "rul_regressor":  rul_metrics,
        "rul_regressor_best_model": best_name,
        "rul_comparison": rul_comparison,
        "isolation_forest": iso_metrics,
        "top_features_xgb": top_features,
        "shap":           shap_result,
        "n_features":     len(feature_cols),
        "n_train_rows":   len(train_df),
        "n_test_rows":    len(test_df),
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)

    print(f"\nAll artifacts saved to {MODELS_DIR}/")
    print(f"  classifier.json         (XGBoost, binary failure gate)")
    print(f"  prob_regressor.json     (XGBoost, smooth 0-1 probability)")
    print(f"  regressor.json          ({best_name}, RUL hours)")
    print(f"  anomaly_detector.joblib (Isolation Forest)")
    print(f"  feature_columns.json    (feature schema)")
    print(f"  metrics.json            (full evaluation metrics)")
    print(f"  model_comparison.json   (RUL model-comparison report)")


if __name__ == "__main__":
    main()
