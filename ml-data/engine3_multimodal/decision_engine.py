"""
EdgeGuard AI — Multi-Modal Decision Engine (Module 08 of ml work.txt)

PURPOSE
-------
Fuse the outputs of every AI subsystem into a single, contextual
decision that tells a maintenance manager:
  - WHAT is the most likely root cause
  - HOW CONFIDENT we are
  - WHAT is the business risk
  - WHAT action to take
  - WHEN to take it

INPUTS (each may be absent — fusion handles partial inputs gracefully)
---------------------------------------------------------------------
  - ai2_prediction:    from AI Engine 2 (Predictive Maintenance)
                      { failure_probability, failure_class, rul_hours, top_features }
  - ai1_vision:        from AI Engine 1 (Computer Vision)
                      { carryback_pct, payload_efficiency, material_occupancy,
                        loading_quality, hydraulic_stress_indicator }
  - maintenance_log:   list of recent maintenance events for this truck
  - business_impact:   from engine4_business (optional)
  - sop_references:    RAG-retrieved SOP titles (optional)

FUSION STRATEGY
---------------
1. Each modality emits a per-cause probability vector over the 5
   failure modes (overheat, bearing, hydraulic, electrical, suspension)
2. A simple weighted-average ensemble fuses the vectors
3. The top cause is selected; confidence = top1 probability
4. Decision is rule-based on (root_cause, confidence, rul_hours)

WHY NOT A TRAINED FUSION MODEL?
-------------------------------
For a hackathon, a heuristic fusion with documented weights is more
auditable, debuggable, and inspectable than a black-box meta-learner.
The structure allows the weights to be tuned in production with
historical failure-labelled data without changing the API.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional
import numpy as np


FAILURE_MODES = ["overheat", "bearing", "hydraulic", "electrical", "suspension"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ai2_cause_distribution(ai2: dict) -> dict:
    """Map AI Engine 2 top-features into a failure-mode probability vector.

    The mapping is heuristic but documented. Production deployments would
    learn these weights from historical failure-labelled data.
    """
    probs = {m: 0.05 for m in FAILURE_MODES}  # prior
    feature_to_mode = {
        "engine_temp":        "overheat",
        "coolant_temp":       "overheat",
        "bearing_temp":       "bearing",
        "vibration":          "bearing",
        "vibration_rms":      "bearing",
        "vibration_x":        "bearing",
        "vibration_y":        "bearing",
        "vibration_z":        "bearing",
        "hydraulic_pressure": "hydraulic",
        "suspension_pressure": "suspension",
        "brake_pressure":     "suspension",
        "brake_temp":         "suspension",
        "battery_voltage":    "electrical",
        "alternator_voltage": "electrical",
        "current_draw":       "electrical",
    }
    # Spread the failure probability mass across the top-feature causes
    top_feats = ai2.get("top_features", [])
    p = float(ai2.get("failure_probability", 0.0))
    if not top_feats or p <= 0.05:
        return probs
    # Total importance of top features
    total_imp = sum(v for _, v in top_feats) or 1.0
    for name, imp in top_feats:
        for key, mode in feature_to_mode.items():
            if key in name:
                probs[mode] += (imp / total_imp) * p * 0.85
    # Normalise
    s = sum(probs.values())
    if s > 0:
        probs = {k: v / s for k, v in probs.items()}
    return probs


def _ai1_cause_distribution(ai1: dict) -> dict:
    """Map AI Engine 1 (Computer Vision) outputs to a failure-mode vector."""
    probs = {m: 0.05 for m in FAILURE_MODES}
    if not ai1:
        return probs
    # Hydraulic stress indicator -> hydraulic
    if ai1.get("hydraulic_stress_indicator", 0) > 0.6:
        probs["hydraulic"] += 0.6
    if ai1.get("loading_quality", 1.0) < 0.5:
        probs["suspension"] += 0.3
    if ai1.get("carryback_pct", 0) > 6:
        probs["suspension"] += 0.3
        probs["hydraulic"]  += 0.2
    if ai1.get("payload_efficiency", 1.0) < 0.7:
        probs["suspension"] += 0.2
    s = sum(probs.values())
    if s > 0:
        probs = {k: v / s for k, v in probs.items()}
    return probs


def _history_penalty(maintenance_log: list) -> dict:
    """Frequent recent failures on the same component raise that cause's weight."""
    probs = {m: 0.0 for m in FAILURE_MODES}
    if not maintenance_log:
        return probs
    for event in maintenance_log[-10:]:  # last 10 events
        mode = event.get("failure_mode") or event.get("component")
        if mode in probs:
            probs[mode] += 0.10
    s = sum(probs.values())
    if s > 0:
        probs = {k: v / s * 0.25 for k, v in probs.items()}
    return probs


# ---------------------------------------------------------------------------
# Decision output
# ---------------------------------------------------------------------------
@dataclass
class DecisionOutput:
    truck_id:            str
    root_cause:          str
    cause_confidence:    float
    risk_score:          float           # 0-100
    risk_band:           str             # "critical" | "high" | "medium" | "low"
    maintenance_priority: str            # critical / high / medium / low
    estimated_downtime_hr: float
    recommended_action:  str
    sop_references:      list = field(default_factory=list)
    fused_probabilities:  dict = field(default_factory=dict)
    reasoning:           str = ""

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Main decision function
# ---------------------------------------------------------------------------
def fuse_and_decide(
    ai2_prediction:  Optional[dict] = None,
    ai1_vision:      Optional[dict] = None,
    maintenance_log: Optional[list] = None,
    business_impact: Optional[dict] = None,
    sop_references:  Optional[list] = None,
    truck_id:        str = "truck1",
) -> DecisionOutput:
    """Run the multi-modal fusion and return a DecisionOutput."""
    ai2 = ai2_prediction or {}
    ai1 = ai1_vision or {}
    log = maintenance_log or []
    sop = sop_references or []

    # 1) Per-modality cause distributions
    p_ai2  = _ai2_cause_distribution(ai2)
    p_ai1  = _ai1_cause_distribution(ai1)
    p_hist = _history_penalty(log)

    # 2) Weighted average. AI2 dominates (it has the strongest signal);
    #    AI1 is a tiebreaker; history is a soft prior.
    weights = {"ai2": 0.65, "ai1": 0.20, "history": 0.15}
    fused = {m: (weights["ai2"] * p_ai2[m]
                + weights["ai1"] * p_ai1[m]
                + weights["history"] * p_hist[m])
             for m in FAILURE_MODES}
    s = sum(fused.values())
    if s > 0:
        fused = {k: v / s for k, v in fused.items()}

    root_cause = max(fused, key=fused.get)
    cause_confidence = round(fused[root_cause], 3)

    # 3) Risk score (0-100)
    p = float(ai2.get("failure_probability", 0.0))
    rul = float(ai2.get("rul_hours", 500.0))
    # Combine failure prob (weight 0.7) with low RUL (weight 0.3)
    rul_risk = max(0.0, 1.0 - min(rul, 24.0) / 24.0)
    risk_score = round(100 * (0.7 * p + 0.3 * rul_risk), 1)

    # 4) Risk band
    if risk_score >= 75:    risk_band = "critical"
    elif risk_score >= 50:  risk_band = "high"
    elif risk_score >= 25:  risk_band = "medium"
    else:                   risk_band = "low"

    # 5) Maintenance priority
    priority = risk_band
    if cause_confidence < 0.30 and risk_band in ("medium", "high"):
        priority = "medium"  # downweight if we don't know why

    # 6) Estimated downtime (next 24h)
    if p >= 0.85 or rul <= 1.0:
        downtime = 6.0
        action = f"HALT operations. Probable {root_cause} failure within 1h. Dispatch maintenance."
    elif p >= 0.65 or rul <= 8.0:
        downtime = 3.0
        action = f"Schedule service within 4h. Probable {root_cause} failure."
    elif p >= 0.40 or rul <= 24.0:
        downtime = 1.5
        action = f"Plan service in next shift. Monitor {root_cause} closely."
    else:
        downtime = 0.0
        action = f"Continue operations. Log {root_cause} reading for trend analysis."

    # 7) Reasoning narrative
    reasons = []
    reasons.append(f"Fused cause distribution: " + ", ".join(
        f"{m}={v:.2f}" for m, v in sorted(fused.items(), key=lambda x: -x[1])[:3]
    ))
    if ai2:
        reasons.append(f"AI2: failure_probability={p:.2f}, rul_hours={rul:.1f}")
    if ai1:
        reasons.append(f"AI1: carryback={ai1.get('carryback_pct', 0):.1f}%, "
                       f"loading_quality={ai1.get('loading_quality', 0):.2f}")
    if business_impact:
        reasons.append(f"Business: ROI={business_impact.get('roi_pct', 0):.0f}%, "
                       f"annual savings=INR {business_impact.get('annualised_savings_inr', 0):,.0f}")
    if log:
        reasons.append(f"History: {len(log)} recent events")

    return DecisionOutput(
        truck_id=truck_id,
        root_cause=root_cause,
        cause_confidence=cause_confidence,
        risk_score=risk_score,
        risk_band=risk_band,
        maintenance_priority=priority,
        estimated_downtime_hr=downtime,
        recommended_action=action,
        sop_references=sop,
        fused_probabilities={k: round(v, 3) for k, v in fused.items()},
        reasoning=" | ".join(reasons),
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    sample_ai2 = {
        "failure_probability": 0.78,
        "failure_class": 1,
        "rul_hours": 4.2,
        "top_features": [
            ("bearing_temp_roc_10", 0.45),
            ("vibration_rms_rmean_5", 0.30),
            ("engine_temp_ema", 0.15),
        ],
    }
    sample_ai1 = {
        "carryback_pct": 7.5,
        "payload_efficiency": 0.78,
        "loading_quality": 0.62,
        "hydraulic_stress_indicator": 0.55,
    }
    sample_log = [
        {"component": "bearing", "date": "2026-06-15"},
        {"component": "bearing", "date": "2026-05-02"},
    ]
    decision = fuse_and_decide(
        ai2_prediction=sample_ai2,
        ai1_vision=sample_ai1,
        maintenance_log=sample_log,
        sop_references=["SOP-BRG-04a: Bearing Overheat — Warning"],
    )
    print(json.dumps(decision.to_dict(), indent=2))

    # ------------------------------------------------------------------
    # NEW (Module 05 wiring): also exercise the CV engine end-to-end.
    # Renders a fresh synthetic frame, runs it through YOLOv8n, and feeds
    # the resulting ai1_vision dict back into fuse_and_decide. If ultralytics
    # isn't installed or the model isn't trained yet, we print a clear note
    # and skip — the synthetic test above already validates fusion logic.
    # ------------------------------------------------------------------
    print("\n=== AI Engine 1 (Computer Vision) end-to-end ===\n")
    try:
        import sys, tempfile, cv2
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from engine1_vision.synth_frames import render_frame
        from engine1_vision.infer import VisionIntelligence

        img, _ = render_frame()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, img)
            tmp_path = f.name

        svc = VisionIntelligence()
        if svc.load_error:
            print(f"  [skip] {svc.load_error}")
            print("         (Train the model with: python engine1_vision/train.py)")
        else:
            ai1_live = svc.predict(tmp_path)
            print("  ai1_vision (from YOLOv8 inference on a synthetic frame):")
            print(json.dumps(ai1_live, indent=4))
            decision_live = fuse_and_decide(
                ai2_prediction=sample_ai2,
                ai1_vision=ai1_live,
                maintenance_log=sample_log,
                sop_references=["SOP-BRG-04a: Bearing Overheat — Warning"],
            )
            print("\n  fused decision (with real YOLO output):")
            print(json.dumps(decision_live.to_dict(), indent=2))
    except Exception as e:
        print(f"  [skip] CV demo failed: {e}")
        import traceback; traceback.print_exc()
