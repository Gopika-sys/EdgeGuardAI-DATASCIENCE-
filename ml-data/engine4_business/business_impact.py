"""
EdgeGuard AI — Business Impact Engine (Module 09 of ml work.txt)

PURPOSE
-------
Convert AI engine predictions into business value. Industrial managers
and CFOs do not care about F1 or MAE — they care about cost savings,
ROI, and avoided downtime. This module translates every AI output
into dollars / INR and a priority ranking.

INDUSTRY INPUT ASSUMPTIONS (Tata Signa 4825.TK, open-pit mining)
---------------------------------------------------------------
Source: industry benchmark figures for open-pit heavy mining truck
operations. These are editable constants; adjust for your deployment.

AVG_CYCLE_DOWNHOURS_NO_PRED    8.0   # unplanned downtime without AI
AVG_CYCLE_DOWNHOURS_WITH_AI   2.0   # with predictive maintenance
HOURLY_REVENUE_PER_TRUCK_INR  18000 # INR per hour (revenue + production)
MAINT_COST_PER_SERVICE_INR    45000 # planned service cost
UNPLANNED_MAINT_MULTIPLIER    3.5   # unscheduled work is ~3.5x planned
DAILY_FUEL_LITRES             1100  # baseline
FUEL_COST_PER_LITRE_INR       95    # India diesel ~INR 95/L
UNPLANNED_DOWN_COST_PER_HR    35000 # tow, idle fleet, contract penalties
RUL_HOURS_OPTIMAL_SERVICE     120   # sweet spot for service window

ROI ASSUMPTION
--------------
ROI = (avoided cost + savings) / (platform cost)

Platform cost is taken as a fixed annual figure (annualised hardware +
ML platform + integration). Override via EDGEGUARD_ANNUAL_PLATFORM_COST
environment variable. Default INR 6,000,000 (USD ~72k at hackathon
benchmark) — appropriate for a small 5-truck pilot.
"""

import os
from dataclasses import dataclass, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
AVG_CYCLE_DOWNHOURS_NO_PRED   = 8.0
AVG_CYCLE_DOWNHOURS_WITH_AI  = 2.0
HOURLY_REVENUE_PER_TRUCK_INR = 18_000
MAINT_COST_PER_SERVICE_INR   = 45_000
UNPLANNED_MAINT_MULTIPLIER   = 3.5
DAILY_FUEL_LITRES            = 1_100
FUEL_COST_PER_LITRE_INR      = 95
UNPLANNED_DOWN_COST_PER_HR   = 35_000
RUL_HOURS_OPTIMAL_SERVICE    = 120

# Default platform cost — overridable via env
DEFAULT_ANNUAL_PLATFORM_COST = 6_000_000  # INR
ANNUAL_PLATFORM_COST = float(
    os.environ.get("EDGEGUARD_ANNUAL_PLATFORM_COST", DEFAULT_ANNUAL_PLATFORM_COST)
)


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------
@dataclass
class BusinessImpact:
    truck_id:            str
    failure_probability: float
    rul_hours:           float

    # Core KPIs
    maintenance_priority:    str  # "critical" | "high" | "medium" | "low"
    estimated_downtime_hr:   float
    downtime_avoided_hr:     float
    maintenance_savings_inr: float
    fuel_savings_inr:        float
    production_recovered_inr: float
    unplanned_cost_avoided_inr: float
    annualised_savings_inr:  float
    roi_pct:                 float
    payback_months:          float

    # Decision rationale
    recommended_action:      str
    reasoning:               str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core calculation
# ---------------------------------------------------------------------------
def compute_business_impact(
    failure_probability: float,
    rul_hours: float,
    truck_id: str = "truck1",
    fleet_size: int = 5,
) -> BusinessImpact:
    """Translate AI predictions to business KPIs.

    Args:
        failure_probability: 0-1 from prob_regressor
        rul_hours: Remaining useful life in hours
        truck_id: for reporting only
        fleet_size: number of trucks in the fleet (used to annualise savings)

    Returns a BusinessImpact dataclass with the full KPI set.
    """
    # --- 1. Maintenance priority ----------------------------------------
    if failure_probability >= 0.85 or rul_hours <= 1.0:
        priority = "critical"
        action = "HALT operations immediately. Dispatch maintenance team."
    elif failure_probability >= 0.65 or rul_hours <= 8.0:
        priority = "high"
        action = "Schedule service within 4 operating hours."
    elif failure_probability >= 0.40 or rul_hours <= 24.0:
        priority = "medium"
        action = "Schedule service within next shift (24h)."
    else:
        priority = "low"
        action = "Continue normal operations. Log reading."

    # --- 2. Estimated downtime & cost avoidance -------------------------
    # If we predict failure and service proactively, we save the difference
    est_downtime_hr = max(0.0, rul_hours) * 0.1  # service takes ~10% of remaining RUL
    downtime_avoided_hr = max(0.0, AVG_CYCLE_DOWNHOURS_NO_PRED - est_downtime_hr)

    # Cost of unplanned failure (no AI)
    unplanned_cost_no_ai = (
        AVG_CYCLE_DOWNHOURS_NO_PRED * UNPLANNED_DOWN_COST_PER_HR
        + MAINT_COST_PER_SERVICE_INR * UNPLANNED_MAINT_MULTIPLIER
    )
    # Cost of planned service (with AI)
    planned_cost_with_ai = (
        est_downtime_hr * HOURLY_REVENUE_PER_TRUCK_INR
        + MAINT_COST_PER_SERVICE_INR
    )
    unplanned_cost_avoided = max(0.0, unplanned_cost_no_ai - planned_cost_with_ai)

    # --- 3. Production recovery -----------------------------------------
    # Production recovered = hours of operation saved x hourly revenue
    production_recovered = downtime_avoided_hr * HOURLY_REVENUE_PER_TRUCK_INR

    # --- 4. Maintenance savings ----------------------------------------
    # Ratio: planned service is X% the cost of unplanned
    maintenance_savings = (
        MAINT_COST_PER_SERVICE_INR * UNPLANNED_MAINT_MULTIPLIER
        - MAINT_COST_PER_SERVICE_INR
    ) * (failure_probability)  # scaled by confidence: a true positive = real savings

    # --- 5. Fuel savings -----------------------------------------------
    # Anomalous operations burn 8-12% more fuel; AI flag → driver corrects
    fuel_savings = (
        DAILY_FUEL_LITRES * FUEL_COST_PER_LITRE_INR * 0.08
        * failure_probability
        * 30  # 30 days/month
    )

    # --- 6. Annualised savings + ROI -----------------------------------
    # If probability >= 0.5, we assume ~12 prevented incidents/year/fleet
    expected_events_per_year = max(1.0, 12.0 * failure_probability) * fleet_size
    annual_savings = (
        (unplanned_cost_avoided + maintenance_savings + fuel_savings + production_recovered)
        * expected_events_per_year
    )
    roi_pct = (annual_savings - ANNUAL_PLATFORM_COST) / ANNUAL_PLATFORM_COST * 100.0
    payback_months = (
        ANNUAL_PLATFORM_COST / max(annual_savings / 12.0, 1.0)
        if annual_savings > 0 else float("inf")
    )

    # --- 7. Reasoning --------------------------------------------------
    reasoning = (
        f"Failure probability {failure_probability:.0%}, RUL {rul_hours:.1f}h. "
        f"Predicted incident would cost INR {unplanned_cost_no_ai:,.0f} (unplanned) "
        f"vs. INR {planned_cost_with_ai:,.0f} (planned) — savings INR "
        f"{unplanned_cost_avoided:,.0f} per avoided incident. "
        f"Annualised fleet impact: INR {annual_savings:,.0f}. "
        f"ROI vs. platform cost: {roi_pct:.0f}%."
    )

    return BusinessImpact(
        truck_id=truck_id,
        failure_probability=failure_probability,
        rul_hours=rul_hours,
        maintenance_priority=priority,
        estimated_downtime_hr=round(est_downtime_hr, 2),
        downtime_avoided_hr=round(downtime_avoided_hr, 2),
        maintenance_savings_inr=round(maintenance_savings, 0),
        fuel_savings_inr=round(fuel_savings, 0),
        production_recovered_inr=round(production_recovered, 0),
        unplanned_cost_avoided_inr=round(unplanned_cost_avoided, 0),
        annualised_savings_inr=round(annual_savings, 0),
        roi_pct=round(roi_pct, 1),
        payback_months=round(payback_months, 1),
        recommended_action=action,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Fleet-level rollup
# ---------------------------------------------------------------------------
def fleet_business_impact(per_truck: list[BusinessImpact]) -> dict:
    """Aggregate per-truck business impacts into a fleet rollup."""
    if not per_truck:
        return {"error": "no trucks provided"}
    n = len(per_truck)
    return {
        "fleet_size":                  n,
        "total_annualised_savings_inr": round(sum(t.annualised_savings_inr for t in per_truck), 0),
        "total_downtime_avoided_hr":    round(sum(t.downtime_avoided_hr for t in per_truck), 2),
        "avg_roi_pct":                  round(sum(t.roi_pct for t in per_truck) / n, 1),
        "critical_trucks":              sum(1 for t in per_truck if t.maintenance_priority == "critical"),
        "high_priority_trucks":         sum(1 for t in per_truck if t.maintenance_priority == "high"),
        "truck_breakdown":              [t.to_dict() for t in per_truck],
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        # (prob, rul, expected_priority)
        (0.95, 0.5, "critical"),
        (0.70, 6.0, "high"),
        (0.50, 18.0, "medium"),
        (0.10, 200.0, "low"),
    ]
    for prob, rul, expected in test_cases:
        impact = compute_business_impact(prob, rul)
        ok = "OK " if impact.maintenance_priority == expected else "FAIL"
        print(f"{ok} prob={prob:.2f} rul={rul:.1f}h -> "
              f"priority={impact.maintenance_priority} "
              f"ROI={impact.roi_pct:.0f}% "
              f"savings/yr=INR {impact.annualised_savings_inr:,.0f}")
        print(f"    action: {impact.recommended_action}")
