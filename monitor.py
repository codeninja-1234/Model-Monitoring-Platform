"""Offline monitoring pipeline (Person 2 – Monitoring and Analytics).

For every simulated month this script computes
  * data / feature drift against the held-out month-1 baseline (Evidently),
  * model quality: accuracy, precision, recall, F1, ROC-AUC,
  * fairness across gender: TPR (equal opportunity) plus FPR, selection-rate
    and actual-outcome-rate gaps, with per-group sample sizes,
and applies the alert rules documented in ALERT_RUNBOOK.md.

Outputs (all under reports/):
  monthly_metrics.csv      one row per month  -> read by app.py / Grafana
  monitoring_results.json  same data as JSON
  drift_by_feature.csv     one row per (month, feature) with test, score, flag
  drift_month_N.html/json  full Evidently reports

Run:  python monitor.py       (then:  python analyze_results.py)
"""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from evidently import Report
from evidently.presets import DataDriftPreset

Path("reports").mkdir(exist_ok=True)

FEATURES = ["age", "income", "debt_ratio", "credit_history", "region"]

# ---------------------------------------------------------------- thresholds
ACCURACY_THRESHOLD = 0.70        # P1: accuracy below this
FAIRNESS_THRESHOLD = 0.12        # P2: |TPR_F - TPR_M| above this
OUTCOME_GAP_THRESHOLD = 0.08     # P2 early warning: |approval-rate_F - approval-rate_M|
DRIFT_SHARE_THRESHOLD = 0.5      # P2: dataset drift when >= 50% of features drift
                                 # (Evidently's own default for dataset drift)
# Evidently's default categorical test (chi-square) mis-fires when the
# reference (300 rows) and current (1200 rows) samples differ in size: it
# flagged `region` in months 2, 3 and 5 although region never changes.
# Jensen-Shannon distance is robust to this, so it is used for categoricals.
CAT_DRIFT_METHOD = "jensenshannon"


# ------------------------------------------------------------------ fairness
def group_metrics(frame):
    """Per-gender TPR, FPR, selection rate, outcome rate and counts."""
    out = {}
    for group, part in frame.groupby("gender"):
        pos = part[part["approved"] == 1]
        neg = part[part["approved"] == 0]
        out[group] = {
            "n": int(len(part)),
            "n_positive": int(len(pos)),
            "tpr": float((pos["prediction"] == 1).mean()) if len(pos) else None,
            "fpr": float((neg["prediction"] == 1).mean()) if len(neg) else None,
            "selection_rate": float((part["prediction"] == 1).mean()),
            "outcome_rate": float((part["approved"] == 1).mean()),
        }
    return out


def gap(groups, key):
    if "F" not in groups or "M" not in groups:
        return None
    a, b = groups["F"][key], groups["M"][key]
    if a is None or b is None:
        return None
    return abs(a - b)


# --------------------------------------------------------------------- drift
def _drifted(value, method, threshold):
    """p-value tests drift when value < threshold, distance tests when >=."""
    if "p_value" in method or method in ("chisquare", "z", "fisher_exact", "g_test"):
        return value < threshold
    return value >= threshold


def extract_drift(snapshot):
    """Return (drift_share, per-feature rows) from an Evidently snapshot."""
    try:
        data = snapshot.dict()
        share, rows = None, []
        for metric in data.get("metrics", []):
            name = metric.get("metric_name", "")
            if "DriftedColumnsCount" in name:
                share = float(metric["value"].get("share", 0.0))
            elif name.startswith("ValueDrift"):
                cfg = metric["config"]
                value = float(metric["value"])
                rows.append({
                    "feature": cfg["column"],
                    "method": cfg["method"],
                    "threshold": cfg["threshold"],
                    "score": value,
                    "drifted": bool(_drifted(value, cfg["method"], cfg["threshold"])),
                })
        return share, rows
    except Exception as exc:  # pragma: no cover - defensive
        print("Drift extraction error:", exc)
        return None, []


# ------------------------------------------------------------------- metrics
def calculate_metrics(batch):
    y, pred = batch["approved"], batch["prediction"]
    groups = group_metrics(batch)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, batch["probability"]),
        "groups": groups,
        "tpr_gap": gap(groups, "tpr"),
        "fpr_gap": gap(groups, "fpr"),
        "selection_gap": gap(groups, "selection_rate"),
        "outcome_gap": gap(groups, "outcome_rate"),
    }


def r4(x):
    return None if x is None else round(float(x), 4)


def main():
    data = pd.read_csv("data/six_months.csv")

    # Held-out portion of month 1 is the baseline / Evidently reference.
    month1 = data[data["month"] == 1].copy()
    _, baseline = train_test_split(
        month1, test_size=0.25, random_state=42, stratify=month1["approved"]
    )
    reference = baseline[FEATURES].copy()

    model = joblib.load("model.joblib")["model"]

    results, feature_rows = [], []

    for month in range(1, 7):
        batch = baseline.copy() if month == 1 else data[data["month"] == month].copy()
        batch["prediction"] = model.predict(batch[FEATURES])
        batch["probability"] = model.predict_proba(batch[FEATURES])[:, 1]

        m = calculate_metrics(batch)
        groups = m["groups"]

        drift_share, drift_detected, drifted_features = None, None, []

        if month > 1:
            report = Report([DataDriftPreset(
                drift_share=DRIFT_SHARE_THRESHOLD,
                cat_method=CAT_DRIFT_METHOD,
            )])
            snapshot = report.run(batch[FEATURES], reference)
            snapshot.save_html(f"reports/drift_month_{month}.html")
            with open(f"reports/drift_month_{month}.json", "w") as file:
                file.write(snapshot.json())

            drift_share, rows = extract_drift(snapshot)
            for row in rows:
                feature_rows.append({"month": month, **row})
            drifted_features = [r["feature"] for r in rows if r["drifted"]]
            if drift_share is not None:
                drift_detected = drift_share >= DRIFT_SHARE_THRESHOLD

        # ------------------------------------------------------------ alerts
        alerts = []
        # No production alerts for the baseline month itself.
        if month > 1 and m["accuracy"] < ACCURACY_THRESHOLD:
            alerts.append("CRITICAL: accuracy below 0.70")
        if m["tpr_gap"] is not None and m["tpr_gap"] > FAIRNESS_THRESHOLD:
            alerts.append("WARNING: fairness TPR gap above 0.12")
        if month > 1 and m["outcome_gap"] is not None and m["outcome_gap"] > OUTCOME_GAP_THRESHOLD:
            alerts.append("WARNING: outcome-rate gap F vs M above 0.08")
        if drift_detected is True:
            alerts.append("WARNING: Evidently dataset drift (>=50% of features)")
        elif drift_detected is None and month > 1:
            alerts.append("CHECK: drift result unavailable")

        results.append({
            "month": month,
            "evaluation_type": "held_out_baseline" if month == 1 else "monthly_production",
            "n_rows": int(len(batch)),
            "accuracy": r4(m["accuracy"]),
            "precision": r4(m["precision"]),
            "recall": r4(m["recall"]),
            "f1": r4(m["f1"]),
            "fairness_tpr": {g: groups[g]["tpr"] for g in groups},
            "fairness_gap": r4(m["tpr_gap"]),
            "evidently_drift_share": r4(drift_share),
            "drift_detected": drift_detected,
            "alerts": alerts,
            # --- additional columns (appended; existing ones unchanged) ---
            "roc_auc": r4(m["roc_auc"]),
            "fpr_gap": r4(m["fpr_gap"]),
            "selection_rate_gap": r4(m["selection_gap"]),
            "outcome_rate_gap": r4(m["outcome_gap"]),
            "n_female": groups["F"]["n"],
            "n_male": groups["M"]["n"],
            "n_positive_female": groups["F"]["n_positive"],
            "n_positive_male": groups["M"]["n_positive"],
            "drifted_features": ",".join(drifted_features),
        })

        print(
            f"Month {month}: acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
            f"rec={m['recall']:.3f} f1={m['f1']:.3f} tpr_gap={r4(m['tpr_gap'])} "
            f"outcome_gap={r4(m['outcome_gap'])} drift_share={drift_share} "
            f"drifted={drifted_features} alerts={alerts}"
        )

    pd.DataFrame(results).to_csv("reports/monthly_metrics.csv", index=False)
    pd.DataFrame(feature_rows).to_csv("reports/drift_by_feature.csv", index=False)
    with open("reports/monitoring_results.json", "w") as file:
        json.dump(results, file, indent=2)

    print("\nMonitoring reports saved successfully.")


if __name__ == "__main__":
    main()
