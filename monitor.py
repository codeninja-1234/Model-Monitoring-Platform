
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split

from evidently import Report
from evidently.presets import DataDriftPreset


Path("reports").mkdir(exist_ok=True)

FEATURES = [
    "age",
    "income",
    "debt_ratio",
    "credit_history",
    "region"
]

ACCURACY_THRESHOLD = 0.70
FAIRNESS_THRESHOLD = 0.12


def fairness_gap(frame):
    rates = {}

    for group, part in frame.groupby("gender"):
        positives = part[part["approved"] == 1]

        if len(positives):
            rates[group] = float(
                (positives["prediction"] == 1).mean()
            )

    if "F" not in rates or "M" not in rates:
        return rates, None

    gap = abs(rates["F"] - rates["M"])
    return rates, gap


def extract_drift_share(snapshot):
    try:
        data = snapshot.dict()

        for metric in data.get("metrics", []):
            if "DriftedColumnsCount" in metric.get("metric_name", ""):
                value = metric.get("value", {})
                return float(value.get("share", 0.0))

        return None

    except Exception as e:
        print("Drift extraction error:", e)
        return None


def calculate_metrics(batch):
    accuracy = accuracy_score(
        batch["approved"], batch["prediction"]
    )

    precision = precision_score(
        batch["approved"],
        batch["prediction"],
        zero_division=0
    )

    recall = recall_score(
        batch["approved"],
        batch["prediction"],
        zero_division=0
    )

    f1 = f1_score(
        batch["approved"],
        batch["prediction"],
        zero_division=0
    )

    group_rates, gap = fairness_gap(batch)

    return accuracy, precision, recall, f1, group_rates, gap


def main():
    data = pd.read_csv("data/six_months.csv")

    # Use a held-out portion of month 1 as the baseline.
    month1 = data[data["month"] == 1].copy()

    _, baseline = train_test_split(
        month1,
        test_size=0.25,
        random_state=42,
        stratify=month1["approved"]
    )

    reference = baseline[FEATURES].copy()

    # Load the model trained on month 1.
    bundle = joblib.load("model.joblib")
    model = bundle["model"]

    results = []

    for month in range(1, 7):

        if month == 1:
            batch = baseline.copy()
        else:
            batch = data[data["month"] == month].copy()

        batch["prediction"] = model.predict(
            batch[FEATURES]
        )

        (
            accuracy,
            precision,
            recall,
            f1,
            group_rates,
            gap
        ) = calculate_metrics(batch)

        drift_share = None
        drift_detected = None

        # Compare every later month against the held-out baseline.
        if month > 1:
            report = Report([DataDriftPreset()])

            snapshot = report.run(
                batch[FEATURES],
                reference
            )

            snapshot.save_html(
                f"reports/drift_month_{month}.html"
            )

            with open(
                f"reports/drift_month_{month}.json", "w"
            ) as file:
                file.write(snapshot.json())

            drift_share = extract_drift_share(snapshot)

            if drift_share is not None:
                drift_detected = drift_share > 0

        alerts = []

        # Do not generate production alerts for the baseline month.
        if month > 1 and accuracy < ACCURACY_THRESHOLD:
            alerts.append(
                "CRITICAL: accuracy below 0.70"
            )

        if gap is not None and gap > FAIRNESS_THRESHOLD:
            alerts.append(
                "WARNING: fairness TPR gap above 0.12"
            )

        if drift_detected is True:
            alerts.append(
                "WARNING: Evidently detected feature drift"
            )
        elif drift_detected is None and month > 1:
            alerts.append(
                "CHECK: drift result unavailable"
            )

        result = {
            "month": month,
            "evaluation_type": (
                "held_out_baseline"
                if month == 1
                else "monthly_production"
            ),
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "fairness_tpr": group_rates,
            "fairness_gap": (
                round(float(gap), 4)
                if gap is not None
                else None
            ),
            "evidently_drift_share": (
                round(float(drift_share), 4)
                if drift_share is not None
                else None
            ),
            "drift_detected": drift_detected,
            "alerts": alerts
        }

        results.append(result)

        print(
            f"Month {month}: "
            f"accuracy={accuracy:.3f}, "
            f"fairness_gap={gap}, "
            f"drift_share={drift_share}, "
            f"alerts={alerts}"
        )

    # Save monitoring results.
    pd.DataFrame(results).to_csv(
        "reports/monthly_metrics.csv",
        index=False
    )

    with open(
        "reports/monitoring_results.json", "w"
    ) as file:
        json.dump(results, file, indent=2)

    print("\nMonitoring reports saved successfully.")


if __name__ == "__main__":
    main()
