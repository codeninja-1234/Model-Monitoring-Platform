"""Run from the repo root after `python monitor.py`:  python -m pytest -q"""
import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import monitor  # noqa: E402


@pytest.fixture(scope="module")
def res():
    path = ROOT / "reports" / "monthly_metrics.csv"
    if not path.exists():
        pytest.skip("run monitor.py first")
    return pd.read_csv(path)


def toy():
    return pd.DataFrame({
        "gender":     ["F"] * 4 + ["M"] * 4,
        "approved":   [1, 1, 0, 0, 1, 1, 0, 0],
        "prediction": [1, 0, 0, 1, 1, 1, 0, 0],
        "probability": [.9, .4, .2, .8, .9, .8, .1, .2],
    })


def test_hand_checked_metrics():
    m = monitor.calculate_metrics(toy())
    # tp=3 fp=1 tn=3 fn=1
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["f1"] == pytest.approx(0.75)


def test_hand_checked_fairness():
    m = monitor.calculate_metrics(toy())
    assert m["groups"]["F"]["tpr"] == pytest.approx(0.5)   # 1 of 2 positives found
    assert m["groups"]["M"]["tpr"] == pytest.approx(1.0)
    assert m["tpr_gap"] == pytest.approx(0.5)
    assert m["groups"]["F"]["fpr"] == pytest.approx(0.5)
    assert m["groups"]["M"]["fpr"] == pytest.approx(0.0)


def test_six_months_present(res):
    assert list(res.month) == [1, 2, 3, 4, 5, 6]


def test_columns_used_by_dashboard(res):
    for col in ["month", "accuracy", "fairness_gap", "evidently_drift_share"]:
        assert col in res.columns


def test_baseline_month_has_no_alerts(res):
    assert ast.literal_eval(res.loc[0, "alerts"]) == []


def test_no_drift_alert_before_planted_drift(res):
    assert not res.loc[res.month == 2, "drift_detected"].iloc[0]


def test_planted_drift_detected_in_month_3(res):
    assert res.loc[res.month == 3, "drift_detected"].iloc[0]


def test_region_never_drifts():
    df = pd.read_csv(ROOT / "reports" / "drift_by_feature.csv")
    assert not df[df.feature == "region"].drifted.any()


def test_accuracy_alert_in_final_months(res):
    last = res[res.month >= 5]
    assert (last.accuracy < monitor.ACCURACY_THRESHOLD).all()


def test_fairness_alert_month_6(res):
    assert res.loc[res.month == 6, "fairness_gap"].iloc[0] > monitor.FAIRNESS_THRESHOLD
    assert res.loc[res.month <= 5, "fairness_gap"].max() <= monitor.FAIRNESS_THRESHOLD
