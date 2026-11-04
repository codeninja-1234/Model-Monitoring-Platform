"""Monitoring analytics & validation (Person 2).

Run AFTER monitor.py:   python analyze_results.py

What it does
  1. Independently re-computes accuracy / precision / recall / F1 and group TPR
     from raw confusion-matrix counts and checks them against monitor.py output.
  2. Builds a fairness table (TPR, FPR, selection rate, outcome rate, gaps with
     95% bootstrap CIs and per-group sample sizes).
  3. Compares what the monitors flagged with the issues planted by
     src_generate_data.py  -> detection month, delay, false positives.
  4. Compares the final Evidently configuration with Evidently's defaults
     (per-feature confusion matrix against the planted ground truth).
  5. Saves tables, charts and a markdown summary under reports/ for the
     dashboard and project report.
Exit code is non-zero if any validation check fails.
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from evidently import Report  # noqa: E402
from evidently.presets import DataDriftPreset  # noqa: E402

import monitor  # noqa: E402  (re-uses thresholds and helpers)

REPORTS = Path("reports")
FIG = REPORTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

FEATURES = monitor.FEATURES
NUMERIC = ["age", "income", "debt_ratio", "credit_history"]
DAYS_PER_BATCH = 30

# Ground truth: copied from the logic in src_generate_data.py
#   shift = max(0, month - 2)      -> numeric features move from month 3
#   noise added when month >= 4    -> concept drift from month 4
#   gender penalty when month >= 5 -> fairness issue from month 5
PLANTED = {
    "feature_drift": 3,
    "concept_drift": 4,
    "fairness_issue": 5,
}

# colour-blind-safe palette (Okabe-Ito) + marker per series
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "red": "#D55E00", "purple": "#CC79A7", "grey": "#6b7280"}


# ------------------------------------------------------------------ helpers
def load_batches():
    data = pd.read_csv("data/six_months.csv")
    model = joblib.load("model.joblib")["model"]
    m1 = data[data.month == 1]
    _, base = train_test_split(m1, test_size=0.25, random_state=42, stratify=m1.approved)
    batches = {}
    for m in range(1, 7):
        b = base.copy() if m == 1 else data[data.month == m].copy()
        b["prediction"] = model.predict(b[FEATURES])
        b["probability"] = model.predict_proba(b[FEATURES])[:, 1]
        batches[m] = b
    return data, base, batches


def confusion(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return (int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum()),
            int(((y == 0) & (p == 0)).sum()), int(((y == 1) & (p == 0)).sum()))


def manual_metrics(y, p):
    tp, fp, tn, fn = confusion(y, p)
    acc = (tp + tn) / (tp + fp + tn + fn)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return acc, prec, rec, f1


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def bootstrap_gaps(b, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(b))
    tpr, outc = [], []
    for _ in range(n_boot):
        s = b.iloc[rng.choice(idx, len(idx))]
        g = monitor.group_metrics(s)
        tpr.append(monitor.gap(g, "tpr"))
        outc.append(monitor.gap(g, "outcome_rate"))
    q = lambda a: tuple(np.round(np.percentile(a, [2.5, 97.5]), 4))  # noqa: E731
    return q(tpr), q(outc)


def evidently_features(data, base, cat_method=None):
    """Run Evidently for months 2-6, return per-feature rows and dataset shares."""
    rows, shares = [], {}
    for m in range(2, 7):
        cur = data[data.month == m][FEATURES]
        kw = {"drift_share": 0.5}
        if cat_method:
            kw["cat_method"] = cat_method
        snap = Report([DataDriftPreset(**kw)]).run(cur, base[FEATURES])
        share, r = monitor.extract_drift(snap)
        shares[m] = share
        rows += [{"month": m, **x} for x in r]
    return pd.DataFrame(rows), shares


def first_true(series):
    for m, v in series.items():
        if v:
            return int(m)
    return None


# -------------------------------------------------------------- 1. validate
def validate(res, batches):
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    for m, b in batches.items():
        row = res[res.month == m].iloc[0]
        acc, prec, rec, f1 = manual_metrics(b.approved, b.prediction)
        for name, val in [("accuracy", acc), ("precision", prec), ("recall", rec), ("f1", f1)]:
            add(f"month {m} {name} matches manual calculation",
                abs(val - row[name]) < 6e-5, f"manual={val:.4f} saved={row[name]:.4f}")
        g = monitor.group_metrics(b)
        add(f"month {m} TPR gap matches manual calculation",
            abs(abs(g["F"]["tpr"] - g["M"]["tpr"]) - row.fairness_gap) < 6e-5,
            f"F={g['F']['tpr']:.4f} M={g['M']['tpr']:.4f} saved gap={row.fairness_gap:.4f}")
        add(f"month {m} group sizes add up to batch size",
            g["F"]["n"] + g["M"]["n"] == len(b) == row.n_rows, f"{g['F']['n']}+{g['M']['n']}={len(b)}")
        add(f"month {m} metrics within [0,1]",
            all(0 <= row[k] <= 1 for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]))
    add("month 1 is flagged as baseline (no production alerts)",
        res.iloc[0].evaluation_type == "held_out_baseline" and res.iloc[0].alerts == "[]")
    add("gender is NOT a model input", "gender" not in FEATURES)
    add("monitoring files exist for dashboard (monthly_metrics.csv, .json, drift_by_feature.csv)",
        all((REPORTS / f).exists() for f in
            ["monthly_metrics.csv", "monitoring_results.json", "drift_by_feature.csv"]))
    add("Evidently HTML report exists for months 2-6",
        all((REPORTS / f"drift_month_{m}.html").exists() for m in range(2, 7)))
    return pd.DataFrame(checks)


# ------------------------------------------------------------------- main
def main():
    res = pd.read_csv("reports/monthly_metrics.csv")
    data, base, batches = load_batches()

    # 1. validation ---------------------------------------------------------
    val = validate(res, batches)
    val.to_csv(REPORTS / "validation_checks.csv", index=False)
    n_fail = int((~val.passed).sum())
    print(f"[1] Validation: {len(val) - n_fail}/{len(val)} checks passed")
    for _, r in val[~val.passed].iterrows():
        print("    FAIL:", r.check, r.detail)

    # 2. fairness table -----------------------------------------------------
    fair_rows = []
    for m, b in batches.items():
        g = monitor.group_metrics(b)
        t_ci, o_ci = bootstrap_gaps(b)
        fair_rows.append({
            "month": m,
            "tpr_F": round(g["F"]["tpr"], 4), "tpr_M": round(g["M"]["tpr"], 4),
            "tpr_gap": round(abs(g["F"]["tpr"] - g["M"]["tpr"]), 4),
            "tpr_gap_ci_low": t_ci[0], "tpr_gap_ci_high": t_ci[1],
            "fpr_F": round(g["F"]["fpr"], 4), "fpr_M": round(g["M"]["fpr"], 4),
            "outcome_rate_F": round(g["F"]["outcome_rate"], 4),
            "outcome_rate_M": round(g["M"]["outcome_rate"], 4),
            "outcome_gap": round(abs(g["F"]["outcome_rate"] - g["M"]["outcome_rate"]), 4),
            "outcome_gap_ci_low": o_ci[0], "outcome_gap_ci_high": o_ci[1],
            "n_F": g["F"]["n"], "n_M": g["M"]["n"],
            "positives_F": g["F"]["n_positive"], "positives_M": g["M"]["n_positive"],
        })
    fair = pd.DataFrame(fair_rows)
    fair.to_csv(REPORTS / "fairness_detail.csv", index=False)
    print("[2] Fairness table saved (reports/fairness_detail.csv)")

    # accuracy CIs
    acc_ci = {}
    for m, b in batches.items():
        k = int((b.approved == b.prediction).sum())
        acc_ci[m] = wilson(k, len(b))
    res["acc_ci_low"] = [round(acc_ci[m][0], 4) for m in res.month]
    res["acc_ci_high"] = [round(acc_ci[m][1], 4) for m in res.month]

    # 3. Evidently: default configuration vs final configuration ------------
    final_feat = pd.read_csv(REPORTS / "drift_by_feature.csv")
    default_feat, default_share = evidently_features(data, base, cat_method=None)

    def confusion_vs_truth(df):
        tp = fp = fn = tn = 0
        for _, r in df.iterrows():
            truth = r.feature in NUMERIC and r.month >= PLANTED["feature_drift"]
            if r.drifted and truth: tp += 1
            elif r.drifted and not truth: fp += 1
            elif not r.drifted and truth: fn += 1
            else: tn += 1
        return tp, fp, fn, tn

    drift_conf = []
    for name, df in [("Evidently defaults (chi-square for region)", default_feat),
                     ("Final (Jensen-Shannon for region)", final_feat)]:
        tp, fp, fn, tn = confusion_vs_truth(df)
        drift_conf.append({"configuration": name, "true_pos": tp, "false_pos": fp,
                           "false_neg": fn, "true_neg": tn,
                           "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                           "recall": round(tp / (tp + fn), 3)})
    drift_conf = pd.DataFrame(drift_conf)
    drift_conf.to_csv(REPORTS / "drift_confusion.csv", index=False)
    print("[3] Per-feature drift vs planted truth:")
    print(drift_conf.to_string(index=False))

    fp_cells = default_feat[(default_feat.drifted) &
                            ~((default_feat.feature.isin(NUMERIC)) & (default_feat.month >= 3))]
    # 4. detection analysis --------------------------------------------------
    prod = res[res.month > 1].set_index("month")
    fair_i = fair.set_index("month")
    default_share_s = pd.Series(default_share)
    signals = [
        # name, issue, rule text, boolean series (months 2..6)
        ("Original rule: ANY feature drifted, Evidently default tests", "feature_drift",
         "drift_share > 0 (defaults)", default_share_s > 0),
        ("Final: dataset drift (>=50% features, JS for region)", "feature_drift",
         "drift_share >= 0.5", prod.evidently_drift_share >= monitor.DRIFT_SHARE_THRESHOLD),
        ("Accuracy < 0.70", "concept_drift", "accuracy < 0.70", prod.accuracy < monitor.ACCURACY_THRESHOLD),
        ("Accuracy < 0.70, statistically confirmed (95% CI upper bound < 0.70)", "concept_drift",
         "Wilson upper < 0.70", prod.acc_ci_high < monitor.ACCURACY_THRESHOLD),
        ("TPR gap (F vs M) > 0.12", "fairness_issue", "tpr_gap > 0.12",
         fair_i.loc[2:6, "tpr_gap"] > monitor.FAIRNESS_THRESHOLD),
        ("Outcome-rate gap (F vs M) > 0.08  [threshold set after seeing data]", "fairness_issue",
         "outcome_gap > 0.08", fair_i.loc[2:6, "outcome_gap"] > monitor.OUTCOME_GAP_THRESHOLD),
    ]
    det_rows = []
    for name, issue, rule, s in signals:
        s = s.astype(bool)
        s.index = s.index.astype(int)
        start = PLANTED[issue]
        first = first_true(s[s.index >= start])  # true detection = at/after planted start
        fp_months = [int(m) for m, v in s.items() if v and m < start]
        det_rows.append({
            "signal": name, "issue": issue, "planted_month": start, "rule": rule,
            "first_alert_month": first,
            "delay_batches": None if first is None else first - start,
            "approx_days_to_alert": None if first is None else (first - start + 1) * DAYS_PER_BATCH,
            "false_positive_months": ",".join(map(str, fp_months)) or "none",
            "alert_months": ",".join(str(m) for m, v in s.items() if v) or "none",
        })
    det = pd.DataFrame(det_rows)
    det.to_csv(REPORTS / "detection_analysis.csv", index=False)
    print("[4] Detection analysis:")
    print(det[["signal", "planted_month", "first_alert_month", "delay_batches",
               "false_positive_months"]].to_string(index=False))

    # 5. charts --------------------------------------------------------------
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .25, "figure.dpi": 150})
    months = list(range(1, 7))

    def mark_planted(ax, ymax=None):
        for label, m in [("feature drift", 3), ("concept drift", 4), ("fairness issue", 5)]:
            ax.axvline(m - .5, color=C["grey"], ls=":", lw=1)
        ax.set_xticks(months)
        ax.set_xlabel("Simulated month (month 1 = held-out baseline)")

    # (a) performance trend
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for col, name, color, mk in [("accuracy", "Accuracy", C["blue"], "o"),
                                 ("precision", "Precision", C["orange"], "s"),
                                 ("recall", "Recall", C["green"], "^"),
                                 ("f1", "F1", C["purple"], "D")]:
        ax.plot(res.month, res[col], marker=mk, color=color, lw=2, label=name)
    ax.axhline(0.70, color=C["red"], ls="--", lw=1.2)
    ax.text(6.05, 0.70, "alert 0.70", color=C["red"], va="center", fontsize=8)
    ax.fill_between(res.month, res.acc_ci_low, res.acc_ci_high, color=C["blue"], alpha=.12,
                    label="Accuracy 95% CI")
    mark_planted(ax)
    ax.set_ylabel("Score"); ax.set_title("Model quality by month (dotted lines = planted issue starts at months 3, 4, 5)", fontsize=9)
    ax.legend(ncol=5, fontsize=8, loc="lower left", frameon=False)
    ax.set_xlim(.7, 6.7)
    fig.tight_layout(); fig.savefig(FIG / "performance_trend.png"); plt.close(fig)

    # (b) fairness
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(fair.month, fair.tpr_F, marker="o", color=C["red"], lw=2, label="TPR female")
    ax.plot(fair.month, fair.tpr_M, marker="s", color=C["blue"], lw=2, label="TPR male")
    ax.set_ylim(.6, 1); ax.set_ylabel("True-positive rate"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("TPR by gender", fontsize=9); mark_planted(ax)
    ax = axes[1]
    ax.plot(fair.month, fair.tpr_gap, marker="o", color=C["purple"], lw=2, label="TPR gap")
    ax.fill_between(fair.month, fair.tpr_gap_ci_low, fair.tpr_gap_ci_high, color=C["purple"], alpha=.12)
    ax.plot(fair.month, fair.outcome_gap, marker="^", color=C["orange"], lw=2, label="Actual approval-rate gap")
    ax.fill_between(fair.month, fair.outcome_gap_ci_low, fair.outcome_gap_ci_high, color=C["orange"], alpha=.12)
    ax.axhline(.12, color=C["purple"], ls="--", lw=1); ax.text(1.0, .123, "TPR alert 0.12", fontsize=8, color=C["purple"])
    ax.axhline(.08, color=C["orange"], ls="--", lw=1); ax.text(1.0, .083, "outcome alert 0.08", fontsize=8, color=C["orange"])
    ax.set_ylabel("|F - M| gap"); ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Gaps with 95% bootstrap CI (shaded)", fontsize=9); mark_planted(ax)
    fig.tight_layout(); fig.savefig(FIG / "fairness.png"); plt.close(fig)

    # (c) drift heatmap (-log10 p / distance would mix scales -> show flag grid)
    piv = final_feat.pivot(index="feature", columns="month", values="drifted").reindex(FEATURES)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.imshow(piv.astype(float).values, cmap=matplotlib.colors.ListedColormap(["#e5e7eb", C["red"]]), aspect="auto")
    for i, f in enumerate(piv.index):
        for j, m in enumerate(piv.columns):
            ax.text(j, i, "DRIFT" if piv.loc[f, m] else "ok", ha="center", va="center", fontsize=8,
                    color="white" if piv.loc[f, m] else "#374151")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.grid(False)
    shares = [prod.evidently_drift_share.get(m) for m in piv.columns]
    ax.set_xlabel("Month   |   dataset drift share: " + "  ".join(f"m{m}={s:.0%}" for m, s in zip(piv.columns, shares)), fontsize=8)
    ax.set_title("Evidently feature drift vs month-1 baseline (final configuration)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "drift_heatmap.png"); plt.close(fig)

    # (d) detection timeline
    fig, ax = plt.subplots(figsize=(8, 3.6))
    rows = det[det.signal.str.startswith(("Final", "Accuracy < 0.70", "TPR", "Outcome"))
               & ~det.signal.str.contains("confirmed")].reset_index(drop=True)
    for i, r in rows.iterrows():
        y = len(rows) - i
        ax.plot([r.planted_month, r.planted_month], [y - .25, y + .25], color=C["grey"], lw=3)
        if r.first_alert_month:
            ax.plot([r.planted_month, r.first_alert_month], [y, y], color=C["grey"], lw=1.5)
            ax.plot(r.first_alert_month, y, "o", color=C["red"], ms=9)
            ax.text(r.first_alert_month + .12, y, f"month {int(r.first_alert_month)} (delay {int(r.delay_batches)})",
                    va="center", fontsize=8)
    ax.set_yticks(range(len(rows), 0, -1))
    ax.set_yticklabels([s.split("[")[0].replace(" (>=50% features, JS for region)", "").strip()
                        for s in rows.signal], fontsize=8)
    ax.set_xticks(months); ax.set_xlim(2.4, 7.6)
    ax.set_xlabel("Month   (grey bar = issue planted, red dot = first alert)")
    ax.set_title("Time to detection per monitoring signal", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "detection_timeline.png"); plt.close(fig)
    print("[5] Charts saved in reports/figures/")

    # 6. summary files -------------------------------------------------------
    summary = {
        "planted_issues": PLANTED,
        "thresholds": {"accuracy": monitor.ACCURACY_THRESHOLD, "tpr_gap": monitor.FAIRNESS_THRESHOLD,
                       "outcome_gap": monitor.OUTCOME_GAP_THRESHOLD,
                       "drift_share": monitor.DRIFT_SHARE_THRESHOLD},
        "validation": {"passed": int(val.passed.sum()), "total": len(val)},
        "detection": det.to_dict(orient="records"),
        "drift_confusion": drift_conf.to_dict(orient="records"),
        "evidently_default_false_positive_cells": fp_cells[["month", "feature", "score"]].round(5).to_dict(orient="records"),
        "monthly": res.drop(columns=["alerts"]).to_dict(orient="records"),
    }
    with open(REPORTS / "monitoring_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    md = ["# Monitoring results (auto-generated by analyze_results.py)\n",
          "## Monthly metrics\n",
          res[["month", "accuracy", "acc_ci_low", "acc_ci_high", "precision", "recall", "f1", "roc_auc",
               "evidently_drift_share", "drifted_features"]].to_markdown(index=False), "\n",
          "## Fairness (female vs male)\n", fair.to_markdown(index=False), "\n",
          "## Detection vs planted issues (delay in monthly batches)\n",
          det[["signal", "planted_month", "first_alert_month", "delay_batches", "approx_days_to_alert",
               "false_positive_months"]].to_markdown(index=False), "\n",
          "## Per-feature drift accuracy vs planted ground truth\n", drift_conf.to_markdown(index=False), "\n",
          "## Validation checks\n", f"{int(val.passed.sum())}/{len(val)} passed. See validation_checks.csv.\n"]
    (REPORTS / "MONITORING_ANALYSIS_TABLES.md").write_text("\n".join(md))
    print("[6] Saved summary JSON + markdown tables")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
