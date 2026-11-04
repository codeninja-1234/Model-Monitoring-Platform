# Monitoring & Analytics Results (Person 2)

All numbers come from `python monitor.py` followed by `python analyze_results.py`
(seeded, fully reproducible). Raw tables: `reports/*.csv`, charts: `reports/figures/*.png`.

## 1. What was verified

| Item | Result |
|---|---|
| Independent recomputation of accuracy/precision/recall/F1 and group TPR from confusion-matrix counts | **46/46 checks pass** (`reports/validation_checks.csv`) |
| Unit tests with hand-computed toy data + result sanity tests | **10/10 pass** (`python -m pytest -q`) |
| Evidently reports exist for months 2–6; results saved as CSV/JSON | Yes (`reports/`) |
| `app.py` still loads `monthly_metrics.csv` and exposes the Grafana gauges | Yes |

## 2. Evidently integration – problems found and fixed

| # | Finding | Evidence | Fix in `monitor.py` |
|---|---|---|---|
| 1 | **Dataset drift alert fired when ANY single column drifted** (`drift_share > 0`). Month 2 has *no* planted change but alerted. | Month 2: only `credit_history` flagged (K-S p = 0.029, one of five tests at α = 0.05). | Alert when **≥ 50 % of features** drift (Evidently's own dataset-drift default): `DRIFT_SHARE_THRESHOLD = 0.5`. |
| 2 | **Evidently's default chi-square test flags `region` although it never changes.** Reference = 300 rows, current = 1 200 rows. | `region` flagged in months 2, 3, 5 (p = 0.0003, 0.003, 0.003). A standard chi-square homogeneity test (scipy) gives p = 0.28, 0.43, 0.43. | `cat_method="jensenshannon"` for categorical columns. `region` is never flagged (JS distance ≤ 0.045 vs threshold 0.1). |
| 3 | Evidently needs `pandas<3` (as pinned in `requirements.txt`); with pandas 3 the `region` column is silently dropped from the report. | Share became 0.25 (4 columns) instead of 0.4 (5 columns). | Keep the pinned versions; use the venv from the README. |

### Per-feature accuracy against the planted truth (months 2–6, 5 features = 25 cells)
Truth = the four numeric features shift from month 3; `region` never shifts.

| Configuration | TP | FP | FN | TN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Evidently defaults | 15 | 4 | 1 | 5 | 0.79 | 0.94 |
| **Final (JS for categorical)** | 15 | **1** | 1 | 8 | **0.94** | 0.94 |

Remaining false positive: `credit_history` in month 2 (chance, α = 0.05).
Remaining false negative: `age` in month 3 (planted shift is only +1 year, SD = 11, undetectable with a 300-row reference).

## 3. Monthly results (final configuration)

| Month | Accuracy (95 % CI) | Precision | Recall | F1 | ROC-AUC | TPR F | TPR M | TPR gap | Drift share | Alerts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 (baseline) | 0.697 (0.642–0.746) | 0.715 | 0.778 | 0.745 | 0.724 | 0.744 | 0.812 | 0.068 | – | – |
| 2 | 0.709 (0.683–0.734) | 0.733 | 0.797 | 0.764 | 0.766 | 0.809 | 0.784 | 0.025 | 0.2 | – |
| 3 | 0.710 (0.684–0.735) | 0.742 | 0.771 | 0.756 | 0.767 | 0.780 | 0.763 | 0.017 | 0.6 | Drift |
| 4 | 0.727 (0.701–0.751) | 0.748 | 0.795 | 0.771 | 0.792 | 0.794 | 0.797 | 0.003 | 0.8 | Drift |
| 5 | 0.693 (0.667–0.719) | 0.668 | 0.800 | 0.728 | 0.773 | 0.836 | 0.771 | 0.065 | 0.8 | Accuracy, outcome-gap, drift |
| 6 | 0.663 (0.635–0.689) | 0.613 | 0.800 | 0.694 | 0.741 | 0.881 | 0.743 | **0.137** | 0.8 | Accuracy, TPR gap, outcome-gap, drift |

Figures: `reports/figures/performance_trend.png`, `fairness.png`, `drift_heatmap.png`, `detection_timeline.png`.

### What the numbers say
* **Accuracy alone is a weak, late signal.** It *rises* in months 2–4 (0.709–0.727) because the shifted population is easier to classify, even though concept drift starts in month 4. It first crosses 0.70 in month 5, but 0.693 is inside the 95 % CI (0.667–0.719) – statistically only month 6 (0.663, CI upper bound 0.689) is confirmed.
* **Precision is the metric that degrades**: 0.748 (m4) → 0.668 (m5) → 0.613 (m6) while recall stays ≈ 0.80. The label rate changes (actual approvals fall) but the model keeps approving at the same rate, i.e. it over-approves.
* **Fairness:** actual approval rate for women falls 0.58 → 0.47 → 0.40 while men stay ≈ 0.56. The model never sees gender, so its TPR for women actually **rises** (0.79 → 0.88) and the gap is **in favour of women**: the model over-approves women relative to their true outcomes (FPR 0.51 vs 0.41 in month 6). The TPR gap is therefore a real but indirect symptom; a bias in the *labels* shows up first in the outcome-rate gap.
* **Sample-size caution (runbook step 5):** the TPR-gap 95 % bootstrap CI in month 6 is 0.078–0.196, i.e. it overlaps the 0.12 threshold; the month-1 baseline gap (0.068) has CI 0.004–0.192 with only 85–86 positives per group. Treat the fairness alert as "investigate", which matches its P2 severity.

## 4. Time to detection

Monitoring is monthly batch monitoring: an issue planted in month *k* can first be alerted when month *k*'s batch is evaluated. Delay is counted in batches (0 = same batch as the issue start; ≈ 30 days from issue start to evaluation).

| Planted issue (start) | Signal / rule | First alert | Delay | ≈ Days | False positives |
|---|---|---:|---:|---:|---|
| Feature drift (month 3) | Original: any feature drifted, Evidently defaults | 3 | 0 | 30 | **month 2** |
| Feature drift (month 3) | **Final:** ≥ 50 % features drifted | **3** | **0** | 30 | none |
| Concept drift (month 4) | Accuracy < 0.70 | 5 | 1 | 60 | none |
| Concept drift (month 4) | Accuracy < 0.70, statistically confirmed | 6 | 2 | 90 | none |
| Fairness issue (month 5) | TPR gap > 0.12 | 6 | 1 | 60 | none |
| Fairness issue (month 5) | Outcome-rate gap > 0.08 *(proposed early-warning signal)* | 5 | 0 | 30 | none |

**Honesty note:** the 0.08 outcome-gap threshold was chosen after looking at the simulated data (non-planted months stay ≤ 0.07), so its zero-delay result is illustrative, not a validated claim. In production, outcome labels also arrive late, which adds to every delay above.

**Delayed detections:** concept drift (1–2 batches) and fairness (1 batch).
**False positives:** month-2 drift alert in the original rule (fixed); `region` drift under Evidently's default chi-square (fixed); a single-feature flag of `credit_history` in month 2 that no longer raises an alert.

## 5. Threshold observations (for the runbook)
* The accuracy threshold 0.70 sits almost exactly on the baseline accuracy (0.697). Keep it as the P1 rule but confirm with the 95 % CI before escalating; consider a relative rule ("drop > 5 points below the best of the last 3 months") for a real deployment.
* Model was trained on all of month 1, so the 300-row "held-out" baseline is actually part of the training data (small effect for a 6-parameter logistic regression). For a stricter protocol, train on the other 75 % of month 1 using the same split.

## 6. Reproduce
```bash
python src_generate_data.py && python src_train_model.py   # Person 1 (already in repo)
python monitor.py                                           # drift + metrics + alerts
python analyze_results.py                                   # validation, detection analysis, charts
python -m pytest -q                                         # unit + sanity tests
```
