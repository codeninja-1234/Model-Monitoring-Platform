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

| Month | Accuracy | Precision | Recall | F1 | ROC-AUC | TPR F | TPR M | TPR gap | Drift share | Alerts |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 (held-out baseline) | 0.6967 | 0.7174 | 0.7719 | 0.7437 | 0.7206 | 0.7442 | 0.8000 | 0.0558 | — | — |
| 2 | 0.7033 | 0.7262 | 0.7980 | 0.7604 | 0.7646 | 0.8089 | 0.7867 | 0.0221 | 0.2000 | — |
| 3 | 0.7075 | 0.7394 | 0.7700 | 0.7544 | — | 0.7742 | 0.7660 | 0.0082 | 0.6000 | Drift |
| 4 | 0.7292 | 0.7503 | 0.7968 | 0.7729 | — | 0.7971 | 0.7966 | 0.0004 | 0.8000 | Drift |
| 5 | 0.6917 | 0.6658 | 0.8019 | 0.7275 | — | 0.8393 | 0.7708 | 0.0685 | 0.8000 | Accuracy, outcome-gap, drift |
| 6 | 0.6650 | 0.6150 | 0.8014 | 0.6959 | — | 0.8809 | 0.7463 | 0.1345 | 0.8000 | Accuracy, TPR gap, outcome-gap, drift |
Figures: `reports/figures/performance_trend.png`, `fairness.png`, `drift_heatmap.png`, `detection_timeline.png`.

### What the numbers say

* **Accuracy:** Accuracy rises in months 2–4 (0.703–0.729), even though concept drift starts in month 4. It first falls below 0.70 in month 5 (0.692). The 95% confidence interval includes 0.70, while month 6 accuracy is 0.665, with a Wilson confidence interval upper bound below 0.70.

* **Precision:** Precision decreases from 0.750 in month 4 to 0.666 in month 5 and 0.615 in month 6, while recall remains around 0.80. This indicates that the model increasingly approves applications that are not actually approved in the labels.

* **Fairness:** The actual approval rate for women falls from 0.581 in month 4 to 0.469 in month 5 and 0.397 in month 6, while the rate for men remains around 0.56. Gender is not used as a model input. In month 6, the TPR is 0.881 for women and 0.746 for men, giving a gap of 0.1345. The FPR is also higher for women (0.501) than men (0.405). These results should be interpreted alongside the outcome-rate gap.

* **Sample-size caution:** The month-6 TPR-gap 95% bootstrap confidence interval is 0.0739–0.1935, which overlaps the 0.12 threshold. The month-1 baseline gap is 0.0558, with a confidence interval of 0.0042–0.1837. Treat the fairness alert as an investigation signal rather than conclusive proof of bias.

## 4. Time to detection

Monitoring is monthly batch monitoring: an issue planted in month *k* can first be alerted when month *k*'s batch is evaluated. Delay is counted in batches (0 = same batch as the issue start; ≈ 30 days from issue start to evaluation).

| Planted issue (start) | Signal / rule | First alert | Delay | ≈ Days | False positives |
|---|---|---:|---:|---:|---|
| Feature drift (month 3) | Original: any feature drifted, Evidently defaults | 3 | 0 | 30 | **month 2** |
| Feature drift (month 3) | Final: >=50% features drifted, JS for region | 3 | 0 | 30 | none |

| Concept drift (month 4) | Accuracy < 0.70 | 5 | 1 | 60 | none |
| Concept drift (month 4) | Accuracy < 0.70, statistically confirmed | 6 | 2 | 90 | none |
| Fairness issue (month 5) | TPR gap > 0.12 | 6 | 1 | 60 | none |
| Fairness issue (month 5) | Outcome-rate gap > 0.08 *(proposed early-warning signal)* | 5 | 0 | 30 | none |

**Honesty note:** the 0.08 outcome-gap threshold was chosen after looking at the simulated data (non-planted months stay ≤ 0.07), so its zero-delay result is illustrative, not a validated claim. In production, outcome labels also arrive late, which adds to every delay above.

**Delayed detections:** concept drift (1–2 batches) and fairness (1 batch).
**False positives:** month-2 drift alert in the original rule (fixed); `region` drift under Evidently's default chi-square (fixed); a single-feature flag of `credit_history` in month 2 that no longer raises an alert.

## 5. Threshold observations (for the runbook)
* The accuracy threshold 0.70 sits almost exactly on the baseline accuracy (0.697). Keep it as the P1 rule but confirm with the 95 % CI before escalating; consider a relative rule ("drop > 5 points below the best of the last 3 months") for a real deployment.
* The training/evaluation leakage identified in the initial implementation was corrected. The model is now trained on 900 records (75% of month 1), while the remaining 300 records (25%) are held out for baseline evaluation. Both training and monitoring use the same stratified split with random_state=42.

## 6. Reproduce
```bash
python src_generate_data.py && python src_train_model.py   # Person 1 (already in repo)
python monitor.py                                           # drift + metrics + alerts
python analyze_results.py                                   # validation, detection analysis, charts
python -m pytest -q                                         # unit + sanity tests
```
