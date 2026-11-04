# Alert Runbook

## Severity levels
- **P1 Critical:** The latest evaluated accuracy is below 0.70. Stop treating the model as validated; escalate to the model owner and evaluate rollback or human review.
- **P2 Warning:** The latest evaluated absolute group TPR gap is above 0.12, or Evidently reports any drifted input features. Investigate before changing or retraining the model.
- **P3 Informational:** Small distribution changes without observed performance loss.

Prometheus evaluates the three model-quality rules every five seconds and requires a breach to persist for one minute before firing. The rules use the latest row in `reports/monthly_metrics.csv`; they do not page or send notifications because this local demo has no Alertmanager receiver configured. The API error and latency signals are dashboard-only.

## Response procedure
1. Acknowledge the alert and record timestamp, model version, affected month/batch and alert value.
2. Check API health at `http://localhost:5001/health`, Prometheus targets at `http://localhost:9090/targets`, active rules at `http://localhost:9090/alerts`, and Grafana request/error/latency panels.
3. Open the relevant Evidently HTML report. Identify which features changed and compare them with the month-1 reference.
4. Check labelled evaluation metrics: accuracy, precision, recall and F1. Do not treat drift alone as proof of model failure.
5. Inspect group-level TPR values and sample counts. Confirm that the gap is not caused by a very small sample.
6. If quality or fairness thresholds are breached, escalate to the model owner. Consider rollback to the last validated model, limiting automated decisions, or human review.
7. Investigate upstream data changes, label quality and changes in the real-world process. Retrain only after validating new data and repeating evaluation.
8. Record root cause, mitigation, owner and resolution time.

## Thresholds for this project
| Signal | Threshold | Severity | First action |
|---|---:|---|---|
| Accuracy | < 0.70 | P1 | Verify labels and evaluate rollback |
| Group TPR gap | > 0.12 | P2 | Inspect group metrics and sample sizes |
| Outcome-rate gap (F vs M) | > 0.08 | P2 (early warning) | Check label quality / upstream process for the affected group |
| Feature drift | Evidently dataset drift: >= 50% of features drifted (JS distance used for categoricals) | P2 | Inspect changed features and pipeline |
| API errors | Any sustained increase | P1/P2 | Check logs and service health |
| p95 latency | > 1 second | P2 | Inspect resource usage and request load |
| Latest monthly accuracy | < 0.70 for 1 minute | P1 / `ModelAccuracyDegraded` | Verify labels and evaluate rollback |
| Latest monthly group TPR gap | > 0.12 for 1 minute | P2 / `FairnessViolation` | Inspect group metrics and sample sizes |
| Latest monthly drifted-feature share | > 0 for 1 minute | P2 / `FeatureDriftDetected` | Inspect the month’s Evidently report and data pipeline |
| API errors | Dashboard signal; no alert rule | Informational | Check container logs and service health |
| p95 latency | Dashboard signal; no alert rule | Informational | Inspect resource usage and request load |

## Time to detection
Using the checked-in six-month results:

| Signal | Planted change | First threshold crossing | Batch delay |
|---|---:|---:|---:|
| Accuracy < 0.70 | Month 4 (additional concept noise) | Month 5 | 1 month |
| Fairness gap > 0.12 | Month 5 (group outcome shift) | Month 6 | 1 month |
| Feature drift share > 0 | Month 3 (population shift) | Month 2 | Crossed before the intentional shift; likely sample variation |

Because this is monthly batch monitoring, report detection delay in monthly batches. Do not claim instant detection for the offline six-month simulation.

### Measured detection delays (from `reports/detection_analysis.csv`)
| Issue (planted month) | Signal | First alert | Delay (batches) |
|---|---|---:|---:|
| Feature drift (3) | >= 50% features drifted | 3 | 0 |
| Concept drift (4) | Accuracy < 0.70 | 5 | 1 |
| Fairness issue (5) | TPR gap > 0.12 | 6 | 1 |
| Fairness issue (5) | Outcome-rate gap > 0.08 | 5 | 0 |

Before confirming an accuracy alert, check the 95% confidence interval (month 5: 0.693, CI 0.667-0.719 includes 0.70; month 6: 0.663, CI upper bound 0.689 is confirmed). See `MONITORING_ANALYSIS.md`.
These are retrospective batch results, not online detection times. The Prometheus rules evaluate the checked-in latest batch and start their one-minute hold only after the API loads it.
