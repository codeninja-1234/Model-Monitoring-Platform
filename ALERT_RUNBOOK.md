# Alert Runbook

## Severity levels
- **P1 Critical:** Accuracy < 0.70, sustained API failure, or severe fairness gap.
- **P2 Warning:** Fairness TPR gap > 0.12 or clear drift in several important features.
- **P3 Informational:** Small distribution changes without observed performance loss.

## Response procedure
1. Acknowledge the alert and record timestamp, model version, affected month/batch and alert value.
2. Check API health at `/health`, Prometheus targets at `http://localhost:9090/targets`, and Grafana request/error/latency panels.
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
| Feature drift | Evidently flags drift | P2 | Inspect changed features and pipeline |
| API errors | Any sustained increase | P1/P2 | Check logs and service health |
| p95 latency | > 1 second | P2 | Inspect resource usage and request load |

## Time to detection
For each planted issue:
- **Issue start:** first month in which the simulation introduces the change.
- **Detection time:** first monitoring batch in which the configured threshold is crossed.
- **Detection delay:** detection time minus issue start time.

Because this is monthly batch monitoring, report detection delay in monthly batches. Do not claim instant detection for the offline six-month simulation.
