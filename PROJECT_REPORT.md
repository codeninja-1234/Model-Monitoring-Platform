# Model Monitoring Platform — Project Report

## Abstract
Machine learning models can lose reliability after deployment because incoming data and real-world relationships change. This project builds a monitoring prototype that evaluates data drift, predictive performance, fairness indicators and API health over six simulated months.

## Problem statement
A deployed model may gradually degrade without being noticed until users experience poor decisions. The project aims to identify these changes through continuous measurement, visual dashboards and an operational alert procedure.

## Objectives
1. Generate six months of simulated production data with known issues.
2. Train a baseline classifier using the first month.
3. compare later data batches against the baseline using Evidently.
4. Track predictive quality and group-level fairness measures.
5. Expose service metrics to Prometheus and visualize them in Grafana.
6. Document alert handling and measure time to detection.

## Methodology
A synthetic credit-approval dataset is generated with demographic and financial attributes. Month 1 is used as the reference period and training data. Later months introduce gradual feature-distribution shifts, increased noise in the target relationship, and a group-specific outcome change beginning in month 5. The model excludes gender from its input features; gender is retained only for post-hoc fairness evaluation.

Evidently compares each later month's input features with the reference month. Scikit-learn metrics are calculated using the simulated ground-truth labels. The Flask API serves predictions and exposes request count, latency, error count and approval probability metrics. Prometheus scrapes these metrics and Grafana displays them.

## Evaluation
The monitoring script produces monthly accuracy, precision, recall, F1, group true-positive rates, fairness gap and Evidently drift reports. Populate the final results table using `reports/monthly_metrics.csv` after running the project.

| Month | Accuracy | Precision | Recall | F1 | Fairness gap | Drift |
|---|---:|---:|---:|---:|---:|---|
| 1 | Run experiment | | | | | Reference |
| 2 | Run experiment | | | | | Measured |
| 3 | Run experiment | | | | | Measured |
| 4 | Run experiment | | | | | Measured |
| 5 | Run experiment | | | | | Measured |
| 6 | Run experiment | | | | | Measured |

## Time-to-detection
Use the first month that each configured threshold is crossed and compare it with the month in which the issue was planted. Record detection delay in monthly batches. The result should be based on the actual output, not an assumed value.

## Limitations
The dataset is synthetic and the project uses monthly offline evaluation for historical batches. The API metrics are live service metrics, while the six-month drift and fairness analysis is batch-based. The fairness gap is only one diagnostic measure and does not establish legal or ethical fairness on its own.

## Conclusion
The prototype demonstrates how model monitoring combines data-quality analysis, model evaluation, fairness checks and service observability. It shows why monitoring must continue after deployment and why alerts should lead to investigation rather than automatic retraining.
