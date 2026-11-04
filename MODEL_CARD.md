# Model Card — Model Monitoring Platform

## 1. Model Overview

**Model type:** Logistic Regression  
**Framework:** Scikit-learn  
**Task:** Binary classification of loan approval outcomes.

The model predicts whether a loan application should be approved using applicant and credit-related features.

## 2. Intended Use

This model is developed for an educational model-monitoring project. It demonstrates how model performance, data drift and fairness can be monitored over a simulated deployment period.

It is not intended for real-world lending or automated financial decisions.

## 3. Dataset

The project uses a synthetic six-month dataset containing 7,200 records, with 1,200 records per month.

A separate reference dataset is provided.

### Input features

- Age
- Income
- Debt ratio
- Credit history
- Region

Gender is retained for fairness evaluation but is not used as a model input.

## 4. Training Methodology

The model is trained using month 1 data.

A stratified 75:25 train-test split is used with random_state=42.

- Training records: 900
- Held-out baseline records: 300

The held-out records are used for baseline evaluation and as the reference dataset for drift monitoring.

The preprocessing pipeline uses median imputation and standard scaling for numerical features, and most-frequent imputation and one-hot encoding for categorical features.

The classifier is Logistic Regression with max_iter=1000.

## 5. Evaluation Metrics

The model is evaluated monthly using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Performance is compared against the held-out month 1 baseline.

## 6. Monitoring Strategy

Evidently is used to monitor feature drift against the month 1 reference data.

The dataset-level drift threshold is 50% of monitored features.

The monitoring pipeline also evaluates:

- True positive rate by gender group
- False positive rate by gender group
- Selection-rate gap
- Outcome-rate gap

Prometheus exposes monitoring metrics and evaluates alert rules. Grafana provides dashboard visualizations.

## 7. Simulated Issues

The dataset includes intentionally introduced changes:

- Feature distribution changes beginning in month 3.
- Concept drift beginning in month 4.
- A gender-related outcome disparity beginning in month 5.

These changes are used to evaluate the monitoring system's ability to identify degradation.

## 8. Observed Results

After correcting the training/test split, the monitoring results show:

- Feature drift detection beginning in month 3.
- Accuracy falling below 70% in month 5.
- Fairness TPR gap exceeding 0.12 in month 6.
- Month 6 accuracy of approximately 66.5%.
- Month 6 fairness TPR gap of approximately 0.1345.

Detection delays are measured in simulated monthly batches, not real elapsed calendar time.

## 9. Fairness Considerations

Gender is excluded from model inputs but retained for group-level evaluation.

The project measures group TPR, FPR, selection rate and outcome rate.

These metrics provide useful indicators but do not establish that the model is fair in every context. Synthetic data also cannot represent all real-world demographic or socioeconomic conditions.

## 10. Limitations

- The dataset is synthetic.
- The model is a simple Logistic Regression classifier.
- Monitoring is demonstrated over six simulated months.
- Historical monitoring metrics are generated offline.
- Prometheus alerts evaluate the latest exported monthly metrics.
- The Flask development server is intended for demonstration, not production deployment.
- No real lending decisions should be based on this model.

## 11. Ethical Considerations

The model is an educational demonstration only. Any real-world lending system would require representative data, legal and regulatory review, independent fairness assessment, privacy safeguards and human oversight.
