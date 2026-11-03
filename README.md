# Model Monitoring Platform

A course project demonstrating production-style monitoring of a deployed binary classification model over six simulated months.

## What it demonstrates
- Synthetic six-month production data with planted population drift, concept drift and a group-level outcome shift.
- A baseline scikit-learn model trained on month 1.
- A Flask prediction API with Prometheus request, latency, error and prediction metrics.
- Evidently data-drift HTML reports for months 2–6.
- Offline monthly accuracy, precision, recall, F1 and group fairness analysis.
- Grafana dashboard provisioned automatically through Docker Compose.
- Alert runbook and time-to-detection analysis.

## Requirements
- macOS/Linux
- Python 3.10 or 3.11 recommended
- Docker Desktop with Docker Compose

## 1. Create environment and install packages
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Generate six months of data and train model
```bash
python src_generate_data.py
python src_train_model.py
```
The dataset has 7,200 rows (1,200 per month). Month 1 is the reference/training period.

## 3. Run offline monitoring
```bash
python monitor.py
```
Outputs are saved under `reports/`. Open `reports/drift_month_4.html` in a browser to inspect a drift report. `monthly_metrics.csv` contains the six-month metric trend.

## 4. Run the prediction API locally
In a terminal:
```bash
python app.py
```
In another terminal, test it:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":35,"income":65000,"debt_ratio":0.25,"credit_history":8,"region":"South"}'
```
Metrics: http://localhost:5000/metrics

## 5. Start Prometheus and Grafana
Docker Compose builds and runs the model API, Prometheus and Grafana:
```bash
docker compose up --build -d
```
Open:
- Grafana: http://localhost:3000 (login: admin / admin)
- Prometheus: http://localhost:9090
- Model API: http://localhost:5000/health

Send several test predictions using the curl command above, then refresh Grafana. Prometheus scrapes the API every five seconds.

Stop:
```bash
docker compose down
```

## 6. Alert thresholds used in the demonstration
- Accuracy below 0.70: critical model-quality alert.
- Absolute group TPR gap above 0.12: fairness warning requiring investigation.
- Drift detected in multiple input columns: investigate the data pipeline and compare current distributions with the reference period.
- API error rate or latency increase: investigate service logs and infrastructure.

The offline script flags the planted drift period from month 4 as a review condition. The actual Evidently HTML report contains the measured feature drift results.

## Architecture
```text
Six-month synthetic data
       |
       +--> Month 1 reference --> Model training --> model.joblib
       |                                           |
       +--> Monthly batches -----------------------+
                    |
          Offline monitoring (Evidently + sklearn)
                    |
          HTML reports + monthly metrics + fairness

Client --> Flask prediction API --> Prometheus --> Grafana
```

## Important project interpretation
This is a reproducible simulation, not a claim that a real company model was monitored. The fairness measure is the difference in group true-positive rates (F vs M); it is a screening metric, not a complete legal fairness assessment. Drift does not automatically prove performance loss, and model performance requires ground-truth labels.

## Suggested evaluation demo
1. Show the six-month dataset and explain the planted changes.
2. Show that the model was trained only on month 1.
3. Open a later-month Evidently HTML drift report.
4. Compare monthly accuracy and fairness gap in `reports/monthly_metrics.csv`.
5. Open Grafana and send live prediction requests.
6. Explain the alert runbook and calculate detection delay from the first threshold-crossing month.

## References
- Evidently documentation: https://docs.evidentlyai.com/
- Prometheus Python client: https://prometheus.github.io/client_python/
- Prometheus: https://prometheus.io/docs/
- Grafana: https://grafana.com/docs/grafana/latest/
