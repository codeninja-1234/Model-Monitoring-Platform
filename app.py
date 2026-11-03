
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from time import perf_counter

app = Flask(__name__)
bundle = joblib.load("model.joblib")
model, FEATURES = bundle["model"], bundle["features"]

REQUESTS = Counter("model_prediction_requests_total", "Total prediction requests", ["status"])
LATENCY = Histogram("model_prediction_latency_seconds", "Prediction latency")
APPROVAL_RATE = Gauge("model_prediction_approval_rate", "Rolling approval rate")
ERRORS = Counter("model_prediction_errors_total", "Prediction errors")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

@app.post("/predict")
def predict():
    start = perf_counter()
    try:
        payload = request.get_json(force=True)
        frame = pd.DataFrame([payload], columns=FEATURES)
        prediction = int(model.predict(frame)[0])
        probability = float(model.predict_proba(frame)[0, 1])
        REQUESTS.labels(status="success").inc()
        APPROVAL_RATE.set(probability)
        return jsonify({"approved": prediction, "approval_probability": round(probability, 4)})
    except Exception as exc:
        ERRORS.inc()
        REQUESTS.labels(status="error").inc()
        return jsonify({"error": str(exc)}), 400
    finally:
        LATENCY.observe(perf_counter() - start)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
