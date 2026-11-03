
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_PER_MONTH = 1200
OUT = Path("data")
OUT.mkdir(exist_ok=True)

def make_month(month: int, n: int = N_PER_MONTH) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + month)
    # Gradual population shift begins in month 3; stronger shift from month 4.
    shift = max(0, month - 2)
    age = np.clip(rng.normal(39 + 1.0 * shift, 11, n), 18, 75)
    income = np.clip(rng.lognormal(np.log(52000) + 0.09 * shift, 0.42, n), 12000, 220000)
    debt_ratio = np.clip(rng.beta(2.0, 5.0, n) + 0.035 * shift, 0.01, 0.98)
    credit_history = np.clip(rng.normal(6.5 - 0.30 * shift, 2.0, n), 0, 20)
    gender = rng.choice(["F", "M"], size=n, p=[0.49, 0.51])
    region = rng.choice(["North", "South", "East", "West"], size=n)

    # Ground-truth approval relationship changes over time.
    score = (
        0.035 * (age - 40)
        + 0.000035 * (income - 50000)
        - 3.2 * (debt_ratio - 0.30)
        + 0.38 * (credit_history - 6)
        + rng.normal(0, 0.85, n)
    )
    # Planted fairness issue: from month 5, outcomes worsen for F independently
    # of the predictive features. The model does not receive gender as an input.
    if month >= 5:
        score = score - (gender == "F") * (0.85 + 0.35 * (month - 5))
    # General concept drift: the relationship becomes noisier in later months.
    if month >= 4:
        score += rng.normal(0, 0.35 * (month - 3), n)

    probability = 1 / (1 + np.exp(-score))
    approved = rng.binomial(1, probability)
    return pd.DataFrame({
        "month": month, "age": age.round(1), "income": income.round(2),
        "debt_ratio": debt_ratio.round(4), "credit_history": credit_history.round(2),
        "gender": gender, "region": region, "approved": approved
    })

def main():
    frames = [make_month(m) for m in range(1, 7)]
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(OUT / "six_months.csv", index=False)
    data[data.month == 1].to_csv(OUT / "reference.csv", index=False)
    print(f"Generated {len(data):,} records across six months.")
    print(data.groupby("month")["approved"].mean().round(3))

if __name__ == "__main__":
    main()
