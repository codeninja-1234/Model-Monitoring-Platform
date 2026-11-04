import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

FEATURES = ["age", "income", "debt_ratio", "credit_history", "region"]
NUMERIC = ["age", "income", "debt_ratio", "credit_history"]
CATEGORICAL = ["region"]


def main():
    data = pd.read_csv("data/six_months.csv")

    # Use only month 1 for training.
    month1 = data[data["month"] == 1].copy()

    # Keep the same 75:25 split used by monitor.py.
    train, test = train_test_split(
        month1,
        test_size=0.25,
        random_state=42,
        stratify=month1["approved"]
    )

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), NUMERIC),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), CATEGORICAL)
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    model.fit(train[FEATURES], train["approved"])

    joblib.dump({
        "model": model,
        "features": FEATURES
    }, "model.joblib")

    print(f"Training records: {len(train)}")
    print(f"Held-out records: {len(test)}")
    print("Model saved successfully to model.joblib")


if __name__ == "__main__":
    main()
