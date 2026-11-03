
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

FEATURES = ["age", "income", "debt_ratio", "credit_history", "region"]
NUMERIC = ["age", "income", "debt_ratio", "credit_history"]
CATEGORICAL = ["region"]

def main():
    data = pd.read_csv("data/six_months.csv")
    train = data[data.month == 1]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL)
    ])
    model = Pipeline([
        ("prep", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    model.fit(train[FEATURES], train["approved"])
    joblib.dump({"model": model, "features": FEATURES}, "model.joblib")
    print(f"Model trained on month 1 ({len(train)} records) and saved to model.joblib")

if __name__ == "__main__":
    main()
