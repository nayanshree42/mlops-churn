"""
Train a simple Logistic Regression.
Run: python src/train.py
This reads: data/churn.csv
This writes: models/model.pkl and metrics/metrics.json
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from utils import FeatureInfo

os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)


# Load data
path = os.path.join("data", "churn.csv")
if not os.path.exists(path):
    raise FileNotFoundError(
        "data/churn.csv not found. Run: python src/data/make_dataset.py"
    )


df = pd.read_csv(path)
X = df[FeatureInfo.numeric]
y = df["churn"].astype(int)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
    ]
)


pipe.fit(X_train, y_train)


# Evaluate
probs = pipe.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)


metrics = {
    "accuracy": float(accuracy_score(y_test, preds)),
    "precision": float(precision_score(y_test, preds)),
    "recall": float(recall_score(y_test, preds)),
    "f1": float(f1_score(y_test, preds)),
    "roc_auc": float(roc_auc_score(y_test, probs)),
}


# Save
joblib.dump(pipe, os.path.join("models", "model.pkl"))
with open(os.path.join("metrics", "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)


print("Saved models/model.pkl and metrics/metrics.json")
print("Metrics:\n", json.dumps(metrics, indent=2))