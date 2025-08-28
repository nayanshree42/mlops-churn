from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
from pathlib import Path


MODEL_PATH = Path("models/model.pkl")



from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "models/model.pkl not found. Train first: python src/train.py"
        )
    global model
    model = joblib.load(MODEL_PATH)
    yield

app = FastAPI(title="Churn Prediction API", version="0.1.0", lifespan=lifespan)


class CustomerFeatures(BaseModel):
    tenure_months: float = Field(..., ge=0)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    support_calls: int = Field(..., ge=0)
    contracts_left: int = Field(..., ge=0)
    is_senior: int = Field(..., ge=0, le=1)



@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: CustomerFeatures):
    x = np.array([
        [
            features.tenure_months,
            features.monthly_charges,
            features.total_charges,
            features.support_calls,
            features.contracts_left,
            features.is_senior,
        ]
    ])
    prob = float(model.predict_proba(x)[0, 1])
    pred = int(prob >= 0.5)
    return {"churn_probability": prob, "churn_prediction": pred}