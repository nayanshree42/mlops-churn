import json
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict():
    payload = {
        "tenure_months": 8.0,
        "monthly_charges": 120.0,
        "total_charges": 960.0,
        "support_calls": 3,
        "contracts_left": 2,
        "is_senior": 0,
    }
    r = client.post("/predict", content=json.dumps(payload))
    assert r.status_code == 200
    body = r.json()
    assert "churn_probability" in body
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)