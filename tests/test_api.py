import numpy as np
from fastapi.testclient import TestClient

from app.main import app, FEATURE_NAMES

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "LogisticRegression"
    assert body["feature_count"] == 30
    assert len(body["features"]) == 30


def test_predict_schema_and_probabilities():
    sample = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
              0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399,
              0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33,
              184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
    response = client.post("/predict", json={"features": sample})
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {
        "prediction",
        "prediction_label",
        "probability_benign",
        "probability_malignant",
    }
    assert body["prediction"] in [0, 1]
    assert body["prediction_label"] in ["benign", "malignant"]
    assert 0.0 <= body["probability_benign"] <= 1.0
    assert 0.0 <= body["probability_malignant"] <= 1.0
    assert np.isclose(
        body["probability_benign"] + body["probability_malignant"], 1.0
    )


def test_predict_rejects_wrong_feature_count():
    response = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert response.status_code == 422
