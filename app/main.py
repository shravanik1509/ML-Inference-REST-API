from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = BASE_DIR / "models" / "champion_model.joblib"

artifact = joblib.load(ARTIFACT_PATH)
model = artifact["model"]
FEATURE_NAMES = artifact["feature_names"]
CLASS_NAMES = artifact["class_names"]

app = FastAPI(
    title="ML Inference REST API",
    version="1.0.0",
    description=(
        "Production-style inference service for the Task 3 champion "
        "Logistic Regression model. Educational use only; not a clinical system."
    ),
)


class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="Exactly 30 numeric features in the documented model feature order.",
    )


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability_benign: float
    probability_malignant: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/model-info")
def model_info():
    return {
        "model": "LogisticRegression",
        "features": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "classes": CLASS_NAMES,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        X = np.asarray(request.features, dtype=float).reshape(1, -1)
        probabilities = model.predict_proba(X)[0]
        prediction = int(model.predict(X)[0])
        return PredictionResponse(
            prediction=prediction,
            prediction_label=CLASS_NAMES[prediction],
            probability_benign=float(probabilities[0]),
            probability_malignant=float(probabilities[1]),
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}")
