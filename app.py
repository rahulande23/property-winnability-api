from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

app = FastAPI(
    title="Property Insurance Winnability API",
    version="1.0.0",
    description="Returns the probability that a property insurance submission will bind."
)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "winnability_model_production.joblib"
model = None

FEATURE_COLS = [
    "state", "primary_occupancy", "occupancy_classification",
    "primary_construction", "frame_construction_percent", "total_tiv",
    "covered_perils", "aop_deductible", "wind_deductible", "flood_deductible",
    "earthquake_deductible", "scs_deductible", "wind_exposure",
    "flood_exposure", "earthquake_exposure", "layer_amount",
    "attachment_point", "equipment_breakdown", "estimated_premium"
]

REQUIRED_FIELDS = ["submission_id", "state", "total_tiv", "estimated_premium"]

OPTIONAL_FIELDS = [
    c for c in FEATURE_COLS
    if c not in {"state", "total_tiv", "estimated_premium"}
]

class ScoringRequest(BaseModel):
    submissions: List[Dict[str, Any]] = Field(min_length=1)

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Loaded model does not support predict_proba().")

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/score")
def score(request: ScoringRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    results = []

    for index, submission in enumerate(request.submissions):
        missing_required = [
            f for f in REQUIRED_FIELDS
            if f not in submission or submission[f] is None
        ]
        if missing_required:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"Submission at index {index} is missing required fields.",
                    "missing_required_fields": missing_required,
                },
            )

        row = submission.copy()

        for field in OPTIONAL_FIELDS:
            if field not in row or row[field] is None:
                row[field] = np.nan

        X = pd.DataFrame([{field: row[field] for field in FEATURE_COLS}])

        try:
            probability = float(model.predict_proba(X)[0, 1])
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"Prediction failed for submission at index {index}.",
                    "error": str(exc),
                },
            )

        probability = max(0.0, min(1.0, probability))

        results.append({
            "submission_id": str(submission["submission_id"]),
            "winnability_probability": round(probability, 6),
            "winnability_percentage": round(probability * 100, 2),
        })

    return {"predictions": results}
