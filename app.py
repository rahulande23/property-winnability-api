from pathlib import Path
from typing import Any, Dict, List
import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

WINNABILITY_MODEL_PATH = Path(
    os.environ.get(
        "WINNABILITY_MODEL_PATH",
        str(MODEL_DIR / "winnability_model_production.joblib"),
    )
)
PROFITABILITY_MODEL_PATH = Path(
    os.environ.get(
        "PROFITABILITY_MODEL_PATH",
        str(MODEL_DIR / "profitability_model_production.joblib"),
    )
)

FEATURE_COLS = [
    "state",
    "primary_occupancy",
    "occupancy_classification",
    "primary_construction",
    "frame_construction_percent",
    "total_tiv",
    "covered_perils",
    "aop_deductible",
    "wind_deductible",
    "flood_deductible",
    "earthquake_deductible",
    "scs_deductible",
    "wind_exposure",
    "flood_exposure",
    "earthquake_exposure",
    "layer_amount",
    "attachment_point",
    "equipment_breakdown",
    "estimated_premium",
]

REQUIRED_FIELDS = ["submission_id", "state", "total_tiv", "estimated_premium"]
OPTIONAL_FIELDS = [
    field for field in FEATURE_COLS if field not in {"state", "total_tiv", "estimated_premium"}
]


app = FastAPI(
    title="Property Insurance Scoring API",
    version="2.0.0",
    description=(
        "Scores property insurance submissions using two independent models: "
        "winnability classification and profitability regression."
    ),
)

winnability_model = None
profitability_model = None


class ScoringRequest(BaseModel):
    submissions: List[Dict[str, Any]] = Field(min_length=1)


def _load_model(path: Path, model_name: str):
    if not path.exists():
        raise RuntimeError(f"{model_name} model file not found: {path}")
    try:
        return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load {model_name} model: {exc}") from exc


def _prepare_submission(submission: Dict[str, Any], index: int) -> pd.DataFrame:
    missing_required = [
        field
        for field in REQUIRED_FIELDS
        if field not in submission or submission[field] is None
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

    return pd.DataFrame([{field: row[field] for field in FEATURE_COLS}])


@app.on_event("startup")
def load_models():
    global winnability_model, profitability_model

    winnability_model = _load_model(WINNABILITY_MODEL_PATH, "Winnability")
    profitability_model = _load_model(PROFITABILITY_MODEL_PATH, "Profitability")

    if not hasattr(winnability_model, "predict_proba"):
        raise RuntimeError("Loaded winnability model does not support predict_proba().")

    if not hasattr(profitability_model, "predict"):
        raise RuntimeError("Loaded profitability model does not support predict().")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "winnability_model_loaded": winnability_model is not None,
        "profitability_model_loaded": profitability_model is not None,
    }


@app.post("/score")
def score(request: ScoringRequest):
    if winnability_model is None or profitability_model is None:
        raise HTTPException(status_code=503, detail="One or more models are not loaded.")

    results = []

    for index, submission in enumerate(request.submissions):
        X = _prepare_submission(submission, index)

        try:
            winnability_probability = float(
                winnability_model.predict_proba(X)[0, 1]
            )
            predicted_profit = float(profitability_model.predict(X)[0])
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"Prediction failed for submission at index {index}.",
                    "error": str(exc),
                },
            )

        winnability_probability = max(0.0, min(1.0, winnability_probability))

        estimated_premium = float(submission["estimated_premium"])
        if estimated_premium > 0:
            predicted_profit_margin = (predicted_profit / estimated_premium) * 100.0
        else:
            predicted_profit_margin = None

        results.append(
            {
                "submission_id": str(submission["submission_id"]),
                "winnability_probability": round(winnability_probability, 6),
                "winnability_percentage": round(winnability_probability * 100, 2),
                "predicted_profit": round(predicted_profit, 2),
                "predicted_profit_margin": (
                    round(predicted_profit_margin, 2)
                    if predicted_profit_margin is not None
                    else None
                ),
            }
        )

    return {"predictions": results}


@app.post("/winnability")
def score_winnability(request: ScoringRequest):
    if winnability_model is None:
        raise HTTPException(status_code=503, detail="Winnability model is not loaded.")

    results = []
    for index, submission in enumerate(request.submissions):
        X = _prepare_submission(submission, index)
        try:
            probability = float(winnability_model.predict_proba(X)[0, 1])
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        probability = max(0.0, min(1.0, probability))
        results.append(
            {
                "submission_id": str(submission["submission_id"]),
                "winnability_probability": round(probability, 6),
                "winnability_percentage": round(probability * 100, 2),
            }
        )

    return {"predictions": results}


@app.post("/profitability")
def score_profitability(request: ScoringRequest):
    if profitability_model is None:
        raise HTTPException(status_code=503, detail="Profitability model is not loaded.")

    results = []
    for index, submission in enumerate(request.submissions):
        X = _prepare_submission(submission, index)
        try:
            predicted_profit = float(profitability_model.predict(X)[0])
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        estimated_premium = float(submission["estimated_premium"])
        predicted_profit_margin = (
            (predicted_profit / estimated_premium) * 100.0
            if estimated_premium > 0
            else None
        )

        results.append(
            {
                "submission_id": str(submission["submission_id"]),
                "predicted_profit": round(predicted_profit, 2),
                "predicted_profit_margin": (
                    round(predicted_profit_margin, 2)
                    if predicted_profit_margin is not None
                    else None
                ),
            }
        )

    return {"predictions": results}
