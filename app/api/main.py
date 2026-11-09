
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from app.models.predictor import explain, metadata, predict
from app.monitoring.drift import drift_report
from app.services.retraining import retrain, should_retrain


app = FastAPI(
    title="ModelOps Sentinel API",
    description=(
        "Production ML monitoring, explainability, "
        "custom model prediction and autonomous retraining."
    ),
    version="2.1.0",
)

ROOT = Path(__file__).resolve().parents[2]
CUSTOM_MODEL_PATH = ROOT / "models" / "custom_model.joblib"


# ------------------------------------------------------
# EXISTING PRODUCTION API
# ------------------------------------------------------

class PredictionRequest(BaseModel):
    age: int = Field(ge=18, le=100)
    income: float = Field(gt=0)
    credit_score: float = Field(ge=300, le=900)
    loan_amount: float = Field(gt=0)
    debt_ratio: float = Field(ge=0, le=1)
    employment_years: float = Field(ge=0, le=60)
    late_payments: int = Field(ge=0)
    account_balance: float = Field(ge=0)


@app.get("/health")
def health():
    try:
        meta = metadata()
        return {
            "status": "healthy",
            "model_version": meta["model_version"],
            "model": meta["model_name"],
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/model-info")
def model_info():
    return metadata()


@app.post("/predict")
def make_prediction(request: PredictionRequest):
    return predict(request.model_dump())


@app.post("/explain")
def explain_prediction(request: PredictionRequest):
    try:
        return explain(request.model_dump())
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Explainability failed: {exc}",
        )


class MonitoringRequest(BaseModel):
    records: list[PredictionRequest] = Field(min_length=20)
    observed_performance: float | None = Field(
        default=None, ge=0, le=1
    )


@app.post("/monitor")
def monitor(request: MonitoringRequest):
    meta = metadata()
    reference_path = ROOT / "data" / "reference.csv"

    if not reference_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Reference dataset is unavailable",
        )

    reference = pd.read_csv(reference_path)
    current = pd.DataFrame(
        [r.model_dump() for r in request.records]
    )

    report = drift_report(
        reference,
        current,
        meta["features"],
    )

    retraining_needed = should_retrain(
        report["max_psi"],
        request.observed_performance,
    )

    result = {
        "drift": report,
        "retraining_triggered": retraining_needed,
    }

    if retraining_needed:
        result["retraining"] = retrain()

    return result


# ------------------------------------------------------
# CUSTOM MODEL API
# ------------------------------------------------------

class CustomPredictionRequest(BaseModel):
    features: dict[str, Any]


def load_custom_model():
    if not CUSTOM_MODEL_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No custom model uploaded. Upload a model first.",
        )

    try:
        package = joblib.load(CUSTOM_MODEL_PATH)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load custom model: {exc}",
        )

    required_keys = {
        "model",
        "target_column",
        "feature_columns",
        "problem_type",
        "algorithm",
        "metrics",
    }

    if not isinstance(package, dict) or not required_keys.issubset(package):
        raise HTTPException(
            status_code=400,
            detail="Invalid custom model package.",
        )

    return package


@app.post("/custom-model/upload")
async def upload_custom_model(
    file: UploadFile = File(...),
):
    if not file.filename or not file.filename.lower().endswith(".joblib"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a .joblib model file.",
        )

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        if len(contents) > 100 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Model file must be smaller than 100 MB.",
            )

        # Only load model files from trusted sources.
        package = joblib.load(BytesIO(contents))

        required_keys = {
            "model",
            "target_column",
            "feature_columns",
            "problem_type",
            "algorithm",
            "metrics",
        }

        if not isinstance(package, dict) or not required_keys.issubset(package):
            raise ValueError("Invalid model package structure.")

        if not hasattr(package["model"], "predict"):
            raise ValueError("Uploaded model does not support prediction.")

        if not isinstance(package["feature_columns"], list):
            raise ValueError("Invalid feature columns.")

        if not package["feature_columns"]:
            raise ValueError("Model has no feature columns.")

        if package["problem_type"] not in ("Classification", "Regression"):
            raise ValueError("Unsupported problem type.")

        CUSTOM_MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        CUSTOM_MODEL_PATH.write_bytes(contents)

        return {
            "status": "success",
            "message": "Custom model uploaded successfully.",
            "algorithm": str(package["algorithm"]),
            "problem_type": str(package["problem_type"]),
            "target_column": str(package["target_column"]),
            "feature_columns": package["feature_columns"],
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model file: {exc}",
        )
    finally:
        await file.close()


@app.get("/custom-model/info")
def custom_model_info():
    package = load_custom_model()

    return {
        "status": "uploaded",
        "algorithm": str(package["algorithm"]),
        "problem_type": str(package["problem_type"]),
        "target_column": str(package["target_column"]),
        "feature_columns": package["feature_columns"],
        "metrics": package["metrics"],
    }


@app.post("/custom-predict")
def custom_prediction(request: CustomPredictionRequest):
    package = load_custom_model()

    model = package["model"]
    feature_columns = package["feature_columns"]

    missing_features = [
        column
        for column in feature_columns
        if column not in request.features
    ]

    if missing_features:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing required input features.",
                "missing_features": missing_features,
            },
        )

    try:
        input_data = pd.DataFrame(
            [
                {
                    column: request.features[column]
                    for column in feature_columns
                }
            ],
            columns=feature_columns,
        )

        prediction = model.predict(input_data)[0]

        if hasattr(prediction, "item"):
            prediction = prediction.item()

        result = {
            "status": "success",
            "algorithm": str(package["algorithm"]),
            "problem_type": str(package["problem_type"]),
            "target_column": str(package["target_column"]),
            "prediction": prediction,
        }

        if (
            package["problem_type"] == "Classification"
            and hasattr(model, "predict_proba")
        ):
            probabilities = model.predict_proba(input_data)[0]
            classes = model.classes_

            result["class_probabilities"] = {
                str(cls): float(prob)
                for cls, prob in zip(classes, probabilities)
            }

        return result

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Custom prediction failed: {exc}",
        )
