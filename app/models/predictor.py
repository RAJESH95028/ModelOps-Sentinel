import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"

FEATURES = [
    "age",
    "income",
    "credit_score",
    "loan_amount",
    "debt_ratio",
    "employment_years",
    "late_payments",
    "account_balance",
]

_model = None
_metadata = None


def load_model():
    global _model, _metadata

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found: {MODEL_PATH}. "
                "Run python training/train.py first."
            )
        _model = joblib.load(MODEL_PATH)

    if _metadata is None:
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as file:
                _metadata = json.load(file)
        else:
            _metadata = {
                "model_name": "Unknown",
                "features": FEATURES,
                "target": "default_risk",
                "metrics": {},
            }

    return _model


def metadata():
    load_model()
    return {
        "model_name": _metadata.get("model_name", "Unknown"),
        "model_version": _metadata.get("model_version", "v1"),
        "features": _metadata.get("features", FEATURES),
        "target": _metadata.get("target", "default_risk"),
        "metrics": _metadata.get("metrics", {}),
    }


def predict(input_data):
    model = load_model()

    missing = [feature for feature in FEATURES if feature not in input_data]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    input_df = pd.DataFrame([
        {feature: input_data[feature] for feature in FEATURES}
    ])

    prediction = int(model.predict(input_df)[0])

    result = {
        "prediction": prediction,
        "risk_label": "High Risk" if prediction == 1 else "Low Risk",
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        result["default_probability"] = float(probabilities[1])

    return result


def explain(input_data):
    result = predict(input_data)
    reasons = []

    if input_data["credit_score"] < 600:
        reasons.append("Low credit score")
    if input_data["debt_ratio"] > 0.5:
        reasons.append("High debt-to-income ratio")
    if input_data["late_payments"] >= 3:
        reasons.append("Multiple late payments")
    if input_data["employment_years"] < 2:
        reasons.append("Limited employment history")
    if input_data["account_balance"] < 2000:
        reasons.append("Low account balance")

    if not reasons:
        reasons.append(
            "No major risk indicators were identified by these simple rules"
        )

    result["explanation"] = reasons
    result["explanation_type"] = "Rule-based explanation"
    return result
