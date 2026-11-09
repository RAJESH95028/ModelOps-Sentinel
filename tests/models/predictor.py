from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models/model.joblib"
META_PATH = ROOT / "models/metadata.json"
REF_PATH = ROOT / "data/reference.csv"


def load_model():
    return joblib.load(MODEL_PATH)


def metadata():
    return json.loads(META_PATH.read_text())


def predict(payload: dict) -> dict:
    model = load_model()
    meta = metadata()
    frame = pd.DataFrame([payload])[meta["features"]]
    probability = float(model.predict_proba(frame)[0, 1])
    prediction = int(probability >= 0.5)
    return {
        "prediction": prediction,
        "probability": round(probability, 6),
        "risk_label": "high_risk" if prediction else "low_risk",
        "model_version": meta["model_version"],
        "model_name": meta["model_name"],
    }


def explain(payload: dict) -> dict:
    import shap

    model = load_model()
    meta = metadata()
    background = pd.read_csv(REF_PATH)[meta["features"]].sample(100, random_state=42)
    row = pd.DataFrame([payload])[meta["features"]]

    def probability_fn(values):
        frame = pd.DataFrame(values, columns=meta["features"])
        return model.predict_proba(frame)[:, 1]

    explainer = shap.Explainer(probability_fn, background)
    shap_values = explainer(row)
    values = shap_values.values[0]
    if values.ndim > 1:
        values = values[:, 0]
    contributions = sorted(
        [
            {"feature": feature, "contribution": round(float(value), 6)}
            for feature, value in zip(meta["features"], values)
        ],
        key=lambda item: abs(item["contribution"]),
        reverse=True,
    )
    return {
        "model_version": meta["model_version"],
        "prediction": predict(payload),
        "explanations": contributions,
    }
