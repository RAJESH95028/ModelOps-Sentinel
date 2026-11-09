"""Run a small end-to-end ModelOps Sentinel demonstration."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.monitoring.drift import drift_report
from app.services.retraining import retrain, should_retrain
from app.models.predictor import predict


def main() -> None:
    reference = pd.read_csv(ROOT / "data/reference.csv")
    features = json.loads((ROOT / "models/metadata.json").read_text())["features"]

    # Simulate a production distribution shift in income and loan amount.
    production = reference[features].sample(250, random_state=7).copy()
    production["income"] *= 2.5
    production["loan_amount"] *= 2.0

    report = drift_report(reference, production, features)
    trigger = should_retrain(report["max_psi"], performance=0.70)

    sample = production.iloc[0].to_dict()
    prediction = predict(sample)

    result = {
        "drift_status": report["status"],
        "max_psi": round(report["max_psi"], 4),
        "drifted_features": report["drifted_features"],
        "retraining_triggered": trigger,
        "sample_prediction": prediction,
    }

    if trigger:
        result["retraining"] = retrain()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
