from __future__ import annotations

import json
from pathlib import Path

import joblib
from training.train import train

ROOT = Path(__file__).resolve().parents[2]


def should_retrain(
    drift_score: float,
    performance: float | None,
    drift_threshold: float = 0.20,
    performance_threshold: float = 0.80,
) -> bool:
    drift_breach = drift_score >= drift_threshold
    performance_breach = (
        performance is not None and performance < performance_threshold
    )
    return drift_breach or performance_breach


def retrain() -> dict:
    candidate = train(seed=2026, save_model=False)
    candidate_pipeline = candidate.pop("pipeline")
    candidate_metrics = candidate["best_metrics"] or {}
    candidate_f1 = float(candidate_metrics.get("f1_score", 0.0))
    candidate_name = candidate.get("best_model", "Unknown")

    meta_path = ROOT / "models" / "metadata.json"
    current = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    current_f1 = float(
        current.get("metrics", {}).get(
            "f1_score",
            current.get("metrics", {}).get("f1", 0.0),
        )
    )

    current_version = current.get("model_version", "v1")
    current_num = int(current_version[1:]) if current_version.startswith("v") else 1

    promoted = candidate_f1 > current_f1

    if promoted:
        version = f"v{current_num + 1}"
        joblib.dump(candidate_pipeline, ROOT / "models" / "model.joblib")

        metadata = {
            "model_name": candidate_name,
            "model_version": version,
            "features": candidate.get("features", [
                "age", "income", "credit_score", "loan_amount",
                "debt_ratio", "employment_years", "late_payments",
                "account_balance",
            ]),
            "target": "default_risk",
            "metrics": candidate_metrics,
            "seed": 2026,
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        status = "promoted"
    else:
        version = current_version
        status = "rejected"

    result = {
        "status": status,
        "candidate_model": candidate_name,
        "candidate_f1": round(candidate_f1, 6),
        "production_f1": round(current_f1, 6),
        "model_version": version,
    }

    (ROOT / "models" / "retraining_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
