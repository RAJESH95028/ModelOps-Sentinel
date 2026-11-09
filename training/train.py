import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# --------------------------------------------------
# Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# MLflow setup (optional)
# --------------------------------------------------

try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

# --------------------------------------------------
# Dataset configuration
# --------------------------------------------------

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

TARGET = "default_risk"

# --------------------------------------------------
# Generate synthetic financial-risk data
# --------------------------------------------------

def generate_dataset(n_samples=5000, seed=42):
    """Generate synthetic data for financial-risk classification."""

    rng = np.random.default_rng(seed)

    age = rng.integers(18, 75, n_samples)
    income = rng.normal(60000, 25000, n_samples).clip(12000, 200000)
    credit_score = rng.normal(680, 70, n_samples).clip(300, 850)
    loan_amount = rng.normal(20000, 12000, n_samples).clip(500, 100000)
    debt_ratio = rng.beta(2, 5, n_samples).clip(0, 1)
    employment_years = rng.integers(0, 41, n_samples)
    late_payments = rng.poisson(1.2, n_samples).clip(0, 12)
    account_balance = rng.normal(12000, 9000, n_samples).clip(0, 100000)

    risk_score = (
        0.035 * (680 - credit_score)
        + 2.0 * debt_ratio
        + 0.35 * late_payments
        + 0.000012 * loan_amount
        - 0.000004 * income
        - 0.015 * employment_years
        - 0.000008 * account_balance
        + 0.008 * (age - 40)
    )

    probability = 1 / (1 + np.exp(-np.clip(risk_score, -20, 20)))
    probability = np.clip(probability, 0.02, 0.98)

    default_risk = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "age": age,
            "income": income,
            "credit_score": credit_score,
            "loan_amount": loan_amount,
            "debt_ratio": debt_ratio,
            "employment_years": employment_years,
            "late_payments": late_payments,
            "account_balance": account_balance,
            "default_risk": default_risk,
        }
    )


# --------------------------------------------------
# Build preprocessing pipeline
# --------------------------------------------------

def build_pipeline(model):
    """Create a preprocessing and classification pipeline."""

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clone(model)),
        ]
    )


# --------------------------------------------------
# Evaluate a model
# --------------------------------------------------

def evaluate_model(pipeline, X_test, y_test):
    """Calculate model evaluation metrics."""

    predictions = pipeline.predict(X_test)

    try:
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, probabilities)
    except (AttributeError, ValueError, IndexError):
        roc_auc = 0.0

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc),
    }


# --------------------------------------------------
# Train and compare models
# --------------------------------------------------

def train(save_model=True, seed=42):
    """Train multiple models and save the best-performing pipeline."""

    print("Generating dataset...")

    dataset = generate_dataset(n_samples=5000, seed=seed)

    dataset.to_csv(DATA_DIR / "reference.csv", index=False)

    X = dataset[FEATURES]
    y = dataset[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=seed,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            random_state=seed,
            n_jobs=-1,
        ),
    }

    # Add XGBoost if installed.
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
    except ImportError:
        print("XGBoost is not installed. Skipping that model.")

    mlflow_enabled = MLFLOW_AVAILABLE

    if mlflow_enabled:
        try:
            mlflow.set_experiment("Default Risk Model Training")
        except Exception as error:
            print(f"MLflow setup skipped: {error}")
            mlflow_enabled = False

    results = []
    best = None

    for name, model in models.items():
        print(f"\nTraining {name}...")

        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)

        result = {
            "model": name,
            **metrics,
        }

        results.append(result)

        print(json.dumps(result, indent=2))

        # Log parameters and metrics when MLflow is available.
        if mlflow_enabled:
            try:
                with mlflow.start_run(run_name=name):
                    mlflow.log_params(
                        {
                            "model": name,
                            "features": len(FEATURES),
                            "seed": seed,
                        }
                    )

                    mlflow.log_metrics(metrics)

                    # MLflow serialization failures should not stop
                    # local model training or saving.
                    try:
                        mlflow.sklearn.log_model(
                            sk_model=pipeline,
                            artifact_path="model",
                        )
                    except Exception as error:
                        print(
                            f"MLflow model logging skipped for "
                            f"{name}: {error}"
                        )

            except Exception as error:
                print(f"MLflow logging skipped for {name}: {error}")

        # Select the model with the highest F1 score.
        if best is None or metrics["f1_score"] > best["metrics"]["f1_score"]:
            best = {
                "name": name,
                "pipeline": pipeline,
                "metrics": metrics,
            }

    # Save model comparison results.
    comparison_path = MODELS_DIR / "comparison.csv"

    pd.DataFrame(results).sort_values(
        by="f1_score",
        ascending=False,
    ).to_csv(comparison_path, index=False)

    # Save the best model locally.
    if save_model and best is not None:
        model_path = MODELS_DIR / "model.joblib"

        joblib.dump(best["pipeline"], model_path)

        metadata = {
            "model_name": best["name"],
            "features": FEATURES,
            "target": TARGET,
            "metrics": best["metrics"],
            "seed": seed,
        }

        metadata_path = MODELS_DIR / "metadata.json"

        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        print("\nBest model saved successfully.")
        print(f"Model: {model_path}")
        print(f"Metadata: {metadata_path}")
        print(f"Comparison: {comparison_path}")
        print(f"Best model: {best['name']}")
        print(f"Best F1 score: {best['metrics']['f1_score']:.4f}")

    return {
        "results": results,
        "best_model": best["name"] if best else None,
        "best_metrics": best["metrics"] if best else None,
        "pipeline": best["pipeline"] if best else None,
    }


# --------------------------------------------------
# Main entry point
# --------------------------------------------------

if __name__ == "__main__":
    output = train(save_model=True)
    output.pop("pipeline", None)

    print("\nTraining completed.")
    print(json.dumps(output, indent=2))