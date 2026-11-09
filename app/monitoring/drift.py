from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_counts / max(len(ref), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(len(cur), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> dict:
    values = {feature: psi(reference[feature], current[feature]) for feature in features}
    max_psi = max(values.values(), default=0.0)
    return {
        "features": values,
        "max_psi": max_psi,
        "drifted_features": [k for k, v in values.items() if v >= 0.20],
        "status": "HIGH" if max_psi >= 0.20 else "LOW",
        "threshold": 0.20,
    }


def performance_report(y_true, y_pred, y_proba) -> dict:
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "samples": len(y_true),
    }
