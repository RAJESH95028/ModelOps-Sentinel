import pandas as pd
from app.monitoring.drift import psi, drift_report


def test_no_drift_is_low():
    s = pd.Series(range(100))
    assert psi(s, s) < 0.01


def test_drift_report_schema():
    ref = pd.DataFrame({"x": range(100)})
    cur = pd.DataFrame({"x": [1000 + i for i in range(100)]})
    report = drift_report(ref, cur, ["x"])
    assert "max_psi" in report
    assert "threshold" in report
    assert report["status"] == "HIGH"
