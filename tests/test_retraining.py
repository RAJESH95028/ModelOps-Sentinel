from app.services.retraining import should_retrain


def test_drift_triggers_retraining():
    assert should_retrain(0.25, 0.95)


def test_good_model_no_retraining():
    assert not should_retrain(0.05, 0.95)


def test_bad_performance_triggers_retraining():
    assert should_retrain(0.05, 0.70)
