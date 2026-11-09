from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def payload():
    return {
        "age": 35,
        "income": 50000,
        "credit_score": 700,
        "loan_amount": 15000,
        "debt_ratio": 0.3,
        "employment_years": 5,
        "late_payments": 1,
        "account_balance": 10000,
    }


def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_prediction():
    response = client.post("/predict", json=payload())
    assert response.status_code == 200

    result = response.json()
    assert "prediction" in result
    assert "risk_label" in result
    assert "default_probability" in result
    assert result["prediction"] in [0, 1]
    assert 0 <= result["default_probability"] <= 1


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200

    result = response.json()
    assert "model_name" in result
    assert "features" in result
    assert "metrics" in result


def test_explanation():
    response = client.post("/explain", json=payload())
    assert response.status_code == 200


def test_monitor():
    response = client.post(
        "/monitor",
        json={"records": [payload() for _ in range(20)]},
    )

    assert response.status_code == 200