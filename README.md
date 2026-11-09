# 🛡️ ModelOps Sentinel

### Production ML Monitoring & Autonomous Retraining Platform

ModelOps Sentinel is a production-oriented MLOps system that turns a trained ML model into a monitored service. It compares candidate models, tracks experiments with MLflow, exposes REST inference APIs, detects production data drift with PSI, generates SHAP explanations, and runs a champion-vs-challenger retraining workflow before promoting a new model.

> **Portfolio focus:** AI/ML Engineering · MLOps · Model Monitoring · Explainable AI · FastAPI · Docker · CI/CD

## Why this project matters

A model does not stop being a software system after `fit()` finishes. Production inputs change, model quality can degrade, and a good ML platform needs observable health signals and controlled model promotion. ModelOps Sentinel demonstrates that lifecycle end to end.

## Architecture

```text
                       ┌──────────────────────┐
                       │   Synthetic / Live   │
                       │       Data           │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │ Training + Evaluation│
                       │ LR / RF / XGBoost    │
                       └──────────┬───────────┘
                                  │
                         experiments + models
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │       MLflow         │
                       │ Tracking + Registry  │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │      FastAPI         │
                       │ /predict /explain    │
                       └──────────┬───────────┘
                                  │
                           production data
                                  │
                                  ▼
                 ┌────────────────────────────────┐
                 │        Monitoring Engine       │
                 │ PSI Drift + Performance Checks │
                 └────────────────┬───────────────┘
                                  │ threshold breach
                                  ▼
                       ┌──────────────────────┐
                       │ Autonomous Retraining│
                       │ Champion vs Candidate│
                       └──────────┬───────────┘
                                  │
                         candidate wins?
                           /             \
                         YES              NO
                          │                │
                          ▼                ▼
                    Promote vN       Reject + record
```

## Key capabilities

- **Automated model selection:** Logistic Regression, Random Forest and XGBoost evaluated using F1, precision, recall and ROC-AUC.
- **Experiment tracking:** MLflow tracking plus optional model registration under `ModelOpsSentinel`.
- **Production inference:** FastAPI endpoints for health, model metadata and predictions.
- **Explainable AI:** SHAP-powered `/explain` endpoint ranks feature contributions for individual predictions.
- **Data drift monitoring:** PSI-based monitoring with configurable `0.20` breach threshold.
- **Autonomous retraining:** Drift/performance breaches trigger candidate training; the candidate is promoted only when its F1 beats the production champion.
- **Operations dashboard:** Streamlit dashboard for model health, drift simulation, model comparison and retraining decisions.
- **Containerized stack:** Docker Compose for API, dashboard and MLflow.
- **CI:** GitHub Actions installs dependencies, retrains a reproducible model, compiles source and executes tests.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service and production model health |
| `GET /model-info` | Version, metrics and feature metadata |
| `POST /predict` | Single prediction |
| `POST /explain` | Prediction + SHAP feature contributions |
| `POST /monitor` | Drift check and conditional retraining |
| `GET /docs` | Interactive Swagger API documentation |

### Example prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":42,"income":65000,"credit_score":720,"loan_amount":15000,"debt_ratio":0.28,"employment_years":8,"late_payments":1,"account_balance":18000}'
```

### Example explainability request

```bash
curl -X POST http://localhost:8000/explain \
  -H 'Content-Type: application/json' \
  -d '{"age":42,"income":65000,"credit_score":720,"loan_amount":15000,"debt_ratio":0.28,"employment_years":8,"late_payments":1,"account_balance":18000}'
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python training/train.py
uvicorn app.api.main:app --reload
```

In another terminal:

```bash
streamlit run monitoring/dashboard.py
```

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`

## Run the full stack with Docker

```bash
docker compose up --build
```

- API → `http://localhost:8000`
- Dashboard → `http://localhost:8501`
- MLflow → `http://localhost:5000`

## Repository structure

```text
modelops-sentinel/
├── app/
│   ├── api/main.py
│   ├── models/predictor.py
│   ├── monitoring/drift.py
│   └── services/retraining.py
├── training/train.py
├── monitoring/dashboard.py
├── tests/
├── data/
├── models/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .github/workflows/ci.yml
```

## Engineering decisions

1. **Synthetic data by default** keeps the repository reproducible and avoids publishing private financial data.
2. **Champion/challenger promotion** prevents an automatic retraining event from blindly replacing a better production model.
3. **PSI + delayed performance** provides two complementary health signals: distribution change and observed model quality.
4. **MLflow** makes experiments and model versions inspectable rather than hidden in local notebooks.
5. **FastAPI + Docker + CI** turns the project into a deployable engineering artifact rather than a standalone ML experiment.

## Production hardening roadmap

For a real financial deployment, add authentication/authorization, encrypted secrets, a governed feature store, persistent metrics, alerting, approval gates, audit logs, model rollback, and a real labeled data source.

## License

MIT

## End-to-end demo

Run the complete monitoring decision flow from the repository root:

```bash
python scripts/demo_end_to_end.py
```

The demo creates a controlled production distribution shift, calculates PSI drift, evaluates the retraining trigger, makes a sample prediction, and—when the trigger fires—runs the champion-vs-candidate retraining workflow.

> The demo uses synthetic data so the repository is reproducible and does not contain private customer information.
