# 🛡️ ModelOps Sentinel

### Multi-Model AI Intelligence and Analytics Platform

**ModelOps Sentinel** is an AI/ML and MLOps platform designed to manage the machine learning lifecycle, from model training and evaluation to deployment, monitoring and retraining.

It provides an interactive dashboard for dataset analysis, custom model training, REST APIs for predictions, data drift detection and model performance monitoring.

The project demonstrates how machine learning models can be developed and managed through a unified platform.

---

## 🚀 Project Overview

Traditional machine learning projects usually focus on training a model and generating predictions. However, real-world applications also need data validation, model evaluation, monitoring and retraining.

ModelOps Sentinel brings these important activities together in one platform.

### Main Objectives

- Analyze and validate uploaded datasets.
- Perform data cleaning and exploratory data analysis.
- Train and compare multiple machine learning algorithms.
- Generate predictions using trained models.
- Monitor data drift and model performance.
- Provide API-based access to machine learning models.
- Support controlled model retraining.

---

## ✨ Key Features

### 1. Dataset Analysis and Visualization
- Upload and explore datasets.
- Preview dataset records.
- Check missing values and data quality.
- Generate statistical summaries.
- Visualize feature distributions.
- Analyze correlations using heatmaps.

### 2. Custom Machine Learning Training
- Upload CSV and Parquet datasets.
- Select the target column.
- Choose classification or regression.
- Train multiple machine learning algorithms.
- Compare model evaluation metrics.
- Download the trained model.

### 3. Multiple Model Comparison
Supported algorithms include:

**Classification**
- Logistic Regression
- Random Forest
- XGBoost

**Regression**
- Linear Regression
- Random Forest
- XGBoost

### 4. Model Monitoring
- Monitor data distribution changes.
- Detect data drift using Population Stability Index (PSI).
- Configure drift detection thresholds.
- Evaluate model performance.
- Identify when retraining may be required.

### 5. FastAPI Integration
The platform provides REST API endpoints for:
- Service health checks.
- Model information.
- Machine learning predictions.
- Prediction explanations.
- Data drift monitoring.

### 6. Automated Retraining Workflow
- Detect monitoring threshold breaches.
- Train a candidate model.
- Compare the candidate with the existing model.
- Promote the candidate only when its F1 score improves.
- Record retraining results.

### 7. Interactive Dashboard
The Streamlit dashboard provides visual access to dataset analysis, model monitoring and related analytics.

---

## 🏗️ System Architecture

```text
                 DATASET UPLOAD
                       |
                       v
             DATA VALIDATION & CLEANING
                       |
                       v
              EXPLORATORY DATA ANALYSIS
                       |
                       v
              MACHINE LEARNING TRAINING
                       |
                       v
              MODEL EVALUATION & COMPARISON
                       |
                       v
                 TRAINED MODEL
                       |
             +---------+---------+
             |                   |
             v                   v
         FASTAPI API        STREAMLIT DASHBOARD
             |                   |
             v                   v
         PREDICTIONS       VISUAL ANALYTICS
             |
             v
       MODEL MONITORING
             |
             v
         DATA DRIFT
             |
             v
       RETRAINING CHECK
             |
             v
      CHAMPION VS CANDIDATE
             |
       +-----+-----+
       |           |
    IMPROVED    NOT IMPROVED
       |           |
       v           v
     PROMOTE      REJECT
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data processing |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning |
| XGBoost | Gradient boosting models |
| Streamlit | Interactive dashboard |
| FastAPI | REST API development |
| MLflow | Experiment tracking |
| SHAP | Model explainability |
| Matplotlib | Data visualization |
| Docker | Application containerization |
| GitHub Actions | Continuous integration |
| Pytest | Automated testing |

---

## 📂 Project Structure

```text
modelops-sentinel-main/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .streamlit/
│   └── config.toml
│
├── app/
│   ├── api/
│   │   └── main.py
│   ├── models/
│   │   └── predictor.py
│   ├── monitoring/
│   │   └── drift.py
│   └── services/
│       └── retraining.py
│
├── data/
│   └── reference.csv
│
├── models/
│   ├── model.joblib
│   ├── metadata.json
│   ├── comparison.csv
│   └── retraining_result.json
│
├── monitoring/
│   ├── dashboard.py
│   └── custom_training.py
│
├── scripts/
│   └── demo_end_to_end.py
│
├── tests/
│   ├── test_api.py
│   ├── test_drift.py
│   └── test_retraining.py
│
├── training/
│   └── train.py
│
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
├── README.md
└── .gitignore
```

---

## 💻 Installation and Setup

### Prerequisites

Install the following before running the project:

- Python 3.11 or later
- Git
- Visual Studio Code (recommended)
- Internet connection for installing dependencies

### Step 1: Clone the repository

```powershell
git clone https://github.com/RAJESH95028/ModelOps-Sentinel.git
cd ModelOps-Sentinel
```

### Step 2: Create a virtual environment

```powershell
python -m venv .venv
```

### Step 3: Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```powershell
python -m pip install -r requirements.txt
```

---

## ▶️ Running the Application

Run each component in a separate PowerShell terminal from the project root.

### 1. Main Dashboard

```powershell
python -m streamlit run monitoring/dashboard.py --server.port 8501
```

Open:

http://localhost:8501

### 2. Custom Model Training

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m streamlit run monitoring/custom_training.py --server.port 8502
```

Open:

http://localhost:8502

### 3. FastAPI Server

```powershell
uvicorn app.api.main:app --reload --port 8000
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

### 4. Model Training

To train the default reference model:

```powershell
python training/train.py
```

### 5. End-to-End Monitoring Demo

```powershell
python scripts/demo_end_to_end.py
```

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks API and model health |
| GET | `/model-info` | Displays model information |
| POST | `/predict` | Generates a prediction |
| POST | `/explain` | Explains a prediction |
| POST | `/monitor` | Checks data drift and retraining conditions |
| GET | `/docs` | Opens interactive API documentation |

### Example Prediction Request

Send a POST request to:

`http://127.0.0.1:8000/predict`

Example JSON:

```json
{
  "age": 42,
  "income": 65000,
  "credit_score": 720,
  "loan_amount": 15000,
  "debt_ratio": 0.28,
  "employment_years": 8,
  "late_payments": 1,
  "account_balance": 18000
}
```

The API returns the prediction and associated risk information.

---

## 📊 Model Evaluation

The platform supports model comparison using relevant evaluation metrics.

**Classification metrics**
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

**Regression metrics**
- Mean Absolute Error
- Mean Squared Error
- Root Mean Squared Error
- R² Score

---

## 📈 Monitoring and Retraining

Model monitoring is an important part of this project.

The platform uses the Population Stability Index (PSI) to identify changes in data distributions.

A PSI threshold of `0.20` is used by the current monitoring workflow.

When monitoring conditions indicate a need for retraining, the system can train and evaluate a candidate model.

The candidate is promoted only when its F1 score is better than the existing champion model.

---

## 🐳 Docker Deployment

The project includes Docker Compose configuration for running the application stack.

Build and start the services using:

```powershell
docker compose up --build
```

The configured services can be accessed through their respective local ports.

---

## 🧪 Testing

Run the automated test suite using:

```powershell
pytest -v
```

The tests cover API functionality, data drift calculations and retraining logic.

---

## 🔮 Future Enhancements

- Cloud deployment and scalable infrastructure.
- User authentication and role-based access.
- Advanced model explainability.
- Real-time monitoring and alert notifications.
- Model versioning and rollback.
- Integration with additional machine learning algorithms.
- Enhanced reporting and analytics.

---

## 🎯 Project Outcome

ModelOps Sentinel demonstrates an integrated machine learning workflow that combines dataset analysis, model training, model evaluation, API deployment, monitoring and retraining.

It aims to make machine learning workflows easier to develop, understand and manage through a unified analytics platform.

---

## 👨‍💻 Developer

**G. Rajesh**

GitHub: https://github.com/RAJESH95028

Project Repository: https://github.com/RAJESH95028/ModelOps-Sentinel

---

*Developed as an AI/ML and MLOps portfolio project.*