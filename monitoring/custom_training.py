import io
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Custom Model Training | ModelOps Sentinel",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Custom Model Training")
st.write(
    "Upload a dataset, select the target column, train a machine "
    "learning model, evaluate performance, and download your model."
)

st.info(
    "This application trains models locally using uploaded data. "
    "The uploaded dataset is not automatically connected to the "
    "production API or production model."
)


# ==========================================================
# DATASET UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "parquet"],
)

if uploaded_file is None:
    st.markdown(
        """
        ### How to use Custom Model Training

        1. Upload a CSV or Parquet dataset.
        2. Select the column you want to predict.
        3. Choose Classification or Regression.
        4. Select a machine learning algorithm.
        5. Train and evaluate the model.
        6. Download the trained model.
        """
    )
    st.stop()


try:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_parquet(uploaded_file)

except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()


if df.empty or len(df.columns) < 2:
    st.error(
        "The dataset must contain at least one feature column "
        "and one target column."
    )
    st.stop()


# ==========================================================
# DATASET OVERVIEW
# ==========================================================

st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Rows", f"{len(df):,}")
col2.metric("Total Columns", len(df.columns))
col3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
col4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")

with st.expander("Preview Dataset", expanded=True):
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("Column Information"):
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[c].dtype) for c in df.columns],
        "Missing Values": [int(df[c].isna().sum()) for c in df.columns],
        "Unique Values": [
            int(df[c].nunique(dropna=True)) for c in df.columns
        ],
    })

    st.dataframe(info_df, use_container_width=True)


# ==========================================================
# TRAINING CONFIGURATION
# ==========================================================

st.divider()
st.subheader("⚙️ Training Configuration")

target_column = st.selectbox(
    "Select the target column",
    options=df.columns.tolist(),
)

problem_type = st.radio(
    "Select prediction type",
    ["Classification", "Regression"],
    horizontal=True,
)

if problem_type == "Classification":
    algorithm = st.selectbox(
        "Select classification algorithm",
        [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
        ],
    )
else:
    algorithm = st.selectbox(
        "Select regression algorithm",
        [
            "Linear Regression",
            "Random Forest",
            "XGBoost",
        ],
    )

test_size = st.slider(
    "Test dataset percentage",
    min_value=10,
    max_value=40,
    value=20,
    step=5,
)

random_state = st.number_input(
    "Random seed",
    min_value=0,
    max_value=99999,
    value=42,
    step=1,
)

st.caption(
    "Choose Classification for categories such as Yes/No or Low/Medium/High. "
    "Choose Regression for continuous numerical values."
)


# ==========================================================
# MODEL TRAINING
# ==========================================================

if st.button(
    "🚀 Train Model",
    type="primary",
    use_container_width=True,
):

    try:
        with st.spinner("Preparing data and training the model..."):

            # Remove rows with missing targets and duplicate rows.
            working_df = df.dropna(subset=[target_column]).copy()
            working_df = working_df.drop_duplicates()

            if len(working_df) < 10:
                raise ValueError(
                    "At least 10 usable rows are required."
                )

            X = working_df.drop(columns=[target_column])
            y = working_df[target_column]

            # Remove completely empty feature columns.
            empty_columns = [
                col for col in X.columns
                if X[col].isna().all()
            ]

            if empty_columns:
                X = X.drop(columns=empty_columns)

            if X.shape[1] == 0:
                raise ValueError(
                    "No usable feature columns remain."
                )

            # --------------------------------------------------
            # TARGET VALIDATION
            # --------------------------------------------------

            if problem_type == "Classification":

                if y.nunique() < 2:
                    raise ValueError(
                        "The target must contain at least two classes."
                    )

                if y.nunique() > 50:
                    raise ValueError(
                        f"The target '{target_column}' contains "
                        f"{y.nunique()} unique values. "
                        "If these are continuous numerical values, "
                        "select Regression. If these are real class "
                        "labels, this version needs an adjusted "
                        "classification setup."
                    )

                class_counts = y.value_counts()

                if class_counts.min() < 2:
                    raise ValueError(
                        "Every class must have at least two rows. "
                        "Please add examples or remove very rare classes."
                    )

                n_classes = y.nunique()
                n_test = int(np.ceil(len(y) * test_size / 100))
                n_train = len(y) - n_test

                stratify_y = (
                    y
                    if n_test >= n_classes and n_train >= n_classes
                    else None
                )

            else:

                # Regression requires a numeric target.
                y = pd.to_numeric(y, errors="coerce")

                valid_rows = y.notna()

                X = X.loc[valid_rows]
                y = y.loc[valid_rows]

                if len(y) < 10 or y.nunique() < 2:
                    raise ValueError(
                        "Regression needs at least 10 usable rows "
                        "and two distinct numerical target values."
                    )

                n_test = int(np.ceil(len(y) * test_size / 100))
                n_train = len(y) - n_test

                stratify_y = None

            if n_test < 1 or n_train < 2:
                raise ValueError(
                    "Not enough rows for the selected train/test split."
                )

            # --------------------------------------------------
            # FEATURE PREPROCESSING
            # --------------------------------------------------

            numeric_features = X.select_dtypes(
                include=["number"]
            ).columns.tolist()

            categorical_features = [
                col for col in X.columns
                if col not in numeric_features
            ]

            transformers = []

            if numeric_features:
                numeric_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ])

                transformers.append(
                    ("numeric", numeric_pipeline, numeric_features)
                )

            if categorical_features:
                categorical_pipeline = Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ])

                transformers.append(
                    (
                        "categorical",
                        categorical_pipeline,
                        categorical_features,
                    )
                )

            preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder="drop",
            )

            # --------------------------------------------------
            # SELECT ALGORITHM
            # --------------------------------------------------

            if problem_type == "Classification":

                if algorithm == "Logistic Regression":
                    estimator = LogisticRegression(
                        max_iter=2000,
                        random_state=int(random_state),
                    )

                elif algorithm == "Random Forest":
                    estimator = RandomForestClassifier(
                        n_estimators=150,
                        random_state=int(random_state),
                        n_jobs=-1,
                    )

                else:
                    from xgboost import XGBClassifier

                    estimator = XGBClassifier(
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                        random_state=int(random_state),
                        n_jobs=-1,
                        eval_metric="mlogloss",
                    )

            else:

                if algorithm == "Linear Regression":
                    estimator = LinearRegression()

                elif algorithm == "Random Forest":
                    estimator = RandomForestRegressor(
                        n_estimators=150,
                        random_state=int(random_state),
                        n_jobs=-1,
                    )

                else:
                    from xgboost import XGBRegressor

                    estimator = XGBRegressor(
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                        random_state=int(random_state),
                        n_jobs=-1,
                        objective="reg:squarederror",
                    )

            model = Pipeline([
                ("preprocessor", preprocessor),
                ("model", estimator),
            ])

            # --------------------------------------------------
            # SPLIT AND TRAIN
            # --------------------------------------------------

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size / 100,
                random_state=int(random_state),
                stratify=stratify_y,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            # --------------------------------------------------
            # EVALUATION
            # --------------------------------------------------

            if problem_type == "Classification":

                metrics = {
                    "Accuracy": accuracy_score(y_test, predictions),
                    "Precision": precision_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    ),
                    "Recall": recall_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    ),
                    "F1 Score": f1_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    ),
                }

                labels = model.named_steps["model"].classes_

                cm = confusion_matrix(
                    y_test,
                    predictions,
                    labels=labels,
                )

                report_df = pd.DataFrame(
                    classification_report(
                        y_test,
                        predictions,
                        output_dict=True,
                        zero_division=0,
                    )
                ).transpose()

                st.session_state["custom_confusion_matrix"] = cm
                st.session_state["custom_class_labels"] = labels.tolist()
                st.session_state["custom_report"] = report_df

            else:

                metrics = {
                    "MAE": mean_absolute_error(y_test, predictions),
                    "RMSE": np.sqrt(
                        mean_squared_error(y_test, predictions)
                    ),
                    "R2 Score": r2_score(y_test, predictions),
                }

            # Save results to Streamlit session state.
            st.session_state["custom_model"] = model
            st.session_state["custom_metrics"] = {
                key: float(value)
                for key, value in metrics.items()
            }
            st.session_state["custom_target"] = target_column
            st.session_state["custom_features"] = X.columns.tolist()
            st.session_state["custom_problem_type"] = problem_type
            st.session_state["custom_algorithm"] = algorithm
            st.session_state["custom_train_rows"] = len(X_train)
            st.session_state["custom_test_rows"] = len(X_test)
            st.session_state["custom_y_test"] = y_test.tolist()
            st.session_state["custom_predictions"] = predictions.tolist()
            st.session_state["custom_training_success"] = True

        st.success("Model trained successfully!")

    except Exception as exc:
        st.session_state["custom_training_success"] = False
        st.error(f"Training failed: {exc}")


# ==========================================================
# DISPLAY RESULTS
# ==========================================================

if st.session_state.get("custom_training_success", False):

    model = st.session_state["custom_model"]
    metrics = st.session_state["custom_metrics"]
    problem_type = st.session_state["custom_problem_type"]

    y_test = st.session_state["custom_y_test"]
    predictions = st.session_state["custom_predictions"]

    st.divider()
    st.subheader("✅ Model Training Results")

    st.write(f"**Algorithm:** {st.session_state['custom_algorithm']}")
    st.write(f"**Prediction type:** {problem_type}")
    st.write(f"**Training rows:** {st.session_state['custom_train_rows']:,}")
    st.write(f"**Testing rows:** {st.session_state['custom_test_rows']:,}")

    metric_columns = st.columns(len(metrics))

    for column, (name, value) in zip(metric_columns, metrics.items()):
        column.metric(name, f"{value:.4f}")

    # ------------------------------------------------------
    # CLASSIFICATION RESULTS
    # ------------------------------------------------------

    if problem_type == "Classification":

        st.subheader("📊 Confusion Matrix")

        labels = st.session_state["custom_class_labels"]
        cm = st.session_state["custom_confusion_matrix"]

        fig = px.imshow(
            cm,
            x=[str(label) for label in labels],
            y=[str(label) for label in labels],
            text_auto=True,
            aspect="auto",
            title="Actual vs Predicted Classes",
            labels={
                "x": "Predicted Class",
                "y": "Actual Class",
                "color": "Count",
            },
            color_continuous_scale="Blues",
        )

        st.plotly_chart(fig, use_container_width=True)

        report_df = st.session_state["custom_report"]

        st.subheader("Classification Report")

        st.dataframe(
            report_df.round(4),
            use_container_width=True,
        )

        st.download_button(
            "📥 Download Classification Report",
            data=report_df.to_csv().encode("utf-8"),
            file_name="classification_report.csv",
            mime="text/csv",
        )

    # ------------------------------------------------------
    # REGRESSION RESULTS
    # ------------------------------------------------------

    else:

        st.subheader("📈 Actual vs Predicted Values")

        results_df = pd.DataFrame({
            "Actual": y_test,
            "Predicted": predictions,
        })

        fig = px.scatter(
            results_df,
            x="Actual",
            y="Predicted",
            title="Actual vs Predicted Values",
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            results_df.head(20),
            use_container_width=True,
        )

        st.download_button(
            "📥 Download Predictions",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="regression_predictions.csv",
            mime="text/csv",
        )

    # ------------------------------------------------------
    # DOWNLOAD TRAINED MODEL
    # ------------------------------------------------------

    st.divider()
    st.subheader("📦 Download Trained Model")

    model_package = {
        "model": model,
        "target_column": st.session_state["custom_target"],
        "feature_columns": st.session_state["custom_features"],
        "problem_type": problem_type,
        "algorithm": st.session_state["custom_algorithm"],
        "metrics": metrics,
    }

    model_buffer = io.BytesIO()
    joblib.dump(model_package, model_buffer)
    model_buffer.seek(0)

    st.download_button(
        "📥 Download Trained Model (.joblib)",
        data=model_buffer.getvalue(),
        file_name="custom_trained_model.joblib",
        mime="application/octet-stream",
    )

    st.caption(
        "Only load joblib files from trusted sources. "
        "The downloaded model is separate from the production API model."
    )
