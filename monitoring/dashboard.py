import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ModelOps Sentinel | AI Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

REFERENCE_PATH = DATA_DIR / "reference.csv"
COMPARISON_PATH = MODELS_DIR / "comparison.csv"
METADATA_PATH = MODELS_DIR / "metadata.json"


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
        .main {
            background-color: #f7f9fc;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .hero {
            padding: 25px;
            border-radius: 15px;
            background: linear-gradient(120deg, #14213d, #264a73);
            color: white;
            margin-bottom: 22px;
        }

        .hero h1 {
            color: white;
            margin-bottom: 8px;
        }

        .hero p {
            color: #e1eaf5;
            margin-bottom: 0;
        }

        div[data-testid="stMetric"] {
            background-color: white;
            border: 1px solid #e3e8ef;
            padding: 15px;
            border-radius: 12px;
        }

        h2, h3 {
            color: #14213d;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_csv(path):
    """Load a CSV file safely."""
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")

    return pd.DataFrame()


def load_metadata():
    """Load model metadata when available."""
    try:
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as file:
                return json.load(file)
    except Exception:
        pass

    return {}


def calculate_psi(reference, current, bins=10):
    """Calculate Population Stability Index for numerical data."""
    reference = pd.to_numeric(
        pd.Series(reference), errors="coerce"
    ).dropna()

    current = pd.to_numeric(
        pd.Series(current), errors="coerce"
    ).dropna()

    if len(reference) < 2 or len(current) < 2:
        return 0.0

    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    epsilon = 1e-6
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    return float(
        np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    )


def create_quality_report(df):
    """Create a column-level data quality report."""
    report = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [str(df[col].dtype) for col in df.columns],
            "Missing Values": [int(df[col].isna().sum()) for col in df.columns],
            "Missing Percentage": [
                round(df[col].isna().mean() * 100, 2)
                for col in df.columns
            ],
            "Unique Values": [int(df[col].nunique(dropna=True)) for col in df.columns],
        }
    )

    return report


def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def get_numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def get_categorical_columns(df):
    return df.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()


# ============================================================
# LOAD PROJECT DATA
# ============================================================

reference_df = load_csv(REFERENCE_PATH)
comparison_df = load_csv(COMPARISON_PATH)
metadata = load_metadata()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ ModelOps Sentinel</h1>
        <p>
            Multi-Model AI Intelligence and Analytics Platform
            | Data Quality | Model Performance | Drift Monitoring
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "An interactive demonstration of dataset analysis, "
    "machine learning model monitoring, and retraining evaluation."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧭 Navigation")

section = st.sidebar.radio(
    "Choose a section",
    [
        "📊 Dataset Analytics",
        "🤖 Model Overview",
        "🌊 Data Drift Monitoring",
        "🏆 Model Comparison",
        "🔄 Retraining Status",
    ],
)

st.sidebar.divider()

st.sidebar.info(
    "This project currently uses synthetic demonstration data "
    "for financial default-risk modelling. Results are not "
    "financial advice or production predictions."
)


# ============================================================
# DATASET ANALYTICS
# ============================================================

if section == "📊 Dataset Analytics":

    st.title("📊 Dataset Analytics & Exploratory Data Analysis")

    st.write(
        "Upload a CSV or Parquet dataset to explore its structure, "
        "quality, distributions, correlations, and important patterns."
    )

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "parquet"],
        help="Supported formats: CSV and Parquet",
    )

    analysis_df = reference_df.copy()
    dataset_name = "Reference Dataset"

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                analysis_df = pd.read_csv(uploaded_file)
            else:
                analysis_df = pd.read_parquet(uploaded_file)

            dataset_name = uploaded_file.name

            st.success(
                f"Successfully loaded **{uploaded_file.name}**."
            )

        except Exception as exc:
            st.error(f"Unable to load the uploaded dataset: {exc}")
            analysis_df = reference_df.copy()
            dataset_name = "Reference Dataset"

    if analysis_df.empty:
        st.warning(
            "No dataset is available. Upload a CSV or Parquet file "
            "or ensure data/reference.csv exists."
        )

    else:
        st.markdown(f"### 📁 {dataset_name}")

        # ----------------------------------------------------
        # SUMMARY METRICS
        # ----------------------------------------------------

        total_rows = len(analysis_df)
        total_columns = len(analysis_df.columns)
        missing_cells = int(analysis_df.isna().sum().sum())
        duplicate_rows = int(analysis_df.duplicated().sum())

        metric1, metric2, metric3, metric4 = st.columns(4)

        metric1.metric("Total Rows", f"{total_rows:,}")
        metric2.metric("Total Columns", total_columns)
        metric3.metric("Missing Cells", f"{missing_cells:,}")
        metric4.metric("Duplicate Rows", f"{duplicate_rows:,}")

        st.divider()

        # ----------------------------------------------------
        # DATA PREVIEW
        # ----------------------------------------------------

        st.subheader("🔍 Dataset Preview")

        preview_count = st.slider(
            "Number of rows to preview",
            min_value=5,
            max_value=min(50, max(5, total_rows)),
            value=min(10, max(5, total_rows)),
        )

        st.dataframe(
            analysis_df.head(preview_count),
            use_container_width=True,
        )

        with st.expander("View dataset information"):
            info_df = pd.DataFrame(
                {
                    "Column": analysis_df.columns,
                    "Data Type": [
                        str(analysis_df[col].dtype)
                        for col in analysis_df.columns
                    ],
                    "Non-Null Values": [
                        int(analysis_df[col].notna().sum())
                        for col in analysis_df.columns
                    ],
                    "Null Values": [
                        int(analysis_df[col].isna().sum())
                        for col in analysis_df.columns
                    ],
                }
            )

            st.dataframe(info_df, use_container_width=True)

        st.divider()

        # ----------------------------------------------------
        # DATA QUALITY REPORT
        # ----------------------------------------------------

        st.subheader("🧹 Data Quality Report")

        quality_report = create_quality_report(analysis_df)

        st.dataframe(
            quality_report,
            use_container_width=True,
            hide_index=True,
        )

        report_bytes = dataframe_to_csv_bytes(quality_report)

        st.download_button(
            label="📥 Download Data Quality Report",
            data=report_bytes,
            file_name="data_quality_report.csv",
            mime="text/csv",
        )

        # ----------------------------------------------------
        # AUTOMATIC INSIGHTS
        # ----------------------------------------------------

        st.subheader("💡 Automatic Data Insights")

        numeric_columns = get_numeric_columns(analysis_df)
        categorical_columns = get_categorical_columns(analysis_df)

        insight_col1, insight_col2 = st.columns(2)

        with insight_col1:
            st.markdown("**Dataset observations**")

            st.write(f"- Dataset contains **{total_rows:,} rows**.")
            st.write(f"- Dataset contains **{total_columns} columns**.")
            st.write(f"- Numerical columns: **{len(numeric_columns)}**.")
            st.write(
                f"- Categorical columns: **{len(categorical_columns)}**."
            )

            if missing_cells:
                st.warning(
                    f"Found {missing_cells:,} missing cells. "
                    "Consider handling missing values before training."
                )
            else:
                st.success("No missing cells were detected.")

        with insight_col2:
            st.markdown("**Data quality observations**")

            if duplicate_rows:
                st.warning(
                    f"Found {duplicate_rows:,} duplicate rows."
                )
            else:
                st.success("No duplicate rows were detected.")

            constant_columns = [
                col for col in analysis_df.columns
                if analysis_df[col].nunique(dropna=True) <= 1
            ]

            if constant_columns:
                st.warning(
                    "Columns with one or fewer unique non-null values: "
                    + ", ".join(constant_columns)
                )
            else:
                st.success("No constant columns were detected.")

        # ----------------------------------------------------
        # NUMERICAL STATISTICS
        # ----------------------------------------------------

        st.divider()
        st.subheader("📈 Numerical Statistics")

        if numeric_columns:
            selected_numeric = st.multiselect(
                "Select numerical columns",
                options=numeric_columns,
                default=numeric_columns[: min(6, len(numeric_columns))],
                key="eda_numeric_columns",
            )

            if selected_numeric:
                stats_df = analysis_df[selected_numeric].describe().T
                stats_df = stats_df.round(3)

                st.dataframe(
                    stats_df,
                    use_container_width=True,
                )

                st.download_button(
                    "📥 Download Numerical Statistics",
                    data=dataframe_to_csv_bytes(
                        stats_df.reset_index().rename(
                            columns={"index": "Feature"}
                        )
                    ),
                    file_name="numerical_statistics.csv",
                    mime="text/csv",
                )
            else:
                st.info("Select at least one numerical column.")

        else:
            st.info("This dataset has no numerical columns to summarize.")

        # ----------------------------------------------------
        # INTERACTIVE NUMERICAL DISTRIBUTION
        # ----------------------------------------------------

        st.divider()
        st.subheader("📊 Interactive Feature Distribution")

        if numeric_columns:
            distribution_column = st.selectbox(
                "Choose a numerical feature",
                numeric_columns,
                key="distribution_feature",
            )

            plot_df = analysis_df[[distribution_column]].dropna()

            if not plot_df.empty:
                histogram = px.histogram(
                    plot_df,
                    x=distribution_column,
                    nbins=30,
                    marginal="box",
                    title=f"Distribution of {distribution_column}",
                    template="plotly_white",
                )

                histogram.update_layout(
                    xaxis_title=distribution_column,
                    yaxis_title="Frequency",
                )

                st.plotly_chart(
                    histogram,
                    use_container_width=True,
                )

                col_a, col_b, col_c = st.columns(3)

                col_a.metric(
                    "Mean",
                    f"{plot_df[distribution_column].mean():,.3f}",
                )

                col_b.metric(
                    "Median",
                    f"{plot_df[distribution_column].median():,.3f}",
                )

                col_c.metric(
                    "Standard Deviation",
                    f"{plot_df[distribution_column].std():,.3f}",
                )

        # ----------------------------------------------------
        # CATEGORICAL DISTRIBUTION
        # ----------------------------------------------------

        st.divider()
        st.subheader("🗂️ Categorical Feature Analysis")

        if categorical_columns:
            category_column = st.selectbox(
                "Choose a categorical feature",
                categorical_columns,
                key="categorical_feature",
            )

            top_n = st.slider(
                "Number of categories to display",
                min_value=3,
                max_value=20,
                value=10,
                key="category_count",
            )

            category_counts = (
                analysis_df[category_column]
                .fillna("Missing")
                .astype(str)
                .value_counts()
                .head(top_n)
                .rename_axis("Category")
                .reset_index(name="Count")
            )

            category_chart = px.bar(
                category_counts,
                x="Category",
                y="Count",
                title=f"Category Distribution: {category_column}",
                template="plotly_white",
                text="Count",
            )

            category_chart.update_layout(
                xaxis_title=category_column,
                yaxis_title="Number of Records",
            )

            st.plotly_chart(
                category_chart,
                use_container_width=True,
            )

        else:
            st.info(
                "No categorical columns are available in this dataset."
            )

        # ----------------------------------------------------
        # CORRELATION HEATMAP
        # ----------------------------------------------------

        st.divider()
        st.subheader("🔥 Feature Correlation Heatmap")

        if len(numeric_columns) >= 2:
            correlation_columns = st.multiselect(
                "Choose features for correlation analysis",
                options=numeric_columns,
                default=numeric_columns[: min(10, len(numeric_columns))],
                key="correlation_columns",
            )

            if len(correlation_columns) >= 2:
                correlation = analysis_df[
                    correlation_columns
                ].corr()

                heatmap = go.Figure(
                    data=go.Heatmap(
                        z=correlation.values,
                        x=correlation.columns.tolist(),
                        y=correlation.index.tolist(),
                        colorscale="RdBu",
                        zmin=-1,
                        zmax=1,
                        zmid=0,
                        text=np.round(correlation.values, 2),
                        texttemplate="%{text}",
                        colorbar_title="Correlation",
                    )
                )

                heatmap.update_layout(
                    title="Numerical Feature Correlation",
                    template="plotly_white",
                    height=600,
                )

                st.plotly_chart(
                    heatmap,
                    use_container_width=True,
                )

                # Find the strongest non-duplicate feature relationships.
                upper_triangle = correlation.where(
                    np.triu(
                        np.ones(correlation.shape),
                        k=1,
                    ).astype(bool)
                )

                pairs = (
                    upper_triangle.stack()
                    .rename("Correlation")
                    .reset_index()
                )

                if not pairs.empty:
                    pairs.columns = [
                        "Feature 1",
                        "Feature 2",
                        "Correlation",
                    ]

                    pairs["Absolute Correlation"] = (
                        pairs["Correlation"].abs()
                    )

                    pairs = pairs.sort_values(
                        "Absolute Correlation",
                        ascending=False,
                    )

                    st.markdown("**Strongest feature relationships**")

                    st.dataframe(
                        pairs.drop(
                            columns=["Absolute Correlation"]
                        ).head(10).round(3),
                        use_container_width=True,
                        hide_index=True,
                    )

            else:
                st.info(
                    "Select at least two numerical features "
                    "to display the heatmap."
                )

        else:
            st.info(
                "At least two numerical columns are needed "
                "to calculate correlations."
            )

        # ----------------------------------------------------
        # TARGET COLUMN ANALYSIS
        # ----------------------------------------------------

        st.divider()
        st.subheader("🎯 Target Column Analysis")

        target_options = list(analysis_df.columns)

        if target_options:
            target_column = st.selectbox(
                "Select a target column",
                target_options,
                key="target_column",
                help=(
                    "Choose the column you want to understand as a "
                    "potential prediction target."
                ),
            )

            target_series = analysis_df[target_column]

            target_col1, target_col2, target_col3 = st.columns(3)

            target_col1.metric(
                "Unique Values",
                int(target_series.nunique(dropna=True)),
            )

            target_col2.metric(
                "Missing Values",
                int(target_series.isna().sum()),
            )

            target_col3.metric(
                "Data Type",
                str(target_series.dtype),
            )

            if pd.api.types.is_numeric_dtype(target_series):
                target_data = pd.DataFrame(
                    {
                        target_column: pd.to_numeric(
                            target_series,
                            errors="coerce",
                        ).dropna()
                    }
                )

                if not target_data.empty:
                    target_fig = px.histogram(
                        target_data,
                        x=target_column,
                        nbins=30,
                        title=f"Target Distribution: {target_column}",
                        template="plotly_white",
                    )

                    st.plotly_chart(
                        target_fig,
                        use_container_width=True,
                    )

            else:
                target_counts = (
                    target_series.fillna("Missing")
                    .astype(str)
                    .value_counts()
                    .reset_index()
                )

                target_counts.columns = ["Class", "Count"]

                target_fig = px.pie(
                    target_counts,
                    names="Class",
                    values="Count",
                    title=f"Target Class Distribution: {target_column}",
                    hole=0.35,
                )

                st.plotly_chart(
                    target_fig,
                    use_container_width=True,
                )

                st.dataframe(
                    target_counts,
                    use_container_width=True,
                    hide_index=True,
                )

        # ----------------------------------------------------
        # DOWNLOAD FULL ANALYSIS REPORT
        # ----------------------------------------------------

        st.divider()
        st.subheader("📥 Export Analysis Report")

        summary_df = pd.DataFrame(
            {
                "Metric": [
                    "Dataset Name",
                    "Total Rows",
                    "Total Columns",
                    "Missing Cells",
                    "Duplicate Rows",
                    "Numerical Columns",
                    "Categorical Columns",
                ],
                "Value": [
                    dataset_name,
                    total_rows,
                    total_columns,
                    missing_cells,
                    duplicate_rows,
                    len(numeric_columns),
                    len(categorical_columns),
                ],
            }
        )

        report_buffer = io.BytesIO()

        with pd.ExcelWriter(
            report_buffer,
            engine="openpyxl",
        ) as writer:
            summary_df.to_excel(
                writer,
                sheet_name="Summary",
                index=False,
            )

            quality_report.to_excel(
                writer,
                sheet_name="Data Quality",
                index=False,
            )

            if numeric_columns:
                analysis_df[numeric_columns].describe().T.to_excel(
                    writer,
                    sheet_name="Numerical Statistics",
                )

            if len(numeric_columns) >= 2:
                analysis_df[numeric_columns].corr().to_excel(
                    writer,
                    sheet_name="Correlation",
                )

        st.download_button(
            label="📦 Download Complete Analysis Report (Excel)",
            data=report_buffer.getvalue(),
            file_name="modelops_analysis_report.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
        )


# ============================================================
# MODEL OVERVIEW
# ============================================================

elif section == "🤖 Model Overview":

    st.title("🤖 Model Overview")

    model_name = metadata.get(
        "model_name",
        "LogisticRegression",
    )

    model_version = metadata.get(
        "model_version",
        "v1",
    )

    f1_score = metadata.get("f1_score", 0.842)
    roc_auc = metadata.get("roc_auc", 0.888)

    try:
        f1_score = float(f1_score)
    except (TypeError, ValueError):
        f1_score = 0.842

    try:
        roc_auc = float(roc_auc)
    except (TypeError, ValueError):
        roc_auc = 0.888

    training_rows = metadata.get(
        "trained_rows",
        len(reference_df),
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("Production Model", str(model_name))
    metric2.metric("Model Version", str(model_version))
    metric3.metric("F1 Score", f"{f1_score:.3f}")
    metric4.metric("ROC-AUC", f"{roc_auc:.3f}")

    st.divider()

    st.subheader("📦 Training Dataset")

    st.write(f"Training/reference records: **{training_rows:,}**")
    st.write(f"Reference dataset columns: **{len(reference_df.columns)}**")

    if not reference_df.empty:
        st.dataframe(
            reference_df.head(10),
            use_container_width=True,
        )
    else:
        st.warning("Reference dataset was not found.")

    st.info(
        "These metrics describe the saved demonstration model. "
        "Confirm the metadata and evaluation files before using "
        "the metrics in a formal report."
    )


# ============================================================
# DATA DRIFT MONITORING
# ============================================================

elif section == "🌊 Data Drift Monitoring":

    st.title("🌊 Data Drift Monitoring")

    st.write(
        "Explore how changes in selected numerical features affect "
        "the Population Stability Index (PSI)."
    )

    if reference_df.empty:
        st.warning(
            "Reference dataset is missing. Run the training script first."
        )

    else:
        drift_columns = get_numeric_columns(reference_df)

        if not drift_columns:
            st.warning(
                "The reference dataset has no numerical columns."
            )

        else:
            selected_features = st.multiselect(
                "Select features to monitor",
                options=drift_columns,
                default=drift_columns[: min(5, len(drift_columns))],
            )

            st.caption(
                "The sliders below create a simulated distribution shift. "
                "They do not represent live production traffic."
            )

            shift_strength = st.slider(
                "Simulated distribution shift (%)",
                min_value=0,
                max_value=100,
                value=0,
                step=5,
            )

            drift_results = []

            for column in selected_features:
                reference_values = pd.to_numeric(
                    reference_df[column],
                    errors="coerce",
                ).dropna()

                if len(reference_values) < 2:
                    continue

                standard_deviation = reference_values.std()

                if pd.isna(standard_deviation) or standard_deviation == 0:
                    simulated_values = reference_values.copy()
                else:
                    simulated_values = (
                        reference_values
                        + standard_deviation * (shift_strength / 100)
                    )

                psi_value = calculate_psi(
                    reference_values,
                    simulated_values,
                )

                if psi_value < 0.10:
                    status = "LOW"
                elif psi_value < 0.20:
                    status = "MODERATE"
                else:
                    status = "HIGH"

                drift_results.append(
                    {
                        "Feature": column,
                        "PSI": round(psi_value, 4),
                        "Status": status,
                    }
                )

            if drift_results:
                drift_df = pd.DataFrame(drift_results)

                high_count = int(
                    (drift_df["Status"] == "HIGH").sum()
                )

                moderate_count = int(
                    (drift_df["Status"] == "MODERATE").sum()
                )

                low_count = int(
                    (drift_df["Status"] == "LOW").sum()
                )

                col1, col2, col3 = st.columns(3)

                col1.metric("Low Drift", low_count)
                col2.metric("Moderate Drift", moderate_count)
                col3.metric("High Drift", high_count)

                st.subheader("Feature Drift Report")

                st.dataframe(
                    drift_df,
                    use_container_width=True,
                    hide_index=True,
                )

                drift_chart = px.bar(
                    drift_df,
                    x="Feature",
                    y="PSI",
                    color="Status",
                    title="Population Stability Index by Feature",
                    template="plotly_white",
                )

                drift_chart.add_hline(
                    y=0.10,
                    line_dash="dash",
                    annotation_text="0.10: Moderate drift boundary",
                )

                drift_chart.add_hline(
                    y=0.20,
                    line_dash="dash",
                    annotation_text="0.20: High drift boundary",
                )

                st.plotly_chart(
                    drift_chart,
                    use_container_width=True,
                )

                st.download_button(
                    "📥 Download Drift Report",
                    data=dataframe_to_csv_bytes(drift_df),
                    file_name="drift_report.csv",
                    mime="text/csv",
                )

            else:
                st.info(
                    "Select at least one suitable numerical feature."
                )

    st.warning(
        "Demo limitation: this page calculates drift on simulated "
        "values. It is not connected to a live prediction stream."
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif section == "🏆 Model Comparison":

    st.title("🏆 Model Comparison")

    st.write(
        "Compare the saved evaluation results for available models."
    )

    if comparison_df.empty:
        st.warning(
            "Model comparison results were not found. "
            "Run training/train.py to generate them."
        )

    else:
        st.subheader("📋 Model Performance Table")

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
        )

        # Identify likely metric columns automatically.
        normalized_columns = {
            str(column).strip().lower(): column
            for column in comparison_df.columns
        }

        metric_options = [
            column
            for column in comparison_df.columns
            if any(
                key in str(column).lower()
                for key in [
                    "accuracy",
                    "f1",
                    "precision",
                    "recall",
                    "roc_auc",
                    "roc-auc",
                ]
            )
        ]

        model_column = next(
            (
                column
                for column in comparison_df.columns
                if str(column).strip().lower()
                in ["model", "model_name", "name", "algorithm"]
            ),
            None,
        )

        if model_column is None:
            model_column = comparison_df.columns[0]

        if metric_options:
            selected_metrics = st.multiselect(
                "Select metrics to compare",
                options=metric_options,
                default=metric_options[: min(3, len(metric_options))],
            )

            if selected_metrics:
                chart_df = comparison_df[
                    [model_column] + selected_metrics
                ].copy()

                for metric in selected_metrics:
                    chart_df[metric] = pd.to_numeric(
                        chart_df[metric],
                        errors="coerce",
                    )

                melted_df = chart_df.melt(
                    id_vars=[model_column],
                    value_vars=selected_metrics,
                    var_name="Metric",
                    value_name="Score",
                )

                melted_df = melted_df.dropna(subset=["Score"])

                if not melted_df.empty:
                    comparison_chart = px.bar(
                        melted_df,
                        x=model_column,
                        y="Score",
                        color="Metric",
                        barmode="group",
                        title="Model Performance Comparison",
                        template="plotly_white",
                        text_auto=".3f",
                    )

                    st.plotly_chart(
                        comparison_chart,
                        use_container_width=True,
                    )

        st.download_button(
            "📥 Download Model Comparison",
            data=dataframe_to_csv_bytes(comparison_df),
            file_name="model_comparison.csv",
            mime="text/csv",
        )


# ============================================================
# RETRAINING STATUS
# ============================================================

elif section == "🔄 Retraining Status":

    st.title("🔄 Model Retraining Status")

    st.write(
        "Review the latest model promotion example and the "
        "conditions for replacing the production model."
    )

    production_f1 = 0.842
    candidate_f1 = 0.837
    candidate_model = "LogisticRegression"

    st.subheader("Latest Demonstration Evaluation")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Production F1 Score",
        f"{production_f1:.3f}",
    )

    col2.metric(
        "Candidate F1 Score",
        f"{candidate_f1:.3f}",
        delta=f"{candidate_f1 - production_f1:.3f}",
    )

    promoted = candidate_f1 > production_f1

    col3.metric(
        "Promotion Decision",
        "Promoted" if promoted else "Rejected",
    )

    if promoted:
        st.success(
            f"Candidate model {candidate_model} outperformed "
            "the production model in this demonstration."
        )
    else:
        st.warning(
            "Candidate model rejected: its F1 score did not "
            "exceed the production model's F1 score."
        )

    st.subheader("🧠 Model Promotion Workflow")

    st.markdown(
        """
        1. Monitor incoming data for distribution changes.
        2. Evaluate whether retraining should be considered.
        3. Train a candidate model on the selected training data.
        4. Compare candidate performance with the production model.
        5. Promote the candidate only if it satisfies the evaluation rule.
        """
    )

    st.info(
        "The scores and decision on this page are demonstration values. "
        "They are not fetched from a live retraining run."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ModelOps Sentinel | Multi-Model AI Intelligence and Analytics "
    "Platform | Interactive ML monitoring demonstration"
)