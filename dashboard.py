import streamlit as st
import pandas as pd
import json
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="XAI Credit Risk & Macro Stress Testing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = BASE_DIR / "figures"


# ============================================================
# FILE PATHS
# ============================================================

MODEL_METADATA_FILE = (
    MODEL_DIR / "model_metadata.json"
)

FINAL_PREDICTIONS_FILE = (
    OUTPUT_DIR / "final_test_predictions.csv"
)

MACRO_SUMMARY_FILE = (
    OUTPUT_DIR / "macro_stress_summary.csv"
)

COUNTRY_MACRO_SUMMARY_FILE = (
    OUTPUT_DIR / "country_macro_stress_summary.csv"
)

SHAP_IMPORTANCE_FILE = (
    OUTPUT_DIR / "SHAP_feature_importance.csv"
)

SHAP_INDIVIDUAL_FILE = (
    OUTPUT_DIR / "SHAP_individual_explanation.csv"
)

COUNTRY_RISK_FILE = (
    OUTPUT_DIR / "country_risk_summary.csv"
)

COUNTRY_PERFORMANCE_FILE = (
    OUTPUT_DIR / "country_model_performance.csv"
)


# ============================================================
# IMAGE FILES
# ============================================================

SHAP_GLOBAL_IMAGE = (
    FIGURES_DIR / "SHAP_global_feature_importance.png"
)

SHAP_SUMMARY_IMAGE = (
    FIGURES_DIR / "SHAP_summary_plot.png"
)

SHAP_INDIVIDUAL_IMAGE = (
    FIGURES_DIR / "XAI_individual_explanation.png"
)

MACRO_PD_IMAGE = (
    FIGURES_DIR / "macro_stress_pd.png"
)

MACRO_RELATIVE_IMAGE = (
    FIGURES_DIR / "macro_stress_relative_change.png"
)

MACRO_ABSOLUTE_IMAGE = (
    FIGURES_DIR / "macro_stress_absolute_change.png"
)

COUNTRY_PD_IMAGE = (
    FIGURES_DIR / "country_actual_vs_predicted_pd.png"
)

COUNTRY_CALIBRATION_IMAGE = (
    FIGURES_DIR / "country_calibration_gap.png"
)

COUNTRY_ROC_IMAGE = (
    FIGURES_DIR / "country_roc_auc.png"
)


# ============================================================
# FILE VALIDATION
# ============================================================

REQUIRED_FILES = {
    "Model metadata": MODEL_METADATA_FILE,
    "Final predictions": FINAL_PREDICTIONS_FILE,
    "Macro summary": MACRO_SUMMARY_FILE,
    "Country macro summary": COUNTRY_MACRO_SUMMARY_FILE,
    "SHAP importance": SHAP_IMPORTANCE_FILE,
    "SHAP individual": SHAP_INDIVIDUAL_FILE,
    "Country risk": COUNTRY_RISK_FILE,
    "Country performance": COUNTRY_PERFORMANCE_FILE,
}


REQUIRED_IMAGES = {
    "SHAP global importance": SHAP_GLOBAL_IMAGE,
    "SHAP summary plot": SHAP_SUMMARY_IMAGE,
    "SHAP individual explanation": SHAP_INDIVIDUAL_IMAGE,
    "Macro PD": MACRO_PD_IMAGE,
    "Macro relative change": MACRO_RELATIVE_IMAGE,
    "Macro absolute change": MACRO_ABSOLUTE_IMAGE,
    "Country actual vs predicted PD": COUNTRY_PD_IMAGE,
    "Country calibration gap": COUNTRY_CALIBRATION_IMAGE,
    "Country ROC-AUC": COUNTRY_ROC_IMAGE,
}


def validate_files():

    missing_files = []

    for name, path in REQUIRED_FILES.items():

        if not path.exists():
            missing_files.append(
                f"{name}: {path}"
            )

    if missing_files:

        st.error(
            "❌ Required data files are missing."
        )

        for file in missing_files:
            st.write(file)

        st.stop()


validate_files()


# ============================================================
# GLOBAL DASHBOARD STYLING
# ============================================================

st.markdown(
    """
    <style>

    [data-testid="stMetric"] {
        padding-top: 4px !important;
        padding-bottom: 4px !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 16px !important;
        font-weight: 600 !important;
        line-height: 1.3 !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 500 !important;
        line-height: 1.15 !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 13px !important;
    }

    .metric-label {
        font-size: 16px;
        font-weight: 600;
        color: #343746;
        margin-bottom: 6px;
        min-height: 22px;
        line-height: 1.25;
    }

    .metric-value {
        font-size: 27px;
        font-weight: 500;
        line-height: 1.15;
        color: #343746;
        min-height: 34px;
        white-space: nowrap;
    }

    .model-value {
        font-size: 24px;
        font-weight: 650;
        line-height: 1.15;
        color: #343746;
        min-height: 34px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .metric-card {
        min-height: 70px;
        padding-right: 8px;
    }

    h1 {
        margin-bottom: 0.35rem !important;
    }

    h2 {
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
    }

    h3 {
        margin-top: 0.8rem !important;
        margin-bottom: 0.4rem !important;
    }

    [data-testid="stDataFrame"] {
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL METADATA
# ============================================================

@st.cache_data
def load_metadata():

    with open(
        MODEL_METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# LOAD CSV
# ============================================================

@st.cache_data
def load_csv(file_path):

    return pd.read_csv(file_path)


# ============================================================
# LOAD ALL DATA
# ============================================================

metadata = load_metadata()

macro_summary = load_csv(
    MACRO_SUMMARY_FILE
)

country_macro_summary = load_csv(
    COUNTRY_MACRO_SUMMARY_FILE
)

shap_importance = load_csv(
    SHAP_IMPORTANCE_FILE
)

shap_individual = load_csv(
    SHAP_INDIVIDUAL_FILE
)

country_risk = load_csv(
    COUNTRY_RISK_FILE
)

country_performance = load_csv(
    COUNTRY_PERFORMANCE_FILE
)


# ============================================================
# TEST DATA SUMMARY
# ============================================================

@st.cache_data
def get_test_summary():

    df = pd.read_csv(
        FINAL_PREDICTIONS_FILE
    )

    observations = len(df)

    actual_defaults = int(
        df["actual_default"].sum()
    )

    mean_pd = (
        df["predicted_probability_of_default"]
        .mean()
    )

    return (
        observations,
        actual_defaults,
        mean_pd
    )


(
    test_observations,
    actual_defaults,
    average_pd
) = get_test_summary()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Dashboard Navigation"
)

st.sidebar.markdown(
    "### Select Section"
)

section = st.sidebar.radio(
    "",
    [
        "Executive Overview",
        "Macro Stress Testing",
        "SHAP Explainability",
        "Country Risk & Fairness",
        "Model Limitations"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "XAI Credit Risk & Macro Stress Testing"
)

st.sidebar.caption(
    "Final calibrated Random Forest model"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "📊 XAI Credit Risk & Macro Stress Testing"
)

st.markdown(
    "**Explainable Credit Risk Modelling with Macro Stress Testing**"
)

st.markdown(
    """
This dashboard combines machine learning predictions,
SHAP-based explainability, macroeconomic stress scenarios,
and country-level risk analysis.
"""
)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

if section == "Executive Overview":

    st.header(
        "Executive Overview"
    )

    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Model</div>
                <div class="model-value">Random Forest</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Test ROC-AUC</div>
                <div class="metric-value">
                    {metadata["test_roc_auc"]:.4f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Test PR-AUC</div>
                <div class="metric-value">
                    {metadata["test_pr_auc"]:.4f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Brier Score</div>
                <div class="metric-value">
                    {metadata["test_brier_score"]:.4f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # --------------------------------------------------------
    # MODEL CONFIGURATION
    # --------------------------------------------------------

    st.subheader(
        "Model Configuration"
    )

    config_col1, config_col2 = st.columns(2)

    with config_col1:

        st.write(
            f"**Training period:** "
            f"{metadata['train_period']}"
        )

        st.write(
            f"**Validation period:** "
            f"{metadata['validation_period']}"
        )

        st.write(
            f"**Test period:** "
            f"{metadata['test_period']}"
        )

        st.write(
            f"**Random seed:** "
            f"{metadata['random_seed']}"
        )

    with config_col2:

        calibration_status = (
            "Applied"
            if metadata["calibration_applied"]
            else "Not Applied"
        )

        st.write(
            f"**Calibration:** "
            f"{calibration_status}"
        )

        st.write(
            f"**Number of features:** "
            f"{len(metadata['features'])}"
        )

        st.write(
            "**Model type:** Random Forest "
            "(tuned and calibrated)"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # TEST DATASET
    # --------------------------------------------------------

    st.subheader(
        "Test Dataset"
    )

    data_col1, data_col2, data_col3 = st.columns(3)

    with data_col1:

        st.metric(
            "Test Observations",
            f"{test_observations:,}"
        )

    with data_col2:

        st.metric(
            "Actual Defaults",
            f"{actual_defaults:,}"
        )

    with data_col3:

        st.metric(
            "Average Predicted PD",
            f"{average_pd * 100:.2f}%"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # PROJECT COMPONENTS
    # --------------------------------------------------------

    st.subheader(
        "Project Components"
    )

    st.markdown(
        """
- **Credit Risk Prediction** — Random Forest classification model
- **XAI** — SHAP-based global and individual explanations
- **Macro Stress Testing** — Interest-rate shock scenarios
- **Country Segmentation** — Country-level risk analysis
- **Fairness Screening** — Prediction and calibration comparison
"""
    )

    st.markdown("---")

    # --------------------------------------------------------
    # MODEL PERFORMANCE NOTE
    # --------------------------------------------------------

    st.subheader(
        "Model Performance Note"
    )

    st.info(
        f"""
The 2023 test ROC-AUC of
**{metadata["test_roc_auc"]:.4f}**
indicates limited discriminative performance on the
future test period.

Therefore, the model outputs and stress scenarios should
be interpreted as risk-sensitivity analysis rather than
as a highly accurate default prediction system.
"""
    )


# ============================================================
# MACRO STRESS TESTING
# ============================================================

elif section == "Macro Stress Testing":

    st.header(
        "📈 Macro Stress Testing"
    )

    st.markdown(
        """
Scenario-based sensitivity analysis showing how
interest-rate shocks affect predicted probability
of default (PD).
"""
    )

    # --------------------------------------------------------
    # SCENARIO TABLE
    # --------------------------------------------------------

    st.subheader(
        "Stress Scenario Results"
    )

    display_macro = macro_summary.copy()

    for column in [
        "mean_pd",
        "median_pd",
        "max_pd",
        "absolute_pd_change"
    ]:

        if column in display_macro.columns:

            display_macro[column] = (
                display_macro[column] * 100
            )

    display_macro = display_macro.rename(
        columns={
            "scenario": "Scenario",
            "interest_rate_shock":
                "Interest Rate Shock",
            "mean_pd": "Mean PD (%)",
            "median_pd": "Median PD (%)",
            "max_pd": "Maximum PD (%)",
            "absolute_pd_change":
                "Absolute PD Change (pp)",
            "relative_pd_change_percent":
                "Relative PD Change (%)"
        }
    )

    if "Interest Rate Shock" in display_macro.columns:

        display_macro[
            "Interest Rate Shock"
        ] = (
            display_macro[
                "Interest Rate Shock"
            ] * 100
        ).map(
            lambda x: f"{x:.0f}%"
        )

    st.dataframe(
        display_macro,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # KEY STRESS METRICS
    # --------------------------------------------------------

    baseline = macro_summary.iloc[0]

    severe_df = macro_summary[
        macro_summary["scenario"]
        == "Severe Stress (+20%)"
    ]

    if not severe_df.empty:

        severe = severe_df.iloc[0]

        mean_pd_change = (
            severe["mean_pd"]
            - baseline["mean_pd"]
        )

        relative_change = (
            severe["relative_pd_change_percent"]
        )

    else:

        severe = None
        mean_pd_change = 0
        relative_change = 0

    st.markdown("---")

    stress_col1, stress_col2, stress_col3 = st.columns(3)

    with stress_col1:

        st.metric(
            "Baseline Mean PD",
            f"{baseline['mean_pd'] * 100:.2f}%"
        )

    with stress_col2:

        st.metric(
            "Severe Stress Mean PD",
            (
                f"{severe['mean_pd'] * 100:.2f}%"
                if severe is not None
                else "N/A"
            )
        )

    with stress_col3:

        st.metric(
            "Severe Stress Relative Change",
            f"{relative_change:.2f}%"
        )

    # --------------------------------------------------------
    # MEAN PD
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Mean Probability of Default by Scenario"
    )

    if MACRO_PD_IMAGE.exists():

        st.image(
            str(MACRO_PD_IMAGE),
            use_container_width=True
        )

    else:

        st.warning(
            f"Image not found: {MACRO_PD_IMAGE}"
        )

    # --------------------------------------------------------
    # RELATIVE CHANGE
    # --------------------------------------------------------

    st.subheader(
        "Relative Increase in Probability of Default"
    )

    if MACRO_RELATIVE_IMAGE.exists():

        st.image(
            str(MACRO_RELATIVE_IMAGE),
            use_container_width=True
        )

    else:

        st.warning(
            f"Image not found: {MACRO_RELATIVE_IMAGE}"
        )

    # --------------------------------------------------------
    # ABSOLUTE CHANGE
    # --------------------------------------------------------

    st.subheader(
        "Absolute Increase in Probability of Default"
    )

    if MACRO_ABSOLUTE_IMAGE.exists():

        st.image(
            str(MACRO_ABSOLUTE_IMAGE),
            use_container_width=True
        )

    else:

        st.warning(
            f"Image not found: {MACRO_ABSOLUTE_IMAGE}"
        )

    # --------------------------------------------------------
    # COUNTRY STRESS
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Country-Level Macro Stress Results"
    )

    country_stress_display = (
        country_macro_summary.copy()
    )

    for column in [
        "mean_pd",
        "median_pd",
        "max_pd",
        "baseline_mean_pd",
        "absolute_pd_change"
    ]:

        if column in country_stress_display.columns:

            country_stress_display[column] = (
                country_stress_display[column] * 100
            )

    country_stress_display = (
        country_stress_display.rename(
            columns={
                "country_grouped": "Country",
                "scenario": "Scenario",
                "interest_rate_shock":
                    "Interest Rate Shock",
                "observations": "Observations",
                "mean_pd": "Mean PD (%)",
                "median_pd": "Median PD (%)",
                "max_pd": "Maximum PD (%)",
                "baseline_mean_pd":
                    "Baseline Mean PD (%)",
                "absolute_pd_change":
                    "Absolute PD Change (pp)",
                "relative_pd_change_percent":
                    "Relative PD Change (%)"
            }
        )
    )

    if "Interest Rate Shock" in country_stress_display.columns:

        country_stress_display[
            "Interest Rate Shock"
        ] = (
            country_stress_display[
                "Interest Rate Shock"
            ] * 100
        ).map(
            lambda x: f"{x:.0f}%"
        )

    st.dataframe(
        country_stress_display,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        """
These stress scenarios are externally imposed sensitivity
assumptions. They are not causal estimates or forecasts
of future macroeconomic conditions.
"""
    )


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

elif section == "SHAP Explainability":

    st.header(
        "🔍 SHAP Explainability"
    )

    st.markdown(
        """
SHAP (SHapley Additive exPlanations) is used to explain
how individual model features contribute to predicted
credit default risk.
"""
    )

    # --------------------------------------------------------
    # GLOBAL FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader(
        "Global Feature Importance"
    )

    st.markdown(
        """
Mean absolute SHAP values show the overall importance
of each feature across the SHAP sample.
"""
    )

    if SHAP_GLOBAL_IMAGE.exists():

        st.image(
            str(SHAP_GLOBAL_IMAGE),
            use_container_width=True
        )

    else:

        st.warning(
            f"Image not found: {SHAP_GLOBAL_IMAGE}"
        )

    st.dataframe(
        shap_importance,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # SHAP SUMMARY PLOT
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "SHAP Summary / Beeswarm Plot"
    )

    st.markdown(
        """
The SHAP summary plot shows both feature importance
and the direction/magnitude of individual feature
contributions to predicted risk.
"""
    )

    if SHAP_SUMMARY_IMAGE.exists():

        st.image(
            str(SHAP_SUMMARY_IMAGE),
            use_container_width=True
        )

    else:

        st.warning(
            f"Image not found: {SHAP_SUMMARY_IMAGE}"
        )

    # --------------------------------------------------------
    # INDIVIDUAL LOAN
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Individual Loan Explanation"
    )

    if SHAP_INDIVIDUAL_IMAGE.exists():

        st.image(
            str(SHAP_INDIVIDUAL_IMAGE),
            use_container_width=True
        )

    else:

        st.warning(
            f"Image not found: {SHAP_INDIVIDUAL_IMAGE}"
        )

    st.markdown(
        "### Individual SHAP Contributions"
    )

    individual_display = (
        shap_individual.copy()
    )

    if "absolute_shap_value" in individual_display.columns:

        individual_display = (
            individual_display.sort_values(
                "absolute_shap_value",
                ascending=False
            )
        )

    st.dataframe(
        individual_display,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # FEATURE LIST
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Model Features"
    )

    feature_df = pd.DataFrame(
        {
            "Feature": metadata["features"]
        }
    )

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# COUNTRY RISK & FAIRNESS
# ============================================================

elif section == "Country Risk & Fairness":

    st.header(
        "🌍 Country Risk & Fairness Screening"
    )

    st.markdown(
        """
Country-level analysis compares observed default rates,
predicted PD, calibration gaps, and model discrimination
across country groups.
"""
    )

    # --------------------------------------------------------
    # COUNTRY RISK TABLE
    # --------------------------------------------------------

    st.subheader(
        "Country Risk Summary"
    )

    country_display = (
        country_risk.copy()
    )

    percentage_columns = [
        "actual_default_rate_percent",
        "mean_predicted_pd_percent",
        "pd_calibration_gap_percent"
    ]

    for column in percentage_columns:

        if column in country_display.columns:

            country_display[column] = (
                country_display[column].round(2)
            )

    country_display = country_display.rename(
        columns={
            "country_grouped":
                "Country",

            "observations":
                "Observations",

            "actual_default_rate":
                "Actual Default Rate",

            "actual_default_rate_percent":
                "Actual Default Rate (%)",

            "mean_predicted_pd":
                "Mean Predicted PD",

            "mean_predicted_pd_percent":
                "Mean Predicted PD (%)",

            "median_predicted_pd":
                "Median Predicted PD",

            "max_predicted_pd":
                "Maximum Predicted PD",

            "pd_calibration_gap":
                "Calibration Gap",

            "pd_calibration_gap_percent":
                "Calibration Gap (pp)"
        }
    )

    st.dataframe(
        country_display,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # ACTUAL VS PREDICTED PD
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Actual Default Rate vs Predicted PD"
    )

    if COUNTRY_PD_IMAGE.exists():

        st.image(
            str(COUNTRY_PD_IMAGE),
            use_container_width=True
        )

    else:

        st.warning(
            f"Image not found: {COUNTRY_PD_IMAGE}"
        )

    # --------------------------------------------------------
    # CALIBRATION GAP
    # --------------------------------------------------------

    st.subheader(
        "Country-Level PD Calibration Gap"
    )

    if COUNTRY_CALIBRATION_IMAGE.exists():

        st.image(
            str(COUNTRY_CALIBRATION_IMAGE),
            use_container_width=True
        )

    else:

        st.warning(
            f"Image not found: {COUNTRY_CALIBRATION_IMAGE}"
        )

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader(
        "Model Discrimination by Country"
    )

    if COUNTRY_ROC_IMAGE.exists():

        st.image(
            str(COUNTRY_ROC_IMAGE),
            use_container_width=True
        )

    else:

        st.warning(
            f"Image not found: {COUNTRY_ROC_IMAGE}"
        )

    # --------------------------------------------------------
    # PERFORMANCE TABLE
    # --------------------------------------------------------

    st.subheader(
        "Country-Level Model Performance"
    )

    performance_display = (
        country_performance.copy()
    )

    st.dataframe(
        performance_display,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # FAIRNESS SCREEN
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Fairness Screening"
    )

    st.info(
        """
This analysis is a fairness screening exercise rather
than a certified bias audit. The available data does not
contain protected attributes such as age or gender.

The analysis therefore focuses on differences in observed
default rates, predicted PD, calibration gaps, and
country-level model performance.
"""
    )


# ============================================================
# MODEL LIMITATIONS
# ============================================================

elif section == "Model Limitations":

    st.header(
        "⚠️ Model Limitations & Methodological Notes"
    )

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader(
        "1. Model Performance"
    )

    st.markdown(
        f"""
The final model is a tuned and calibrated Random Forest.
On the 2023 test period:

- **ROC-AUC:** {metadata['test_roc_auc']:.4f}
- **PR-AUC:** {metadata['test_pr_auc']:.4f}
- **Brier Score:** {metadata['test_brier_score']:.4f}

The test ROC-AUC indicates limited discrimination on the
future test period.
"""
    )

    # --------------------------------------------------------
    # GENERALIZATION
    # --------------------------------------------------------

    st.subheader(
        "2. Validation-to-Test Generalization"
    )

    st.warning(
        """
Validation performance was notably higher than true
2023 test performance. This indicates that the model
generalizes less reliably across future time periods
than the validation results alone might suggest.
"""
    )

    # --------------------------------------------------------
    # MACRO STRESS TESTING
    # --------------------------------------------------------

    st.subheader(
        "3. Macro Stress Testing"
    )

    st.markdown(
        """
The macro stress-testing component is a sensitivity
analysis. The borrower-level dataset does not contain
direct unemployment or inflation variables, so the
analysis does not estimate causal macroeconomic effects.

The stress scenarios are externally imposed assumptions.
"""
    )

    # --------------------------------------------------------
    # FAIRNESS
    # --------------------------------------------------------

    st.subheader(
        "4. Fairness Analysis"
    )

    st.markdown(
        """
Country-level analysis should be interpreted as a
fairness screening rather than a formal or certified
bias audit.

The analysis is limited by the available attributes
and does not include protected characteristics such
as age or gender.
"""
    )

    # --------------------------------------------------------
    # DATA LIMITATIONS
    # --------------------------------------------------------

    st.subheader(
        "5. Data and Feature Limitations"
    )

    st.markdown(
        """
The model uses eight origination-time features:

- `issued_amount`
- `initial_interest_rate`
- `initial_loan_duration`
- `loan_amount_band`
- `loan_term_bucket`
- `interest_rate_bucket`
- `country_grouped`
- `customer_risk_rating_was_missing`

`combined_income` was excluded because it was structurally
missing for the relevant training period.
"""
    )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    st.subheader(
        "6. Interpretation Guidance"
    )

    st.info(
        """
Model predictions and stress-test results should be used
as analytical decision-support information. They should
not be interpreted as guaranteed individual-level default
outcomes or causal forecasts of macroeconomic events.
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "XAI Credit Risk & Macro Stress Testing | "
    "Random Forest + SHAP + Macro Stress Testing + "
    "Country Segmentation"
)