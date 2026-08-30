# XAI Credit Risk & Macro Stress Testing

## Explainable Credit-Risk Modelling, Segmentation & Macroeconomic Stress Testing

---

## 📌 Project Overview

**XAI Credit Risk & Macro Stress Testing** is an end-to-end machine learning and Explainable AI project designed to assess credit risk, explain model predictions, analyse risk across country groups, and evaluate how predicted Probability of Default (PD) changes under adverse economic scenarios.

The project combines:

- Machine learning-based credit-risk prediction
- Model validation and evaluation
- Probability calibration
- Explainable AI using SHAP / TreeSHAP
- Feature importance analysis
- Country-level risk segmentation
- Country-level fairness screening
- Macroeconomic stress testing
- Interest-rate sensitivity analysis
- Interactive Streamlit dashboard
- Model limitations and governance considerations

The objective is to move beyond simply predicting credit risk and answer three important questions:

> **What is the predicted credit risk?**

> **Why does the model produce that prediction?**

> **How could predicted credit risk change under adverse economic conditions?**

---

# 🎯 Project Objectives

The main objectives of the project are:

1. Prepare and clean a real-world loan-level credit-risk dataset.
2. Prevent data leakage using a predefined blacklist of post-outcome variables.
3. Create a time-based training, validation, and test framework.
4. Compare multiple machine learning classification models.
5. Select and tune the best-performing model.
6. Calibrate the model's predicted probabilities.
7. Evaluate final model performance on a future-period test set.
8. Explain model predictions using SHAP / TreeSHAP.
9. Identify the most influential model features.
10. Analyse credit risk across country groups.
11. Perform a country-level fairness screening.
12. Conduct macroeconomic scenario-based stress testing.
13. Analyse interest-rate sensitivity.
14. Present the results through an interactive Streamlit dashboard.
15. Clearly document model limitations and interpretation constraints.

---

# 💼 Business Problem

Credit-risk models can estimate the probability that a borrower will default, but predictions alone are often difficult to interpret.

For financial institutions and risk-management teams, it is important to understand:

- Why a borrower is classified as higher or lower risk.
- Which variables contribute most to the prediction.
- Whether the model behaves differently across country groups.
- Whether predicted probabilities are appropriately calibrated.
- How predicted portfolio risk changes under adverse economic scenarios.
- Whether model performance remains stable when applied to future data.

This project addresses these requirements by combining:

**Credit Risk + Machine Learning + Explainable AI + Risk Segmentation + Fairness Screening + Macroeconomic Stress Testing**

---

# 🌍 Project Context

The project uses real loan-level data from **Bondora**, a European peer-to-peer lending platform.

The modelling framework focuses on **Probability of Default (PD)**.

The project does not build a complete Expected Credit Loss framework because Loss Given Default (LGD) and Exposure at Default (EAD) are not modelled.

The overall analytical framework is:

    Raw Loan Data
          ↓
    Data Preparation
          ↓
    Leakage Prevention
          ↓
    Feature Engineering
          ↓
    Time-Based Data Split
          ↓
    Model Comparison
          ↓
    Model Tuning
          ↓
    Probability Calibration
          ↓
    Final Test Evaluation
          ↓
       ┌──┴──┐
       ↓     ↓
      XAI   Segmentation
       ↓     ↓
      SHAP  Country Analysis
       │     Fairness Screen
       └──┬──┘
          ↓
    Macro Stress Testing
          ↓
    Interest-Rate Sensitivity
          ↓
    Interactive Dashboard

---

# 📊 Dataset

## Data Source

The project uses real loan-level data from the **Bondora** peer-to-peer lending platform.

The raw dataset contained:

- **737,889 loan records**
- **32 attributes**

The modelling population was restricted to loans issued between:

**1 January 2018 and 31 December 2023**

A hard maturity cutoff was applied to ensure that loans had sufficient time to reach a known 12-month outcome.

After filtering and preparation, the final modelling population contained:

- **306,470 loans**
- **16.64% overall default rate**

---

# 🧹 Data Preparation

The data preparation process included multiple quality and leakage-control steps.

## 1. Study Window

Loans were restricted to the period:

**2018–2023**

A hard 12-month maturity cutoff was applied to reduce survivorship bias.

---

## 2. Duplicate Checking

Loan identifiers were checked for duplicate records.

---

## 3. Placeholder Checking

Numeric fields were checked for placeholder values such as:

- `-1`
- `0`
- Other "not set" codes

---

## 4. Data Type and Range Validation

Data types were checked and numerical variables were examined for invalid or suspicious values.

---

## 5. Missing Values

### Customer Risk Rating

The original `customer_risk_rating` variable had a very high proportion of missing values.

Instead of directly using the categorical variable, the project retained a binary indicator:

    customer_risk_rating_was_missing

The original customer risk rating field was excluded.

### Combined Income

`combined_income` was investigated because of its high missingness.

Importantly, it was found to be:

**100% missing for loans issued during 2018–2021.**

This was treated as a structural data gap rather than random missingness.

Therefore:

- `combined_income` was excluded from the model.
- Derived income ratios were also excluded.
- Artificial imputation was avoided.

This prevented the model from being trained on fabricated income values for most of the training period.

---

# 🔐 Data Leakage Prevention

Preventing data leakage was a key design principle.

A **22-column blacklist** was enforced programmatically.

Examples of excluded post-outcome variables include:

    is_default
    days_past_due_principal
    loan_status

These variables contain information that is only available after the loan outcome and therefore cannot be used as origination-time predictors.

The feature-preparation process was designed to prevent blacklisted variables from entering the modelling feature set.

---

# 🔎 Exploratory Data Analysis

Exploratory analysis was performed to understand:

- Default-rate patterns
- Country-level differences
- Missing values
- Feature distributions
- Outliers
- Temporal changes
- Potential leakage variables

The analysis identified substantial differences in default behaviour across years and country groups.

---

# 📅 Default Rate by Year

The observed default rates changed substantially across the study period.

| Year | Loans | Default Rate |
|------|------:|-------------:|
| 2018 | 25,359 | 26.84% |
| 2019 | 56,506 | 29.55% |
| 2020 | 27,874 | 17.28% |
| 2021 | 51,742 | 10.30% |
| 2022 | 63,594 | 13.37% |
| 2023 | 81,395 | 10.87% |

The variation over time supports the use of a time-based validation framework.

---

# 🌍 Country-Level Risk

The dataset also showed substantial differences in observed default rates across countries.

| Country | Number of Loans | Default Rate |
|---------|----------------:|-------------:|
| Finland | 142,336 | 15.27% |
| Estonia | 138,040 | 13.52% |
| Spain | 20,645 | 48.97% |
| Netherlands | 5,448 | 8.76% |

Spain showed a substantially higher observed default rate than the other major country groups.

Because of this, country was examined as:

- A predictive feature
- A segmentation variable
- A robustness dimension
- A fairness-screening dimension

---

# 🛠️ Feature Engineering

The final model uses **8 origination-time features**.

    issued_amount
    initial_interest_rate
    initial_loan_duration
    loan_amount_band
    loan_term_bucket
    interest_rate_bucket
    country_grouped
    customer_risk_rating_was_missing

## Continuous Features

- `issued_amount`
- `initial_interest_rate`
- `initial_loan_duration`

## Bucketed Features

- `loan_amount_band`
- `loan_term_bucket`
- `interest_rate_bucket`

Quantile-based binning was used for loan amount and interest rate.

Loan duration was transformed into fixed-range buckets.

## Country Grouping

The `country_grouped` variable retains meaningful country groups.

Countries with fewer than 100 observations were grouped into:

    Other

Spain was retained separately because of its substantially higher observed default rate.

---

# ⏳ Train / Validation / Test Split

A **time-based split** was used instead of a random split.

| Dataset | Period | Observations |
|---------|--------|-------------:|
| Training | 2018–2021 | 161,481 |
| Validation | 2022 | 63,594 |
| Test | 2023 | 81,395 |

This creates a realistic temporal modelling structure:

    Historical Data
          ↓
       Training
          ↓
    Future Validation
          ↓
       Future Test

The 2023 test set was reserved for final reporting and was not used to make further modelling decisions.

---

# 🤖 Machine Learning Models

Three classification algorithms were considered:

1. Logistic Regression
2. Random Forest
3. XGBoost

Logistic Regression provided an interpretable baseline.

Random Forest and XGBoost were used to evaluate nonlinear and ensemble-based approaches.

---

# ⚖️ Class Imbalance

The default class represented a minority of observations.

Class imbalance was handled through model-specific weighting.

### Logistic Regression

    class_weight = "balanced"

### Random Forest

    class_weight = "balanced"

### XGBoost

A calculated `scale_pos_weight` was used.

---

# 🏆 Model Comparison

The initial validation results were:

| Model | ROC-AUC | PR-AUC |
|-------|--------:|-------:|
| Logistic Regression | 0.6600 | 0.2504 |
| Random Forest | **0.6753** | **0.2655** |
| XGBoost | 0.6673 | 0.2479 |

Random Forest achieved the strongest validation performance on both:

- ROC-AUC
- PR-AUC

It was therefore selected for further tuning.

---

# ⚙️ Hyperparameter Tuning

The Random Forest model was tuned using:

    RandomizedSearchCV

The tuning process used:

- 20 randomly sampled parameter combinations
- 3-fold cross-validation
- 60 total model fits
- PR-AUC as the optimization metric

Parameters considered included:

- Number of estimators
- Maximum tree depth
- Minimum samples per split
- Minimum samples per leaf
- Maximum features

---

# 🎯 Final Model

The final model is a:

**Tuned and Calibrated Random Forest**

The finalized model is stored in:

    FINAL_model.pkl

The model metadata is maintained in:

    model_metadata.json

The model should be loaded directly rather than retrained when performing the downstream XAI and stress-testing analysis.

Example:

    import joblib
    import pandas as pd

    model = joblib.load("FINAL_model.pkl")

    X_val = pd.read_csv("X_val.csv")
    y_val = pd.read_csv("y_val.csv").squeeze()

    probs = model.predict_proba(X_val)[:, 1]

---

# 📐 Probability Calibration

The raw Random Forest probabilities were calibrated because the model systematically over-predicted default risk across probability deciles.

Isotonic calibration was applied using:

    CalibratedClassifierCV

The Brier Score improved from:

    0.1407 → 0.1079

Calibration improves the reliability of predicted probabilities.

This is different from discrimination:

- **Discrimination** measures how well the model ranks higher-risk versus lower-risk observations.
- **Calibration** measures how closely predicted probabilities correspond to observed outcomes.

---

# 📊 Final Model Performance

The final model was evaluated on the future 2023 test set.

| Metric | Validation 2022 | Test 2023 |
|--------|----------------:|----------:|
| ROC-AUC | 0.678 | 0.564 |
| PR-AUC | 0.270 | 0.152 |
| Brier Score | 0.108 | 0.097 |

---

# ⚠️ Important Model Finding

A major finding of the project is the decline in performance between the validation period and the genuine future test period.

    Validation ROC-AUC = 0.678
    Test ROC-AUC       = 0.564

This indicates that the model does not generalize as strongly to the 2023 future period as the validation results alone might suggest.

This is **not a calibration bug**.

Calibration rescales predicted probabilities but does not change the ranking of observations.

Therefore, the reduction in ROC-AUC represents a genuine reduction in predictive discrimination on future data.

This limitation must be considered when interpreting:

- Model predictions
- SHAP explanations
- Country segmentation
- Stress-testing results
- # 🔎 Explainable AI

Explainable AI is a central component of the project.

The project uses:

**SHAP — SHapley Additive exPlanations**

Because the final model is a Random Forest, **TreeSHAP** is used for tree-based explanations.

SHAP analysis provides two major perspectives.

## Global Explainability

Global SHAP analysis identifies which features have the greatest influence on model predictions across the portfolio.

## Local Explainability

Local SHAP analysis explains why a particular observation received a particular predicted Probability of Default.

The SHAP framework helps answer:

- Which features influence model predictions the most?
- Does a feature contribute toward higher or lower predicted risk?
- Why did the model produce a particular prediction?
- Which variables are consistently influential?

SHAP values describe model behaviour and should not automatically be interpreted as causal relationships.

---

# 🧠 SHAP Feature Importance

The final model contains eight features.

The mean absolute SHAP analysis produced the following ranking:

| Rank | Feature | Mean Absolute SHAP |
|-----:|---------|-------------------:|
| 1 | Country | 0.02075 |
| 2 | Initial interest rate | 0.01894 |
| 3 | Interest-rate bucket | 0.01642 |
| 4 | Initial loan duration | 0.01283 |
| 5 | Loan amount band | 0.01205 |
| 6 | Issued amount | 0.01160 |
| 7 | Loan-term bucket | 0.00771 |
| 8 | Missing customer risk rating indicator | 0.00093 |

Country was the most influential feature according to mean absolute SHAP value.

It was followed by:

1. Initial interest rate
2. Interest-rate bucket
3. Initial loan duration
4. Loan amount band
5. Issued amount
6. Loan-term bucket
7. Missing customer risk rating indicator

Mean absolute SHAP values measure the magnitude of model influence.

They do **not** independently indicate whether a feature increases or decreases risk.

---

# 🌍 Risk Segmentation

Country-level segmentation was performed using:

    country_grouped

The segmentation analysis compares:

- Observed default rate
- Mean predicted PD
- Predicted risk distribution
- Model performance
- Calibration behaviour
- Stress sensitivity

Country was specifically investigated because the EDA showed substantial differences in observed default rates.

---

# ⚖️ Fairness Screening

A country-level fairness screening was conducted.

The analysis examines:

- Group-wise observed default rate
- Group-wise predicted PD
- Calibration across country groups

This analysis is explicitly a:

> **Fairness screen, not a certified fairness audit.**

The dataset does not contain protected attributes such as:

- Age
- Gender

Therefore, differences in predicted risk between country groups should not automatically be interpreted as algorithmic bias.

Observed differences may reflect genuine differences in the underlying portfolio.

---

# 🧪 Country-Dominance Robustness Check

Because country was identified as an important model feature, an additional robustness experiment was performed.

The Random Forest was evaluated:

1. With `country_grouped`
2. Without `country_grouped`

| Metric | With Country | Without Country | Change |
|--------|-------------:|----------------:|-------:|
| ROC-AUC | 0.6753 | 0.6615 | -0.0138 |
| PR-AUC | 0.2655 | 0.2351 | -0.0304 |

Removing country resulted in:

- Approximately 2.0% relative decline in ROC-AUC
- Approximately 11.5% relative decline in PR-AUC

This indicates that country contains predictive information, while other features still retain substantial predictive value.

Country was therefore retained in the final model.

---

# 🌍 Macroeconomic Stress Testing

The project adds a scenario-based macroeconomic stress-testing layer on top of the finalized PD model.

An important methodological distinction is:

> **This is sensitivity analysis, not econometric estimation.**

The borrower-level dataset does not contain borrower-level unemployment or inflation variables.

Therefore, the project does not statistically estimate relationships such as:

    Unemployment → Default
    Inflation → Default

Instead, externally imposed sensitivity assumptions are used.

The final calibrated model is **not retrained** during stress testing.

---

# 📉 Unemployment & Inflation Stress Testing

Three macroeconomic scenarios were evaluated.

| Scenario | Unemployment | Inflation |
|----------|-------------:|----------:|
| Baseline | 0 pp | 0 pp |
| Moderate Stress | +1 pp | +2 pp |
| Severe Stress | +3 pp | +4 pp |

The analysis uses an odds-based adjustment.

### Sensitivity Assumptions

A:

    +1 percentage-point unemployment

increase corresponds to a:

    10% increase in default odds

A:

    +1 percentage-point inflation

increase corresponds to a:

    5% increase in default odds

The resulting combined odds multipliers are:

| Scenario | Odds Multiplier |
|----------|----------------:|
| Baseline | 1.00 |
| Moderate Stress | 1.21 |
| Severe Stress | 1.56 |

These are **scenario assumptions** and are not coefficients estimated from the borrower-level dataset.

---

# 📊 Unemployment & Inflation Stress Results

The locked 2023 test population was used as the baseline reference population.

| Scenario | Mean Predicted PD |
|----------|------------------:|
| Baseline | **11.1454%** |
| Moderate Stress | **13.0717%** |
| Severe Stress | **16.0540%** |

## Moderate Stress

Mean predicted PD increases from:

    11.1454% → 13.0717%

Absolute change:

    +1.9263 percentage points

Relative change:

    +17.2834%

## Severe Stress

Mean predicted PD increases from:

    11.1454% → 16.0540%

Absolute change:

    +4.9085 percentage points

Relative change:

    +44.0409%

---

# ✅ Stress-Test Sanity Checks

The stress-testing framework includes directional sanity checks.

### Check 1

    Moderate Stress > Baseline

**Result: PASS**

### Check 2

    Severe Stress > Baseline

**Result: PASS**

### Check 3

    Severe Stress > Moderate Stress

**Result: PASS**

Therefore:

    Baseline < Moderate Stress < Severe Stress

The stress scenarios produced the expected monotonic increase in predicted risk.

---

# 📈 Interest-Rate Sensitivity Analysis

A separate interest-rate sensitivity analysis was also performed.

This analysis examines how the finalized PD model responds to increasing interest-rate stress.

The model was not retrained.

The 2023 test population containing:

**81,395 observations**

was used as the baseline reference population.

Four scenarios were evaluated:

| Scenario | Interest-Rate Shock |
|----------|--------------------:|
| Baseline | 0% |
| Mild Stress | +5% |
| Moderate Stress | +10% |
| Severe Stress | +20% |

---

# 📊 Interest-Rate Stress Results

| Scenario | Mean Predicted PD | Relative Change |
|----------|------------------:|----------------:|
| Baseline | 11.15% | 0.00% |
| Mild Stress (+5%) | 19.34% | +73.49% |
| Moderate Stress (+10%) | 21.39% | +91.92% |
| Severe Stress (+20%) | 24.43% | +119.17% |

Under the severe +20% interest-rate stress scenario:

    Mean PD
    11.15% → 24.43%

Absolute increase:

    +13.28 percentage points

Relative increase:

    +119.17%

---

# 🌍 Country-Level Interest-Rate Stress Sensitivity

The impact of interest-rate stress differs substantially across country groups.

Key findings include:

- **Finland** shows the highest sensitivity, with a **152.46% relative increase** in mean PD under severe stress.
- **Spain** has the highest baseline mean PD at approximately **28.88%** and reaches approximately **39.17%** under severe stress.
- **Estonia** shows a **21.00% relative increase** under severe stress.
- **Netherlands** shows the lowest sensitivity, with a **1.24% relative increase** under severe stress.

These results indicate that the sensitivity of predicted credit risk to stress is not uniform across country groups.

---

# ⚠️ Stress-Test Interpretation

The stress-testing results should not be interpreted as:

- Economic forecasts
- Causal estimates
- Actual future default rates
- Econometrically estimated macroeconomic relationships

Instead, they represent:

> **Conditional sensitivity of the finalized credit-risk model under externally imposed adverse scenarios.**

The results depend on:

1. The assumptions used to define the stress scenarios.
2. The behaviour of the underlying PD model.

Therefore, limitations in the underlying PD model also affect the interpretation of the stress-testing results.

---

# 🖥️ Interactive Streamlit Dashboard

The project includes an interactive Streamlit dashboard that brings together the model, explainability, segmentation, fairness screening, and stress-testing analyses.

The dashboard provides sections for:

## 📌 Project Overview

Provides an overview of the project, objectives, dataset, and analytical framework.

## 🤖 Model Information

Displays information about the finalized model and model configuration.

## 📊 Model Validation

Presents model performance metrics and validation results.

## 🎯 Credit-Risk Predictions

Displays predicted Probability of Default and related risk information.

## 🔎 Explainable AI

Provides SHAP-based:

- Feature importance
- Global explanations
- Individual prediction explanations

## 🌍 Macro Stress Testing

Displays how predicted PD changes under different stress scenarios.

## 🗺️ Country Risk Analysis

Compares risk and model behaviour across country groups.

## ⚖️ Fairness Screening

Provides country-level comparisons of observed and predicted risk.

## ⚠️ Model Limitations

Communicates the important limitations of the model and stress-testing framework.

---

# ▶️ Running the Dashboard

Create a Python virtual environment:

    python -m venv venv

Activate it on Windows:

    venv\Scripts\activate

Install the required dependencies:

    pip install -r requirements.txt

Run the dashboard:

    streamlit run dashboard.py

The Streamlit application will open in the browser.

---

# 🗂️ Project Structure

    xai-credit-risk-macro-stress-testing/
    │
    ├── .gitignore
    ├── README.md
    ├── requirements.txt
    ├── dashboard.py
    ├── FINAL_model.pkl
    ├── model_metadata.json
    │
    ├── X_train.csv
    ├── X_val.csv
    ├── X_test.csv
    ├── y_train.csv
    ├── y_val.csv
    ├── y_test.csv
    │
    ├── final_test_predictions.csv
    ├── macro_stress_results.csv
    ├── macro_stress_summary.csv
    ├── macro_stress_unemployment_inflation_results.csv
    ├── macro_stress_unemployment_inflation_summary.csv
    │
    ├── XAI.ipynb
    ├── macro_stress_testing.ipynb
    │
    └── src/
        └── analysis and modelling scripts

Additional analysis outputs and visualizations may be generated during the modelling workflow.

---

# 🧰 Technologies Used

## Programming

- Python
- Jupyter Notebook
- Visual Studio Code

## Data Processing

- Pandas
- NumPy

## Machine Learning

- Scikit-learn
- XGBoost

## Explainable AI

- SHAP
- TreeSHAP

## Visualization

- Matplotlib
- Plotly

## Dashboard

- Streamlit

## Model Serialization

- Joblib

## Version Control

- Git
- GitHub

---

# 🔬 End-to-End Methodology

The complete methodology is:

    Bondora Loan Dataset
            ↓
    Data Quality Checks
            ↓
    Study Window & Maturity Cutoff
            ↓
    Leakage Prevention
            ↓
    Missing-Value Analysis
            ↓
    Outlier Treatment
            ↓
    Exploratory Data Analysis
            ↓
    Feature Engineering
            ↓
    Time-Based Train / Validation / Test Split
            ↓
    Model Comparison
            ↓
    Random Forest Selection
            ↓
    Hyperparameter Tuning
            ↓
    Probability Calibration
            ↓
    Final 2023 Test Evaluation
            ↓
    ┌───────────────────────────────┐
    │                               │
    ↓                               ↓
    SHAP Explainability        Risk Segmentation
    │                               │
    ↓                               ↓
    Feature Importance         Country Analysis
    Local Explanations         Fairness Screening
    │                               │
    └───────────────┬───────────────┘
                    ↓
            Macro Stress Testing
                    ↓
          ┌─────────┴─────────┐
          ↓                   ↓
    Unemployment &        Interest-Rate
    Inflation Stress      Sensitivity
          ↓                   ↓
          └─────────┬─────────┘
                    ↓
           Streamlit Dashboard

---

# 🔐 Model Governance & Responsible AI

The project incorporates several governance and responsible-AI practices.

## Data Leakage Prevention

A 22-column blacklist prevents post-outcome variables from entering the model.

## Temporal Validation

Time-based train, validation, and test periods better reflect future deployment conditions.

## Probability Calibration

Isotonic calibration improves the reliability of predicted probabilities.

## Country Robustness

The effect of country information was tested by comparing models with and without the country feature.

## Explainability

SHAP provides global and local explanations of model behaviour.

## Fairness Screening

Country-level differences in observed and predicted risk were examined.

## Honest Test Evaluation

The 2023 test set was reserved for final evaluation.

## Transparent Limitations

The decline between validation and future test performance is explicitly reported.

---

# ⚠️ Limitations

## 1. Temporal Generalization

The validation ROC-AUC was:

    0.678

while the 2023 test ROC-AUC was:

    0.564

This indicates weaker performance when the model is applied to a future period.

---

## 2. Limited Macroeconomic Variables

The borrower-level dataset does not contain unemployment or inflation variables.

Therefore, the macroeconomic component is scenario-based sensitivity analysis rather than econometric estimation.

---

## 3. Externally Imposed Stress Assumptions

The unemployment and inflation sensitivity parameters are assumptions.

They are not coefficients statistically estimated from the borrower-level dataset.

---

## 4. Interest-Rate Stress Assumptions

The interest-rate stress scenarios are also externally imposed sensitivity assumptions.

They should not be interpreted as causal estimates of the actual effect of interest rates on default.

---

## 5. Country Concentration

Country is an important predictor and observed default rates differ substantially between country groups.

Although a country-dominance robustness test was performed, geographic concentration remains an important consideration.

---

## 6. Fairness Scope

The fairness analysis is a screening exercise rather than a formal fairness audit.

Protected demographic attributes such as age and gender are not available.

---

## 7. PD-Only Framework

The project focuses on Probability of Default.

It does not estimate:

- Loss Given Default (LGD)
- Exposure at Default (EAD)
- Full Expected Credit Loss (ECL)

---

## 8. No Causal Interpretation

SHAP explanations describe model behaviour.

They do not establish causality.

Similarly, stress-testing results describe model sensitivity under specified assumptions rather than actual economic causality.

---

# 🚀 Future Enhancements

Potential future improvements include:

1. Extend the framework to a full Expected Credit Loss model.
2. Add LGD and EAD estimation when appropriate data becomes available.
3. Incorporate additional macroeconomic indicators.
4. Use borrower-level or country-time-level macroeconomic variables.
5. Replace externally imposed stress assumptions with empirically estimated relationships.
6. Implement rolling-origin or multi-period validation.
7. Compare additional machine learning algorithms.
8. Add automated model monitoring.
9. Implement model-drift detection.
10. Expand SHAP-based interactive analysis.
11. Expand fairness analysis if protected demographic attributes become available.
12. Add formal demographic fairness metrics.
13. Add additional country and regional segmentation.
14. Introduce more sophisticated stress scenarios.
15. Improve dashboard interactivity and reporting.
16. Incorporate regularly updated macroeconomic data.
17. Develop portfolio-level expected-loss analysis under stress scenarios.

---

# 📚 Key Concepts

The project integrates:

    Credit Risk
          +
    Probability of Default
          +
    Machine Learning
          +
    Random Forest
          +
    Probability Calibration
          +
    Explainable AI
          +
    SHAP / TreeSHAP
          +
    Risk Segmentation
          +
    Fairness Screening
          +
    Macroeconomic Stress Testing
          +
    Interest-Rate Sensitivity
          +
    Model Governance
          +
    Interactive Dashboard

The framework therefore aims to understand not only:

> **What the model predicts**

but also:

> **Why the model predicts it**

and:

> **How those predictions may change under adverse scenarios.**

---

# 📌 Key Results at a Glance

| Component | Result |
|-----------|--------|
| Raw Dataset | 737,889 loans |
| Final Modelling Population | 306,470 loans |
| Overall Default Rate | 16.64% |
| Training Period | 2018–2021 |
| Validation Period | 2022 |
| Test Period | 2023 |
| Training Observations | 161,481 |
| Validation Observations | 63,594 |
| Test Observations | 81,395 |
| Final Model | Tuned & Calibrated Random Forest |
| Validation ROC-AUC | 0.678 |
| Test ROC-AUC | 0.564 |
| Test PR-AUC | 0.152 |
| Test Brier Score | 0.097 |
| Calibration Brier Score | Improved from 0.1407 to 0.1079 |
| Most Influential SHAP Feature | Country |
| Baseline Mean PD | 11.1454% |
| Moderate Macro Stress PD | 13.0717% |
| Severe Macro Stress PD | 16.0540% |
| Severe Macro Stress Increase | +4.9085 pp / +44.04% |
| Severe Interest-Rate Stress PD | 24.43% |
| Country-Dominance Check | Passed |
| Stress Monotonicity Checks | Passed |

---

# 📊 Key Findings

## Model Performance

The tuned Random Forest performed best during validation among the three compared models.

However, performance decreased when evaluated on the future 2023 test period.

This highlights the importance of genuine out-of-time evaluation.

---

## Calibration

Probability calibration improved the Brier Score from:

    0.1407 → 0.1079

This indicates improved reliability of predicted PD values.

---

## Explainability

Country was the most influential feature according to mean absolute SHAP value.

Initial interest rate and interest-rate bucket were the next most influential features.

---

## Country Risk

Substantial differences were observed across country groups.

Spain showed a particularly high observed default rate.

---

## Macro Stress

Under the moderate unemployment/inflation stress scenario:

    Baseline PD = 11.1454%
    Moderate Stress PD = 13.0717%

Under severe stress:

    Baseline PD = 11.1454%
    Severe Stress PD = 16.0540%

---

## Interest-Rate Sensitivity

Under severe +20% interest-rate stress:

    Baseline PD = 11.15%
    Severe Stress PD = 24.43%

This represents an approximately:

    +119.17% relative increase

in mean predicted PD.

---

# ⚠️ Important Interpretation

The project should not be presented as a highly accurate production credit-scoring system.

The key model-risk finding is:

    Validation performance ≠ guaranteed future performance

The 2023 test result demonstrates a meaningful reduction in predictive discrimination compared with the validation period.

Similarly:

    Stress-test sensitivity ≠ economic forecast

The stress scenarios show how the finalized model's predictions respond to specified assumptions.

They do not estimate actual future default rates or causal macroeconomic relationships.

Therefore, stress-testing results should always be interpreted together with the underlying model's limitations.

---

# 📖 References

1. Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *The Journal of Finance*, 23(4), 589–609.

2. Basel Committee on Banking Supervision. (2017). *Supervisory and bank stress testing: range of practices*. Bank for International Settlements.

3. Bondora AS. *Public Loan Dataset*.

4. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.

5. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

6. IFRS Foundation. (2014). *IFRS 9 Financial Instruments*. International Accounting Standards Board.

7. Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. *European Journal of Operational Research*, 247(1), 124–136.

8. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

9. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

---

# 👥 Project Team

**XAI Credit Risk & Macro Stress Testing**

Academic Machine Learning and Data Analytics Project

### Team Members

- **Janani K**
- **Vishnu Priya**
- **Rubina**

---

# 🔗 Repository

GitHub Repository:

https://github.com/janani-k14/xai-credit-risk-macro-stress-testing

---

# ⭐ Final Takeaway

This project demonstrates an end-to-end explainable credit-risk framework:

    Leakage-Free Data Preparation
                ↓
    Machine Learning
                ↓
    Probability Calibration
                ↓
    Future-Period Evaluation
                ↓
    SHAP Explainability
                ↓
    Country Risk Segmentation
                ↓
    Fairness Screening
                ↓
    Macroeconomic Stress Testing
                ↓
    Interest-Rate Sensitivity Analysis
                ↓
    Interactive Dashboard

The project prioritizes:

- Methodological rigor
- Data leakage prevention
- Temporal validation
- Model explainability
- Probability calibration
- Risk segmentation
- Responsible AI
- Transparent stress-test assumptions
- Honest reporting of model limitations

The central conclusion is that a useful credit-risk framework should not only answer:

> **"What does the model predict?"**

but also:

> **"Why does it predict this?"**

and:

> **"How sensitive is the predicted risk when economic conditions deteriorate?"**
> 
