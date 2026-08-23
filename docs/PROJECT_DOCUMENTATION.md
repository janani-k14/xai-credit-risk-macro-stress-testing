
# XAI Credit Risk & Macro Stress Testing — Project Documentation

## 1. Introduction

This project develops an explainable machine learning framework for credit-risk assessment and macroeconomic stress testing.

The framework combines machine learning predictions with Explainable AI (XAI) and macroeconomic scenario analysis to understand both credit-risk predictions and their sensitivity to changing economic conditions.

---

## 2. Project Objectives

* Develop a machine learning model for credit-risk prediction.
* Validate the predictive performance of the model.
* Explain model predictions using SHAP.
* Identify important factors influencing credit-risk predictions.
* Perform macroeconomic stress testing.
* Analyze the effect of unemployment and inflation scenarios.
* Present the results through an interactive Streamlit dashboard.

---

## 3. Project Workflow

The overall workflow is:

**Data → Preprocessing → Model Training → Validation → Prediction → SHAP Analysis → Stress Testing → Dashboard**

### 3.1 Data Preparation

The dataset is divided into training, validation, and testing datasets.

```text
data/
├── X_train.csv
├── X_val.csv
├── X_test.csv
├── y_train.csv
├── y_val.csv
└── y_test.csv
```

### 3.2 Model Development

The training dataset is used to develop the credit-risk classification model.

The validation dataset is used during model evaluation and development, while the test dataset is reserved for final performance assessment.

### 3.3 Model Validation

The model is evaluated using classification metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

The final metric values are reported based on the actual model evaluation results.

---

## 4. Explainable AI

Explainable AI is used to understand how the machine learning model arrives at its predictions.

SHAP (SHapley Additive exPlanations) is used to identify the contribution of individual features to model predictions.

The analysis focuses on:

* Global feature importance
* Direction of feature influence
* Individual prediction explanations

SHAP explanations describe the behaviour of the trained model and should not automatically be interpreted as causal relationships.

---

## 5. Macroeconomic Stress Testing

The project evaluates the sensitivity of credit-risk predictions to changes in macroeconomic conditions.

The stress-testing analysis focuses on:

* Unemployment
* Inflation

The baseline scenario is compared with stressed economic conditions to examine changes in predicted credit-risk outcomes.

The generated stress-testing outputs include:

```text
macro_stress_unemployment_inflation_results.csv
macro_stress_unemployment_inflation_summary.csv
```

---

## 5.1 Key Results

### Overall Macro Stress Testing

The baseline mean predicted probability of default (PD) is **11.15%**.

As the interest-rate shock increases, the mean predicted PD also increases:

| Scenario               | Mean PD | Relative Change |
| ---------------------- | ------: | --------------: |
| Baseline               |  11.15% |           0.00% |
| Mild Stress (+5%)      |  19.34% |         +73.49% |
| Moderate Stress (+10%) |  21.39% |         +91.92% |
| Severe Stress (+20%)   |  24.43% |        +119.17% |

Under the severe +20% interest-rate stress scenario, mean predicted PD increases from **11.15% to 24.43%**, representing a **119.17% relative increase** compared with the baseline.

### Country-Level Stress Testing

The country-level analysis shows different sensitivities to macroeconomic stress.

* **Finland:** The most sensitive country group. Under severe stress, mean PD increases by **152.46%** relative to its baseline.
* **Spain:** Has the highest baseline mean PD at approximately **28.88%** and reaches approximately **39.17%** under severe stress.
* **Estonia:** Shows a **21.00%** relative increase in mean PD under severe stress.
* **Netherlands:** Shows the lowest sensitivity, with a **1.24%** relative increase under severe stress.

These results demonstrate that the impact of macroeconomic stress is not uniform across country groups.

### Explainable AI Findings

SHAP analysis identifies the following features as the most influential based on mean absolute SHAP value:

| Rank | Feature                                | Mean Absolute SHAP |
| ---: | -------------------------------------- | -----------------: |
|    1 | Country                                |            0.02075 |
|    2 | Initial interest rate                  |            0.01894 |
|    3 | Interest-rate bucket                   |            0.01642 |
|    4 | Initial loan duration                  |            0.01283 |
|    5 | Loan amount band                       |            0.01205 |
|    6 | Issued amount                          |            0.01160 |
|    7 | Loan-term bucket                       |            0.00771 |
|    8 | Missing customer risk rating indicator |            0.00093 |

Country is the most influential feature according to mean absolute SHAP value, followed by initial interest rate and interest-rate bucket.

These values indicate the magnitude of feature contribution to model predictions. They do not, by themselves, indicate whether a feature increases or decreases predicted risk.

## 6. Dashboard

The project includes an interactive Streamlit dashboard implemented using:

```text
dashboard.py
```

The dashboard provides access to the major analytical components of the project, including:

* Project overview
* Model information
* Model validation
* Credit-risk predictions
* Explainable AI analysis
* Macro stress testing
* Unemployment and inflation scenarios

---

## 7. Project Structure

```text
xai-credit-risk-macro-stress-testing/
│
├── README.md
├── requirements.txt
├── dashboard.py
│
├── data/
│   └── Dataset files
│
├── docs/
│   └── PROJECT_DOCUMENTATION.md
│
├── figures/
│   └── Analysis figures
│
├── model/
│   └── Model artifacts
│
├── notebooks/
│   └── Analysis notebooks
│
├── outputs/
│   └── Generated outputs
│
└── src/
    └── Source code
```

The `venv/` directory is a local Python virtual environment and should not be committed to the GitHub repository.

---

## 8. Technologies

The project uses:

* Python
* Pandas
* NumPy
* Scikit-learn
* SHAP
* Matplotlib
* Plotly
* Streamlit
* Jupyter Notebook

---

## 9. Running the Project

Create and activate a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run dashboard.py
```

---

## 10. Limitations

The results depend on the quality and representativeness of the available data.

Macroeconomic stress-testing results also depend on the assumptions used to define the stress scenarios.

SHAP explains model behaviour but does not establish causal relationships between variables and credit risk.

The model should therefore be independently validated before being used for real-world credit decisions.

---

## 11. Future Enhancements

Potential future enhancements include:

* Additional macroeconomic indicators.
* More extensive stress scenarios.
* Comparison of multiple machine learning algorithms.
* Model monitoring and drift detection.
* Automated updating of macroeconomic data.
* Enhanced dashboard interactivity.
* Additional regional or country-level risk analysis.

---

## 12. Conclusion

The project demonstrates an integrated approach to credit-risk modelling by combining machine learning, Explainable AI, and macroeconomic stress testing.

The framework aims to provide a more transparent understanding of credit-risk predictions while examining how those predictions may change under adverse economic conditions.
