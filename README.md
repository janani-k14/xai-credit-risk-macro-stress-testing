# XAI Credit Risk & Macro Stress Testing

## 📊 Project Overview

**XAI Credit Risk & Macro Stress Testing** is an explainable machine learning project designed to support credit-risk assessment under changing macroeconomic conditions.

The project combines:

* Machine learning-based credit-risk prediction
* Model validation and performance evaluation
* Explainable AI using SHAP
* Macroeconomic stress testing
* Unemployment and inflation stress scenarios
* Interactive Streamlit dashboard visualization

The objective is to make credit-risk predictions more **interpretable, transparent, and useful for risk analysis and decision-making**.

---

## 🎯 Project Objectives

The key objectives of this project are:

1. Develop a machine learning model for credit-risk prediction.
2. Evaluate the model using appropriate validation and test datasets.
3. Explain individual predictions using Explainable AI techniques.
4. Identify the key features influencing credit-risk predictions.
5. Evaluate how changes in macroeconomic conditions can affect credit risk.
6. Simulate unemployment and inflation stress scenarios.
7. Present the results through an interactive dashboard.

---

## 💼 Business Problem

Credit-risk models can provide accurate predictions, but their decisions are often difficult to interpret.

For financial institutions, it is important to understand:

* Why a borrower is classified as high or low risk.
* Which variables contribute most to the prediction.
* How model predictions change under adverse economic conditions.
* How macroeconomic shocks may affect overall credit risk.

This project addresses these requirements by combining **machine learning, Explainable AI, and macroeconomic stress testing** in a single analytical framework.

---

## 🔬 Project Methodology

The project follows an end-to-end workflow:

```text
Raw Data
   ↓
Data Preparation
   ↓
Train / Validation / Test Split
   ↓
Machine Learning Model
   ↓
Model Validation
   ↓
Credit-Risk Predictions
   ↓
SHAP Explainability
   ↓
Macroeconomic Stress Testing
   ↓
Unemployment & Inflation Scenarios
   ↓
Interactive Dashboard
```

---

## 📁 Dataset

The processed dataset is divided into training, validation, and testing datasets.

The repository contains:

```text
data/
├── X_train.csv
├── X_val.csv
├── X_test.csv
├── y_train.csv
├── y_val.csv
└── y_test.csv
```

### Dataset Splits

* **Training set** — used to train the machine learning model.
* **Validation set** — used during model development and validation.
* **Test set** — used for final evaluation of model performance.

---

## 🤖 Machine Learning

A machine learning classification model is used to estimate credit-risk outcomes.

The trained model and supporting metadata are maintained as part of the project workflow.

The model is evaluated using appropriate classification metrics to assess its predictive performance.

The evaluation process includes measures such as:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

The final reported values should be interpreted together rather than relying on accuracy alone, particularly when dealing with potentially imbalanced credit-risk classes.

---

## 🔎 Explainable AI

Explainable AI is an important component of this project.

The project uses **SHAP (SHapley Additive exPlanations)** to understand how individual features influence model predictions.

SHAP analysis helps answer questions such as:

* Which features have the greatest influence on credit-risk predictions?
* Does a feature increase or decrease predicted risk?
* Why did the model produce a particular prediction?
* Which variables are consistently important across observations?

This improves model transparency and makes machine learning predictions easier to interpret from a credit-risk perspective.

---

## 🌍 Macroeconomic Stress Testing

Credit risk can change when economic conditions deteriorate.

The project therefore incorporates macroeconomic stress testing to examine how credit-risk predictions may respond to changes in economic variables.

The stress-testing framework considers scenarios involving:

* **Unemployment**
* **Inflation**

The analysis evaluates how changes in these variables can influence predicted credit risk.

The generated results are stored in:

```text
macro_stress_unemployment_inflation_results.csv
macro_stress_unemployment_inflation_summary.csv
```

These results are subsequently presented through the project dashboard.

---

## 📈 Stress Scenarios

The macroeconomic stress-testing component allows different economic conditions to be examined.

Conceptually, the framework compares:

```text
Baseline Economic Conditions
            ↓
      Stress Scenario
            ↓
Change in Model Predictions
            ↓
Change in Credit-Risk Outcomes
```

This provides a way to assess the potential sensitivity of credit-risk predictions to adverse macroeconomic conditions.

---

## 🖥️ Interactive Dashboard

The project includes a **Streamlit dashboard** for presenting the analytical results.

The dashboard provides an interactive interface for exploring:

* Project overview
* Model information
* Credit-risk predictions
* Model validation
* Explainable AI results
* Macro stress-testing results
* Unemployment and inflation scenarios

The dashboard is implemented in:

```text
dashboard.py
```

---

## 🗂️ Project Structure

```text
xai-credit-risk-macro-stress-testing/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── dashboard.py
│
├── data/
│   ├── X_train.csv
│   ├── X_val.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_val.csv
│   └── y_test.csv
│
├── macro_stress_unemployment_inflation_results.csv
└── macro_stress_unemployment_inflation_summary.csv
```

Additional model and analysis artifacts may be generated during the modelling workflow.

---

## 🛠️ Technologies Used

### Programming

* Python

### Machine Learning

* Scikit-learn
* Pandas
* NumPy

### Explainable AI

* SHAP

### Data Visualization

* Matplotlib
* Plotly

### Dashboard

* Streamlit

### Development Environment

* Visual Studio Code
* Jupyter Notebook

---

## ⚙️ Installation

Clone the repository and navigate to the project directory.

Create and activate a Python virtual environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Dashboard

After installing the dependencies, run:

```bash
streamlit run dashboard.py
```

The Streamlit application will open in the browser.

---

## 📊 Results & Key Findings

### Model Performance by Country

The country-level evaluation produced the following results:

| Country     | Observations | ROC-AUC | Brier Score |
| ----------- | -----------: | ------: | ----------: |
| Estonia     |       16,599 |   0.617 |       0.081 |
| Finland     |       58,014 |   0.536 |       0.100 |
| Netherlands |        5,203 |   0.568 |       0.072 |
| Spain       |        1,578 |   0.560 |       0.247 |

Estonia has the highest ROC-AUC among the evaluated country groups, while the Netherlands has the lowest Brier score.

Spain has the highest Brier score, indicating comparatively weaker probabilistic calibration among the country groups evaluated.

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

These values measure the magnitude of feature influence on the model predictions. They do not independently indicate whether a feature increases or decreases predicted risk.

### Macroeconomic Stress Testing

The baseline mean predicted probability of default (PD) is **11.15%**.

Increasing interest-rate stress produces progressively higher predicted default probabilities:

| Scenario               | Mean PD | Relative Change |
| ---------------------- | ------: | --------------: |
| Baseline               |  11.15% |           0.00% |
| Mild Stress (+5%)      |  19.34% |         +73.49% |
| Moderate Stress (+10%) |  21.39% |         +91.92% |
| Severe Stress (+20%)   |  24.43% |        +119.17% |

Under the severe +20% stress scenario, mean predicted PD increases from **11.15% to 24.43%**, representing a **119.17% relative increase** compared with the baseline.

### Country-Level Stress Testing

The impact of interest-rate stress differs substantially across country groups.

* **Finland:** Shows the highest sensitivity, with a **152.46%** relative increase in mean PD under severe stress.
* **Spain:** Has the highest baseline mean PD at approximately **28.88%** and reaches approximately **39.17%** under severe stress.
* **Estonia:** Shows a **21.00%** relative increase under severe stress.
* **Netherlands:** Shows the lowest sensitivity, with a **1.24%** relative increase under severe stress.

These results demonstrate that macroeconomic stress does not affect all country groups uniformly.

---

## 🔐 Model Interpretability

A major objective of the project is to move beyond a traditional "black-box" prediction approach.

Instead of reporting only:

> "The model predicts high risk."

the XAI framework helps explain:

> "The model predicts high risk because specific borrower characteristics and economic variables contributed to the prediction."

This provides greater transparency for analytical and risk-management applications.

---

## ⚠️ Limitations

The project has several limitations:

* Stress-testing results depend on the assumptions used to define macroeconomic scenarios.
* Historical relationships may not necessarily remain unchanged during future economic conditions.
* Machine learning predictions are dependent on the quality and representativeness of the underlying dataset.
* SHAP explanations describe model behaviour; they should not automatically be interpreted as causal relationships.
* The model should be independently validated before being used for real-world credit decisions.

---

## 🚀 Future Enhancements

Potential future improvements include:

1. Incorporating additional macroeconomic indicators.
2. Adding more sophisticated stress scenarios.
3. Comparing multiple machine learning algorithms.
4. Implementing automated model monitoring.
5. Adding model-drift detection.
6. Expanding SHAP-based interactive analysis.
7. Adding country-level or regional risk comparisons.
8. Incorporating real-time or regularly updated macroeconomic data.
9. Improving dashboard interactivity and reporting capabilities.

---

## 📚 Key Concepts

This project integrates the following concepts:

**Credit Risk + Machine Learning + Explainable AI + SHAP + Macroeconomic Stress Testing + Data Visualization**

The combination provides an end-to-end framework for understanding not only **what the model predicts**, but also **why it predicts it and how those predictions may change under economic stress**.

---

## 👩‍💻 Project

**XAI Credit Risk & Macro Stress Testing**

Developed as an academic machine learning and data analytics project.
