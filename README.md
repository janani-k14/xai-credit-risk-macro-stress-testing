# XAI Credit Risk — Macro Stress-Testing Project

**Risk Model Handoff — JANANI **

This repo contains the finalized Probability of Default (PD) model and everything needed to build on it. Read this file fully before using any other file here.

---

## What's in This Repo

| File | What It Is |
|---|---|
| `FINAL_model.pkl` | The trained, tuned, and calibrated model. **Use this one — not an earlier version.** |
| `model_metadata.json` | Summary of the model: periods used, final metrics, feature list, calibration status |
| `requirements.txt` | Exact Python package versions used to build this model — install these to avoid version mismatches |
| `X_train.csv` / `y_train.csv` | Training data (2018–2021) — features and target, already split apart |
| `X_val.csv` / `y_val.csv` | Validation data (2022) |
| `X_test.csv` / `y_test.csv` | Test data (2023) — **do not use this to make further modeling decisions.** It was touched once, for final reporting only. |

---

## How to Load and Use the Model

```python
import joblib
import pandas as pd

# Load the model
model = joblib.load("FINAL_model.pkl")

# Load whichever data split you need
X_val = pd.read_csv("X_val.csv")
y_val = pd.read_csv("y_val.csv").squeeze()

# Get predicted probability of default for each loan
probs = model.predict_proba(X_val)[:, 1]
```

**Before running anything, install the exact package versions this model was built with:**
```bash
pip install -r requirements.txt
```

---

## Final Model Results (from `model_metadata.json`)

- **Model type:** Random Forest (tuned, calibrated)
- **Train period:** 2018–2021 | **Validation:** 2022 | **Test:** 2023
- **Test set (2023) ROC-AUC:** 0.564
- **Test set (2023) PR-AUC:** 0.152
- **Test set (2023) Brier Score:** 0.0969

## Important: Read This Before Using the Model's Output

**Validation performance (0.678 ROC-AUC) was notably higher than true test performance (0.564 ROC-AUC).** This is not a calibration bug — calibration only rescales probabilities, it cannot change ranking. This is genuine evidence that the model generalizes less reliably across future time periods than a single validation year suggested.

**What this means for your work specifically:**
- Any conclusions drawn from this model's predictions should note that discriminative power is moderate, not strong, particularly on 2023-like future data
- This should be stated as a limitation in both the stress-testing and segmentation write-ups, not treated as a fully solved, highly accurate model

---

## Feature List (8 features, all origination-time only — no post-loan behavior)

```
issued_amount, initial_interest_rate, initial_loan_duration,
loan_amount_band, loan_term_bucket, interest_rate_bucket,
country_grouped, customer_risk_rating_was_missing
```

**Note:** `combined_income` was investigated and deliberately excluded — it was 100% missing for loans issued 2018–2021 (a structural data gap, not random missingness). Imputing it would have meant training on fabricated values for most of the dataset.

---

## Vishnu Priya — Macro Stress-Testing

1. Load `FINAL_model.pkl` as shown above
2. Do **not** retrain it — take the model as-is and re-score it by feeding in different macroeconomic-adjusted feature values under your Normal / Mild / Severe scenarios
3. Use `X_test.csv` and `y_test.csv` as your baseline reference population, since that's the most realistic "current" snapshot (2023)
4. Sanity-check your output: severe stress should always show a higher predicted default rate than baseline. If it doesn't, stop and flag it before proceeding
5. Keep in mind the validation-to-test performance gap noted above when interpreting how much confidence to place in scenario-level shifts

## Rubina — Segmentation, Explainability, Fairness

1. Load `FINAL_model.pkl` — this is the exact model to run **TreeSHAP** on (it's a Random Forest, so TreeSHAP applies directly)
2. Segment using `country_grouped` and any income-band logic you're building separately — remember `combined_income` was dropped from the model itself, so segmentation by income (if done) should be treated as a secondary analysis, not based on a model feature
3. For the fairness screen: group-wise default rate, group-wise predicted PD, and calibration by country — scoped as a fairness **screen**, not a certified bias audit, since we don't have protected attributes like age or gender
4. Reference `model_metadata.json` for exact feature names when building your segment breakdowns

---

## Data Preparation Summary (Full Detail in `DATA_PREPARATION_REPORT.md`)

- Source: real Bondora loan data (European P2P lending platform)
- Scoped to loans issued 2018–2023 with a hard 12-month maturity cutoff (avoids survivorship bias)
- 306,470 loans, 16.64% overall default rate
- Full cleaning: duplicates, placeholder codes, text consistency, data types, range checks — all verified
- Leakage prevention: 22-column blacklist enforced in code (e.g. `is_default`, `days_past_due_principal`, `loan_status` — all excluded since they're only known after the loan's outcome)

---

 
