import pandas as pd
import numpy as np
import joblib

print("=" * 70)
print("COUNTRY-LEVEL MACRO STRESS ANALYSIS")
print("=" * 70)


# ============================================================
# 1. FILE PATHS
# ============================================================

MODEL_PATH = "FINAL_model.pkl"
X_TEST_PATH = "X_test.csv"


# ============================================================
# 2. LOAD MODEL AND DATA
# ============================================================

print("\nLoading model and test data...")

final_model = joblib.load(MODEL_PATH)
X_test = pd.read_csv(X_TEST_PATH)

print("X_test shape:", X_test.shape)


# ============================================================
# 3. VERIFY REQUIRED FEATURES
# ============================================================

required_features = [
    "issued_amount",
    "initial_interest_rate",
    "initial_loan_duration",
    "loan_amount_band",
    "loan_term_bucket",
    "interest_rate_bucket",
    "country_grouped",
    "customer_risk_rating_was_missing"
]

missing = [
    feature for feature in required_features
    if feature not in X_test.columns
]

if missing:
    raise ValueError(f"Missing features: {missing}")

X_test = X_test[required_features]


# ============================================================
# 4. COUNTRIES
# ============================================================

country_counts = X_test["country_grouped"].value_counts()

MIN_GROUP_SIZE = 30

valid_countries = country_counts[
    country_counts >= MIN_GROUP_SIZE
].index.tolist()

print("\nCountries included:")
print(valid_countries)

analysis_df = X_test[
    X_test["country_grouped"].isin(valid_countries)
].copy()


# ============================================================
# 5. STRESS SCENARIOS
# ============================================================

STRESS_SCENARIOS = {
    "Baseline": 0.00,
    "Mild Stress (+5%)": 0.05,
    "Moderate Stress (+10%)": 0.10,
    "Severe Stress (+20%)": 0.20
}


# ============================================================
# 6. COUNTRY-LEVEL STRESS TESTING
# ============================================================

results = []


for scenario, shock in STRESS_SCENARIOS.items():

    print(f"\nRunning: {scenario}")

    X_stressed = analysis_df.copy()

    # Increase initial interest rate
    X_stressed["initial_interest_rate"] = (
        X_stressed["initial_interest_rate"] * (1 + shock)
    )

    # Recalculate interest-rate bucket
    def assign_interest_bucket(rate):

        if rate < 0.20:
            return "low"

        elif rate < 0.30:
            return "medium"

        else:
            return "high"

    X_stressed["interest_rate_bucket"] = (
        X_stressed["initial_interest_rate"]
        .apply(assign_interest_bucket)
    )

    # Predict probability of default
    stressed_pd = final_model.predict_proba(
        X_stressed[required_features]
    )[:, 1]

    X_stressed["predicted_pd"] = stressed_pd

    # Country-level results
    country_result = (
        X_stressed
        .groupby("country_grouped")["predicted_pd"]
        .agg(
            observations="count",
            mean_pd="mean",
            median_pd="median",
            max_pd="max"
        )
        .reset_index()
    )

    country_result["scenario"] = scenario
    country_result["interest_rate_shock"] = shock

    results.append(country_result)


# ============================================================
# 7. COMBINE RESULTS
# ============================================================

stress_df = pd.concat(
    results,
    ignore_index=True
)

stress_df = stress_df[
    [
        "country_grouped",
        "scenario",
        "interest_rate_shock",
        "observations",
        "mean_pd",
        "median_pd",
        "max_pd"
    ]
]


# ============================================================
# 8. CALCULATE CHANGE FROM BASELINE
# ============================================================

baseline = (
    stress_df[
        stress_df["scenario"] == "Baseline"
    ][
        ["country_grouped", "mean_pd"]
    ]
    .rename(
        columns={"mean_pd": "baseline_mean_pd"}
    )
)

stress_df = stress_df.merge(
    baseline,
    on="country_grouped",
    how="left"
)

stress_df["absolute_pd_change"] = (
    stress_df["mean_pd"]
    - stress_df["baseline_mean_pd"]
)

stress_df["relative_pd_change_percent"] = np.where(
    stress_df["baseline_mean_pd"] != 0,
    (
        stress_df["absolute_pd_change"]
        / stress_df["baseline_mean_pd"]
    ) * 100,
    np.nan
)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("COUNTRY-LEVEL MACRO STRESS RESULTS")
print("=" * 70)

display_columns = [
    "country_grouped",
    "scenario",
    "mean_pd",
    "absolute_pd_change",
    "relative_pd_change_percent"
]

print(
    stress_df[
        display_columns
    ].to_string(index=False)
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

stress_df.to_csv(
    "country_macro_stress_results.csv",
    index=False
)

print(
    "\nSaved: country_macro_stress_results.csv"
)


# ============================================================
# 11. SUMMARY
# ============================================================

summary = (
    stress_df[
        stress_df["scenario"] != "Baseline"
    ]
    .sort_values(
        "relative_pd_change_percent",
        ascending=False
    )
)

summary.to_csv(
    "country_macro_stress_summary.csv",
    index=False
)

print(
    "Saved: country_macro_stress_summary.csv"
)


print("\n" + "=" * 70)
print("COUNTRY MACRO STRESS ANALYSIS COMPLETED")
print("=" * 70)