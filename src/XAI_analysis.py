import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

print("Libraries loaded successfully.")


# ============================================================
# 1. FILE PATHS
# ============================================================

MODEL_PATH = "FINAL_model.pkl"
X_TEST_PATH = "X_test.csv"
Y_TEST_PATH = "y_test.csv"


# ============================================================
# 2. EXPECTED FEATURES
# ============================================================

EXPECTED_FEATURES = [
    "issued_amount",
    "initial_interest_rate",
    "initial_loan_duration",
    "loan_amount_band",
    "loan_term_bucket",
    "interest_rate_bucket",
    "country_grouped",
    "customer_risk_rating_was_missing"
]


# ============================================================
# 3. LOAD MODEL AND TEST DATA
# ============================================================

print("\nLoading model and test data...")

final_model = joblib.load(MODEL_PATH)

X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).squeeze()

print("Final model:", type(final_model))
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("\nTest features:")
print(X_test.columns.tolist())


# ============================================================
# 4. FEATURE VERIFICATION
# ============================================================

missing_features = [
    feature
    for feature in EXPECTED_FEATURES
    if feature not in X_test.columns
]

if missing_features:
    raise ValueError(
        f"Missing expected features: {missing_features}"
    )

print("\nFeature verification: PASSED")


# Make sure the columns are in exactly the expected order.
X_test = X_test[EXPECTED_FEATURES]


# ============================================================
# 5. GENERATE PREDICTED PROBABILITY OF DEFAULT
# ============================================================

print("\nGenerating predictions...")

test_pd = final_model.predict_proba(X_test)[:, 1]

results = X_test.copy()

results["actual_default"] = y_test.values
results["predicted_pd"] = test_pd

print("\nPredictions generated successfully.")

print(
    results[
        ["actual_default", "predicted_pd"]
    ].head()
)

print("\nPredicted PD summary:")

print(
    results["predicted_pd"].describe()
)


# ============================================================
# 6. START SHAP ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("STARTING SHAP ANALYSIS")
print("=" * 60)


# We use a representative sample rather than
# all 81,395 test observations.

SHAP_SAMPLE_SIZE = 200

if len(X_test) > SHAP_SAMPLE_SIZE:

    X_shap = X_test.sample(
        n=SHAP_SAMPLE_SIZE,
        random_state=42
    )

else:

    X_shap = X_test.copy()


# Make absolutely sure this is a DataFrame.
X_shap = pd.DataFrame(
    X_shap,
    columns=EXPECTED_FEATURES
)

print(
    "SHAP sample shape:",
    X_shap.shape
)


# ============================================================
# 7. SHAP PREDICTION FUNCTION
# ============================================================

def predict_pd(X):

    """
    Prediction function used by SHAP.

    SHAP's KernelExplainer converts DataFrames into
    NumPy arrays internally.

    The saved sklearn pipeline uses a ColumnTransformer
    that requires pandas column names.

    Therefore, convert NumPy arrays back to DataFrames.
    """

    # If SHAP gives us a NumPy array,
    # convert it back to a DataFrame.
    if not isinstance(X, pd.DataFrame):

        X = pd.DataFrame(
            X,
            columns=EXPECTED_FEATURES
        )

    # Make sure the column order is correct.
    X = X[EXPECTED_FEATURES]

    # Return probability of default.
    return final_model.predict_proba(X)[:, 1]


# ============================================================
# 8. TEST SHAP PREDICTION FUNCTION
# ============================================================

print("\nTesting SHAP prediction function...")

test_prediction = predict_pd(
    X_shap.head(5)
)

print(
    "Prediction function test PASSED."
)

print(
    "Sample predictions:",
    test_prediction
)


# ============================================================
# 9. CREATE BACKGROUND DATA
# ============================================================

print("\nCreating SHAP background data...")

BACKGROUND_SIZE = 50

background_data = shap.sample(
    X_shap,
    BACKGROUND_SIZE,
    random_state=42
)

background_data = pd.DataFrame(
    background_data,
    columns=EXPECTED_FEATURES
)

print(
    "Background data shape:",
    background_data.shape
)


# ============================================================
# 10. CREATE KERNEL SHAP EXPLAINER
# ============================================================

print(
    "\nCreating SHAP KernelExplainer..."
)

explainer = shap.KernelExplainer(
    predict_pd,
    background_data
)

print(
    "SHAP KernelExplainer created successfully."
)


# ============================================================
# 11. CALCULATE SHAP VALUES
# ============================================================

print(
    "\nCalculating SHAP values..."
)

print(
    "This may take some time..."
)

shap_values = explainer.shap_values(
    X_shap,
    nsamples=100
)


# ============================================================
# 12. FORMAT SHAP OUTPUT
# ============================================================

if isinstance(shap_values, list):

    shap_values_array = np.array(
        shap_values[0]
    )

else:

    shap_values_array = np.array(
        shap_values
    )


print(
    "\nSHAP values calculated successfully."
)

print(
    "SHAP values shape:",
    shap_values_array.shape
)


# ============================================================
# 13. CHECK SHAP SHAPE
# ============================================================

if shap_values_array.shape != (
    len(X_shap),
    len(EXPECTED_FEATURES)
):

    raise ValueError(
        "Unexpected SHAP value shape: "
        f"{shap_values_array.shape}"
    )

print(
    "SHAP shape verification: PASSED"
)


# ============================================================
# 14. GLOBAL FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("GLOBAL FEATURE IMPORTANCE")
print("=" * 60)


mean_abs_shap = (
    np.abs(shap_values_array)
    .mean(axis=0)
)


feature_importance = pd.DataFrame({

    "feature": EXPECTED_FEATURES,

    "mean_abs_shap": mean_abs_shap

})


feature_importance = (
    feature_importance
    .sort_values(
        by="mean_abs_shap",
        ascending=False
    )
)


print(
    "\nFeature importance:"
)

print(
    feature_importance.to_string(
        index=False
    )
)


# ============================================================
# 15. SAVE GLOBAL FEATURE IMPORTANCE
# ============================================================

feature_importance.to_csv(
    "SHAP_feature_importance.csv",
    index=False
)

print(
    "\nSaved:"
)

print(
    "SHAP_feature_importance.csv"
)


# ============================================================
# 16. GLOBAL SHAP BAR PLOT
# ============================================================

print(
    "\nCreating global SHAP bar plot..."
)

plt.figure()

shap.summary_plot(
    shap_values_array,
    X_shap,
    plot_type="bar",
    show=False
)

plt.title(
    "Global Feature Importance - SHAP"
)

plt.tight_layout()

plt.savefig(
    "SHAP_global_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved:"
)

print(
    "SHAP_global_feature_importance.png"
)


# ============================================================
# 17. SHAP SUMMARY / BEESWARM PLOT
# ============================================================

print(
    "\nCreating SHAP summary plot..."
)

shap.summary_plot(
    shap_values_array,
    X_shap,
    show=False
)

plt.title(
    "SHAP Summary Plot - Credit Default Risk"
)

plt.tight_layout()

plt.savefig(
    "SHAP_summary_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved:"
)

print(
    "SHAP_summary_plot.png"
)


# ============================================================
# 18. INDIVIDUAL LOAN EXPLANATION
# ============================================================

print("\n" + "=" * 60)
print("INDIVIDUAL LOAN EXPLANATION")
print("=" * 60)


# Explain the first observation in the SHAP sample.

observation_index = 0


individual_features = X_shap.iloc[
    observation_index
]


individual_shap = (
    shap_values_array[
        observation_index
    ]
)


individual_pd = predict_pd(
    individual_features.to_frame().T
)[0]


print(
    "\nLoan features:"
)

for feature, value in individual_features.items():

    print(
        f"{feature}: {value}"
    )


print(
    f"\nPredicted PD: "
    f"{individual_pd:.4f}"
)


print(
    f"Predicted PD percentage: "
    f"{individual_pd * 100:.2f}%"
)


# ============================================================
# 19. INDIVIDUAL SHAP CONTRIBUTIONS
# ============================================================

print(
    "\nSHAP contribution:"
)


individual_explanation = pd.DataFrame({

    "feature": EXPECTED_FEATURES,

    "feature_value":
        individual_features.values,

    "shap_value":
        individual_shap

})


individual_explanation[
    "absolute_shap_value"
] = np.abs(
    individual_explanation[
        "shap_value"
    ]
)


individual_explanation = (
    individual_explanation
    .sort_values(
        by="absolute_shap_value",
        ascending=False
    )
)


print(
    individual_explanation[
        [
            "feature",
            "feature_value",
            "shap_value"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 20. SAVE INDIVIDUAL EXPLANATION
# ============================================================

individual_explanation.to_csv(
    "SHAP_individual_explanation.csv",
    index=False
)

print(
    "\nSaved:"
)

print(
    "SHAP_individual_explanation.csv"
)


# ============================================================
# 21. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("XAI ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    "\nGenerated files:"
)

print(
    "1. SHAP_feature_importance.csv"
)

print(
    "2. SHAP_global_feature_importance.png"
)

print(
    "3. SHAP_summary_plot.png"
)

print(
    "4. SHAP_individual_explanation.csv"
)

print(
    "\nAll XAI analysis steps completed."
)