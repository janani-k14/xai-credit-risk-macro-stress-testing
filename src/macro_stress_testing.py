import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt


# ============================================================
# MACRO STRESS TESTING
# XAI CREDIT RISK & MACRO STRESS TESTING PROJECT
# ============================================================

print("Libraries loaded successfully.")


# ============================================================
# 1. FILE PATHS
# ============================================================

MODEL_PATH = "FINAL_model.pkl"

X_TRAIN_PATH = "X_train.csv"
X_VAL_PATH = "X_val.csv"
X_TEST_PATH = "X_test.csv"

SUMMARY_OUTPUT = "macro_stress_summary.csv"
RESULTS_OUTPUT = "macro_stress_results.csv"

PD_PLOT_OUTPUT = "macro_stress_pd.png"
RELATIVE_CHANGE_PLOT = "macro_stress_relative_change.png"


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
# 3. INTEREST-RATE BUCKETS
# ============================================================

BUCKET_ORDER = [
    "very_low",
    "low",
    "medium",
    "high",
    "very_high"
]


# ============================================================
# 4. LOAD MODEL AND DATA
# ============================================================

print("\nLoading model and datasets...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

if not os.path.exists(X_TRAIN_PATH):
    raise FileNotFoundError(
        f"Training data not found: {X_TRAIN_PATH}"
    )

if not os.path.exists(X_VAL_PATH):
    raise FileNotFoundError(
        f"Validation data not found: {X_VAL_PATH}"
    )

if not os.path.exists(X_TEST_PATH):
    raise FileNotFoundError(
        f"Test data not found: {X_TEST_PATH}"
    )


final_model = joblib.load(MODEL_PATH)

X_train = pd.read_csv(X_TRAIN_PATH)
X_val = pd.read_csv(X_VAL_PATH)
X_test = pd.read_csv(X_TEST_PATH)


print("Final model:", type(final_model))
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)


# ============================================================
# 5. FEATURE VERIFICATION
# ============================================================

missing_features = [
    feature
    for feature in EXPECTED_FEATURES
    if feature not in X_test.columns
]

if missing_features:
    raise ValueError(
        f"Missing required features in X_test: {missing_features}"
    )


# Keep exactly the features used by the final model
X_test = X_test[EXPECTED_FEATURES].copy()

print("\nFeature verification: PASSED")


# ============================================================
# 6. VERIFY INTEREST-RATE BUCKET VALUES
# ============================================================

print("\nChecking original interest-rate buckets...")

existing_buckets = (
    X_test["interest_rate_bucket"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

unexpected_buckets = [
    bucket
    for bucket in existing_buckets
    if bucket not in BUCKET_ORDER
]

if unexpected_buckets:
    raise ValueError(
        f"Unexpected interest-rate buckets found: "
        f"{unexpected_buckets}"
    )


print("\nOriginal bucket distribution in X_test:")

original_distribution = (
    X_test["interest_rate_bucket"]
    .value_counts()
    .reindex(BUCKET_ORDER, fill_value=0)
)

print(original_distribution)


# ============================================================
# 7. INFER BUCKET BOUNDARIES FROM ORIGINAL LABELS
# ============================================================
#
# IMPORTANT:
#
# We DO NOT use pd.qcut again.
#
# The original model was already trained using the five
# categorical interest-rate buckets.
#
# The CSV files contain the original bucket assigned to each
# observation. Therefore we infer the boundaries from the
# actual rate/bucket relationship.
#
# This avoids changing the original baseline bucket assignments.
# ============================================================

print("\nInferring original interest-rate bucket boundaries...")

all_data = pd.concat(
    [
        X_train,
        X_val,
        X_test
    ],
    ignore_index=True
)

all_data = all_data[
    [
        "initial_interest_rate",
        "interest_rate_bucket"
    ]
].copy()

all_data = all_data.dropna(
    subset=[
        "initial_interest_rate",
        "interest_rate_bucket"
    ]
)

all_data["interest_rate_bucket"] = (
    all_data["interest_rate_bucket"]
    .astype(str)
)


# Check bucket validity
all_buckets = sorted(
    all_data["interest_rate_bucket"].unique()
)

unexpected_all = [
    bucket
    for bucket in all_buckets
    if bucket not in BUCKET_ORDER
]

if unexpected_all:
    raise ValueError(
        f"Unexpected bucket values found: {unexpected_all}"
    )


# ============================================================
# 8. FIND RATE RANGE FOR EACH ORIGINAL BUCKET
# ============================================================

bucket_ranges = {}

for bucket in BUCKET_ORDER:

    bucket_rates = all_data.loc[
        all_data["interest_rate_bucket"] == bucket,
        "initial_interest_rate"
    ]

    if len(bucket_rates) == 0:
        raise ValueError(
            f"No observations found for bucket: {bucket}"
        )

    bucket_ranges[bucket] = {
        "min": bucket_rates.min(),
        "max": bucket_rates.max()
    }


print("\nOriginal bucket rate ranges:")

for bucket in BUCKET_ORDER:

    print(
        f"{bucket:10s} : "
        f"{bucket_ranges[bucket]['min']:.6f}"
        f" -> "
        f"{bucket_ranges[bucket]['max']:.6f}"
    )


# ============================================================
# 9. CONSTRUCT BUCKET BOUNDARIES
# ============================================================
#
# We create boundaries between adjacent observed buckets.
#
# Example:
#
# very_low max = A
# low min      = B
#
# boundary = midpoint(A, B)
#
# This preserves the ordering of the original five categories.
# ============================================================

bucket_boundaries = []

for i in range(len(BUCKET_ORDER) - 1):

    current_bucket = BUCKET_ORDER[i]
    next_bucket = BUCKET_ORDER[i + 1]

    current_max = bucket_ranges[current_bucket]["max"]
    next_min = bucket_ranges[next_bucket]["min"]

    boundary = (current_max + next_min) / 2

    bucket_boundaries.append(boundary)


print("\nInferred bucket boundaries:")

for i, boundary in enumerate(bucket_boundaries):

    print(
        f"Between {BUCKET_ORDER[i]} "
        f"and {BUCKET_ORDER[i + 1]}: "
        f"{boundary:.6f}"
    )


# ============================================================
# 10. BUCKET ASSIGNMENT FUNCTION
# ============================================================

def assign_interest_bucket(rate):

    if pd.isna(rate):
        return "medium"

    if rate <= bucket_boundaries[0]:
        return "very_low"

    elif rate <= bucket_boundaries[1]:
        return "low"

    elif rate <= bucket_boundaries[2]:
        return "medium"

    elif rate <= bucket_boundaries[3]:
        return "high"

    else:
        return "very_high"


# ============================================================
# 11. TEST THE RECONSTRUCTION
# ============================================================

print("\nTesting bucket reconstruction...")

reconstructed_test_buckets = (
    X_test["initial_interest_rate"]
    .apply(assign_interest_bucket)
)

reconstruction_comparison = pd.DataFrame({
    "original": X_test["interest_rate_bucket"].astype(str),
    "reconstructed": reconstructed_test_buckets.astype(str)
})

reconstruction_accuracy = (
    reconstruction_comparison["original"]
    ==
    reconstruction_comparison["reconstructed"]
).mean()


print(
    f"Bucket reconstruction agreement: "
    f"{reconstruction_accuracy * 100:.2f}%"
)


# ============================================================
# 12. BASELINE PREDICTIONS
# ============================================================
#
# VERY IMPORTANT:
#
# The baseline MUST use X_test exactly as stored.
#
# We do NOT recalculate the bucket.
# We do NOT modify the interest rate.
#
# This guarantees that baseline PD represents the actual
# model prediction on the original test dataset.
# ============================================================

print("\nCalculating baseline PD...")

baseline_pd = final_model.predict_proba(
    X_test
)[:, 1]


baseline_mean_pd = float(
    np.mean(baseline_pd)
)

baseline_median_pd = float(
    np.median(baseline_pd)
)

baseline_max_pd = float(
    np.max(baseline_pd)
)


print("\nBaseline PD calculated.")

print(
    f"Baseline mean PD: "
    f"{baseline_mean_pd:.6f}"
)

print(
    f"Baseline mean PD (%): "
    f"{baseline_mean_pd * 100:.2f}%"
)

print(
    f"Baseline median PD: "
    f"{baseline_median_pd:.6f}"
)

print(
    f"Baseline maximum PD: "
    f"{baseline_max_pd:.6f}"
)


# ============================================================
# 13. STRESS SCENARIOS
# ============================================================

STRESS_SCENARIOS = {

    "Baseline": 0.00,

    "Mild Stress (+5%)": 0.05,

    "Moderate Stress (+10%)": 0.10,

    "Severe Stress (+20%)": 0.20

}


# ============================================================
# 14. RUN MACRO STRESS TESTING
# ============================================================

stress_results = []

print("\n" + "=" * 60)
print("MACRO STRESS TESTING")
print("=" * 60)


for scenario, shock in STRESS_SCENARIOS.items():

    print(f"\nScenario: {scenario}")


    # ========================================================
    # BASELINE
    # ========================================================
    #
    # Do NOT modify X_test.
    #
    # This is important because the original X_test already
    # contains the bucket assignments used during training.
    # ========================================================

    if shock == 0.00:

        stressed_pd = baseline_pd.copy()

        scenario_bucket_distribution = (
            X_test["interest_rate_bucket"]
            .value_counts()
            .reindex(
                BUCKET_ORDER,
                fill_value=0
            )
        )

    else:

        # ====================================================
        # COPY ORIGINAL TEST DATA
        # ====================================================

        X_stressed = X_test.copy()


        # ====================================================
        # APPLY INTEREST-RATE SHOCK
        # ====================================================

        X_stressed["initial_interest_rate"] = (
            X_test["initial_interest_rate"]
            * (1 + shock)
        )


        # ====================================================
        # RECREATE INTEREST-RATE BUCKET
        # ====================================================

        X_stressed["interest_rate_bucket"] = (
            X_stressed[
                "initial_interest_rate"
            ]
            .apply(assign_interest_bucket)
        )


        # ====================================================
        # VERIFY BUCKET VALUES
        # ====================================================

        invalid_stress_buckets = (
            set(
                X_stressed[
                    "interest_rate_bucket"
                ].dropna().astype(str)
            )
            -
            set(BUCKET_ORDER)
        )

        if invalid_stress_buckets:

            raise ValueError(
                "Invalid stress-test buckets: "
                f"{invalid_stress_buckets}"
            )


        # ====================================================
        # PRINT BUCKET DISTRIBUTION
        # ====================================================

        scenario_bucket_distribution = (
            X_stressed[
                "interest_rate_bucket"
            ]
            .value_counts()
            .reindex(
                BUCKET_ORDER,
                fill_value=0
            )
        )


        # ====================================================
        # PREDICT STRESSED PD
        # ====================================================

        stressed_pd = final_model.predict_proba(
            X_stressed[EXPECTED_FEATURES]
        )[:, 1]


    # ========================================================
    # PRINT BUCKET DISTRIBUTION
    # ========================================================

    print("\nInterest-rate bucket distribution:")

    print(
        scenario_bucket_distribution
    )


    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    mean_pd = float(
        np.mean(stressed_pd)
    )

    median_pd = float(
        np.median(stressed_pd)
    )

    max_pd = float(
        np.max(stressed_pd)
    )


    # ========================================================
    # CHANGE FROM ORIGINAL BASELINE
    # ========================================================

    absolute_pd_change = (
        mean_pd -
        baseline_mean_pd
    )


    if baseline_mean_pd != 0:

        relative_pd_change = (
            absolute_pd_change /
            baseline_mean_pd
        ) * 100

    else:

        relative_pd_change = 0.0


    # ========================================================
    # REMOVE FLOATING-POINT NOISE
    # ========================================================

    if abs(absolute_pd_change) < 1e-12:

        absolute_pd_change = 0.0

    if abs(relative_pd_change) < 1e-10:

        relative_pd_change = 0.0


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        f"Mean PD: "
        f"{mean_pd:.6f}"
    )

    print(
        f"Mean PD (%): "
        f"{mean_pd * 100:.2f}%"
    )

    print(
        f"Change from baseline: "
        f"{absolute_pd_change * 100:.2f} "
        f"percentage points"
    )

    print(
        f"Relative change: "
        f"{relative_pd_change:.2f}%"
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    stress_results.append({

        "scenario": scenario,

        "interest_rate_shock": shock,

        "mean_pd": round(
            mean_pd,
            6
        ),

        "median_pd": round(
            median_pd,
            6
        ),

        "max_pd": round(
            max_pd,
            6
        ),

        "absolute_pd_change": round(
            absolute_pd_change,
            6
        ),

        "relative_pd_change_percent": round(
            relative_pd_change,
            4
        )

    })


# ============================================================
# 15. CREATE SUMMARY DATAFRAME
# ============================================================

stress_summary = pd.DataFrame(
    stress_results
)


# ============================================================
# 16. SAVE CSV FILES
# ============================================================

stress_summary.to_csv(
    SUMMARY_OUTPUT,
    index=False
)


# ============================================================
# 17. CREATE DETAILED RESULTS
# ============================================================

detailed_results = []

for scenario, shock in STRESS_SCENARIOS.items():

    if shock == 0.00:

        scenario_data = X_test.copy()

        scenario_pd = baseline_pd.copy()

    else:

        scenario_data = X_test.copy()

        scenario_data[
            "initial_interest_rate"
        ] = (
            X_test[
                "initial_interest_rate"
            ]
            * (1 + shock)
        )

        scenario_data[
            "interest_rate_bucket"
        ] = (
            scenario_data[
                "initial_interest_rate"
            ]
            .apply(assign_interest_bucket)
        )

        scenario_pd = final_model.predict_proba(
            scenario_data[EXPECTED_FEATURES]
        )[:, 1]


    detailed_df = pd.DataFrame({

        "scenario": scenario,

        "interest_rate_shock": shock,

        "initial_interest_rate": (
            scenario_data[
                "initial_interest_rate"
            ].values
        ),

        "interest_rate_bucket": (
            scenario_data[
                "interest_rate_bucket"
            ].values
        ),

        "predicted_pd": scenario_pd

    })


    detailed_results.append(
        detailed_df
    )


# Combine all scenarios
macro_results = pd.concat(
    detailed_results,
    ignore_index=True
)


# Save detailed results
macro_results.to_csv(
    RESULTS_OUTPUT,
    index=False
)


# ============================================================
# 18. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STRESS TEST SUMMARY")
print("=" * 60)

print(
    stress_summary.to_string(
        index=False
    )
)


# ============================================================
# 19. CREATE PD STRESS PLOT
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    stress_summary["scenario"],
    stress_summary["mean_pd"] * 100,
    marker="o"
)

plt.xlabel(
    "Stress Scenario"
)

plt.ylabel(
    "Mean Probability of Default (%)"
)

plt.title(
    "Macro Stress Testing - Mean Probability of Default"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    PD_PLOT_OUTPUT,
    dpi=300
)

plt.close()


# ============================================================
# 20. CREATE RELATIVE CHANGE PLOT
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    stress_summary["scenario"],
    stress_summary[
        "relative_pd_change_percent"
    ],
    marker="o"
)

plt.xlabel(
    "Stress Scenario"
)

plt.ylabel(
    "Relative PD Change (%)"
)

plt.title(
    "Macro Stress Testing - Relative Change in PD"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    RELATIVE_CHANGE_PLOT,
    dpi=300
)

plt.close()


# ============================================================
# 21. FINAL OUTPUT
# ============================================================

print(
    f"\nSaved: {SUMMARY_OUTPUT}"
)

print(
    f"Saved: {RESULTS_OUTPUT}"
)

print(
    f"Saved: {PD_PLOT_OUTPUT}"
)

print(
    f"Saved: {RELATIVE_CHANGE_PLOT}"
)


print("\n" + "=" * 60)
print("MACRO STRESS TESTING COMPLETED")
print("=" * 60)

print("\nGenerated files:")

print(
    "1. macro_stress_summary.csv"
)

print(
    "2. macro_stress_results.csv"
)

print(
    "3. macro_stress_pd.png"
)

print(
    "4. macro_stress_relative_change.png"
)

print(
    "\nOriginal five interest-rate categories preserved:"
)

print(
    "very_low, low, medium, high, very_high"
)

print(
    "\nBaseline PD is calculated using the ORIGINAL X_test "
    "without modifying its interest-rate bucket."
)

print(
    "\nAll macro stress testing steps completed successfully."
)