import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss

print("=" * 70)
print("COUNTRY SEGMENTATION AND FAIRNESS ANALYSIS")
print("=" * 70)

# ============================================================
# 1. LOAD TEST PREDICTIONS
# ============================================================

FILE = "final_test_predictions.csv"
df = pd.read_csv(FILE)
print("\nDataset loaded successfully.")
print("Shape:", df.shape)

# ============================================================
# 2. VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = [
    "country_grouped",
    "actual_default",
    "predicted_probability_of_default",
    "predicted_class"
]
missing = [
    col for col in required_columns
    if col not in df.columns
]
if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )
print("\nRequired columns verified.")

# ============================================================
# 3. COUNTRY COUNTS
# ============================================================

country_counts = (
    df["country_grouped"]
    .value_counts(dropna=False)
)
print("\nCountry distribution:")
print(country_counts.to_string())

# ============================================================
# 4. REMOVE VERY SMALL GROUPS
# ============================================================

MIN_GROUP_SIZE = 30
valid_countries = (
    country_counts[
        country_counts >= MIN_GROUP_SIZE
    ]
    .index
)
analysis_df = df[
    df["country_grouped"].isin(valid_countries)
].copy()
print(
    f"\nCountries included in analysis "
    f"(minimum {MIN_GROUP_SIZE} observations):"
)
print(
    analysis_df["country_grouped"]
    .value_counts()
    .to_string()
)

# ============================================================
# 5. COUNTRY-LEVEL RISK METRICS
# ============================================================

country_summary = (
    analysis_df
    .groupby("country_grouped")
    .agg(
        observations=("actual_default", "size"),
        actual_default_rate=("actual_default", "mean"),
        mean_predicted_pd=(
            "predicted_probability_of_default",
            "mean"
        ),
        median_predicted_pd=(
            "predicted_probability_of_default",
            "median"
        ),
        max_predicted_pd=(
            "predicted_probability_of_default",
            "max"
        )
    )
    .reset_index()
)

# Convert rates to percentages
country_summary["actual_default_rate_percent"] = (
    country_summary["actual_default_rate"] * 100
)
country_summary["mean_predicted_pd_percent"] = (
    country_summary["mean_predicted_pd"] * 100
)

# Difference between model prediction and observed default rate
country_summary["pd_calibration_gap"] = (
    country_summary["mean_predicted_pd"]
    - country_summary["actual_default_rate"]
)
country_summary["pd_calibration_gap_percent"] = (
    country_summary["pd_calibration_gap"] * 100
)

# Sort by actual default rate
country_summary = country_summary.sort_values(
    "actual_default_rate",
    ascending=False
)
print("\n" + "=" * 70)
print("COUNTRY RISK SUMMARY")
print("=" * 70)
print(
    country_summary[
        [
            "country_grouped",
            "observations",
            "actual_default_rate_percent",
            "mean_predicted_pd_percent",
            "pd_calibration_gap_percent"
        ]
    ].to_string(index=False)
)

# ============================================================
# 6. COUNTRY-LEVEL AUC AND BRIER SCORE
# ============================================================

metrics = []
for country, group in analysis_df.groupby(
    "country_grouped"
):
    y_true = group["actual_default"]
    y_prob = group[
        "predicted_probability_of_default"
    ]
    # AUC requires both classes to be present
    if y_true.nunique() == 2:
        auc = roc_auc_score(
            y_true,
            y_prob
        )
    else:
        auc = np.nan
    brier = brier_score_loss(
        y_true,
        y_prob
    )
    metrics.append({
        "country_grouped": country,
        "observations": len(group),
        "roc_auc": auc,
        "brier_score": brier
    })
country_metrics = pd.DataFrame(metrics)
print("\n" + "=" * 70)
print("COUNTRY MODEL PERFORMANCE")
print("=" * 70)
print(
    country_metrics.to_string(
        index=False
    )
)

# ============================================================
# 7. FAIRNESS RANGE
# ============================================================

max_actual_rate = (
    country_summary["actual_default_rate"].max()
)
min_actual_rate = (
    country_summary["actual_default_rate"].min()
)
max_predicted_pd = (
    country_summary["mean_predicted_pd"].max()
)
min_predicted_pd = (
    country_summary["mean_predicted_pd"].min()
)
actual_rate_range = (
    max_actual_rate - min_actual_rate
)
predicted_pd_range = (
    max_predicted_pd - min_predicted_pd
)
print("\n" + "=" * 70)
print("COUNTRY FAIRNESS SCREEN")
print("=" * 70)
print(
    f"\nActual default-rate range: "
    f"{actual_rate_range * 100:.2f} percentage points"
)
print(
    f"Predicted PD range: "
    f"{predicted_pd_range * 100:.2f} percentage points"
)

# ============================================================
# 8. SAVE RESULTS
# ============================================================

country_summary.to_csv(
    "country_risk_summary.csv",
    index=False
)
country_metrics.to_csv(
    "country_model_performance.csv",
    index=False
)
print("\nSaved:")
print("1. country_risk_summary.csv")
print("2. country_model_performance.csv")

# ============================================================
# 9. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("COUNTRY SEGMENTATION COMPLETED")
print("=" * 70)