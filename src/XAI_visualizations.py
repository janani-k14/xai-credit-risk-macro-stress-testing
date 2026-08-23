import pandas as pd
import matplotlib.pyplot as plt


print("Loading XAI results...")


# ============================================================
# 1. LOAD SHAP RESULTS
# ============================================================

SHAP_IMPORTANCE_FILE = "SHAP_feature_importance.csv"
SHAP_INDIVIDUAL_FILE = "SHAP_individual_explanation.csv"

MACRO_SUMMARY_FILE = "macro_stress_summary.csv"


try:

    shap_importance = pd.read_csv(
        SHAP_IMPORTANCE_FILE
    )

    shap_individual = pd.read_csv(
        SHAP_INDIVIDUAL_FILE
    )

except FileNotFoundError as e:

    print("\nERROR: Required SHAP file not found.")
    print(e)

    raise SystemExit(1)


# ============================================================
# 2. VERIFY SHAP FILES
# ============================================================

required_shap_columns = [
    "feature",
    "mean_abs_shap"
]

required_individual_columns = [
    "feature",
    "feature_value",
    "shap_value"
]


for column in required_shap_columns:

    if column not in shap_importance.columns:

        raise ValueError(
            f"Missing column '{column}' "
            f"in {SHAP_IMPORTANCE_FILE}"
        )


for column in required_individual_columns:

    if column not in shap_individual.columns:

        raise ValueError(
            f"Missing column '{column}' "
            f"in {SHAP_INDIVIDUAL_FILE}"
        )


print("SHAP files verified successfully.")


# ============================================================
# 3. GLOBAL SHAP FEATURE IMPORTANCE
# ============================================================

print("\nCreating global SHAP feature importance plot...")


shap_importance = shap_importance.sort_values(
    "mean_abs_shap",
    ascending=True
)


plt.figure(
    figsize=(10, 7)
)


plt.barh(
    shap_importance["feature"],
    shap_importance["mean_abs_shap"]
)


plt.xlabel(
    "Mean Absolute SHAP Value"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Global SHAP Feature Importance - Credit Risk"
)


plt.tight_layout()


plt.savefig(
    "XAI_global_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "Saved: XAI_global_feature_importance.png"
)


# ============================================================
# 4. INDIVIDUAL LOAN SHAP CONTRIBUTIONS
# ============================================================

print("\nCreating individual loan SHAP explanation...")


shap_individual = shap_individual.sort_values(
    "shap_value"
)


plt.figure(
    figsize=(10, 7)
)


plt.barh(
    shap_individual["feature"],
    shap_individual["shap_value"]
)


plt.axvline(
    0,
    linewidth=1
)


plt.xlabel(
    "SHAP Contribution"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Individual Loan SHAP Explanation"
)


plt.tight_layout()


plt.savefig(
    "XAI_individual_explanation.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "Saved: XAI_individual_explanation.png"
)


# ============================================================
# 5. LOAD MACRO STRESS SUMMARY
# ============================================================

print("\nLoading macro stress testing summary...")


try:

    macro_results = pd.read_csv(
        MACRO_SUMMARY_FILE
    )

except FileNotFoundError:

    print(
        f"\nERROR: {MACRO_SUMMARY_FILE} "
        "was not found."
    )

    raise SystemExit(1)


print(
    "\nMacro stress summary columns:"
)

print(
    list(macro_results.columns)
)


# ============================================================
# 6. VERIFY MACRO STRESS COLUMNS
# ============================================================

required_macro_columns = [

    "scenario",

    "interest_rate_shock",

    "mean_pd",

    "median_pd",

    "max_pd",

    "absolute_pd_change",

    "relative_pd_change_percent"

]


missing_columns = [

    column

    for column in required_macro_columns

    if column not in macro_results.columns

]


if missing_columns:

    print(
        "\nERROR: Missing columns in "
        "macro_stress_summary.csv:"
    )

    print(
        missing_columns
    )

    print(
        "\nAvailable columns:"
    )

    print(
        list(macro_results.columns)
    )

    raise SystemExit(1)


print(
    "\nMacro stress summary verified successfully."
)


# ============================================================
# 7. MACRO STRESS - MEAN PD
# ============================================================

print(
    "\nCreating macro stress PD plot..."
)


plt.figure(
    figsize=(10, 6)
)


plt.plot(

    macro_results["scenario"],

    macro_results["mean_pd"],

    marker="o",

    linewidth=2
)


plt.xlabel(
    "Stress Scenario"
)

plt.ylabel(
    "Mean Probability of Default"
)

plt.title(
    "Macro Stress Testing - Mean Probability of Default"
)


plt.xticks(
    rotation=20
)


plt.grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    "macro_stress_pd.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "Saved: macro_stress_pd.png"
)


# ============================================================
# 8. MACRO STRESS - RELATIVE PD CHANGE
# ============================================================

print(
    "\nCreating relative stress impact plot..."
)


plt.figure(
    figsize=(10, 6)
)


plt.bar(

    macro_results["scenario"],

    macro_results[
        "relative_pd_change_percent"
    ]
)


plt.xlabel(
    "Stress Scenario"
)

plt.ylabel(
    "Relative PD Change (%)"
)

plt.title(
    "Relative Increase in Probability of Default"
)


plt.xticks(
    rotation=20
)


plt.grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    "macro_stress_relative_change.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "Saved: macro_stress_relative_change.png"
)


# ============================================================
# 9. ADDITIONAL PD CHANGE PLOT
# ============================================================

print(
    "\nCreating absolute PD change plot..."
)


plt.figure(
    figsize=(10, 6)
)


plt.bar(

    macro_results["scenario"],

    macro_results["absolute_pd_change"]
)


plt.xlabel(
    "Stress Scenario"
)

plt.ylabel(
    "Absolute PD Change"
)

plt.title(
    "Absolute Increase in Probability of Default"
)


plt.xticks(
    rotation=20
)


plt.grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    "macro_stress_absolute_change.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "Saved: macro_stress_absolute_change.png"
)


# ============================================================
# 10. SAVE COMBINED VISUALIZATION DATA
# ============================================================

macro_results.to_csv(
    "XAI_macro_stress_visualization_data.csv",
    index=False
)


print(
    "\nSaved: "
    "XAI_macro_stress_visualization_data.csv"
)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)

print(
    "VISUALIZATION ANALYSIS COMPLETED SUCCESSFULLY"
)

print("=" * 60)


print("\nGenerated files:")

print(
    "1. XAI_global_feature_importance.png"
)

print(
    "2. XAI_individual_explanation.png"
)

print(
    "3. macro_stress_pd.png"
)

print(
    "4. macro_stress_relative_change.png"
)

print(
    "5. macro_stress_absolute_change.png"
)

print(
    "6. XAI_macro_stress_visualization_data.csv"
)


print("\nAll XAI visualization steps completed successfully.")