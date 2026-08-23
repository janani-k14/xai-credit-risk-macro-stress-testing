import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv("country_risk_summary.csv")
print("Country risk summary loaded.")
print(df.to_string(index=False))

# ============================================================
# 1. ACTUAL VS PREDICTED PD
# ============================================================

x = np.arange(len(df))
width = 0.35
plt.figure(figsize=(10, 6))
plt.bar(
    x - width / 2,
    df["actual_default_rate_percent"],
    width,
    label="Actual Default Rate"
)
plt.bar(
    x + width / 2,
    df["mean_predicted_pd_percent"],
    width,
    label="Mean Predicted PD"
)
plt.xticks(
    x,
    df["country_grouped"]
)
plt.ylabel("Percentage (%)")
plt.xlabel("Country")
plt.title(
    "Actual Default Rate vs Predicted PD by Country"
)
plt.legend()
plt.tight_layout()
plt.savefig(
    "country_actual_vs_predicted_pd.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
print(
    "\nSaved: country_actual_vs_predicted_pd.png"
)

# ============================================================
# 2. CALIBRATION GAP
# ============================================================

plt.figure(figsize=(10, 6))
bars = plt.bar(
    df["country_grouped"],
    df["pd_calibration_gap_percent"]
)
plt.axhline(
    0,
    linewidth=1
)
plt.ylabel(
    "Predicted PD - Actual Default Rate (percentage points)"
)
plt.xlabel("Country")
plt.title(
    "Country-Level PD Calibration Gap"
)
plt.tight_layout()
plt.savefig(
    "country_calibration_gap.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
print(
    "Saved: country_calibration_gap.png"
)

# ============================================================
# 3. MODEL PERFORMANCE
# ============================================================

performance = pd.read_csv(
    "country_model_performance.csv"
)
plt.figure(figsize=(10, 6))
plt.bar(
    performance["country_grouped"],
    performance["roc_auc"]
)
plt.axhline(
    0.5,
    linestyle="--",
    linewidth=1,
    label="Random baseline (0.50)"
)
plt.ylabel("ROC-AUC")
plt.xlabel("Country")
plt.title(
    "Model Discrimination by Country"
)
plt.legend()
plt.tight_layout()
plt.savefig(
    "country_roc_auc.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
print(
    "Saved: country_roc_auc.png"
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("COUNTRY VISUALIZATION COMPLETED")
print("=" * 60)
print("""
Generated files:
1. country_actual_vs_predicted_pd.png
2. country_calibration_gap.png
3. country_roc_auc.png
""")