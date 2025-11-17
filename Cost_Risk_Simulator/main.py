import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# Sample vendor data
# ----------------------
# Replace this section with your own CSV if needed
vendors = [
    ("Vendor A", 5.25, 7, 98, 96, 0.5, 0.8, 30, 1000, "Low", 1.0),
    ("Vendor B", 4.60, 18, 82, 85, 3.5, 2.0, 14, 5000, "High", 6.0),
    ("Vendor C", 6.00, 5, 99, 99, 0.2, 0.3, 45, 200, "Low", 0.5),
    ("Vendor D", 4.20, 25, 75, 70, 5.0, 4.0, 60, 8000, "High", 8.0),
    ("Vendor E", 5.00, 12, 90, 92, 1.2, 1.0, 30, 1500, "Medium", 2.5),
    ("Vendor F", 4.80, 10, 88, 87, 2.0, 1.8, 30, 1200, "Medium", 3.0),
    ("Vendor G", 5.75, 9, 95, 94, 0.8, 0.9, 30, 600, "Low", 1.2),
    ("Vendor H", 4.00, 30, 70, 65, 6.5, 5.5, 7, 10000, "High", 10.0),
]

df = pd.DataFrame(vendors, columns=[
    "vendor", "cost_per_unit", "lead_time_days", "on_time_pct", "fill_rate_pct",
    "defect_rate_pct", "return_rate_pct", "payment_terms_days", "MOQ", "geo_risk", "fx_volatility_pct"
])

# ----------------------
# Scoring helpers
# ----------------------
def normalize_inverse(series):
    return (series.max() - series) / (series.max() - series.min()) * 100

def normalize_direct(series):
    return (series - series.min()) / (series.max() - series.min()) * 100

# ----------------------
# Cost Score (lower = better)
# ----------------------
df["cost_score"] = normalize_inverse(df["cost_per_unit"]).round(2)

# ----------------------
# Lead Time Score (lower = better)
# ----------------------
df["lead_time_score"] = normalize_inverse(df["lead_time_days"]).round(2)

# ----------------------
# Reliability Score
# ----------------------
df["reliability_score"] = (
    0.50 * df["on_time_pct"] +
    0.30 * df["fill_rate_pct"] +
    0.20 * df["lead_time_score"]
).round(2)

# ----------------------
# Quality Score
# ----------------------
df["quality_score"] = (
    100 - (df["defect_rate_pct"] * 1.5 + df["return_rate_pct"])
).clip(0,100).round(2)

# ----------------------
# Risk Score
# ----------------------
geo_map = {"Low": 100, "Medium": 70, "High": 40}
df["geo_score"] = df["geo_risk"].map(geo_map)

df["fx_score"] = normalize_inverse(df["fx_volatility_pct"])
df["moq_score"] = normalize_inverse(df["MOQ"])

df["risk_score"] = (
    0.40 * df["geo_score"] +
    0.30 * df["fx_score"] +
    0.30 * df["moq_score"]
).round(2)

# ----------------------
# Weighting
# ----------------------
weights = {
    "cost": 0.25,
    "reliability": 0.35,
    "quality": 0.20,
    "risk": 0.20
}

df["total_score"] = (
    df["cost_score"] * weights["cost"] +
    df["reliability_score"] * weights["reliability"] +
    df["quality_score"] * weights["quality"] +
    df["risk_score"] * weights["risk"]
).round(2)

# ----------------------
# Recommendation rules
# ----------------------
def recommend(score):
    if score >= 85: return "Expand"
    if score >= 70: return "Maintain"
    if score >= 50: return "Monitor"
    return "Replace"

df["recommendation"] = df["total_score"].apply(recommend)

# Reorder to present clean output
df = df.sort_values("total_score", ascending=False)

print(df)

# ----------------------
# Save to CSV
# ----------------------
df.to_csv("vendor_scorecard.csv", index=False)
print("\nSaved as vendor_scorecard.csv")

# ----------------------
# Visualization 1 — Total Score Bar Chart
# ----------------------
plt.figure(figsize=(10,5))
plt.barh(df["vendor"], df["total_score"])
plt.xlabel("Total Score")
plt.title("Vendor Total Score Ranking")
plt.gca().invert_yaxis()
plt.show()

# ----------------------
# Visualization 2 — Cost vs Reliability (bubble = total score)
# ----------------------
plt.figure(figsize=(8,6))
sizes = (df["total_score"] - df["total_score"].min() + 5) * 20
plt.scatter(df["cost_per_unit"], df["reliability_score"], s=sizes)
for i, row in df.iterrows():
    plt.text(row["cost_per_unit"], row["reliability_score"] + 0.5, row["vendor"], fontsize=8)

plt.xlabel("Cost per Unit")
plt.ylabel("Reliability Score")
plt.title("Cost vs Reliability (bubble shows Total Score)")
plt.show()
