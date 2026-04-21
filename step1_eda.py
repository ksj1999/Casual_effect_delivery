import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11

# ── 1. Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv("data.csv")
print(f"Shape: {df.shape}")

# ── 2. Column audit ──────────────────────────────────────────────────────────
DROP_COLS = [
    # identifiers
    "order_id", "customer_id", "customer_unique_id", "product_id",
    "seller_id", "order_item_id",
    # raw timestamps (month already extracted)
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_customer_date", "order_estimated_delivery_date",
    "review_creation_date", "review_answer_timestamp",
    # raw strings / redundant geo (distance_km already computed)
    "order_status", "product_category_name",
    "customer_city", "customer_state", "customer_zip_code_prefix",
    "seller_city", "seller_state", "seller_zip_code_prefix",
    "geolocation_lat_x", "geolocation_lng_x",
    "geolocation_lat_y", "geolocation_lng_y",
    # redundant raw dimensions (kg/size already engineered)
    "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm",
    # duplicate of price
    "product_price",
    # duplicate of product_photos_qty
    "no_photos",
    # potential outcome leakage — check below
    "customer_experience",
    # continuous treatment version (keep separate)
    "late_delivery_in_days",
    # payment_value = price + freight, redundant
    "payment_value",
    # rainfall is a string region label, not numeric — drop or encode
    "rainfall",
]

# Check if customer_experience == Rating
leak_check = (df["customer_experience"] == df["Rating"]).mean()
print(f"\ncustomer_experience == Rating: {leak_check:.1%}  → dropping as leakage")

# Check rainfall dtype
print(f"rainfall unique values: {df['rainfall'].unique()}")

df_clean = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

TREATMENT = "is_delivery_late"
OUTCOME   = "Rating"
COVARIATES = [c for c in df_clean.columns if c not in [TREATMENT, OUTCOME]]

print(f"\nRetained columns ({len(df_clean.columns)}): {df_clean.columns.tolist()}")
print(f"Covariates ({len(COVARIATES)}): {COVARIATES}")

# ── 3. Missing values ────────────────────────────────────────────────────────
miss = df_clean.isnull().sum()
miss = miss[miss > 0]
print(f"\nMissing values:\n{miss if len(miss) else 'None'}")

# ── 4. Treatment balance ─────────────────────────────────────────────────────
treat_counts = df_clean[TREATMENT].value_counts()
treat_pct    = df_clean[TREATMENT].value_counts(normalize=True) * 100
print(f"\nTreatment balance:")
print(f"  Not delayed (0): {treat_counts[0]:,}  ({treat_pct[0]:.1f}%)")
print(f"  Delayed     (1): {treat_counts[1]:,}  ({treat_pct[1]:.1f}%)")

# ── 5. Outcome distribution ──────────────────────────────────────────────────
print(f"\nRating distribution:")
print(df_clean[OUTCOME].value_counts().sort_index())
print(f"Mean rating (not delayed): {df_clean.loc[df_clean[TREATMENT]==0, OUTCOME].mean():.3f}")
print(f"Mean rating (delayed):     {df_clean.loc[df_clean[TREATMENT]==1, OUTCOME].mean():.3f}")
raw_diff = (df_clean.loc[df_clean[TREATMENT]==0, OUTCOME].mean()
          - df_clean.loc[df_clean[TREATMENT]==1, OUTCOME].mean())
print(f"Raw mean difference:       {raw_diff:.3f}  (naive, unadjusted)")

# ── 6. Confounder correlations with T and Y ──────────────────────────────────
corr_T = df_clean[COVARIATES].corrwith(df_clean[TREATMENT]).abs().sort_values(ascending=False)
corr_Y = df_clean[COVARIATES].corrwith(df_clean[OUTCOME]).abs().sort_values(ascending=False)
corr_summary = pd.DataFrame({"corr_with_T": corr_T, "corr_with_Y": corr_Y}).round(3)
print(f"\nConfounder correlations:\n{corr_summary}")

# ── 7. Plots ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# 7a. Treatment balance
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(["Not delayed", "Delayed"], treat_pct.values, color=["#4C72B0", "#DD8452"], width=0.5)
ax1.set_ylabel("% of orders")
ax1.set_title("Treatment Balance")
for i, v in enumerate(treat_pct.values):
    ax1.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)

# 7b. Rating distribution by treatment
ax2 = fig.add_subplot(gs[0, 1])
for t, label, color in [(0, "Not delayed", "#4C72B0"), (1, "Delayed", "#DD8452")]:
    sub = df_clean[df_clean[TREATMENT] == t][OUTCOME].value_counts(normalize=True).sort_index() * 100
    ax2.plot(sub.index, sub.values, marker="o", label=label, color=color)
ax2.set_xlabel("Rating")
ax2.set_ylabel("% of group")
ax2.set_title("Rating Distribution by Treatment")
ax2.legend()
ax2.set_xticks([1, 2, 3, 4, 5])

# 7c. Mean rating by treatment
ax3 = fig.add_subplot(gs[0, 2])
means = df_clean.groupby(TREATMENT)[OUTCOME].mean()
ax3.bar(["Not delayed", "Delayed"], means.values, color=["#4C72B0", "#DD8452"], width=0.5)
ax3.set_ylabel("Mean Rating")
ax3.set_title(f"Mean Rating by Treatment\n(raw diff = {raw_diff:.3f})")
for i, v in enumerate(means.values):
    ax3.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
ax3.set_ylim(0, 5.5)

# 7d. Correlation heatmap with T and Y
ax4 = fig.add_subplot(gs[1, :2])
top_vars = corr_summary.assign(
    combined=corr_summary["corr_with_T"] + corr_summary["corr_with_Y"]
).sort_values("combined", ascending=False).head(10).index.tolist()
plot_df = corr_summary.loc[top_vars]
sns.heatmap(plot_df.T, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax4,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
ax4.set_title("Top 10 Covariates: |Correlation| with Treatment & Outcome")
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=30, ha="right")

# 7e. Distance distribution by treatment
ax5 = fig.add_subplot(gs[1, 2])
for t, label, color in [(0, "Not delayed", "#4C72B0"), (1, "Delayed", "#DD8452")]:
    sub = df_clean[df_clean[TREATMENT] == t]["distance_km"]
    ax5.hist(sub, bins=50, alpha=0.5, label=label, color=color, density=True)
ax5.set_xlabel("distance_km")
ax5.set_ylabel("Density")
ax5.set_title("Distance Distribution by Treatment")
ax5.legend()

# 7f. Freight value distribution by treatment
ax6 = fig.add_subplot(gs[2, 0])
for t, label, color in [(0, "Not delayed", "#4C72B0"), (1, "Delayed", "#DD8452")]:
    sub = df_clean[df_clean[TREATMENT] == t]["freight_value"].clip(upper=200)
    ax6.hist(sub, bins=50, alpha=0.5, label=label, color=color, density=True)
ax6.set_xlabel("freight_value (clipped at 200)")
ax6.set_ylabel("Density")
ax6.set_title("Freight Value by Treatment")
ax6.legend()

# 7g. Price distribution by treatment
ax7 = fig.add_subplot(gs[2, 1])
for t, label, color in [(0, "Not delayed", "#4C72B0"), (1, "Delayed", "#DD8452")]:
    sub = np.log1p(df_clean[df_clean[TREATMENT] == t]["price"])
    ax7.hist(sub, bins=50, alpha=0.5, label=label, color=color, density=True)
ax7.set_xlabel("log(1 + price)")
ax7.set_ylabel("Density")
ax7.set_title("log-Price by Treatment")
ax7.legend()

# 7h. Seller avg rating by treatment
ax8 = fig.add_subplot(gs[2, 2])
for t, label, color in [(0, "Not delayed", "#4C72B0"), (1, "Delayed", "#DD8452")]:
    sub = df_clean[df_clean[TREATMENT] == t]["seller_avg_rating"]
    ax8.hist(sub, bins=30, alpha=0.5, label=label, color=color, density=True)
ax8.set_xlabel("seller_avg_rating")
ax8.set_ylabel("Density")
ax8.set_title("Seller Avg Rating by Treatment")
ax8.legend()

plt.suptitle("Step 1 EDA — Delivery Delay & Customer Ratings", fontsize=14, y=1.01)
plt.savefig("step1_eda.png", bbox_inches="tight")
plt.show()
print("\nPlot saved to step1_eda.png")

# ── 8. Statistical tests for key confounders ─────────────────────────────────
print("\n── Two-sample t-tests: delayed vs not-delayed ──")
numeric_covs = df_clean[COVARIATES].select_dtypes(include=np.number).columns.tolist()
for col in numeric_covs:
    g0 = df_clean.loc[df_clean[TREATMENT] == 0, col].dropna()
    g1 = df_clean.loc[df_clean[TREATMENT] == 1, col].dropna()
    t, p = stats.ttest_ind(g0, g1, equal_var=False)
    smd = (g1.mean() - g0.mean()) / np.sqrt((g0.std()**2 + g1.std()**2) / 2)
    print(f"  {col:<35} SMD={smd:+.3f}  p={p:.3e}")

# ── 9. Save clean dataset for next steps ────────────────────────────────────
df_model = df_clean[COVARIATES + [TREATMENT, OUTCOME]].copy()
df_model.to_csv("data_model.csv", index=False)
print(f"\nSaved data_model.csv  shape={df_model.shape}")
print("Columns:", df_model.columns.tolist())
