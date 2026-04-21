import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import statsmodels.api as sm

TREATMENT  = "is_delivery_late"
OUTCOME    = "Rating"
COVARIATES = ["price","freight_value","product_photos_qty","month",
              "product_category_name_encoded","seller_id_encoded",
              "product_weight_kg","product_size","distance_km","seller_avg_rating"]
BLUE   = "#2C6FAC"
ORANGE = "#E8703A"
LGRAY  = "#F4F4F4"

df = pd.read_csv("data_model.csv")
df["product_photos_qty"] = df["product_photos_qty"].fillna(df["product_photos_qty"].median())
Y = df[OUTCOME].values.astype(float)
T = df[TREATMENT].values.astype(float)
X_ctrl = df[COVARIATES].values

COV_LABELS = ["Price","Freight Value","Photos","Month","Category",
              "Seller ID","Weight","Size","Distance (km)","Seller Avg Rating"]
SMDS = {"seller_avg_rating":0.434,"distance_km":0.280,"freight_value":0.172,
        "month":0.113,"product_weight_kg":0.094,"price":0.085,
        "product_size":0.079,"product_category_name_encoded":0.041,
        "product_photos_qty":0.015,"seller_id_encoded":0.013}

# Regression models
m1 = sm.OLS(Y, sm.add_constant(T)).fit(cov_type="HC1")
m2 = sm.OLS(Y, sm.add_constant(np.column_stack([T, X_ctrl]))).fit(cov_type="HC1")
coef1, se1 = m1.params[1], m1.bse[1]
coef2, se2 = m2.params[1], m2.bse[1]
ci1 = f"[{coef1-1.96*se1:.3f}, {coef1+1.96*se1:.3f}]"
ci2 = f"[{coef2-1.96*se2:.3f}, {coef2+1.96*se2:.3f}]"

# ── Combined figure: 3 panels top, 1 table bottom ────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.38,
                       height_ratios=[1.15, 1])

# ── TOP ROW ───────────────────────────────────────────────────────────────────

# A: Dataset table
ax1 = fig.add_subplot(gs[0, 0])
ax1.axis("off")
rows = [["Orders",       "114,841"],
        ["Delayed",      "7,368  (6.4%)"],
        ["On Time",      "107,473  (93.6%)"],
        ["Treatment T",  "is_delivery_late"],
        ["Outcome Y",    "Star Rating (1-5)"],
        ["Covariates",   "10"],
        ["Period",       "2017-2018"],
        ["Source",       "Olist (Brazil)"]]
tbl = ax1.table(cellText=rows, colLabels=["Variable", "Value"],
                loc="center", cellLoc="left")
tbl.auto_set_font_size(False)
tbl.set_fontsize(11.5)
tbl.scale(1, 1.72)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#DDDDDD")
    if r == 0:
        cell.set_facecolor("#D8D4E8"); cell.set_text_props(fontweight="bold")
    else:
        cell.set_facecolor(LGRAY if r % 2 == 0 else "white")
ax1.set_title("A.  Dataset Overview", fontsize=13, fontweight="bold", pad=10)

# B: Rating distribution
ax2 = fig.add_subplot(gs[0, 1])
rating_vals = [1, 2, 3, 4, 5]
for t, label, color in [(0, "On Time", BLUE), (1, "Delayed", ORANGE)]:
    r = (pd.Series(Y[T==t]).value_counts(normalize=True)
           .reindex(rating_vals, fill_value=0) * 100)
    ax2.plot(rating_vals, r.values, marker="o", color=color,
             linewidth=2.2, markersize=8, label=label)
    ax2.fill_between(rating_vals, r.values, alpha=0.08, color=color)
mean0 = Y[T==0].mean(); mean1 = Y[T==1].mean()
ax2.axvline(mean0, color=BLUE,   linestyle="--", linewidth=1, alpha=0.6)
ax2.axvline(mean1, color=ORANGE, linestyle="--", linewidth=1, alpha=0.6)
ax2.text(mean0 - 0.55, 44, f"mean={mean0:.2f}", color=BLUE,   fontsize=10)
ax2.text(mean1 + 0.08, 38, f"mean={mean1:.2f}", color=ORANGE, fontsize=10)
ax2.set_xlabel("Star Rating", fontsize=12)
ax2.set_ylabel("% of Group",  fontsize=12)
ax2.set_title("B.  Rating by Delivery Status", fontsize=13, fontweight="bold", pad=10)
ax2.set_xticks(rating_vals); ax2.legend(fontsize=11)
ax2.spines[["top","right"]].set_visible(False)

# C: Covariate imbalance
ax3 = fig.add_subplot(gs[0, 2])
smd_sorted  = dict(sorted(SMDS.items(), key=lambda x: x[1]))
ylabels     = [COV_LABELS[COVARIATES.index(k)] for k in smd_sorted]
bar_colors  = [ORANGE if v >= 0.10 else "#AAAAAA" for v in smd_sorted.values()]
ax3.barh(range(len(smd_sorted)), list(smd_sorted.values()),
         color=bar_colors, alpha=0.85, height=0.6)
ax3.axvline(0.10, color="black", linestyle="--", linewidth=1.3)
ax3.set_yticks(range(len(smd_sorted)))
ax3.set_yticklabels(ylabels, fontsize=10)
ax3.set_xlabel("|SMD|", fontsize=12)
ax3.set_title("C.  Covariate Imbalance (Pre-Matching)", fontsize=13, fontweight="bold", pad=10)
p1 = mpatches.Patch(color=ORANGE,    alpha=0.85, label="|SMD| >= 0.10")
p2 = mpatches.Patch(color="#AAAAAA", alpha=0.85, label="|SMD| < 0.10")
ax3.legend(handles=[p1, p2], fontsize=10, loc="lower right")
ax3.spines[["top","right"]].set_visible(False)

# ── BOTTOM ROW: regression table spanning all 3 columns ──────────────────────
ax4 = fig.add_subplot(gs[1, :])
ax4.axis("off")

table_data = [
    ["",             "Bivariate OLS", "OLS + Controls"],
    ["Coefficient",  f"{coef1:.3f}",  f"{coef2:.3f}"],
    ["Std. Error",   f"{se1:.3f}",    f"{se2:.3f}"],
    ["95% CI",       ci1,             ci2],
    ["p-value",      "< 0.001",       "< 0.001"],
    ["R-squared",    f"{m1.rsquared:.3f}", f"{m2.rsquared:.3f}"],
    ["N",            f"{int(m1.nobs):,}",  f"{int(m2.nobs):,}"],
    ["Controls",     "No",            "Yes"],
]
tbl2 = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                 loc="center", cellLoc="center")
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(13)
tbl2.scale(1, 2.2)
for (r, c), cell in tbl2.get_celld().items():
    cell.set_edgecolor("#DDDDDD")
    if r == 0:
        cell.set_facecolor("#D8D4E8"); cell.set_text_props(fontweight="bold")
    elif c == 0:
        cell.set_facecolor(LGRAY); cell.set_text_props(fontweight="bold")
    else:
        cell.set_facecolor("white")
ax4.set_title("OLS Regression: Effect of Delivery Delay on Rating",
              fontsize=13, fontweight="bold", pad=14)

fig.suptitle("Background: Brazilian E-Commerce Delivery & Customer Ratings",
             fontsize=15, fontweight="bold", y=1.01)

plt.savefig("slide_background_regression.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved slide_background_regression.png")
