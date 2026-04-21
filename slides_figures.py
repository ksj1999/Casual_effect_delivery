import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import GradientBoostingClassifier

TREATMENT  = "is_delivery_late"
OUTCOME    = "Rating"
COVARIATES = ['price', 'freight_value', 'product_photos_qty', 'month',
              'product_category_name_encoded', 'seller_id_encoded',
              'product_weight_kg', 'product_size', 'distance_km', 'seller_avg_rating']
COV_LABELS = ['Price', 'Freight Value', 'Product Photos', 'Month',
              'Product Category', 'Seller ID', 'Product Weight',
              'Product Size', 'Distance (km)', 'Seller Avg Rating']

BLUE   = "#2C6FAC"
ORANGE = "#E8703A"
GRAY   = "#AAAAAA"

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data_model.csv")
df["product_photos_qty"] = df["product_photos_qty"].fillna(df["product_photos_qty"].median())
X  = df[COVARIATES].values
T  = df[TREATMENT].values.astype(float)
Y  = df[OUTCOME].values.astype(float)

# ── Propensity scores (needed for both slides) ────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
ps_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
ps_model.fit(X_scaled, T)
ps = ps_model.predict_proba(X_scaled)[:, 1]

ps_min = max(ps[T==1].min(), ps[T==0].min())
ps_max = min(ps[T==1].max(), ps[T==0].max())
in_support = (ps >= ps_min) & (ps <= ps_max)
df_trim = df[in_support].copy()
ps_trim = ps[in_support]; T_trim = T[in_support]; Y_trim = Y[in_support]

logit_ps = np.log(ps_trim / (1 - ps_trim + 1e-9))
caliper  = 0.2 * logit_ps.std()
treated_idx = np.where(T_trim == 1)[0]
control_idx = np.where(T_trim == 0)[0]
nn = NearestNeighbors(n_neighbors=1).fit(ps_trim[control_idx].reshape(-1, 1))
distances, matched_pos = nn.kneighbors(ps_trim[treated_idx].reshape(-1, 1))
distances = distances.flatten(); matched_pos = matched_pos.flatten()
within_caliper = distances <= caliper
matched_treated_idx = treated_idx[within_caliper]
matched_control_idx = control_idx[matched_pos[within_caliper]]
_, unique_pos = np.unique(matched_control_idx, return_index=True)
matched_treated_idx = matched_treated_idx[unique_pos]
matched_control_idx = matched_control_idx[unique_pos]
n_pairs = len(matched_treated_idx)
matched_idx = np.concatenate([matched_treated_idx, matched_control_idx])
df_matched  = df_trim.iloc[matched_idx].copy()

def smd(data, treatment, covariates):
    out = {}
    for col in covariates:
        g1 = data.loc[data[treatment]==1, col]
        g0 = data.loc[data[treatment]==0, col]
        ps = np.sqrt((g1.std()**2 + g0.std()**2) / 2)
        out[col] = (g1.mean() - g0.mean()) / ps if ps > 0 else 0
    return pd.Series(out)

smd_before = smd(df_trim,    TREATMENT, COVARIATES)
smd_after  = smd(df_matched, TREATMENT, COVARIATES)

Y_treated = Y_trim[matched_treated_idx]
Y_control = Y_trim[matched_control_idx]
ATT = (Y_treated - Y_control).mean()
np.random.seed(42)
boot_att = [(Y_treated[np.random.choice(n_pairs, n_pairs, replace=True)] -
             Y_control[np.random.choice(n_pairs, n_pairs, replace=True)]).mean()
            for _ in range(2000)]
ci_lo, ci_hi = np.percentile(boot_att, [2.5, 97.5])

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
fig.patch.set_facecolor("white")
plt.rcParams.update({'font.size': 13, 'font.family': 'sans-serif'})

# Panel A: Treatment balance
ax = axes[0]
counts = [int((T==0).sum()), int((T==1).sum())]
pcts   = [c / sum(counts) * 100 for c in counts]
bars   = ax.bar(["On Time", "Delayed"], pcts, color=[BLUE, ORANGE],
                width=0.45, edgecolor="white", linewidth=1.5)
for bar, p in zip(bars, pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{p:.1f}%", ha="center", fontsize=13, fontweight="bold")
ax.set_ylabel("Percentage of Orders (%)", fontsize=12)
ax.set_title("A.  Treatment Distribution", fontsize=14, fontweight="bold", pad=10)
ax.set_ylim(0, 105)
ax.spines[["top","right"]].set_visible(False)
ax.tick_params(labelsize=12)

# Panel B: Rating distribution by group
ax = axes[1]
rating_vals = [1, 2, 3, 4, 5]
for t, label, color, ls in [(0, "On Time", BLUE, "-"), (1, "Delayed", ORANGE, "-")]:
    r = (pd.Series(Y[T==t]).value_counts(normalize=True)
           .reindex(rating_vals, fill_value=0) * 100)
    ax.plot(rating_vals, r.values, marker="o", color=color,
            linestyle=ls, linewidth=2.2, markersize=8, label=label)
mean0 = Y[T==0].mean(); mean1 = Y[T==1].mean()
ax.axvline(mean0, color=BLUE,   linestyle="--", linewidth=1.2, alpha=0.6)
ax.axvline(mean1, color=ORANGE, linestyle="--", linewidth=1.2, alpha=0.6)
ax.text(mean0 + 0.08, 42, f"μ={mean0:.2f}", color=BLUE,   fontsize=10)
ax.text(mean1 + 0.08, 38, f"μ={mean1:.2f}", color=ORANGE, fontsize=10)
ax.set_xlabel("Star Rating", fontsize=12)
ax.set_ylabel("% of Group", fontsize=12)
ax.set_title("B.  Rating by Delivery Status", fontsize=14, fontweight="bold", pad=10)
ax.set_xticks(rating_vals)
ax.legend(fontsize=11, framealpha=0.8)
ax.spines[["top","right"]].set_visible(False)
ax.tick_params(labelsize=12)

# Panel C: Confounder |SMD| (before matching — motivation for adjustment)
ax = axes[2]
order    = smd_before.abs().sort_values(ascending=True)
y_pos    = np.arange(len(order))
bar_colors = [ORANGE if v >= 0.10 else GRAY for v in order.values]
ax.barh(y_pos, order.values, color=bar_colors, alpha=0.85, height=0.6)
ax.axvline(0.10, color="black", linestyle="--", linewidth=1.3, label="Threshold (0.10)")
ax.set_yticks(y_pos)
labels_ordered = [COV_LABELS[COVARIATES.index(c)] for c in order.index]
ax.set_yticklabels(labels_ordered, fontsize=10.5)
ax.set_xlabel("|Standardized Mean Difference|", fontsize=12)
ax.set_title("C.  Pre-Matching Covariate Imbalance", fontsize=14, fontweight="bold", pad=10)
ax.legend(fontsize=10, loc="lower right")
ax.spines[["top","right"]].set_visible(False)
patch_sig  = mpatches.Patch(color=ORANGE, alpha=0.85, label="|SMD| ≥ 0.10")
patch_ok   = mpatches.Patch(color=GRAY,   alpha=0.85, label="|SMD| < 0.10")
ax.legend(handles=[patch_sig, patch_ok], fontsize=10, loc="lower right")

fig.suptitle("Exploratory Data Analysis", fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
plt.savefig("slide_eda.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved slide_eda.png")

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — PSM
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
fig.patch.set_facecolor("white")

# Panel A: Propensity score overlap
ax = axes[0]
ax.hist(ps_trim[T_trim==0], bins=60, alpha=0.55, density=True,
        label="On Time", color=BLUE,   edgecolor="white", linewidth=0.3)
ax.hist(ps_trim[T_trim==1], bins=60, alpha=0.55, density=True,
        label="Delayed", color=ORANGE, edgecolor="white", linewidth=0.3)
ax.set_xlabel("Propensity Score", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("A.  Common Support (Overlap)", fontsize=14, fontweight="bold", pad=10)
ax.legend(fontsize=11)
ax.spines[["top","right"]].set_visible(False)
ax.tick_params(labelsize=12)

# Panel B: Love plot (before & after)
ax = axes[1]
order_idx = smd_before.abs().sort_values().index
y_pos     = np.arange(len(order_idx))
ax.scatter(smd_before.abs()[order_idx], y_pos, marker="o", color=ORANGE,
           s=70, zorder=3, label="Before Matching")
ax.scatter(smd_after.abs()[order_idx],  y_pos, marker="D", color=BLUE,
           s=70, zorder=3, label="After Matching")
for y, col in enumerate(order_idx):
    ax.plot([smd_before.abs()[col], smd_after.abs()[col]], [y, y],
            color=GRAY, linewidth=1, zorder=2)
ax.axvline(0.10, color="black", linestyle="--", linewidth=1.3, label="Threshold (0.10)")
ax.set_yticks(y_pos)
labels_ordered = [COV_LABELS[COVARIATES.index(c)] for c in order_idx]
ax.set_yticklabels(labels_ordered, fontsize=10.5)
ax.set_xlabel("|Standardized Mean Difference|", fontsize=12)
ax.set_title("B.  Covariate Balance: Before vs After", fontsize=14, fontweight="bold", pad=10)
ax.legend(fontsize=10, loc="lower right")
ax.spines[["top","right"]].set_visible(False)
ax.tick_params(labelsize=12)

# Panel C: ATT bootstrap distribution
ax = axes[2]
ax.hist(boot_att, bins=60, color=BLUE, edgecolor="white", linewidth=0.3, alpha=0.85)
ax.axvline(ATT,   color=ORANGE, linewidth=2.5, label=f"ATT = {ATT:.3f}")
ax.axvline(ci_lo, color="black", linestyle="--", linewidth=1.8)
ax.axvline(ci_hi, color="black", linestyle="--", linewidth=1.8,
           label=f"95% CI  [{ci_lo:.3f}, {ci_hi:.3f}]")
ax.set_xlabel("ATT Estimate", fontsize=12)
ax.set_ylabel("Bootstrap Frequency", fontsize=12)
ax.set_title("C.  ATT: Bootstrap Distribution", fontsize=14, fontweight="bold", pad=10)
ax.legend(fontsize=11)
ax.spines[["top","right"]].set_visible(False)
ax.tick_params(labelsize=12)

fig.suptitle("Propensity Score Matching (PSM)", fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
plt.savefig("slide_psm.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved slide_psm.png")
