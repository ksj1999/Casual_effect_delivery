import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

mpl.rcParams['font.family'] = 'sans-serif'

TREATMENT  = "is_delivery_late"
COVARIATES = ['price', 'freight_value', 'product_photos_qty', 'month',
              'product_category_name_encoded', 'seller_id_encoded',
              'product_weight_kg', 'product_size', 'distance_km', 'seller_avg_rating']
COV_LABELS = ['Price', 'Freight Value', 'Product Photos', 'Month',
              'Product Category', 'Seller ID', 'Product Weight',
              'Product Size', 'Distance (km)', 'Seller Avg Rating']

BLUE   = "#2C6FAC"
ORANGE = "#E8703A"

# ── Load & match ──────────────────────────────────────────────────────────────
df = pd.read_csv("data_model.csv")
df["product_photos_qty"] = df["product_photos_qty"].fillna(df["product_photos_qty"].median())
X  = df[COVARIATES].values
T  = df[TREATMENT].values.astype(float)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
ps_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
ps_model.fit(X_scaled, T)
ps = ps_model.predict_proba(X_scaled)[:, 1]

ps_min = max(ps[T==1].min(), ps[T==0].min())
ps_max = min(ps[T==1].max(), ps[T==0].max())
in_support = (ps >= ps_min) & (ps <= ps_max)
df_trim = df[in_support].copy(); T_trim = T[in_support]; ps_trim = ps[in_support]

logit_ps = np.log(ps_trim / (1 - ps_trim + 1e-9))
caliper  = 0.2 * logit_ps.std()
treated_idx = np.where(T_trim == 1)[0]; control_idx = np.where(T_trim == 0)[0]
nn = NearestNeighbors(n_neighbors=1).fit(ps_trim[control_idx].reshape(-1, 1))
distances, matched_pos = nn.kneighbors(ps_trim[treated_idx].reshape(-1, 1))
within_caliper = distances.flatten() <= caliper
matched_treated_idx = treated_idx[within_caliper]
matched_control_idx = control_idx[matched_pos.flatten()[within_caliper]]
_, unique_pos = np.unique(matched_control_idx, return_index=True)
matched_idx = np.concatenate([matched_treated_idx[unique_pos], matched_control_idx[unique_pos]])
df_matched = df_trim.iloc[matched_idx].copy()

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

# ── Plot ──────────────────────────────────────────────────────────────────────
order     = smd_before.abs().sort_values(ascending=True)
y_pos     = np.arange(len(order))
labels    = [COV_LABELS[COVARIATES.index(c)] for c in order.index]

fig, ax = plt.subplots(figsize=(7, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Connecting lines
for y, col in enumerate(order.index):
    ax.plot([smd_before.abs()[col], smd_after.abs()[col]], [y, y],
            color="#CCCCCC", linewidth=1.2, zorder=1)

# Points
ax.scatter(smd_before.abs()[order.index], y_pos, marker="o", color=ORANGE,
           s=80, zorder=3, label="Before Matching")
ax.scatter(smd_after.abs()[order.index],  y_pos, marker="D", color=BLUE,
           s=80, zorder=3, label="After Matching")

# Threshold line
ax.axvline(0.10, color="black", linestyle="--", linewidth=1.2, label="Threshold = 0.10")

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=12)
ax.set_xlabel("Absolute Standardized Mean Difference", fontsize=12)
ax.set_title("Love Plot: Covariate Balance Before and After PSM", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=11, loc="lower right", framealpha=0.85)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", labelsize=11)
ax.set_xlim(-0.01, max(smd_before.abs().max(), 0.12) * 1.1)

plt.tight_layout()
plt.savefig("slide_love_plot.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved slide_love_plot.png")
