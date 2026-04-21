import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

mpl.rcParams['font.family'] = 'sans-serif'

TREATMENT  = "is_delivery_late"
COVARIATES = ['price', 'freight_value', 'product_photos_qty', 'month',
              'product_category_name_encoded', 'seller_id_encoded',
              'product_weight_kg', 'product_size', 'distance_km', 'seller_avg_rating']

BLUE   = "#2C6FAC"
ORANGE = "#E8703A"

df = pd.read_csv("data_model.csv")
df["product_photos_qty"] = df["product_photos_qty"].fillna(df["product_photos_qty"].median())
X = df[COVARIATES].values
T = df[TREATMENT].values.astype(float)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
ps_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
ps_model.fit(X_scaled, T)
ps = ps_model.predict_proba(X_scaled)[:, 1]

ps_min = max(ps[T==1].min(), ps[T==0].min())
ps_max = min(ps[T==1].max(), ps[T==0].max())

from scipy.stats import gaussian_kde

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("#F8F8F8")
ax.grid(color="white", linewidth=1.2, zorder=0)

x_grid = np.linspace(-0.2, 1.2, 1000)

for t, label, color in [(0, "T=0", BLUE), (1, "T=1", ORANGE)]:
    kde  = gaussian_kde(ps[T==t], bw_method=0.15)
    dens = kde(x_grid)
    ax.plot(x_grid, dens, color=color, linewidth=2, zorder=2)
    ax.fill_between(x_grid, dens, alpha=0.35, color=color, label=label, zorder=2)

ax.set_xlim(-0.2, 1.2)
ax.set_ylim(0, None)
ax.set_xlabel("Predicted propensity P(T=1|X)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Propensity density by treatment group", fontsize=13, fontweight="bold", pad=10)
ax.legend(fontsize=11, framealpha=0.9, loc="upper right")
ax.spines[["top", "right", "left", "bottom"]].set_visible(True)
ax.spines[:].set_color("#CCCCCC")
ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig("slide_overlap.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved slide_overlap.png")
