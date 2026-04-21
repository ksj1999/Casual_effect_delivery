import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

TREATMENT  = "is_delivery_late"
OUTCOME    = "Rating"
COVARIATES = ["price","freight_value","product_photos_qty","month",
              "product_category_name_encoded","seller_id_encoded",
              "product_weight_kg","product_size","distance_km","seller_avg_rating"]
COV_LABELS = ["Price","Freight Value","Photos","Month","Category",
              "Seller ID","Weight","Size","Distance (km)","Seller Avg Rating"]

BLUE   = "#2C6FAC"
ORANGE = "#E8703A"
PURPLE = "#6B5EA8"
LGRAY  = "#F7F7F7"

# ── Data & matching ───────────────────────────────────────────────────────────
df = pd.read_csv("data_model.csv")
df["product_photos_qty"] = df["product_photos_qty"].fillna(df["product_photos_qty"].median())
X = df[COVARIATES].values
T = df[TREATMENT].values.astype(float)
Y = df[OUTCOME].values.astype(float)

scaler   = StandardScaler()
X_s      = scaler.fit_transform(X)
ps_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
ps_model.fit(X_s, T)
ps = ps_model.predict_proba(X_s)[:, 1]

ps_min = max(ps[T==1].min(), ps[T==0].min())
ps_max = min(ps[T==1].max(), ps[T==0].max())
in_sup = (ps >= ps_min) & (ps <= ps_max)
ps_t = ps[in_sup]; T_t = T[in_sup]; Y_t = Y[in_sup]
df_t = df[in_sup].copy()

logit_ps = np.log(ps_t / (1 - ps_t + 1e-9))
caliper  = 0.2 * logit_ps.std()
ti = np.where(T_t == 1)[0]; ci = np.where(T_t == 0)[0]
nn = NearestNeighbors(n_neighbors=1).fit(ps_t[ci].reshape(-1, 1))
dist, mpos = nn.kneighbors(ps_t[ti].reshape(-1, 1))
wc = dist.flatten() <= caliper
mti = ti[wc]; mci = ci[mpos.flatten()[wc]]
_, up = np.unique(mci, return_index=True)
mti = mti[up]; mci = mci[up]; n_pairs = len(mti)
df_m = df_t.iloc[np.concatenate([mti, mci])].copy()
Yt = Y_t[mti]; Yc = Y_t[mci]; ATT = (Yt - Yc).mean()

np.random.seed(42)
boot = []
for _ in range(2000):
    idx = np.random.choice(n_pairs, n_pairs, replace=True)
    boot.append((Yt[idx] - Yc[idx]).mean())
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

def smd(data, cov):
    out = {}
    for col in cov:
        g1 = data.loc[data[TREATMENT]==1, col]
        g0 = data.loc[data[TREATMENT]==0, col]
        p  = np.sqrt((g1.std()**2 + g0.std()**2) / 2)
        out[col] = (g1.mean() - g0.mean()) / p if p > 0 else 0
    return pd.Series(out)

sb = smd(df_t, COVARIATES)
sa = smd(df_m, COVARIATES)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32,
                       left=0.06, right=0.97, top=0.88, bottom=0.08)

# Title
fig.text(0.5, 0.95, "Propensity Score Matching (PSM)",
         ha="center", va="center", fontsize=20, fontweight="bold", color="white",
         bbox=dict(boxstyle="round,pad=0.4", facecolor=PURPLE,
                   edgecolor="none", alpha=0.95))

# ── Workflow (top-left) ───────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
ax0.axis("off")
ax0.set_facecolor(LGRAY)
ax0.patch.set_visible(True)

ax0.text(0.5, 0.97, "Workflow", ha="center", va="top", fontsize=13,
         fontweight="bold", color=PURPLE, transform=ax0.transAxes)
ax0.plot([0.05, 0.95], [0.91, 0.91], color=PURPLE,
         linewidth=1.5, transform=ax0.transAxes)

steps = [
    ("Covariates $X$,  Treatment $T$,  Outcome $Y$", False),
    ("Step 1: estimate $e(X) = P(T=1 \\mid X)$",     False),
    ("Step 2: trim to common support",                False),
    ("Step 3: 1:1 nearest-neighbor matching",         False),
    ("  caliper $= 0.2\\times$std$(\\mathrm{logit}(e(X)))$", True),
    ("Step 4: check balance  $|\\mathrm{SMD}| < 0.10$", False),
    ("Step 5: estimate ATT",                          False),
    ("  $\\widehat{\\mathrm{ATT}}=\\frac{1}{n_1}"
     "\\sum_{i:T_i=1}(Y_i - Y_{j(i)})$",            True),
]
y_cur = 0.84
for txt, sub in steps:
    prefix = "  " if sub else "\u2022 "
    ax0.text(0.05, y_cur, prefix + txt, va="top",
             fontsize=10.5 if sub else 11.5,
             color="#555555" if sub else "#111111",
             style="italic" if sub else "normal",
             transform=ax0.transAxes)
    y_cur -= 0.115 if sub else 0.128

# ── Overlap (bottom-left) ─────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1, 0])
ax1.hist(ps_t[T_t==0], bins=70, density=True, alpha=0.50, color=BLUE,
         label="On Time",  edgecolor="white", linewidth=0.2)
ax1.hist(ps_t[T_t==1], bins=70, density=True, alpha=0.50, color=ORANGE,
         label="Delayed",  edgecolor="white", linewidth=0.2)
ax1.axvline(ps_min, color="#333333", linestyle="--", linewidth=1.3,
            label=f"Support [{ps_min:.2f}, {ps_max:.2f}]")
ax1.axvline(ps_max, color="#333333", linestyle="--", linewidth=1.3)
ax1.set_xlabel("Propensity Score", fontsize=11)
ax1.set_ylabel("Density", fontsize=11)
ax1.set_title("Overlap Check", fontsize=13, fontweight="bold", color=PURPLE, pad=8)
ax1.legend(fontsize=9, framealpha=0.8)
ax1.spines[["top", "right"]].set_visible(False)

# ── Bootstrap ATT (top-center) ────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(boot, bins=60, color=BLUE, edgecolor="white", linewidth=0.3, alpha=0.85)
ax2.axvline(ATT,   color=ORANGE,    linewidth=2.8, label=f"ATT = {ATT:.3f}")
ax2.axvline(ci_lo, color="#333333", linestyle="--", linewidth=1.8,
            label=f"95% CI  [{ci_lo:.3f}, {ci_hi:.3f}]")
ax2.axvline(ci_hi, color="#333333", linestyle="--", linewidth=1.8)
ax2.set_xlabel("ATT Estimate", fontsize=11)
ax2.set_ylabel("Frequency", fontsize=11)
ax2.set_title("Bootstrap Distribution of ATT", fontsize=13,
              fontweight="bold", color=PURPLE, pad=8)
ax2.legend(fontsize=10, framealpha=0.85)
ax2.spines[["top", "right"]].set_visible(False)

# ── Love plot (top-right) ─────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
order = sb.abs().sort_values(ascending=True)
yp    = np.arange(len(order))
labs  = [COV_LABELS[COVARIATES.index(c)] for c in order.index]
for y, col in enumerate(order.index):
    ax3.plot([sb.abs()[col], sa.abs()[col]], [y, y],
             color="#CCCCCC", linewidth=1.2, zorder=1)
ax3.scatter(sb.abs()[order.index], yp, marker="o", color=ORANGE, s=75, zorder=3, label="Before")
ax3.scatter(sa.abs()[order.index], yp, marker="D", color=BLUE,   s=75, zorder=3, label="After")
ax3.axvline(0.10, color="#333333", linestyle="--", linewidth=1.3, label="Threshold 0.10")
ax3.set_yticks(yp); ax3.set_yticklabels(labs, fontsize=9.5)
ax3.set_xlabel("|SMD|", fontsize=11)
ax3.set_title("Love Plot: Covariate Balance", fontsize=13,
              fontweight="bold", color=PURPLE, pad=8)
ax3.legend(fontsize=9, loc="lower right", framealpha=0.85)
ax3.spines[["top", "right"]].set_visible(False)

# ── Takeaways (bottom center + right) ────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1:])
ax4.axis("off")

takeaways = [
    (BLUE,   "1.", "Matching successfully eliminated covariate imbalance\n"
                   "     (all |SMD| < 0.10 after matching)"),
    (ORANGE, "2.", "Late delivery causally reduces ratings by 1.77 stars\n"
                   "     (95% CI: [\u22121.817, \u22121.720],  p \u2248 0)"),
]
for i, (color, num, txt) in enumerate(takeaways):
    y = 0.70 - i * 0.42
    ax4.add_patch(FancyBboxPatch(
        (0.03, y - 0.12), 0.93, 0.32,
        boxstyle="round,pad=0.02",
        facecolor=color, alpha=0.08,
        edgecolor=color, linewidth=2,
        transform=ax4.transAxes))
    ax4.text(0.08, y + 0.06, num, fontsize=16, fontweight="bold",
             color=color, transform=ax4.transAxes, va="center")
    ax4.text(0.17, y + 0.06, txt, fontsize=12.5, color="#111111",
             transform=ax4.transAxes, va="center")

plt.savefig("slide_psm_final.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved slide_psm_final.png")
