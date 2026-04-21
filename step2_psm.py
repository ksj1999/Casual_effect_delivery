import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11

TREATMENT  = "is_delivery_late"
OUTCOME    = "Rating"
COVARIATES = ['price', 'freight_value', 'product_photos_qty', 'month',
              'product_category_name_encoded', 'seller_id_encoded',
              'product_weight_kg', 'product_size', 'distance_km', 'seller_avg_rating']

# ── 1. Load & impute ──────────────────────────────────────────────────────────
df = pd.read_csv("data_model.csv")
df["product_photos_qty"] = df["product_photos_qty"].fillna(df["product_photos_qty"].median())
print(f"Loaded: {df.shape}  |  Treated: {df[TREATMENT].sum():,}  Control: {(df[TREATMENT]==0).sum():,}")

X = df[COVARIATES].values
T = df[TREATMENT].values
Y = df[OUTCOME].values

# ── 2. Propensity score model ─────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ps_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
ps_model.fit(X_scaled, T)
ps = ps_model.predict_proba(X_scaled)[:, 1]

print(f"\nPropensity scores — treated:  mean={ps[T==1].mean():.3f}  min={ps[T==1].min():.3f}  max={ps[T==1].max():.3f}")
print(f"Propensity scores — control:  mean={ps[T==0].mean():.3f}  min={ps[T==0].min():.3f}  max={ps[T==0].max():.3f}")

# ── 3. Common support trim ────────────────────────────────────────────────────
ps_min = max(ps[T==1].min(), ps[T==0].min())
ps_max = min(ps[T==1].max(), ps[T==0].max())
in_support = (ps >= ps_min) & (ps <= ps_max)
print(f"\nCommon support: [{ps_min:.4f}, {ps_max:.4f}]")
print(f"Dropped (outside support): {(~in_support).sum()}  ({(~in_support).mean()*100:.2f}%)")

df_trim = df[in_support].copy()
ps_trim = ps[in_support]
T_trim  = T[in_support]
Y_trim  = Y[in_support]
X_trim  = X_scaled[in_support]

# ── 4. Caliper matching (1:1 NN without replacement) ─────────────────────────
# Standard caliper: 0.2 × std of logit(PS)
logit_ps  = np.log(ps_trim / (1 - ps_trim + 1e-9))
caliper   = 0.2 * logit_ps.std()
print(f"\nCaliper (0.2 × std logit-PS): {caliper:.4f}")

treated_idx  = np.where(T_trim == 1)[0]
control_idx  = np.where(T_trim == 0)[0]

ps_treated = ps_trim[treated_idx].reshape(-1, 1)
ps_control = ps_trim[control_idx].reshape(-1, 1)

nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
nn.fit(ps_control)
distances, matched_positions = nn.kneighbors(ps_treated)

distances   = distances.flatten()
matched_pos = matched_positions.flatten()

# Apply caliper filter
within_caliper = distances <= caliper
n_matched = within_caliper.sum()
print(f"Treated matched within caliper: {n_matched:,} / {len(treated_idx):,}  ({n_matched/len(treated_idx)*100:.1f}%)")

matched_treated_idx = treated_idx[within_caliper]
matched_control_idx = control_idx[matched_pos[within_caliper]]

# Build matched dataset (deduplicate controls — without replacement approximation)
_, unique_pos = np.unique(matched_control_idx, return_index=True)
matched_treated_idx = matched_treated_idx[unique_pos]
matched_control_idx = matched_control_idx[unique_pos]
n_pairs = len(matched_treated_idx)
print(f"Unique pairs after deduplication: {n_pairs:,}")

matched_idx = np.concatenate([matched_treated_idx, matched_control_idx])
df_matched  = df_trim.iloc[matched_idx].copy()
T_matched   = T_trim[matched_idx]
Y_matched   = Y_trim[matched_idx]
X_matched   = df_trim.iloc[matched_idx][COVARIATES].values

# ── 5. Covariate balance (SMD before/after) ───────────────────────────────────
def smd(data, treatment, covariates):
    smds = {}
    for col in covariates:
        g1 = data.loc[data[treatment] == 1, col]
        g0 = data.loc[data[treatment] == 0, col]
        pooled_std = np.sqrt((g1.std()**2 + g0.std()**2) / 2)
        smds[col] = (g1.mean() - g0.mean()) / pooled_std if pooled_std > 0 else 0
    return pd.Series(smds)

smd_before = smd(df_trim, TREATMENT, COVARIATES)
smd_after  = smd(df_matched, TREATMENT, COVARIATES)

balance = pd.DataFrame({
    "SMD Before": smd_before.round(3),
    "SMD After":  smd_after.round(3),
    "|SMD| Before": smd_before.abs().round(3),
    "|SMD| After":  smd_after.abs().round(3),
}).sort_values("|SMD| Before", ascending=False)

print(f"\n── Covariate Balance ──")
print(balance[["SMD Before", "SMD After"]].to_string())
print(f"\nMax |SMD| after matching: {smd_after.abs().max():.3f}  (threshold: 0.10)")
print(f"Variables above 0.10 after matching: {(smd_after.abs() > 0.10).sum()}")

# ── 6. ATT estimate ───────────────────────────────────────────────────────────
Y_treated = Y_trim[matched_treated_idx]
Y_control = Y_trim[matched_control_idx]
ATT = (Y_treated - Y_control).mean()
print(f"\n── ATT Estimate ──")
print(f"Mean Rating (treated):  {Y_treated.mean():.3f}")
print(f"Mean Rating (matched control): {Y_control.mean():.3f}")
print(f"ATT = {ATT:.3f}")

# Paired t-test
t_stat, p_val = stats.ttest_rel(Y_treated, Y_control)
print(f"Paired t-test: t={t_stat:.3f}, p={p_val:.3e}")

# Bootstrap 95% CI
np.random.seed(42)
n_boot = 2000
boot_att = []
for _ in range(n_boot):
    idx = np.random.choice(n_pairs, n_pairs, replace=True)
    boot_att.append((Y_treated[idx] - Y_control[idx]).mean())
ci_lo, ci_hi = np.percentile(boot_att, [2.5, 97.5])
print(f"Bootstrap 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

# ── 7. Plots ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 7a. Propensity score overlap (before matching)
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(ps_trim[T_trim==0], bins=60, alpha=0.5, density=True, label="Control", color="#4C72B0")
ax1.hist(ps_trim[T_trim==1], bins=60, alpha=0.5, density=True, label="Treated", color="#DD8452")
ax1.set_xlabel("Propensity Score")
ax1.set_ylabel("Density")
ax1.set_title("Overlap: Propensity Score Distribution")
ax1.legend()

# 7b. Propensity score after matching
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(ps_trim[matched_control_idx], bins=40, alpha=0.5, density=True, label="Matched Control", color="#4C72B0")
ax2.hist(ps_trim[matched_treated_idx], bins=40, alpha=0.5, density=True, label="Treated", color="#DD8452")
ax2.set_xlabel("Propensity Score")
ax2.set_ylabel("Density")
ax2.set_title("Overlap: After Matching")
ax2.legend()

# 7c. Love plot (SMD before/after)
ax3 = fig.add_subplot(gs[0, 2])
y_pos = np.arange(len(COVARIATES))
ax3.scatter(smd_before.abs(), y_pos, marker="o", color="#DD8452", s=60, label="Before", zorder=3)
ax3.scatter(smd_after.abs(),  y_pos, marker="D", color="#4C72B0", s=60, label="After",  zorder=3)
ax3.axvline(0.10, color="gray", linestyle="--", linewidth=1, label="Threshold 0.10")
ax3.set_yticks(y_pos)
ax3.set_yticklabels(COVARIATES, fontsize=9)
ax3.set_xlabel("|SMD|")
ax3.set_title("Love Plot: Covariate Balance")
ax3.legend(fontsize=9)
ax3.grid(axis="x", alpha=0.3)

# 7d. ATT bootstrap distribution
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(boot_att, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.3)
ax4.axvline(ATT,   color="#DD8452", linewidth=2, label=f"ATT = {ATT:.3f}")
ax4.axvline(ci_lo, color="gray",   linestyle="--", linewidth=1.5, label=f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
ax4.axvline(ci_hi, color="gray",   linestyle="--", linewidth=1.5)
ax4.set_xlabel("ATT")
ax4.set_ylabel("Bootstrap Frequency")
ax4.set_title("Bootstrap Distribution of ATT")
ax4.legend(fontsize=9)

# 7e. Rating distribution before vs after matching
ax5 = fig.add_subplot(gs[1, 1])
rating_vals = [1, 2, 3, 4, 5]
for t, label, color in [(0, "Control (before)", "#4C72B0"), (1, "Treated (before)", "#DD8452")]:
    r = pd.Series(Y_trim[T_trim==t]).value_counts(normalize=True).reindex(rating_vals, fill_value=0) * 100
    ax5.plot(rating_vals, r.values, marker="o", linestyle="--", alpha=0.5, label=label, color=color)
for t, label, color, marker in [
    (0, "Control (matched)", "#4C72B0", "D"),
    (1, "Treated (matched)", "#DD8452", "D")
]:
    r = pd.Series(Y_matched[T_matched==t]).value_counts(normalize=True).reindex(rating_vals, fill_value=0) * 100
    ax5.plot(rating_vals, r.values, marker=marker, linestyle="-", label=label, color=color)
ax5.set_xlabel("Rating")
ax5.set_ylabel("% of group")
ax5.set_title("Rating Distribution Before/After Matching")
ax5.set_xticks(rating_vals)
ax5.legend(fontsize=8)

# 7f. SMD before/after bar chart
ax6 = fig.add_subplot(gs[1, 2])
x = np.arange(len(COVARIATES))
width = 0.35
ax6.bar(x - width/2, smd_before.abs(), width, label="Before", color="#DD8452", alpha=0.8)
ax6.bar(x + width/2, smd_after.abs(),  width, label="After",  color="#4C72B0", alpha=0.8)
ax6.axhline(0.10, color="gray", linestyle="--", linewidth=1)
ax6.set_xticks(x)
ax6.set_xticklabels(COVARIATES, rotation=40, ha="right", fontsize=8)
ax6.set_ylabel("|SMD|")
ax6.set_title("Balance: |SMD| Before vs After Matching")
ax6.legend()

plt.suptitle("Step 2: Propensity Score Matching", fontsize=14, y=1.01)
plt.savefig("step2_psm.png", bbox_inches="tight")
plt.show()
print("\nPlot saved to step2_psm.png")

# ── 8. Summary ────────────────────────────────────────────────────────────────
print("\n══════════════════════════════════════")
print("  PSM SUMMARY")
print("══════════════════════════════════════")
print(f"  Matched pairs:          {n_pairs:,}")
print(f"  ATT:                    {ATT:.3f}")
print(f"  95% Bootstrap CI:       [{ci_lo:.3f}, {ci_hi:.3f}]")
print(f"  p-value (paired t):     {p_val:.3e}")
print(f"  Max |SMD| after:        {smd_after.abs().max():.3f}")
print(f"  Vars above 0.10:        {(smd_after.abs() > 0.10).sum()}")
print("══════════════════════════════════════")
