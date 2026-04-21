import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

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

X  = df[COVARIATES].values
T  = df[TREATMENT].values.astype(float)
Y  = df[OUTCOME].values.astype(float)
n  = len(Y)
print(f"n={n:,}  treated={int(T.sum()):,}  control={int((1-T).sum()):,}")

# ── 2. Cross-fit nuisance models (5-fold) ────────────────────────────────────
print("\nFitting nuisance models (5-fold cross-fitting)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# E[Y | X]  — regression
m_model = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                     learning_rate=0.05, random_state=42)
Y_hat = cross_val_predict(m_model, X, Y, cv=kf)

# E[T | X]  — classification (class_weight balanced for imbalanced treatment)
e_model = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                      learning_rate=0.05, random_state=42)
T_hat = cross_val_predict(e_model, X, T, cv=kf, method="predict_proba")[:, 1]

# Residuals
Y_res = Y - Y_hat   # outcome residual
T_res = T - T_hat   # treatment residual

r2_y = r2_score(Y, Y_hat)
r2_t = r2_score(T, T_hat)
print(f"Nuisance R2 — Y model: {r2_y:.3f}  |  T model: {r2_t:.3f}")

# ── 3. ATE via PLR (Frisch-Waugh) ────────────────────────────────────────────
# theta = (T_res' T_res)^{-1} T_res' Y_res
T_res_sq = (T_res ** 2).sum()
ATE      = (T_res * Y_res).sum() / T_res_sq

# Sandwich (HC1) standard error
psi    = T_res * (Y_res - ATE * T_res)          # score
var_se = (psi ** 2).sum() / (T_res_sq ** 2)
SE     = np.sqrt(var_se)
z      = ATE / SE
p_val  = 2 * (1 - stats.norm.cdf(abs(z)))
ci_lo  = ATE - 1.96 * SE
ci_hi  = ATE + 1.96 * SE

print(f"\nATE (DML) = {ATE:.3f}  SE={SE:.4f}  95% CI=[{ci_lo:.3f}, {ci_hi:.3f}]  p={p_val:.3e}")

# ── 4. Interactive DML: CATE ~ X  (linear heterogeneity) ─────────────────────
# Regress Y_res on T_res and T_res * (X - mean(X))
# Coefficient on T_res * x_k  = how x_k modifies the treatment effect
X_c    = X - X.mean(axis=0)                     # demeaned covariates
W      = np.column_stack([T_res] +               # intercept = ATE
                          [T_res * X_c[:, k] for k in range(X_c.shape[1])])
weights = T_res ** 2                             # WLS weights

wls = LinearRegression(fit_intercept=False)
wls.fit(W * weights[:, None], Y_res * weights)

cate_coefs = pd.Series(wls.coef_, index=["ATE"] + [f"HTE_{c}" for c in COVARIATES])
print(f"\nInteractive DML (linear CATE):")
print(cate_coefs.round(4).to_string())

# Bootstrap CI for CATE coefficients
np.random.seed(42)
n_boot   = 1000
boot_ate = []
boot_hte = {c: [] for c in COVARIATES}

for _ in range(n_boot):
    idx  = np.random.choice(n, n, replace=True)
    W_b  = W[idx];  Yr_b = Y_res[idx];  w_b = weights[idx]
    m    = LinearRegression(fit_intercept=False)
    m.fit(W_b * w_b[:, None], Yr_b * w_b)
    boot_ate.append(m.coef_[0])
    for i, c in enumerate(COVARIATES):
        boot_hte[c].append(m.coef_[i + 1])

ci_ate = np.percentile(boot_ate, [2.5, 97.5])
print(f"\nBootstrap 95% CI for ATE: [{ci_ate[0]:.3f}, {ci_ate[1]:.3f}]")

hte_ci = {}
print("\nHeterogeneity modifiers (HTE coefficients):")
for c in COVARIATES:
    lo, hi = np.percentile(boot_hte[c], [2.5, 97.5])
    sig    = "*" if (lo > 0 or hi < 0) else ""
    hte_ci[c] = (cate_coefs[f"HTE_{c}"], lo, hi)
    print(f"  {c:<35} {cate_coefs[f'HTE_{c}']:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {sig}")

# ── 5. Per-unit CATE predictions ──────────────────────────────────────────────
CATE_hat = wls.predict(W) / T_res   # approximate individual CATE
# Better: CATE_i = ATE + sum_k beta_k * (x_ik - mean(x_k))
CATE_hat = cate_coefs["ATE"] + X_c.dot(cate_coefs[[f"HTE_{c}" for c in COVARIATES]].values)
df["CATE"] = CATE_hat
print(f"\nCATEs: mean={CATE_hat.mean():.3f}  std={CATE_hat.std():.3f}  "
      f"min={CATE_hat.min():.3f}  max={CATE_hat.max():.3f}")

# ── 6. Sensitivity: partial R² (omitted variable bias) ───────────────────────
# Oster (2019): how strong would omitted confounders need to be to zero out ATE?
# delta = (beta_restricted / (beta_unrestricted - beta_restricted)) * (R2_max - R2_unrestricted) / R2_restricted
# Simplified: report coefficient stability
R2_Y_full = r2_y
# Restricted model: regress Y on T only
ATE_naive = np.cov(T, Y)[0, 1] / np.var(T)
R2_Y_naive = (np.corrcoef(T, Y)[0, 1]) ** 2
delta = (ATE_naive / (ATE_naive - ATE)) * ((1.0 - R2_Y_full) / R2_Y_naive) if R2_Y_naive > 0 else np.nan
print(f"\nSensitivity (Oster delta): {delta:.3f}")
print("  (delta > 1 means omitted confounders would need to explain more than observed X to zero ATE)")

# ── 7. Plots ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

# 7a. Treatment residual distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(T_res[T==0], bins=60, alpha=0.5, density=True, label="Control", color="#4C72B0")
ax1.hist(T_res[T==1], bins=60, alpha=0.5, density=True, label="Treated", color="#DD8452")
ax1.set_xlabel("Treatment Residual (T - E[T|X])")
ax1.set_ylabel("Density")
ax1.set_title("Treatment Residuals After Partialing Out X")
ax1.legend()

# 7b. ATE with 95% CI vs PSM
ax2 = fig.add_subplot(gs[0, 1])
methods = ["Naive\n(raw diff)", "PSM\n(ATT)", "DML\n(ATE)"]
ates    = [-1.952,             -1.767,         ATE]
ci_los  = [-1.952,             -1.817,         ci_lo]
ci_his  = [-1.952,             -1.720,         ci_hi]
colors  = ["#999999",           "#DD8452",      "#4C72B0"]
for i, (m, a, lo, hi, c) in enumerate(zip(methods, ates, ci_los, ci_his, colors)):
    ax2.errorbar(i, a, yerr=[[a-lo], [hi-a]], fmt="o", color=c,
                 capsize=8, capthick=2, markersize=10, linewidth=2)
ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax2.set_xticks(range(3)); ax2.set_xticklabels(methods)
ax2.set_ylabel("Estimated Effect on Rating")
ax2.set_title("ATE Estimates: Naive vs PSM vs DML")
ax2.set_ylim(-2.5, 0.3)

# 7c. HTE coefficients with CI
ax3 = fig.add_subplot(gs[0, 2])
hte_vals  = [hte_ci[c][0] for c in COVARIATES]
hte_lo    = [hte_ci[c][1] for c in COVARIATES]
hte_hi    = [hte_ci[c][2] for c in COVARIATES]
colors_hte = ["#DD8452" if (lo > 0 or hi < 0) else "#999999"
              for lo, hi in zip(hte_lo, hte_hi)]
y_pos = np.arange(len(COVARIATES))
ax3.barh(y_pos, hte_vals, xerr=[np.array(hte_vals)-np.array(hte_lo),
                                  np.array(hte_hi)-np.array(hte_vals)],
         color=colors_hte, capsize=4, alpha=0.85)
ax3.axvline(0, color="black", linestyle="--", linewidth=1)
ax3.set_yticks(y_pos); ax3.set_yticklabels(COVARIATES, fontsize=9)
ax3.set_xlabel("HTE Coefficient")
ax3.set_title("Heterogeneity Modifiers\n(orange = significant)")

# 7d. CATE distribution
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(CATE_hat, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.3)
ax4.axvline(ATE, color="#DD8452", linewidth=2, label=f"ATE={ATE:.3f}")
ax4.set_xlabel("CATE (Individual Treatment Effect)")
ax4.set_ylabel("Frequency")
ax4.set_title("Distribution of CATEs (Linear DML)")
ax4.legend()

# 7e. CATE vs top heterogeneity driver
top_driver = max(COVARIATES, key=lambda c: abs(hte_ci[c][0]))
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(df[top_driver], CATE_hat, alpha=0.05, s=3, color="#4C72B0")
# Lowess-style bin means
bins = pd.qcut(df[top_driver], q=20, duplicates="drop")
bin_means = df.groupby(bins, observed=True)[top_driver].mean()
cate_means = CATE_hat.copy()
cate_bin_means = pd.Series(CATE_hat).groupby(bins).mean()
ax5.plot(bin_means.values, cate_bin_means.values, color="#DD8452",
         linewidth=2.5, label="Bin mean")
ax5.axhline(ATE, color="gray", linestyle="--", linewidth=1, label=f"ATE={ATE:.3f}")
ax5.set_xlabel(top_driver)
ax5.set_ylabel("CATE")
ax5.set_title(f"CATE vs {top_driver}\n(top heterogeneity driver)")
ax5.legend(fontsize=9)

# 7f. Bootstrap distribution of ATE
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(boot_ate, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.3)
ax6.axvline(ATE,      color="#DD8452", linewidth=2, label=f"ATE={ATE:.3f}")
ax6.axvline(ci_ate[0], color="gray",  linestyle="--", linewidth=1.5,
            label=f"95% CI [{ci_ate[0]:.3f}, {ci_ate[1]:.3f}]")
ax6.axvline(ci_ate[1], color="gray",  linestyle="--", linewidth=1.5)
ax6.set_xlabel("ATE (Bootstrap)")
ax6.set_ylabel("Frequency")
ax6.set_title("Bootstrap Distribution of ATE")
ax6.legend(fontsize=9)

plt.suptitle("Step 3: Double Machine Learning (DML)", fontsize=14, y=1.01)
plt.savefig("step3_dml.png", bbox_inches="tight")
plt.show()
print("\nPlot saved to step3_dml.png")

# ── 8. Save CATE for next step ────────────────────────────────────────────────
df.to_csv("data_with_cate_dml.csv", index=False)

# ── 9. Summary ────────────────────────────────────────────────────────────────
print("\n" + "="*44)
print("  DML SUMMARY")
print("="*44)
print(f"  ATE:                {ATE:.3f}")
print(f"  SE (sandwich):      {SE:.4f}")
print(f"  95% CI:             [{ci_lo:.3f}, {ci_hi:.3f}]")
print(f"  Bootstrap 95% CI:   [{ci_ate[0]:.3f}, {ci_ate[1]:.3f}]")
print(f"  p-value:            {p_val:.3e}")
print(f"  Nuisance R2 (Y):    {r2_y:.3f}")
print(f"  Nuisance R2 (T):    {r2_t:.3f}")
print(f"  Oster delta:        {delta:.3f}")
print(f"  Top HTE driver:     {top_driver}")
print("="*44)
