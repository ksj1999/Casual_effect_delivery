# Causal Effect of Delivery Delay on Customer Ratings
**Course:** Causal Inference Final Project  
**Authors:** Runyan Xin, SeongJun Kwon

---

## Research Question

Does late delivery **causally reduce** customer star ratings on an e-commerce platform (Olist/Brazilian e-commerce), and by how much?

- **Treatment** `T` = `is_delivery_late` (binary: 1 if delivered after estimated date)
- **Outcome** `Y` = `Rating` (1–5 stars)
- **Identification assumption:** Unconfoundedness — conditional on observed covariates X, treatment assignment is as good as random

---

## Dataset

| | |
|---|---|
| Source | Olist Brazilian E-Commerce (processed) |
| Raw file | `data.csv` — 114,841 orders × 46 columns |
| Model file | `data_model.csv` — 114,841 × 12 (cleaned, used for all analyses) |

### Covariates used (X)

| Variable | Description |
|---|---|
| `price` | Item price (BRL) |
| `freight_value` | Shipping cost |
| `product_photos_qty` | Number of product photos |
| `month` | Month of purchase |
| `product_category_name_encoded` | Encoded product category |
| `seller_id_encoded` | Encoded seller ID |
| `product_weight_kg` | Product weight |
| `product_size` | Product volume (cm³) |
| `distance_km` | Seller-to-customer distance |
| `seller_avg_rating` | Seller's historical average rating |

### Columns dropped and why

| Column(s) | Reason |
|---|---|
| `order_id`, `customer_id`, `product_id`, etc. | Identifiers — no causal signal |
| All timestamps | Already encoded into `month` |
| Raw geo coordinates | Already encoded into `distance_km` |
| `product_weight_g`, raw dimensions | Superseded by `product_weight_kg`, `product_size` |
| `payment_value` | Collinear with `price + freight_value` |
| `customer_experience` | 99.8% identical to `Rating` — leakage |
| `rainfall` | String region label, not numeric |
| `late_delivery_in_days` | Continuous version of treatment — kept separate |

---

## Step 1 — Exploratory Data Analysis

**Script:** `step1_eda.py` | **Output:** `step1_eda.png`

### Key findings

**Treatment imbalance**
- 93.6% of orders on time (107,473), 6.4% delayed (7,368)
- Severe class imbalance — relevant for propensity score model calibration

**Raw outcome gap**
- Mean rating (on time): **4.206**
- Mean rating (delayed): **2.253**
- Raw difference: **−1.952** — large but confounded

**Top confounders by |SMD|**

| Covariate | |SMD| | Interpretation |
|---|---|---|
| `seller_avg_rating` | 0.434 | Delayed orders come from lower-rated sellers — dominant confounder |
| `distance_km` | 0.280 | Delayed orders travel farther |
| `freight_value` | 0.172 | Higher freight correlates with delay |
| `month` | 0.113 | Seasonality affects delays |

**Missing data:** `product_photos_qty` has 1,607 NAs (~1.4%) — median-imputed before modeling.

---

## Step 2 — Propensity Score Matching (PSM)

**Script:** `step2_psm.py` | **Output:** `step2_psm.png`

### Method
1. Fit logistic regression for `e(X) = P(T=1 | X)` with `class_weight='balanced'`
2. Trim to common support: propensity scores in [0.2035, 0.9829] (drops 63 units, 0.05%)
3. 1:1 nearest-neighbor matching without replacement
4. Caliper = 0.2 × std(logit(e(X))) = 0.108
5. Check covariate balance via |SMD|; estimate ATT on matched sample

### Results

| | |
|---|---|
| Matched pairs | 6,821 |
| Max \|SMD\| after matching | 0.052 (all covariates below threshold 0.10) |
| **ATT** | **−1.767** |
| 95% Bootstrap CI | [−1.817, −1.720] |
| p-value (paired t-test) | ≈ 0 |

**Interpretation:** Late delivery causally reduces a customer's rating by ~1.77 stars on average, among orders that were actually delayed. Confounders explained ~0.19 of the raw 1.95-point gap.

---

## Step 3 — Double Machine Learning (DML)

**Script:** `step3_dml.py` | **Output:** `step3_dml.png`, `data_with_cate_dml.csv`

### Method (Partially Linear Regression)
1. Cross-fit nuisance models (5-fold) using Gradient Boosting:
   - `m(X) = E[Y | X]` → outcome residual Ỹ = Y − m̂(X)
   - `e(X) = P(T=1 | X)` → treatment residual T̃ = T − ê(X)
2. ATE via Frisch-Waugh: θ̂ = (T̃ᵀT̃)⁻¹ T̃ᵀỸ
3. Inference via HC1 sandwich standard error
4. CATE via interactive DML: regress Ỹ on T̃ and T̃ × (X − X̄) — linear heterogeneity

### Results

| | |
|---|---|
| Nuisance R² — Y model | 0.145 |
| Nuisance R² — T model | 0.079 |
| **ATE** | **−1.708** |
| SE (sandwich) | 0.019 |
| 95% CI | [−1.745, −1.671] |
| Bootstrap 95% CI | [−1.710, −1.621] |
| p-value | ≈ 0 |
| **Oster delta** | **54.3** — omitted confounders would need to explain 54× more variance than all observed X to zero out the ATE |

### Heterogeneity (significant HTE modifiers)

| Covariate | Coefficient | Direction |
|---|---|---|
| `seller_avg_rating` | −0.092 | Better sellers → customers more disappointed by delay |
| `distance_km` | −0.0002 | Longer distances amplify the penalty |
| `month` | +0.028 | Later months slightly dampen the penalty |
| `price` | −0.0003 | Higher-priced orders penalized slightly more |

CATE std = 0.151 — modest linear heterogeneity. Causal Forest (next step) will capture nonlinear structure.

---

## Estimates Comparison

| Method | Estimate | CI |
|---|---|---|
| Naive (raw difference) | −1.952 | — |
| PSM (ATT) | −1.767 | [−1.817, −1.720] |
| DML (ATE) | −1.708 | [−1.745, −1.671] |

All three converge in the −1.7 to −1.8 range. The consistency across methods strengthens causal identification.

---

## Remaining Steps

- [ ] **Step 4 — Causal Forest:** Nonparametric CATE estimation; CATE feature importance; subgroup analysis
- [ ] **Step 5 — Sensitivity Analysis:** Rosenbaum bounds (PSM); partial R² / Oster delta (DML)
- [ ] **Step 6 — Policy Design:** Rank orders by CATE magnitude; evaluate targeting policy via quintile analysis

---

## File Index

| File | Description |
|---|---|
| `data.csv` | Raw processed dataset (114,841 × 46) |
| `data_model.csv` | Cleaned model-ready dataset (114,841 × 12) |
| `data_with_cate_dml.csv` | Model dataset + DML CATE estimates |
| `step1_eda.py` | EDA script |
| `step2_psm.py` | PSM script |
| `step3_dml.py` | DML script |
| `slides_figures.py` | Slide-quality EDA + PSM figures |
| `slide_psm_workflow.py` | PSM workflow diagram |
| `step1_eda.png` | EDA diagnostic plots |
| `step2_psm.png` | PSM diagnostic plots |
| `step3_dml.png` | DML diagnostic plots |
| `slide_eda.png` | Slide figure — EDA |
| `slide_psm.png` | Slide figure — PSM results |
| `slide_psm_workflow.png` | Slide figure — PSM workflow diagram |
