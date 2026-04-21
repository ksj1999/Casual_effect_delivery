import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

# Header background
header = mpatches.FancyBboxPatch((0, 8.5), 10, 1.5,
    boxstyle="square,pad=0", linewidth=0,
    facecolor="#D8D4E8")
ax.add_patch(header)

# Body background
body = mpatches.FancyBboxPatch((0, 0), 10, 8.5,
    boxstyle="square,pad=0", linewidth=0,
    facecolor="#F7F7FC")
ax.add_patch(body)

# Left accent bar
accent = mpatches.FancyBboxPatch((0, 0), 0.22, 10,
    boxstyle="square,pad=0", linewidth=0,
    facecolor="#7B6BAE")
ax.add_patch(accent)

# Title
ax.text(0.45, 9.2, "Workflow:", fontsize=18, fontweight="bold",
        color="#3B3060", va="center", fontfamily="sans-serif")

kw = dict(fontsize=13.5, va="top", fontfamily="sans-serif", color="#1a1a1a")
mkw = dict(fontsize=13.5, va="top", color="#1a1a1a")

bullet = "\u2022"
indent_bullet = "    \u2022"

lines = [
    # (x,    y,    text,                        is_math)
    (0.55,  8.05, f"{bullet}  Covariates  ",    False),
    (0.55,  7.35, f"{bullet}  Treatment  ",     False),
    (0.55,  6.65, f"{bullet}  Outcome  ",       False),
    (0.55,  5.80, f"{bullet}  Step 1: Estimate propensity score", False),
    (0.55,  5.10, f"{indent_bullet}  Fit logistic regression: ", False),
    (0.55,  4.30, f"{bullet}  Step 2: Check overlap", False),
    (0.55,  3.55, f"{indent_bullet}  Trim units outside common support", False),
    (0.55,  2.80, f"{bullet}  Step 3: 1:1 nearest-neighbor matching", False),
    (0.55,  2.05, f"{indent_bullet}  Caliper = ", False),
    (0.55,  1.25, f"{bullet}  Step 4: Check balance  (|SMD| < 0.10)", False),
    (0.55,  0.50, f"{bullet}  Step 5: Estimate ATT", False),
]

for x, y, text, _ in lines:
    ax.text(x, y, text, **kw)

# Math annotations inline
ax.text(3.62, 8.05, "$X$",                              fontsize=14, va="top", color="#1a1a1a")
ax.text(3.58, 7.35, "$T$ = is_delivery_late",           fontsize=13.5, va="top", color="#1a1a1a")
ax.text(3.35, 6.65, "$Y$ = Rating",                     fontsize=13.5, va="top", color="#1a1a1a")
ax.text(4.62, 5.10, r"$e(X) = P(T=1\mid X)$",          fontsize=13.5, va="top", color="#1a1a1a")
ax.text(2.02, 2.05, r"$0.2 \times \mathrm{std}(\mathrm{logit}(e(X)))$",
        fontsize=13.5, va="top", color="#1a1a1a")

# ATT formula on Step 5 line
ax.text(4.12, 0.50, r"$\widehat{\mathrm{ATT}} = \frac{1}{n_1}"
        r"\sum_{i:T_i=1}(Y_i - Y_{j(i)})$",
        fontsize=12.5, va="top", color="#1a1a1a")

plt.tight_layout(pad=0)
plt.savefig("slide_psm_workflow.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved slide_psm_workflow.png")
