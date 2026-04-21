import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'

fig, ax = plt.subplots(figsize=(8, 6.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

b  = "\u2022"   # bullet
sb = "\u25E6"   # sub-bullet (◦)

items = [
    # (x,    y,    text,                                       size,   bold)
    (0.2,  9.5,  r"$\bullet$  Covariates $X$",                 15,     False),
    (0.2,  8.7,  r"$\bullet$  Treatment $T$ = is_delivery_late", 15,  False),
    (0.2,  7.9,  r"$\bullet$  Outcome $Y$ = Rating",          15,     False),
    (0.2,  6.9,  r"$\bullet$  Step 1: estimate propensity score", 15, False),
    (0.7,  6.15, r"$\circ$  $e(X) = P(T = 1 \mid X)$",        14.5,   False),
    (0.7,  5.45, r"$\circ$  logistic regression",              14.5,   False),
    (0.2,  4.55, f"{b}  Step 2: check overlap / trim",        15,     False),
    (0.7,  3.85, r"$\circ$  keep units in common support",    14.5,   False),
    (0.2,  2.95, f"{b}  Step 3: 1:1 nearest-neighbor matching", 15,   False),
    (0.7,  2.25, r"$\circ$  caliper $= 0.2 \times$ std$(\mathrm{logit}(e(X)))$", 14.5, False),
    (0.2,  1.35, r"$\bullet$  Step 4: check balance  $|\mathrm{SMD}| < 0.10$",   15,   False),
    (0.2,  0.45, r"$\bullet$  Step 5: estimate ATT",          15,     False),
]

for x, y, text, size, bold in items:
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, fontsize=size, va="top", color="black",
            fontweight=weight, fontfamily="serif")

# ATT formula on its own line below Step 5
ax.text(0.7, -0.30,
        r"$\widehat{\mathrm{ATT}} = \frac{1}{n_1}\sum_{i:\,T_i=1}(Y_i - Y_{j(i)})$",
        fontsize=14, va="top", color="black", fontfamily="serif")

ax.set_ylim(-1.2, 10)
plt.tight_layout(pad=0.4)
plt.savefig("slide_psm_workflow2.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved slide_psm_workflow2.png")
