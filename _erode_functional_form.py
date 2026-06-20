"""Determine the functional form of 1 - P_er(ell, ell_er) from the main
high-stats run, and overlay the two L_sys-perturbed runs as a sanity check
that P_er does not depend on system size.

Reads:
- data/ising_sliding_erosion_v3.50_p3to8_4pts_n40k.jld2     (main, 40k samples)
- data/ising_sliding_erosion_v3.50_p3to8_4pts_Lsys0.5.jld2  (L_sys * 0.5, 10k samples)
- data/ising_sliding_erosion_v3.50_p3to8_4pts_Lsys2.jld2    (L_sys * 2.0, 10k samples)

Tests three candidate forms:
- exp:        1 - P_er ~ exp[-alpha * (1 - x)],    where x = ell / ell_er
- gaussian:   1 - P_er ~ exp[-alpha * (1 - x)^2]
- power:      1 - P_er ~ (1 - x)^beta
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def load(p):
    with h5py.File(p, "r") as f:
        return dict(
            p_values=np.asarray(f["p_values"]),
            lc_values=np.asarray(f["lc_values"]),
            l_matrix=np.asarray(f["erode_l_values"]).T,    # (n_l, n_p)
            prob_matrix=np.asarray(f["erode_probs"]).T,
            thresh=float(f["thresh_prob"][()]),
            num_trials=int(f["num_trials"][()]),
            v=float(f["v"][()]),
            t_evolve_factor=(float(f["t_evolve_factor"][()])
                             if "t_evolve_factor" in f else 2.0),
            L_sys_factor=(float(f["L_sys_factor"][()])
                          if "L_sys_factor" in f else 1.0),
        )


def extract_xy(d, p_idx, min_count=20, x_min=None):
    """Return (x = ell/ell_er, y = 1-p_shrink, ye) for one p column. Mask out
    points whose y is bounded by zero-count noise (i.e., not all trials shrunk).
    If x_min given, also restrict to x >= x_min (drops the small-l artifact)."""
    ls = d["l_matrix"][:, p_idx].astype(float)
    ps = d["prob_matrix"][:, p_idx]
    keep = ls > 0
    ls, ps = ls[keep], ps[keep]
    n = d["num_trials"]
    y = 1.0 - ps
    ye = np.sqrt(np.clip(ps * (1 - ps) / n, 1e-30, None))
    lc = d["lc_values"][p_idx]
    x = ls / lc
    nfail = y * n
    mask = (nfail >= min_count) & (y > 0)
    if x_min is not None:
        mask &= (x >= x_min)
    return x[mask], y[mask], ye[mask]


main = load(DATA_DIR / "ising_sliding_erosion_v3.50_p3to8_4pts_n40k.jld2")
l05 = load(DATA_DIR / "ising_sliding_erosion_v3.50_p3to8_4pts_Lsys0.5.jld2")
l20 = load(DATA_DIR / "ising_sliding_erosion_v3.50_p3to8_4pts_Lsys2.jld2")

p_values = main["p_values"]
n_p = len(p_values)
cmap = plt.cm.viridis(np.linspace(0.1, 0.9, n_p))

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ax_lin, ax_exp = axes[0]
ax_gauss, ax_pow = axes[1]

print(f"main run: v={main['v']}, num_trials={main['num_trials']}, "
      f"t_evolve_factor={main['t_evolve_factor']}, L_sys_factor={main['L_sys_factor']}")
print(f"lc (main):  {main['lc_values']}")
print(f"lc (L×0.5): {l05['lc_values']}")
print(f"lc (L×2.0): {l20['lc_values']}")
print()

for i, p in enumerate(p_values):
    x_m, y_m, ye_m = extract_xy(main, i)
    if len(x_m) == 0:
        continue
    label = f"p = {p:.2f}, ℓ_er = {main['lc_values'][i]}"
    # Plot main on all panels
    for ax, xtransform, name in [
        (ax_lin, lambda x: x, "x"),
        (ax_exp, lambda x: 1 - x, "1-x"),
        (ax_gauss, lambda x: (1 - x)**2, "(1-x)^2"),
        (ax_pow, lambda x: 1 - x, "1-x (log-log)"),
    ]:
        ax.errorbar(xtransform(x_m), y_m, yerr=ye_m, fmt="o-",
                    color=cmap[i], ms=5, lw=1.5, capsize=2, label=label)

for ax, title, xlab in [
    (ax_lin,   "linear ℓ/ℓ_er (default sliding_plotter view)", r"$\ell / \ell_{\rm er}$"),
    (ax_exp,   r"exp form: log $(1-P_{\rm er})$ vs $1-\ell/\ell_{\rm er}$",  r"$1 - \ell/\ell_{\rm er}$"),
    (ax_gauss, r"gaussian form: log $(1-P_{\rm er})$ vs $(1-\ell/\ell_{\rm er})^2$",  r"$(1 - \ell/\ell_{\rm er})^2$"),
    (ax_pow,   r"power-law form: log–log",  r"$1 - \ell/\ell_{\rm er}$"),
]:
    ax.set_ylabel(r"$1 - P_{\rm er}(\ell)$")
    ax.set_xlabel(xlab)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3, which="both")
    if ax is ax_lin:
        pass
    elif ax is ax_pow:
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        ax.set_yscale("log")

plt.tight_layout()
out_form = DATA_DIR.parent / "erode_functional_form.png"
plt.savefig(out_form, dpi=140, bbox_inches="tight")
print(f"saved: {out_form}")

# === Fits: restrict to clean regime past the small-l artifact (x ≥ 0.5)
# and compare exponential vs gaussian vs power-law.
X_MIN = 0.5
print()
print(f"Fits restricted to ℓ/ℓ_er ≥ {X_MIN}")
print(f"{'p':>5}  {'ℓ_er':>5}  {'#pts':>4}  {'exp α':>8}  {'exp R²':>7}  "
      f"{'gauss α':>9}  {'gauss R²':>9}  {'pow β':>6}  {'pow R²':>7}")
fits = {}
for i, p in enumerate(p_values):
    x, y, _ = extract_xy(main, i, x_min=X_MIN)
    if len(x) < 4:
        continue
    u = 1 - x
    logy = np.log(y)
    def linfit(X, target):
        coef, *_ = np.linalg.lstsq(X, target, rcond=None)
        pred = X @ coef
        ss_res = np.sum((target - pred)**2)
        ss_tot = np.sum((target - target.mean())**2)
        return coef, 1 - ss_res / max(ss_tot, 1e-30)
    (a1, b1), R2_1 = linfit(np.vstack([np.ones_like(u), u]).T, logy)
    (a2, b2), R2_2 = linfit(np.vstack([np.ones_like(u), u**2]).T, logy)
    pos = u > 0
    if pos.sum() >= 2:
        (a3, b3), R2_3 = linfit(
            np.vstack([np.ones_like(u[pos]), np.log(u[pos])]).T, logy[pos])
    else:
        a3, b3, R2_3 = 0.0, 0.0, float("nan")
    fits[i] = dict(exp=(a1, -b1, R2_1), gauss=(a2, -b2, R2_2), power=(a3, b3, R2_3))
    print(f"{p:5.2f}  {main['lc_values'][i]:5d}  {len(x):4d}  {-b1:8.3f}  {R2_1:7.3f}  "
          f"{-b2:9.3f}  {R2_2:9.3f}  {b3:6.3f}  {R2_3:7.3f}")

# Re-plot the exp panel with fit overlays
fig3, ax3 = plt.subplots(figsize=(7, 5.5))
for i, p in enumerate(p_values):
    x_all, y_all, ye_all = extract_xy(main, i)  # full curve
    ax3.errorbar(1 - x_all, y_all, yerr=ye_all, fmt="o", color=cmap[i],
                 ms=5, capsize=2, alpha=0.7,
                 label=f"p = {p:.2f}, ℓ_er = {main['lc_values'][i]}")
    if i in fits:
        a, alpha, R2 = fits[i]["exp"]
        u_fit = np.linspace(0, 1 - X_MIN, 100)
        ax3.plot(u_fit, np.exp(a - alpha * u_fit), "--", color=cmap[i],
                 lw=1.2, alpha=0.9)
ax3.set_yscale("log")
ax3.set_xlabel(r"$1 - \ell/\ell_{\rm er}$")
ax3.set_ylabel(r"$1 - P_{\rm er}(\ell)$")
ax3.set_title(rf"Exponential fits restricted to $\ell/\ell_{{\rm er}} \geq {X_MIN}$" "\n"
              "dashed = $\\exp(a - \\alpha(1-\\ell/\\ell_{\\rm er}))$")
ax3.legend(loc="upper left", fontsize=9)
ax3.grid(alpha=0.3, which="both")
ax3.axvline(1 - X_MIN, color="gray", ls=":", lw=1)
ax3.text(1 - X_MIN, ax3.get_ylim()[1] * 0.7, " fit cutoff",
         color="gray", fontsize=8, ha="left", va="top")
plt.tight_layout()
out_fit = DATA_DIR.parent / "erode_exp_fits.png"
plt.savefig(out_fit, dpi=140, bbox_inches="tight")
print(f"saved: {out_fit}")

# === Sanity check panel: overlay main + L0.5 + L2 at each p
fig2, axes2 = plt.subplots(1, n_p, figsize=(4 * n_p, 4), sharey=True)
if n_p == 1:
    axes2 = [axes2]
for i, p in enumerate(p_values):
    ax = axes2[i]
    for d, label, ls, ms in [(main, "main (L×1, 40k)", "-", 5),
                              (l05, "L×0.5 (10k)", "--", 4),
                              (l20, "L×2 (10k)", ":", 4)]:
        # match this p value in this dataset
        j = np.where(np.abs(d["p_values"] - p) < 1e-6)[0]
        if len(j) == 0:
            continue
        x, y, ye = extract_xy(d, j[0])
        ax.errorbar(x, y, yerr=ye, fmt="o" + ls, ms=ms, lw=1.5, capsize=2,
                    label=f"{label}, ℓ_er={d['lc_values'][j[0]]}")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\ell / \ell_{\rm er}$")
    if i == 0:
        ax.set_ylabel(r"$1 - P_{\rm er}(\ell)$")
    ax.set_title(f"p = {p:.2f}")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")

plt.tight_layout()
out_sanity = DATA_DIR.parent / "erode_L_sys_sanity.png"
plt.savefig(out_sanity, dpi=140, bbox_inches="tight")
print(f"saved: {out_sanity}")
