"""Plot p_shrink(l) curves at v=3.5 for 5 p values, illustrating how
tightly the erosion-vs-l transition is concentrated about l_er.

Reads data/ising_sliding_erosion_v3.50_p3to8_erodevsl.jld2.
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
import sys
infile = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    DATA_DIR / "ising_sliding_erosion_v3.50_p3to8_erodevsl.jld2"

with h5py.File(infile, "r") as f:
    p_values = np.asarray(f["p_values"])
    lc_values = np.asarray(f["lc_values"])
    lc_stderrs = np.asarray(f["lc_stderrs"])
    # h5py reads Julia column-major arrays as transposed; restore (n_l, n_p)
    l_matrix = np.asarray(f["erode_l_values"]).T     # (n_l, n_p)
    prob_matrix = np.asarray(f["erode_probs"]).T     # (n_l, n_p)
    thresh = float(f["thresh_prob"][()])
    num_trials = int(f["num_trials"][()])
    v = float(f["v"][()])

n_l, n_p = l_matrix.shape
print(f"v = {v}, num_trials = {num_trials}, thresh = {thresh}")
print(f"p values: {p_values}")
print(f"lc values: {lc_values}")

cmap = plt.cm.viridis(np.linspace(0.1, 0.9, n_p))
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: raw p_shrink(l) curves, one per p
ax = axes[0]
for i, p in enumerate(p_values):
    ls = l_matrix[:, i]
    ps = prob_matrix[:, i]
    keep = ls > 0
    se = np.sqrt(ps[keep] * (1 - ps[keep]) / num_trials)
    ax.errorbar(ls[keep], ps[keep], yerr=se, fmt="o-", color=cmap[i],
                label=f"p = {p:.2f}, ℓ_er = {lc_values[i]}", ms=5, lw=1.5,
                capsize=2)
    ax.axvline(lc_values[i], color=cmap[i], ls=":", alpha=0.5, lw=1)
ax.axhline(thresh, color="black", ls="--", lw=1, alpha=0.6, label=f"thresh = {thresh}")
ax.set_xlabel(r"domain size $\ell$")
ax.set_ylabel(r"$p_{\rm shrink}(\ell)$")
ax.set_title(rf"$v={v}$, $p_{{\rm shrink}}$ vs $\ell$ ({num_trials} trials each)")
ax.legend(loc="lower left", fontsize=8)
ax.grid(alpha=0.3)

# Right: rescale x by lc to compare shapes (concentration check)
ax2 = axes[1]
for i, p in enumerate(p_values):
    ls = l_matrix[:, i]
    ps = prob_matrix[:, i]
    keep = ls > 0
    x = ls[keep] / lc_values[i]
    se = np.sqrt(ps[keep] * (1 - ps[keep]) / num_trials)
    ax2.errorbar(x, ps[keep], yerr=se, fmt="o-", color=cmap[i],
                 label=f"p = {p:.2f}", ms=5, lw=1.5, capsize=2)
ax2.axhline(thresh, color="black", ls="--", lw=1, alpha=0.6)
ax2.axvline(1.0, color="gray", ls=":", lw=1, alpha=0.5)
ax2.set_xlabel(r"$\ell / \ell_{\rm er}$")
ax2.set_ylabel(r"$p_{\rm shrink}(\ell)$")
ax2.set_title("Rescaled: do transitions get sharper with $p$?")
ax2.legend(loc="lower left", fontsize=8)
ax2.grid(alpha=0.3)

plt.tight_layout()
out = DATA_DIR.parent / "erode_concentration_v3.5.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved: {out}")

# Per-curve "transition width" diagnostic: width of region where
# p_shrink drops from 0.95 to 0.75 (the most informative band)
print()
print(f"{'p':>5} {'lc':>4} {'l(p=0.95)':>10} {'l(p=0.75)':>10} {'width':>7} {'width/lc':>9}")
for i, p in enumerate(p_values):
    ls = l_matrix[:, i].astype(float)
    ps = prob_matrix[:, i]
    keep = ls > 0
    ls, ps = ls[keep], ps[keep]
    # Sort by l (should already be sorted)
    order = np.argsort(ls)
    ls = ls[order]; ps = ps[order]
    # Linear interpolation for crossing levels (interpolate in p, monotone decreasing)
    def l_at(p_target):
        # find first i where ps[i] < p_target (assuming ps decreasing)
        idx = np.where(ps < p_target)[0]
        if len(idx) == 0:
            return np.nan
        j = idx[0]
        if j == 0:
            return ls[0]
        # interp between (ls[j-1], ps[j-1]) and (ls[j], ps[j])
        denom = ps[j-1] - ps[j]
        if denom == 0:
            return ls[j]
        frac = (ps[j-1] - p_target) / denom
        return ls[j-1] + frac * (ls[j] - ls[j-1])
    l95 = l_at(0.95)
    l75 = l_at(0.75)
    width = l75 - l95
    rel = width / lc_values[i]
    print(f"{p:5.2f} {lc_values[i]:4d} {l95:10.2f} {l75:10.2f} {width:7.2f} {rel:9.3f}")
