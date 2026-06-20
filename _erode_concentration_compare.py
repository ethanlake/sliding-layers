"""Compare p_shrink(l) curves at v=3.5 between the original (T_evolve = 2l)
and 3x-longer (T_evolve = 6l) trials. Plots both side-by-side, rescaled
by lc, plus the transition widths."""

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
files = {
    "T = 2ℓ (original)": DATA_DIR / "ising_sliding_erosion_v3.50_p3to8_erodevsl.jld2",
    "T = 6ℓ (3× longer)": DATA_DIR / "ising_sliding_erosion_v3.50_p3to8_erodevsl_t6.jld2",
}


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
        )


datasets = {label: load(p) for label, p in files.items()}

n_p = len(next(iter(datasets.values()))["p_values"])
cmap = plt.cm.viridis(np.linspace(0.1, 0.9, n_p))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for ax, (label, d) in zip(axes, datasets.items()):
    for i, p in enumerate(d["p_values"]):
        ls = d["l_matrix"][:, i]
        ps = d["prob_matrix"][:, i]
        keep = ls > 0
        x = ls[keep] / d["lc_values"][i]
        se = np.sqrt(ps[keep] * (1 - ps[keep]) / d["num_trials"])
        ax.errorbar(x, ps[keep], yerr=se, fmt="o-", color=cmap[i],
                    label=f"p = {p:.2f}, ℓ_er = {d['lc_values'][i]}",
                    ms=5, lw=1.5, capsize=2)
    ax.axhline(d["thresh"], color="black", ls="--", lw=1, alpha=0.6)
    ax.axvline(1.0, color="gray", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel(r"$\ell / \ell_{\rm er}$")
    ax.set_ylabel(r"$p_{\rm shrink}(\ell)$")
    ax.set_title(label)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0.7, 1.01)

plt.tight_layout()
out = DATA_DIR.parent / "erode_concentration_v3.5_t2vst6.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved: {out}")


def l_at(ls, ps, p_target):
    order = np.argsort(ls)
    ls, ps = ls[order], ps[order]
    idx = np.where(ps < p_target)[0]
    if len(idx) == 0:
        return np.nan
    j = idx[0]
    if j == 0:
        return float(ls[0])
    denom = ps[j-1] - ps[j]
    if denom <= 0:
        return float(ls[j])
    frac = (ps[j-1] - p_target) / denom
    return float(ls[j-1] + frac * (ls[j] - ls[j-1]))


print(f"\n{'p':>5}  {'ℓ_er(2t)':>9} {'width/ℓ_er (2t)':>16}  {'ℓ_er(6t)':>9} {'width/ℓ_er (6t)':>16}")
d2 = datasets["T = 2ℓ (original)"]
d6 = datasets["T = 6ℓ (3× longer)"]
for i, p in enumerate(d2["p_values"]):
    out_row = [f"{p:5.2f}"]
    for d in (d2, d6):
        ls = d["l_matrix"][:, i].astype(float)
        ps = d["prob_matrix"][:, i]
        keep = ls > 0
        ls, ps = ls[keep], ps[keep]
        lc = d["lc_values"][i]
        l95 = l_at(ls, ps, 0.95)
        l75 = l_at(ls, ps, 0.75)
        w = l75 - l95 if (np.isfinite(l95) and np.isfinite(l75)) else np.nan
        rel = w / lc if np.isfinite(w) else np.nan
        out_row.append(f"{lc:9d}")
        out_row.append(f"{rel:16.3f}" if np.isfinite(rel) else f"{'—':>16}")
    print("  ".join(out_row))
