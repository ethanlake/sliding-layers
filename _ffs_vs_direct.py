"""Compare FFS-estimated mixing times against direct simulation at v=2.5.

Reads JLD2 files written by simulation_driver.jl for both modes at matched
M_threshold=0.75. Plots log10(tau) vs p, overlaying direct mixing (lines)
and FFS (markers with error bars), one color per L.
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

Ls = [500, 1000, 2000]
mix_pattern = "ising_sliding_mixing_L{L}_v2.50_M0.75_p2to2.6.jld2"
ffs_pattern = "ising_sliding_ffs_L{L}_v2.50_p2.00to2.80_M0.75_p2to2.8.jld2"
ffs2_pattern = "ising_sliding_ffs_L{L}_v2.50_p2.00to2.80_M0.75_p2to2.8_nobailout.jld2"


def read_jld2(path):
    with h5py.File(path, "r") as f:
        return {k: f[k][()] for k in f.keys()}


fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax = axes[0]
colors = {500: "C0", 1000: "C1", 2000: "C2"}

for L in Ls:
    mix_path = DATA_DIR / mix_pattern.format(L=L)
    ffs_path = DATA_DIR / ffs_pattern.format(L=L)
    ffs2_path = DATA_DIR / ffs2_pattern.format(L=L)
    if mix_path.exists():
        d = read_jld2(mix_path)
        p_mix = np.asarray(d["p_values"])
        tau_mix = np.asarray(d["mean_mixing_times"])
        max_t = float(d["max_time"])
        capped = tau_mix > 0.8 * max_t
        ax.plot(p_mix[~capped], np.log10(tau_mix[~capped]), "-o",
                color=colors[L], label=f"direct, L={L}", lw=2, ms=7)
        if capped.any():
            ax.plot(p_mix[capped], np.log10(tau_mix[capped]), "x",
                    color=colors[L], ms=10, mew=2)
    if ffs_path.exists():
        d = read_jld2(ffs_path)
        p_ffs = np.asarray(d["p_values"])
        log_tau = np.asarray(d["log_mixing_times"])
        std = np.asarray(d["log_mixing_times_std"])
        ax.errorbar(p_ffs - 0.005, log_tau, yerr=std, fmt="s", color=colors[L],
                    mfc="white", mew=1.5, ms=7, capsize=3,
                    label=f"FFS (orig), L={L}", alpha=0.6)
    if ffs2_path.exists():
        d = read_jld2(ffs2_path)
        p_ffs = np.asarray(d["p_values"])
        log_tau = np.asarray(d["log_mixing_times"])
        std = np.asarray(d["log_mixing_times_std"])
        ax.errorbar(p_ffs + 0.005, log_tau, yerr=std, fmt="D", color=colors[L],
                    mew=1.5, ms=8, capsize=4,
                    label=f"FFS (patched), L={L}")

ax.set_xlabel(r"$p = e^{\beta J}$")
ax.set_ylabel(r"$\log_{10} \tau_{\rm mem}$")
ax.set_title(r"$v=2.5$, $M_{\rm thr}=0.75$")
ax.legend(loc="upper left", fontsize=7, ncol=3)
ax.grid(alpha=0.3)

# Second panel: ratios
ax2 = axes[1]
for L in Ls:
    mix_path = DATA_DIR / mix_pattern.format(L=L)
    if not mix_path.exists():
        continue
    dm = read_jld2(mix_path)
    p_mix = np.asarray(dm["p_values"])
    tau_mix = np.asarray(dm["mean_mixing_times"])
    max_t = float(dm["max_time"])
    for label, pat, fmt, alpha in (("orig", ffs_pattern, "s", 0.55),
                                   ("patched", ffs2_pattern, "D", 1.0)):
        fp = DATA_DIR / pat.format(L=L)
        if not fp.exists():
            continue
        df = read_jld2(fp)
        p_ffs = np.asarray(df["p_values"])
        tau_ffs = 10.0 ** np.asarray(df["log_mixing_times"])
        std_ffs = np.asarray(df["log_mixing_times_std"])
        ratios = []; p_match = []; rel_err = []
        for i, p in enumerate(p_mix):
            j = np.where(np.abs(p_ffs - p) < 1e-6)[0]
            if len(j) == 0 or tau_mix[i] > 0.8 * max_t:
                continue
            r = tau_ffs[j[0]] / tau_mix[i]
            ratios.append(r); p_match.append(p); rel_err.append(std_ffs[j[0]] * np.log(10))
        if ratios:
            p_match = np.asarray(p_match); ratios = np.asarray(ratios); rel_err = np.asarray(rel_err)
            shift = -0.005 if label == "orig" else 0.005
            ax2.errorbar(p_match + shift, ratios, yerr=ratios * rel_err,
                         fmt=fmt + "-", color=colors[L], alpha=alpha, ms=7, lw=1.5,
                         capsize=3,
                         label=f"L={L} ({label})",
                         mfc=("white" if label == "orig" else colors[L]))

ax2.axhline(1.0, color="black", lw=1.2, linestyle="--", alpha=0.7)
ax2.set_yscale("log")
ax2.set_xlabel(r"$p = e^{\beta J}$")
ax2.set_ylabel(r"$\tau_{\rm FFS} / \tau_{\rm direct}$")
ax2.set_title("Ratio to ground truth (1.0 = perfect)")
ax2.legend(fontsize=8, ncol=2, loc="lower left")
ax2.grid(alpha=0.3, which="both")

plt.tight_layout()
out = DATA_DIR.parent / "ffs_vs_direct_v2.5_patched.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved: {out}")

# Print summary table
print()
print(f"{'p':>5} {'L':>5} {'direct':>13} {'FFS orig':>13} {'r_orig':>7} "
      f"{'FFS patch':>13} {'r_patch':>7}")
for L in Ls:
    mix_path = DATA_DIR / mix_pattern.format(L=L)
    ffs_path = DATA_DIR / ffs_pattern.format(L=L)
    ffs2_path = DATA_DIR / ffs2_pattern.format(L=L)
    if not mix_path.exists():
        continue
    dm = read_jld2(mix_path)
    d_orig = read_jld2(ffs_path) if ffs_path.exists() else None
    d_patch = read_jld2(ffs2_path) if ffs2_path.exists() else None
    p_mix = np.asarray(dm["p_values"])
    tau_mix = np.asarray(dm["mean_mixing_times"])
    max_t = float(dm["max_time"])
    # use FFS p grid (covers more points)
    ref = d_patch if d_patch is not None else d_orig
    p_grid = np.asarray(ref["p_values"])
    for p in p_grid:
        j_mix = np.where(np.abs(p_mix - p) < 1e-6)[0]
        d_val_str, r_orig_str, r_patch_str = "—", "—", "—"
        d_num = None
        if len(j_mix):
            d_num = tau_mix[j_mix[0]]
            if d_num > 0.8 * max_t:
                d_val_str = f">{d_num:.1e} cap"
                d_num = None
            else:
                d_val_str = f"{d_num:.3e}"
        for src, ratio_holder in ((d_orig, "orig"), (d_patch, "patch")):
            if src is None:
                continue
            j = np.where(np.abs(np.asarray(src["p_values"]) - p) < 1e-6)[0]
            if not len(j):
                continue
            tau_f = 10.0 ** np.asarray(src["log_mixing_times"])[j[0]]
            if d_num is not None:
                r_str = f"{tau_f / d_num:.2f}"
            else:
                r_str = "—"
            if ratio_holder == "orig":
                fffs_orig_str = f"{tau_f:.3e}"
                r_orig_str = r_str
            else:
                fffs_patch_str = f"{tau_f:.3e}"
                r_patch_str = r_str
        if d_orig is None:
            fffs_orig_str = "—"
        if d_patch is None:
            fffs_patch_str = "—"
        print(f"{p:5.2f} {L:5d} {d_val_str:>13} {fffs_orig_str:>13} {r_orig_str:>7} "
              f"{fffs_patch_str:>13} {r_patch_str:>7}")
