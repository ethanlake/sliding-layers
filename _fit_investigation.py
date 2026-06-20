"""Investigate fit quality for xi_er(v, p) across erosion data files."""
import numpy as np
import h5py
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from sliding_plotter import load_jld2_data

files = [
    "data/ising_sliding_erosion_v4.50.jld2",
    "data/ising_sliding_erosion_v4.00.jld2",
    "data/ising_sliding_erosion_v3.50.jld2",
    "data/ising_sliding_erosion_v3.00.jld2",
    "data/ising_sliding_erosion_v2.50.jld2",
    "data/ising_sliding_erosion_v2.00.jld2",
]

records = []  # list of dicts per file: v, p_arr, lc_arr, err_arr
for f in files:
    d = load_jld2_data(f)
    v = float(d["v"])
    p = np.asarray(d["p_values"], dtype=float)
    lc = np.asarray(d["lc_values"], dtype=float)
    err = d.get("lc_stderrs")
    err = np.asarray(err, dtype=float) if err is not None else None
    mask = lc >= 5
    rec = {
        "v": v,
        "p": p[mask],
        "lc": lc[mask],
        "err": err[mask] if err is not None else None,
        "file": f,
    }
    records.append(rec)

# Per-file linear fits: lc = a_v * p + c_v
print("\n=== Per-file fits: lc = a(v) * p + c(v) ===")
print(f"{'v':>6} {'a(v)':>10} {'c(v)':>10} {'RMSE':>10} {'a/v':>10}")
for r in records:
    if r["p"].size < 2:
        continue
    if r["err"] is not None and np.all(np.isfinite(r["err"])) and np.all(r["err"] > 0):
        w = 1.0 / r["err"] ** 2
    else:
        w = np.ones_like(r["p"])
    A = np.column_stack([r["p"], np.ones_like(r["p"])])
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(A * sw[:, None], r["lc"] * sw, rcond=None)
    a_v, c_v = float(beta[0]), float(beta[1])
    pred = A @ beta
    rmse = float(np.sqrt(np.mean((r["lc"] - pred) ** 2)))
    print(f"{r['v']:6.2f} {a_v:10.4f} {c_v:10.4f} {rmse:10.4f} {a_v/r['v']:10.4f}")

# Concatenate
v_all = np.concatenate([np.full(r["p"].size, r["v"]) for r in records])
p_all = np.concatenate([r["p"] for r in records])
y_all = np.concatenate([r["lc"] for r in records])
err_all = np.concatenate([
    r["err"] if r["err"] is not None else np.full(r["p"].size, np.nan)
    for r in records
])
have_err = np.all(np.isfinite(err_all)) and np.all(err_all > 0)
if have_err:
    w_all = 1.0 / err_all ** 2
else:
    w_all = np.ones_like(y_all)


def fit(M, y, w, label):
    sw = np.sqrt(w)
    A = M * sw[:, None]
    rhs = y * sw
    beta, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    pred = M @ beta
    resid = y - pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    # weighted chi^2 / dof
    chi2 = float(np.sum(w * resid ** 2))
    dof = max(1, y.size - beta.size)
    chi2_red = chi2 / dof
    print(f"\n--- {label} ---")
    print(f"  params = {beta}")
    print(f"  RMSE   = {rmse:.4f}")
    print(f"  chi2/dof = {chi2_red:.4f}")
    # per-v residuals
    for r in records:
        sel = (v_all == r["v"])
        rr = (y_all - pred)[sel]
        print(f"   v={r['v']:.2f}: resid mean={rr.mean():+.3f}  max|r|={np.max(np.abs(rr)):.3f}")
    return beta, pred


# Model 1: lc = s * v * p + b
M1 = np.column_stack([v_all * p_all, np.ones_like(p_all)])
fit(M1, y_all, w_all, "Model 1: lc = s*v*p + b")

# Model 2: lc = s * (v - v0) * p + b   (nonlinear; reparam: s*v*p - s*v0*p + b)
M2 = np.column_stack([v_all * p_all, p_all, np.ones_like(p_all)])
beta2, pred2 = fit(M2, y_all, w_all, "Model 2: lc = s*v*p - s*v0*p + b (linear in 3 params; v0 = -beta2[1]/beta2[0])")
print(f"  → implied v0 = {-beta2[1]/beta2[0]:.4f}")

# Model 3: lc = s*v*p + b(v) where b(v) = b0 + b1*v
M3 = np.column_stack([v_all * p_all, np.ones_like(p_all), v_all])
fit(M3, y_all, w_all, "Model 3: lc = s*v*p + b0 + b1*v")

# Model 4: lc = s*(v-v0)*p + b(v) = s*v*p - s*v0*p + b0 + b1*v
M4 = np.column_stack([v_all * p_all, p_all, np.ones_like(p_all), v_all])
beta4, _ = fit(M4, y_all, w_all, "Model 4: lc = s*v*p + c*p + b0 + b1*v")
print(f"  → implied v0 = {-beta4[1]/beta4[0]:.4f}")

# Model 5: lc = a(v)*p + c(v) freely (i.e. allow both intercept and slope to vary with v).
# Fit a(v) and c(v) per file (already did above), then check whether a(v) and c(v) are linear in v.
print("\n=== How does a(v), c(v) scale with v? ===")
av = []
cv = []
vs = []
for r in records:
    if r["p"].size < 2:
        continue
    if r["err"] is not None and np.all(np.isfinite(r["err"])) and np.all(r["err"] > 0):
        w = 1.0 / r["err"] ** 2
    else:
        w = np.ones_like(r["p"])
    A = np.column_stack([r["p"], np.ones_like(r["p"])])
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(A * sw[:, None], r["lc"] * sw, rcond=None)
    av.append(float(beta[0]))
    cv.append(float(beta[1]))
    vs.append(r["v"])
av = np.array(av)
cv = np.array(cv)
vs = np.array(vs)
# Linear fit a(v) = s*v + a0
A1 = np.column_stack([vs, np.ones_like(vs)])
sb, *_ = np.linalg.lstsq(A1, av, rcond=None)
print(f"a(v) ≈ {sb[0]:.4f} * v + ({sb[1]:+.4f})")
print(f"   → implied v0 (a(v)=0) = {-sb[1]/sb[0]:.4f}")
# Linear fit c(v) = b1*v + b0
cb, *_ = np.linalg.lstsq(A1, cv, rcond=None)
print(f"c(v) ≈ {cb[0]:.4f} * v + ({cb[1]:+.4f})")
