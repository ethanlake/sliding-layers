#=
Parameter study for FFS estimates of t_mem at (L=500, v=3, p in {2,3,4}).
For each parameter setting we run N independent ffs_single_run calls and
report the mean log_tau, empirical std across runs, and the binomial-noise
estimate computed from the per-run variances.

The point: we want to know which knobs actually reduce the spread in
log_tau, and whether the saved log_mixing_times_std (= sqrt(var/n)) honestly
reflects the true scatter.
=#

using Pkg; Pkg.activate(".")
using SlidingLayers
using Statistics, Printf

import SlidingLayers: measure_metastable_magnetization, ffs_single_run

const L         = 500
const v         = 3.0
const h         = 0.0
const max_t     = 1_000_000

# Cache lambda_0 per p so we don't re-measure metastable state each time.
const lam0_cache = Dict{Float64, Float64}()
function get_lambda0(p::Float64)
    if !haskey(lam0_cache, p)
        beta = log(p)
        (m_meta, m_std) = measure_metastable_magnetization(L, beta, h, v)
        lam0_cache[p] = m_meta - 2.5 * m_std
        @printf("lambda_0 for p=%.2f: %.4f (m*=%.4f ± %.4f)\n", p, lam0_cache[p], m_meta, m_std)
    end
    return lam0_cache[p]
end

"""
Run N independent ffs_single_run calls and report stats.
"""
function trial(label::String, p::Float64; n_per_run::Int, M_thr::Float64,
               tcp::Float64, N::Int=12)
    beta = log(p)
    lam0 = get_lambda0(p)
    logs = Float64[]
    vars = Float64[]
    n_fail = 0
    t0 = time()
    for i in 1:N
        (lt, vl, phi, failed) = ffs_single_run(L, beta, h, v, lam0, M_thr,
                                                n_per_run, max_t, tcp;
                                                single_layer=false, verbose=false)
        if failed
            n_fail += 1
        else
            push!(logs, lt)
            push!(vars, vl)
        end
    end
    elapsed = time() - t0
    n_ok = length(logs)
    if n_ok == 0
        @printf("  %-50s  ALL FAILED (took %.1fs)\n", label, elapsed)
        return
    end
    m  = mean(logs)
    se = n_ok > 1 ? std(logs) : NaN
    bin = sqrt(mean(vars) / n_ok) / log(10)  # internal stderr (per-N) in log10
    @printf("  %-50s  log_tau = %.3f, empirical_std = %.3f, binomial_stderr = %.3f, n_ok = %d/%d, %.1fs\n",
            label, m, se, bin, n_ok, N, elapsed)
end

p = 3.0  # workhorse point

println("=== Sweep n_configs_per_run at p=$p (defaults M=0.75, tcp=0.15) ===")
for nper in [40, 100, 200, 400]
    trial(@sprintf("n_per_run=%d", nper), p;
          n_per_run=nper, M_thr=0.75, tcp=0.15, N=12)
end

println("\n=== Sweep target_crossing_prob at p=$p (n_per_run=200, M=0.75) ===")
for tcp in [0.10, 0.25, 0.40]
    trial(@sprintf("tcp=%.2f", tcp), p;
          n_per_run=200, M_thr=0.75, tcp=tcp, N=10)
end

println("\n=== Sweep M_threshold at p=$p (n_per_run=200, tcp=0.15) ===")
for Mth in [0.5, 0.25]
    trial(@sprintf("M_thr=%.2f", Mth), p;
          n_per_run=200, M_thr=Mth, tcp=0.15, N=10)
end

# Sanity at higher p, just one fat setting
println("\n=== At p=4 (n_per_run=200, M=0.75, tcp=0.15) — sanity ===")
trial("p=4, n_per_run=200", 4.0;
      n_per_run=200, M_thr=0.75, tcp=0.15, N=10)
