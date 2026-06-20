#!/usr/bin/env julia
#=
Simulation driver for SlidingLayers.

Usage from command line:
    julia --project=. -t auto simulation_driver.jl --mode=mixing --L=1000
    julia --project=. -t 4 simulation_driver.jl --mode=energy --energy_p=3.0
    julia --project=. simulation_driver.jl --mode=history --beta=1.5 --T_steps=2000
    julia --project=. -t auto simulation_driver.jl --mode=erosion_test --v=2.0

Usage from REPL:
    using Pkg
    Pkg.activate(".")
    using Revise
    include("simulation_driver.jl")
=#

try
    using Revise
catch
    @warn "Revise.jl not loaded: restart the REPL to see changes to SlidingLayers module."
end

using SlidingLayers

### parse command-line arguments ###

function parse_arg(key::String, default_value)
    for raw_arg in ARGS
        # Guard against a stale `\<space>` (escaped space, not line-continuation)
        # in a multi-line command — bash glues the space onto the next token,
        # leaving a leading-space "arg" that startswith() then misses, and the
        # flag silently falls back to its default. Strip leading whitespace.
        arg = lstrip(raw_arg)
        if startswith(arg, "--$key=")
            value_str = split(arg, "=", limit=2)[2]
            if lowercase(value_str) == "nothing" || value_str == ""
                return nothing
            elseif default_value isa Bool
                return lowercase(value_str) in ["true", "1", "yes"]
            elseif default_value isa Int
                return parse(Int, value_str)
            elseif default_value isa Float64
                return parse(Float64, value_str)
            elseif default_value isa String
                return value_str
            elseif default_value isa Vector{Float64}
                return parse.(Float64, split(value_str, ","))
            else
                return value_str
            end
        end
    end
    return default_value
end

### p / q / τ parametrization helpers ###
#
# In the sliding-Ising chain code, p = exp(-β J) ∈ (0, 1] is the primary
# parameter. The user may equivalently set ranges via q = 1/p (= exp(β J)) or
# τ = exp(4 (β J)²). Per the user's request, the sweep is taken to be linear
# in *whichever parameter the user specifies*. The CLI dispatch translates the
# user's chosen range to p values, then passes them to the mode function via
# `sweep_values_override`.

# Convert τ ↔ p for the Ising chain: τ = exp(4 (β J)²), β J = -log p.
function _p_from_tau_ising(t::Float64)
    lt = log(t)
    lt >= 0 || error("--tau must be ≥ 1 in the Ising parametrization (got τ=$t).")
    return exp(-sqrt(lt / 4))
end
function _tau_from_p_ising(p::Float64)
    p > 0 || error("p must be > 0 (got p=$p).")
    bJ = -log(p)
    return exp(4 * bJ * bJ)
end

# Convert r ↔ p for the Ising chain, where r ≡ (β J)² and p = exp(-β J).
# Inverse: p = exp(-√r).
_p_from_r_ising(r::Float64) = (r >= 0 || error("--r must be ≥ 0 (got r=$r)."); exp(-sqrt(r)))

"""
Build `(p_min, p_max, p, sweep_override)` from the user's CLI args, applying
overrides in priority order r > tau > q > p. `sweep_override` is `nothing` when
the user only used --p_min/--p_max (the mode function then does its own linspace);
otherwise it's a precomputed Vector{Float64} of p values matching a linear
sweep in the user's chosen parametrization.

`p_min_default`, `p_max_default`, `p_default` are the mode's existing defaults.
"""
function _resolve_pqtau_sliding(p_min_default::Float64, p_max_default::Float64,
                                p_default::Float64, n_steps::Int)
    # Start from defaults; --p_min/--p_max/--p overrides directly.
    p_min = parse_arg("p_min", p_min_default)
    p_max = parse_arg("p_max", p_max_default)
    p = parse_arg("p", p_default)

    q_min = parse_arg("q_min", NaN)
    q_max = parse_arg("q_max", NaN)
    q = parse_arg("q", NaN)
    tau_min = parse_arg("tau_min", NaN)
    tau_max = parse_arg("tau_max", NaN)
    tau = parse_arg("tau", NaN)
    r_min = parse_arg("r_min", NaN)
    r_max = parse_arg("r_max", NaN)
    r = parse_arg("r", NaN)

    sweep_override::Union{Nothing,Vector{Float64}} = nothing

    if !isnan(r_min) && !isnan(r_max)
        r_values = collect(range(r_min, r_max, length=n_steps))
        sweep_override = [_p_from_r_ising(rv) for rv in r_values]
        p_min, p_max = minimum(sweep_override), maximum(sweep_override)
    elseif !isnan(tau_min) && !isnan(tau_max)
        tau_values = collect(range(tau_min, tau_max, length=n_steps))
        sweep_override = [_p_from_tau_ising(t) for t in tau_values]
        # p_min/p_max here are just the bounds of the override (informational).
        p_min, p_max = minimum(sweep_override), maximum(sweep_override)
    elseif !isnan(q_min) && !isnan(q_max)
        q_values = collect(range(q_min, q_max, length=n_steps))
        sweep_override = [1.0 / qv for qv in q_values]
        p_min, p_max = minimum(sweep_override), maximum(sweep_override)
    else
        # Honour partial overrides on the bounds.
        !isnan(q_min) && (p_max = 1.0 / q_min)
        !isnan(q_max) && (p_min = 1.0 / q_max)
        !isnan(tau_min) && (p_max = _p_from_tau_ising(tau_min))
        !isnan(tau_max) && (p_min = _p_from_tau_ising(tau_max))
        !isnan(r_min) && (p_max = _p_from_r_ising(r_min))
        !isnan(r_max) && (p_min = _p_from_r_ising(r_max))
    end

    # Fixed-value override (used when sweeping v, etc.).
    if !isnan(r)
        p = _p_from_r_ising(r)
    elseif !isnan(tau)
        p = _p_from_tau_ising(tau)
    elseif !isnan(q)
        p = 1.0 / q
    end

    return p_min, p_max, p, sweep_override
end

### default parameters ###

mode = "mixing"
mode = String(parse_arg("mode", mode))

# Set mode-dependent defaults
if mode == "history"
    L = 2000
    v = 2.0
    beta = 2.0
    h = 0.0
    T_steps = 1000
    init = "domain"
    domain_start = 3L ÷ 8
    domain_end = 5L ÷ 8
    show_plot = false
elseif mode == "mixing"
    # Convention: p = exp(-β J), 0 < p ≤ 1 (rare-event regime at small p).
    L = 2000
    v = 2.0
    h = 0.0
    p = 0.2
    p_min = 0.29
    p_max = 0.5
    v_min = 0.5
    v_max = 5.0
    n_steps = 4
    vary_v = false
    n_trials = 100
    M_threshold = 0.65
    max_time = 1000000
    single_layer = false
elseif mode == "ffs"
    # Convention: p = exp(-β J), 0 < p ≤ 1 (rare-event regime at small p).
    L = 2000
    v = 2.0
    h = 0.0
    p = 0.2
    p_min = 0.2
    p_max = 0.5
    v_min = 0.5
    v_max = 5.0
    n_steps = 6
    vary_v = false
    n_configs_per_run = 100
    n_repeats = 8
    target_crossing_prob = 0.15
    n_interfaces = 0   # 0 = adaptive (default). >0 = canonical FFS with that many fixed interfaces and λ_fail = λ_0.
    adaptive_L = false  # set L = adaptive_factor × ℓ_er per sweep point (requires v > 0).
    adaptive_factor = 3.0
    M_threshold = 0.75
    max_time_per_trial = 200000000
    single_layer = false
    lambda_0_override = NaN
    seed_droplet_size = 0   # 0 = off; >0 = instantly inject a k-spin minority droplet on both chains whenever the state hits all-+ (modified-clock convention).
    randshift = false       # if true, place top-chain shifts via a Poisson process at rate v instead of the deterministic Bresenham schedule. Removes integer-vs-half-integer artifacts in v sweeps.
elseif mode == "energy"
    # Convention: p = exp(-β J), 0 < p ≤ 1.
    L = 2000
    v = 2.0
    h = 0.0
    p = 0.385
    p_min = 0.2
    p_max = 0.5
    v_min = 0.0
    v_max = 5.0
    n_steps = 6
    vary_v = false
    T_equil = 125000
    T_sample = 200000
elseif mode == "teff"
    L = 2000
    v = 2.0
    h = 0.0
    T = 0.5
    T_min = 0.3
    T_max = 1.0
    v_min = 0.0
    v_max = 5.0
    n_steps = 6
    vary_v = false
    T_equil = 125000
    T_sample = 200000
    demon_interval = 200
    demon = false
    h_pert = 0.01
    n_perturbations = 200
    T_response = 0
    n_samples = 1
    single_layer = false
    show_plots = false
elseif mode == "erosion_test"
    # Convention: p = exp(-β J), 0 < p ≤ 1.
    v = 2.0
    h = 0.0
    p = 0.2
    p_min = 0.0667
    p_max = 0.4
    v_min = 0.5
    v_max = 5.0
    n_steps = 6
    vary_v = false
    thresh_prob = 0.75
    erosion_num_trials = 5000
    min_erosion_length = 2
    show_histories = false
    doublon_mode = false
    erode_vs_l = false
    t_evolve_factor = nothing  # nothing → use the per-mode default (2 non-doublon, 5 doublon)
    L_sys_factor = 1.0  # multiplier on L_sys = 2*l*(v+1); for sanity checks
    first_passage_mode = false  # use variable-time first-passage escape measurement
    min_doublons = 10  # lower boundary for first-passage erode condition
    show_plots = false
elseif mode == "phase_diagram"
    # Convention: p = exp(-β J), 0 < p ≤ 1.
    L = 2000
    h = 0.0
    p_min = 0.2
    p_max = 0.5
    n_p_steps = 6
    v_min = 0.5
    v_max = 5.0
    n_v_steps = 10
    observable = "mixing"
    onset_threshold = 10.0
    # mixing-only:
    n_trials = 100
    M_threshold = 0.65
    max_time = 1000000
    # erosion-only:
    erosion_num_trials = 5000
    thresh_prob = 0.75
    min_erosion_length = 2
    doublon_mode = false
    # shared:
    show_plots = false
    single_layer = false
else
    error("Unknown mode: $mode. Use 'history', 'mixing', 'ffs', 'energy', 'teff', 'erosion_test', or 'phase_diagram'.")
end

# Parse command-line overrides
save = parse_arg("save", true)
adj = String(parse_arg("adj", ""))

if mode == "history"
    L = parse_arg("L", L)
    v = parse_arg("v", v)
    beta = parse_arg("beta", beta)
    h = parse_arg("h", h)
    T_steps = parse_arg("T_steps", T_steps)
    init = String(parse_arg("init", init))
    domain_start = parse_arg("domain_start", 3L ÷ 8)
    domain_end = parse_arg("domain_end", 5L ÷ 8)
    show_plot = parse_arg("show_plot", show_plot)
elseif mode == "mixing"
    L = parse_arg("L", L)
    v = parse_arg("v", v)
    h = parse_arg("h", h)
    p = parse_arg("p", p)
    p_min = parse_arg("p_min", p_min)
    p_max = parse_arg("p_max", p_max)
    v_min = parse_arg("v_min", v_min)
    v_max = parse_arg("v_max", v_max)
    n_steps = parse_arg("n_steps", n_steps)
    vary_v = parse_arg("vary_v", vary_v)
    n_trials = parse_arg("n_trials", n_trials)
    M_threshold = parse_arg("M_threshold", M_threshold)
    max_time = parse_arg("max_time", max_time)
    single_layer = parse_arg("single_layer", single_layer)
    (p_min, p_max, p, sweep_values_override) = _resolve_pqtau_sliding(p_min, p_max, p, n_steps)
elseif mode == "ffs"
    L = parse_arg("L", L)
    v = parse_arg("v", v)
    h = parse_arg("h", h)
    p = parse_arg("p", p)
    p_min = parse_arg("p_min", p_min)
    p_max = parse_arg("p_max", p_max)
    # Sniff for explicit --v_min/--v_max with a NaN sentinel so we can auto-flip
    # vary_v=true when both are passed (otherwise --v_min=2 --v_max=5 silently
    # gets ignored because vary_v defaults to false and the script does a p-sweep).
    _v_min_user = parse_arg("v_min", NaN)
    _v_max_user = parse_arg("v_max", NaN)
    v_min = isnan(_v_min_user) ? v_min : _v_min_user
    v_max = isnan(_v_max_user) ? v_max : _v_max_user
    if !isnan(_v_min_user) && !isnan(_v_max_user)
        vary_v = true   # auto-infer; explicit --vary_v below still wins
    end
    n_steps = parse_arg("n_steps", n_steps)
    vary_v = parse_arg("vary_v", vary_v)
    n_repeats = parse_arg("n_repeats", n_repeats)
    n_configs_per_run = parse_arg("n_configs_per_run", n_configs_per_run)
    # Backward compat: --n_configs is the deprecated total-budget knob.
    # If given, derive n_configs_per_run = n_configs / n_repeats (the old behavior).
    n_configs_legacy = parse_arg("n_configs", -1)
    if n_configs_legacy > 0
        n_configs_per_run = max(1, round(Int, n_configs_legacy / n_repeats))
        @warn "--n_configs is deprecated; use --n_configs_per_run instead. " *
              "Translating: n_configs_per_run = $(n_configs_legacy) ÷ $(n_repeats) = $(n_configs_per_run)"
    end
    target_crossing_prob = parse_arg("target_crossing_prob", target_crossing_prob)
    n_interfaces = parse_arg("n_interfaces", n_interfaces)
    adaptive_L = parse_arg("adaptive_L", adaptive_L)
    adaptive_factor = parse_arg("adaptive_factor", adaptive_factor)
    M_threshold = parse_arg("M_threshold", M_threshold)
    max_time_per_trial = parse_arg("max_time_per_trial", max_time_per_trial)
    single_layer = parse_arg("single_layer", single_layer)
    lambda_0_override = parse_arg("lambda_0", lambda_0_override)
    seed_droplet_size = parse_arg("seed_droplet_size", seed_droplet_size)
    randshift = parse_arg("randshift", randshift)
    (p_min, p_max, p, sweep_values_override) = _resolve_pqtau_sliding(p_min, p_max, p, n_steps)
elseif mode == "energy"
    L = parse_arg("L", L)
    v = parse_arg("v", v)
    h = parse_arg("h", h)
    p = parse_arg("p", p)
    p_min = parse_arg("p_min", p_min)
    p_max = parse_arg("p_max", p_max)
    v_min = parse_arg("v_min", v_min)
    v_max = parse_arg("v_max", v_max)
    n_steps = parse_arg("n_steps", n_steps)
    vary_v = parse_arg("vary_v", vary_v)
    T_equil = parse_arg("T_equil", T_equil)
    T_sample = parse_arg("T_sample", T_sample)
    (p_min, p_max, p, sweep_values_override) = _resolve_pqtau_sliding(p_min, p_max, p, n_steps)
elseif mode == "teff"
    L = parse_arg("L", L)
    v = parse_arg("v", v)
    h = parse_arg("h", h)
    T = parse_arg("T", T)
    T_min = parse_arg("T_min", T_min)
    T_max = parse_arg("T_max", T_max)
    v_min = parse_arg("v_min", v_min)
    v_max = parse_arg("v_max", v_max)
    n_steps = parse_arg("n_steps", n_steps)
    vary_v = parse_arg("vary_v", vary_v)
    T_equil = parse_arg("T_equil", T_equil)
    T_sample = parse_arg("T_sample", T_sample)
    demon_interval = parse_arg("demon_interval", demon_interval)
    demon = parse_arg("demon", demon)
    h_pert = parse_arg("h_pert", h_pert)
    n_perturbations = parse_arg("n_perturbations", n_perturbations)
    T_response = parse_arg("T_response", T_response)
    n_samples = parse_arg("n_samples", n_samples)
    single_layer = parse_arg("single_layer", single_layer)
    show_plots = parse_arg("show_plots", show_plots)
elseif mode == "erosion_test"
    v = parse_arg("v", v)
    h = parse_arg("h", h)
    p = parse_arg("p", p)
    p_min = parse_arg("p_min", p_min)
    p_max = parse_arg("p_max", p_max)
    v_min = parse_arg("v_min", v_min)
    v_max = parse_arg("v_max", v_max)
    n_steps = parse_arg("n_steps", n_steps)
    vary_v = parse_arg("vary_v", vary_v)
    thresh_prob = parse_arg("thresh_prob", thresh_prob)
    erosion_num_trials = parse_arg("erosion_num_trials", erosion_num_trials)
    min_erosion_length = parse_arg("min_erosion_length", min_erosion_length)
    show_histories = parse_arg("show_histories", show_histories)
    doublon_mode = parse_arg("doublon_mode", doublon_mode)
    erode_vs_l = parse_arg("erode_vs_l", erode_vs_l)
    t_evolve_factor_raw = parse_arg("t_evolve_factor", -1.0)
    t_evolve_factor = t_evolve_factor_raw > 0 ? t_evolve_factor_raw : nothing
    L_sys_factor = parse_arg("L_sys_factor", L_sys_factor)
    first_passage_mode = parse_arg("first_passage_mode", first_passage_mode)
    min_doublons = parse_arg("min_doublons", min_doublons)
    show_plots = parse_arg("show_plots", show_plots)
    (p_min, p_max, p, sweep_values_override) = _resolve_pqtau_sliding(p_min, p_max, p, n_steps)
elseif mode == "phase_diagram"
    L = parse_arg("L", L)
    h = parse_arg("h", h)
    p_min = parse_arg("p_min", p_min)
    p_max = parse_arg("p_max", p_max)
    n_p_steps = parse_arg("n_p_steps", n_p_steps)
    v_min = parse_arg("v_min", v_min)
    v_max = parse_arg("v_max", v_max)
    n_v_steps = parse_arg("n_v_steps", n_v_steps)
    observable = String(parse_arg("observable", observable))
    onset_threshold = parse_arg("onset_threshold", onset_threshold)
    n_trials = parse_arg("n_trials", n_trials)
    M_threshold = parse_arg("M_threshold", M_threshold)
    max_time = parse_arg("max_time", max_time)
    erosion_num_trials = parse_arg("erosion_num_trials", erosion_num_trials)
    thresh_prob = parse_arg("thresh_prob", thresh_prob)
    min_erosion_length = parse_arg("min_erosion_length", min_erosion_length)
    doublon_mode = parse_arg("doublon_mode", doublon_mode)
    show_plots = parse_arg("show_plots", show_plots)
    single_layer = parse_arg("single_layer", single_layer)
    # phase_diagram has no fixed p; ignore returned p
    (p_min, p_max, _ignored_p, sweep_values_override) = _resolve_pqtau_sliding(p_min, p_max, NaN, n_p_steps)
end

### check thread count ###

n_threads = Threads.nthreads()
if n_threads == 1
    @warn "Running with only 1 thread. For parallel trials, start Julia with: julia -t auto\n  Or set JULIA_NUM_THREADS=auto in your shell profile."
end

### build kwargs and run ###

kwargs = Dict{Symbol, Any}()

if mode == "history"
    kwargs[:L] = L
    kwargs[:v] = v
    kwargs[:beta] = beta
    kwargs[:h] = h
    kwargs[:T_steps] = T_steps
    kwargs[:init] = init
    kwargs[:domain_start] = domain_start
    kwargs[:domain_end] = domain_end
    kwargs[:show_plot] = show_plot
elseif mode == "mixing"
    kwargs[:L] = L
    kwargs[:v] = v
    kwargs[:h] = h
    kwargs[:p] = p
    kwargs[:p_min] = p_min
    kwargs[:p_max] = p_max
    kwargs[:v_min] = v_min
    kwargs[:v_max] = v_max
    kwargs[:n_steps] = n_steps
    kwargs[:vary_v] = vary_v
    kwargs[:n_trials] = n_trials
    kwargs[:M_threshold] = M_threshold
    kwargs[:max_time] = max_time
    kwargs[:single_layer] = single_layer
    kwargs[:sweep_values_override] = sweep_values_override
elseif mode == "ffs"
    kwargs[:L] = L
    kwargs[:v] = v
    kwargs[:h] = h
    kwargs[:p] = p
    kwargs[:p_min] = p_min
    kwargs[:p_max] = p_max
    kwargs[:v_min] = v_min
    kwargs[:v_max] = v_max
    kwargs[:n_steps] = n_steps
    kwargs[:vary_v] = vary_v
    kwargs[:n_configs_per_run] = n_configs_per_run
    kwargs[:n_repeats] = n_repeats
    kwargs[:target_crossing_prob] = target_crossing_prob
    kwargs[:n_interfaces] = n_interfaces
    kwargs[:adaptive_L] = adaptive_L
    kwargs[:adaptive_factor] = adaptive_factor
    kwargs[:M_threshold] = M_threshold
    kwargs[:max_time_per_trial] = max_time_per_trial
    kwargs[:single_layer] = single_layer
    kwargs[:lambda_0_override] = lambda_0_override
    kwargs[:seed_droplet_size] = seed_droplet_size
    kwargs[:randshift] = randshift
    kwargs[:sweep_values_override] = sweep_values_override
elseif mode == "energy"
    kwargs[:L] = L
    kwargs[:v] = v
    kwargs[:h] = h
    kwargs[:p] = p
    kwargs[:p_min] = p_min
    kwargs[:p_max] = p_max
    kwargs[:v_min] = v_min
    kwargs[:v_max] = v_max
    kwargs[:n_steps] = n_steps
    kwargs[:vary_v] = vary_v
    kwargs[:T_equil] = T_equil
    kwargs[:T_sample] = T_sample
    kwargs[:sweep_values_override] = sweep_values_override
elseif mode == "teff"
    kwargs[:L] = L
    kwargs[:v] = v
    kwargs[:h] = h
    kwargs[:T] = T
    kwargs[:T_min] = T_min
    kwargs[:T_max] = T_max
    kwargs[:v_min] = v_min
    kwargs[:v_max] = v_max
    kwargs[:n_steps] = n_steps
    kwargs[:vary_v] = vary_v
    kwargs[:T_equil] = T_equil
    kwargs[:T_sample] = T_sample
    kwargs[:demon_interval] = demon_interval
    kwargs[:demon] = demon
    kwargs[:h_pert] = h_pert
    kwargs[:n_perturbations] = n_perturbations
    kwargs[:T_response] = T_response
    kwargs[:n_samples] = n_samples
    kwargs[:single_layer] = single_layer
    kwargs[:show_plots] = show_plots
elseif mode == "erosion_test"
    kwargs[:v] = v
    kwargs[:h] = h
    kwargs[:p] = p
    kwargs[:p_min] = p_min
    kwargs[:p_max] = p_max
    kwargs[:v_min] = v_min
    kwargs[:v_max] = v_max
    kwargs[:n_steps] = n_steps
    kwargs[:vary_v] = vary_v
    kwargs[:thresh_prob] = thresh_prob
    kwargs[:erosion_num_trials] = erosion_num_trials
    kwargs[:min_erosion_length] = min_erosion_length
    kwargs[:show_histories] = show_histories
    kwargs[:doublon_mode] = doublon_mode
    kwargs[:erode_vs_l] = erode_vs_l
    kwargs[:t_evolve_factor] = t_evolve_factor
    kwargs[:L_sys_factor] = L_sys_factor
    kwargs[:first_passage_mode] = first_passage_mode
    kwargs[:min_doublons] = min_doublons
    kwargs[:show_plots] = show_plots
    kwargs[:sweep_values_override] = sweep_values_override
elseif mode == "phase_diagram"
    kwargs[:L] = L
    kwargs[:h] = h
    kwargs[:p_min] = p_min
    kwargs[:p_max] = p_max
    kwargs[:n_p_steps] = n_p_steps
    kwargs[:v_min] = v_min
    kwargs[:v_max] = v_max
    kwargs[:n_v_steps] = n_v_steps
    kwargs[:observable] = observable
    kwargs[:onset_threshold] = onset_threshold
    kwargs[:n_trials] = n_trials
    kwargs[:M_threshold] = M_threshold
    kwargs[:max_time] = max_time
    kwargs[:erosion_num_trials] = erosion_num_trials
    kwargs[:thresh_prob] = thresh_prob
    kwargs[:min_erosion_length] = min_erosion_length
    kwargs[:doublon_mode] = doublon_mode
    kwargs[:show_plots] = show_plots
    kwargs[:single_layer] = single_layer
    kwargs[:sweep_values_override] = sweep_values_override
end

println("Running simulation: mode=$mode, threads=$n_threads")
run_simulation(mode; save=save, adj=adj, kwargs...)
println("Simulation complete!")
try run(`afplay /System/Library/Sounds/Glass.aiff`) catch end
