#!/usr/bin/env julia
#=
gkl.jl — driver for the noisy GKL (Gács-Kurdyumov-Levin) 1D cellular automaton.

Standalone companion to `simulation_driver.jl`; does NOT depend on
the SlidingLayers module.

Update rule (canonical, periodic boundaries):
  - if σ_i = +1: new σ_i = sign(σ_{i-3} + σ_{i-1} + σ_i)
  - if σ_i = -1: new σ_i = sign(σ_i + σ_{i+1} + σ_{i+3})
After the deterministic step, with probability p_noise the cell is overwritten
with +1 (with probability (1+η)/2) or -1 (with probability (1-η)/2).

Update modes: --update=sync (default) or --update=async. One async step =
L random-with-replacement single-cell updates.

Convention (FFS / ler): η > 0 ⇒ deterministic GKL drives the chain toward +1;
the metastable phase (anti-aligned with η) is then all -1. We force eta = abs(eta)
at the top of FFS / ler entry points and start from the metastable phase.

Modes:
  --mode=history   Spacetime history of σ(x, t).
  --mode=ffs       Forward Flux Sampling for the memory time τ.
  --mode=ler       Erosion length ℓ_c (smallest minority cluster that survives).

Usage:
  julia --project=. gkl.jl --mode=history --L=200 --T_steps=300 --eta=0.10 --p_noise=0.05
  julia --project=. -t auto gkl.jl --mode=ffs --L=200 --eta=0.05 --p_min=0.06 --p_max=0.10 --n_steps=2
  julia --project=. -t auto gkl.jl --mode=ler --eta=0.10 --p_min=0.04 --p_max=0.08 --n_steps=2
=#

try
    using Revise
catch
end

using Printf, Random, JLD2, Statistics, ProgressMeter, InteractiveUtils

### =====================================================================
### §0  argument parsing  (copied from simulation_driver.jl)
### =====================================================================

function parse_arg(key::String, default_value)
    for arg in ARGS
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

### =====================================================================
### §1  GKL state and dynamics primitives
### =====================================================================

mutable struct GKLState
    L::Int
    eta::Float64
    p_noise::Float64
    update::Symbol           # :sync or :async
    sigma::Vector{Int8}      # current configuration (length L)
    buffer::Vector{Int8}     # double-buffer for sync update
    mag_sum::Int             # running Σ σ_i; kept in sync by every mutator
end

function GKLState(L::Int, eta::Float64, p_noise::Float64; update::Symbol=:sync,
                  init_value::Int8=Int8(1))
    L >= 7 || error("GKL requires L >= 7 (rule references offsets ±3); got L=$L.")
    sigma = fill(init_value, L)
    buffer = similar(sigma)
    return GKLState(L, eta, p_noise, update, sigma, buffer, Int(init_value) * L)
end

@inline function recompute_mag_sum!(state::GKLState)
    s = 0
    @inbounds for i in 1:state.L
        s += state.sigma[i]
    end
    state.mag_sum = s
end

@inline copy_config(state::GKLState) = copy(state.sigma)

function load_config!(state::GKLState, cfg::Vector{Int8})
    length(cfg) == state.L || error("Config length $(length(cfg)) ≠ L=$(state.L).")
    copyto!(state.sigma, cfg)
    recompute_mag_sum!(state)
end

@inline compute_total_magnetization(state::GKLState) =
    state.mag_sum / state.L  # in [-1, 1]

# ---------------------------------------------------------------------
# Deterministic GKL update for site i (using current `sigma`).
# Returns the new ±1 value (Int8).
# ---------------------------------------------------------------------
@inline function gkl_det_new(sigma::Vector{Int8}, L::Int, i::Int)::Int8
    σ_i = sigma[i]
    if σ_i > 0
        a = sigma[mod1(i - 3, L)]
        b = sigma[mod1(i - 1, L)]
        c = σ_i
    else
        a = σ_i
        b = sigma[mod1(i + 1, L)]
        c = sigma[mod1(i + 3, L)]
    end
    s = Int(a) + Int(b) + Int(c)
    return s >= 0 ? Int8(1) : Int8(-1)
end

# ---------------------------------------------------------------------
# Apply per-cell noise to `new_val` and return the noised value.
# ---------------------------------------------------------------------
@inline function gkl_noise(new_val::Int8, p_noise::Float64, eta::Float64)::Int8
    if rand() < p_noise
        return rand() < (1.0 + eta) / 2.0 ? Int8(1) : Int8(-1)
    end
    return new_val
end

# ---------------------------------------------------------------------
# Synchronous step: write into `buffer`, swap, recompute mag_sum.
# ---------------------------------------------------------------------
function gkl_sync_step!(state::GKLState)
    L = state.L
    p = state.p_noise
    eta = state.eta
    @inbounds for i in 1:L
        new_val = gkl_det_new(state.sigma, L, i)
        state.buffer[i] = gkl_noise(new_val, p, eta)
    end
    copyto!(state.sigma, state.buffer)
    recompute_mag_sum!(state)
    return nothing
end

# ---------------------------------------------------------------------
# Asynchronous step: L random-with-replacement single-cell updates.
# Each update reads & writes `state.sigma`; mag_sum incrementally updated.
# ---------------------------------------------------------------------
function gkl_async_step!(state::GKLState)
    L = state.L
    p = state.p_noise
    eta = state.eta
    @inbounds for _ in 1:L
        i = rand(1:L)
        σ_old = state.sigma[i]
        new_val = gkl_det_new(state.sigma, L, i)
        σ_new = gkl_noise(new_val, p, eta)
        if σ_new != σ_old
            state.sigma[i] = σ_new
            state.mag_sum += Int(σ_new) - Int(σ_old)
        end
    end
    return nothing
end

# ---------------------------------------------------------------------
# One timestep, dispatched on update mode.
# ---------------------------------------------------------------------
@inline function gkl_step!(state::GKLState)
    state.update === :sync ? gkl_sync_step!(state) : gkl_async_step!(state)
end

# ---------------------------------------------------------------------
# evolve_time! / evolve_and_measure!: contract matches src/core.jl.
# `on_measure(step) -> Bool` is called after each timestep; return false
# to stop early. Returns the number of completed steps.
# ---------------------------------------------------------------------
function evolve_time!(state::GKLState, T::Int)
    for _ in 1:T
        gkl_step!(state)
    end
    return T
end

function evolve_and_measure!(state::GKLState, T_max::Int, on_measure::Function)::Int
    for step in 1:T_max
        gkl_step!(state)
        if !on_measure(step)
            return step
        end
    end
    return T_max
end

### =====================================================================
### §2  init helpers shared by all modes
### =====================================================================

"""
Stable phase (the sign that deterministic GKL + biased noise drives toward).
With eta > 0, +1 is stable; with eta < 0, -1 is stable.
"""
@inline stable_phase(eta::Float64)::Int8 = eta >= 0 ? Int8(1) : Int8(-1)

"""
FFS starting configuration: at η > 0 the metastable basin is all-(−1) (anti-aligned
with the bias); at η ≤ 0 we start in all-(+1) by convention and watch the
magnetization fall.
"""
@inline metastable_value(eta::Float64)::Int8 = eta > 0 ? Int8(-1) : Int8(+1)

"""
Escape direction: +1 if the FFS order parameter increases away from m_meta
(η > 0 case, m rises from −1 toward M_threshold > 0); −1 if it decreases
(η ≤ 0 case, m falls from +1 toward M_threshold ≥ 0).
"""
@inline escape_dir(eta::Float64)::Int = eta > 0 ? +1 : -1

# Comparison helpers that fold the direction sign in:
#   `forward_of(m, x, d)` is true when m has progressed past x in the escape direction.
#   `back_of(m, x, d)`    is true when m has returned (loosely) back across x.
@inline forward_of(m::Real, x::Real, d::Int) = d > 0 ? (m > x) : (m < x)
@inline back_of(m::Real, x::Real, d::Int) = d > 0 ? (m <= x) : (m >= x)
@inline reached_thresh(m::Real, M::Real, d::Int) = d > 0 ? (m >= M) : (m <= M)

"""
Initialize `state.sigma` according to `init`:
  - "domain": fill with stable phase, flip `domain_start:domain_end` to -stable.
  - "random": uniform ±1.
  - "all_plus" / "all_minus": trivial.
  - "metastable": all anti-aligned with η (used as the FFS / ler starting basin).
"""
function init_state!(state::GKLState, init::String;
                     domain_start::Int=0, domain_end::Int=0)
    L = state.L
    stable = stable_phase(state.eta)
    if init == "domain"
        ds = domain_start <= 0 ? 3L ÷ 8 : domain_start
        de = domain_end <= 0   ? 5L ÷ 8 : domain_end
        fill!(state.sigma, stable)
        @inbounds for i in ds:de
            state.sigma[i] = -stable
        end
    elseif init == "random"
        @inbounds for i in 1:L
            state.sigma[i] = rand() < 0.5 ? Int8(-1) : Int8(1)
        end
    elseif init == "all_plus"
        fill!(state.sigma, Int8(1))
    elseif init == "all_minus"
        fill!(state.sigma, Int8(-1))
    elseif init == "metastable"
        fill!(state.sigma, -stable)
    else
        error("Unknown init: $init. Use 'domain', 'random', 'all_plus', 'all_minus', or 'metastable'.")
    end
    recompute_mag_sum!(state)
    return nothing
end

### =====================================================================
### §3  history mode
### =====================================================================

function run_history_mode(; L::Int, eta::Float64, p_noise::Float64,
                            update::Symbol=:sync, T_steps::Int=500,
                            init::String="domain",
                            domain_start::Int=3L÷8, domain_end::Int=5L÷8,
                            kwargs...)
    println("=== Noisy GKL: History Mode ===")
    println("L = $L, η = $eta, p_noise = $p_noise, update = $update, T = $T_steps, init = $init")

    state = GKLState(L, eta, p_noise; update)
    init_state!(state, init; domain_start, domain_end)

    history = zeros(Int8, T_steps + 1, L)
    history[1, :] = state.sigma

    println("Running simulation...")
    @time evolve_and_measure!(state, T_steps, step -> begin
        history[step + 1, :] = state.sigma
        if step % 200 == 0
            @printf("  Step %d/%d, m = %.4f\n", step, T_steps, compute_total_magnetization(state))
        end
        true
    end)
    println("Done!")

    return Dict{String, Any}(
        "magnetization_history" => history,
        "L" => L, "T_steps" => T_steps,
        "eta" => eta, "p_noise" => p_noise,
        "update" => String(update),
        "init" => init,
        "domain_start" => domain_start, "domain_end" => domain_end,
        "dynamics" => "gkl",
        # Plotter-compat shims (sliding_plotter.py expects these scalars)
        "v" => 0.0, "beta" => 0.0, "h" => 0.0,
    )
end

### =====================================================================
### §4  FFS mode
###
### Convention: the user passes η > 0; the noise then biases +1, the
### stable phase is +1, and the metastable phase (where the system starts)
### is all -1. The reaction coordinate m = Σσ/L therefore starts near -1
### and escapes upward toward M_threshold (positive). All FFS inequalities
### are flipped relative to src/ffs.jl (which had m decreasing from +1):
### interfaces are an INCREASING sequence lambda_0 < lambda_1 < ... < M_thr,
### and probes track MAX m rather than MIN m.
### =====================================================================

const FFSConfig = Vector{Int8}

# ---------------------------------------------------------------------
# Phase 0: initial flux
# (mirror of src/ffs.jl:24-99)
# ---------------------------------------------------------------------
function measure_initial_flux(L::Int, eta::Float64, p_noise::Float64,
                               update::Symbol, lambda_0::Float64,
                               M_threshold::Float64,
                               n_configs::Int, max_time::Int;
                               verbose::Bool=false)
    n_threads = Threads.nthreads()
    configs_per_thread = cld(n_configs, n_threads)
    meta_val = metastable_value(eta)  # +1 if η ≤ 0, −1 if η > 0
    dir = escape_dir(eta)             # +1 (m rises) or −1 (m falls)

    all_configs = Vector{Vector{FFSConfig}}(undef, n_threads)
    all_times = zeros(Int, n_threads)
    all_n_trajectories = zeros(Int, n_threads)
    collected_total = Threads.Atomic{Int}(0)
    progress = verbose ? Progress(n_configs; desc="    Initial flux: ", dt=1.0) : nothing

    Threads.@threads for tid in 1:n_threads
        my_configs = FFSConfig[]
        my_target = min(configs_per_thread, n_configs - (tid - 1) * configs_per_thread)
        if my_target <= 0
            all_configs[tid] = my_configs
            continue
        end

        total_time = 0
        n_trajectories = 0

        while length(my_configs) < my_target
            state = GKLState(L, eta, p_noise; update, init_value=meta_val)
            in_basin = true  # m starts at m_meta (on the metastable side of λ₀)
            n_trajectories += 1

            final_step = evolve_and_measure!(state, max_time, step -> begin
                m = compute_total_magnetization(state)
                if in_basin && forward_of(m, lambda_0, dir)
                    # Forward crossing of λ₀ — store config
                    push!(my_configs, copy_config(state))
                    Threads.atomic_add!(collected_total, 1)
                    if progress !== nothing
                        update!(progress, collected_total[])
                    end
                    in_basin = false
                    if length(my_configs) >= my_target
                        return false
                    end
                elseif !in_basin && back_of(m, lambda_0, dir)
                    # Returned to basin A — ready for next crossing
                    in_basin = true
                end
                if !in_basin && reached_thresh(m, M_threshold, dir)
                    return false  # left basin A — abandon trajectory
                end
                return true
            end)

            if final_step >= max_time && length(my_configs) < my_target
                @warn "Phase 0: $(length(my_configs)) / $my_target configs collected, trajectory timed out at max_time = $max_time. Increase --max_time_per_trial."
            end
            total_time += final_step
        end

        all_configs[tid] = my_configs
        all_times[tid] = total_time
        all_n_trajectories[tid] = n_trajectories
    end

    if progress !== nothing
        finish!(progress)
    end

    configs = reduce(vcat, all_configs)
    total_time = sum(all_times)
    n_crossings = length(configs)
    flux_rate = total_time > 0 ? n_crossings / total_time : 0.0
    return (flux_rate, configs)
end

# ---------------------------------------------------------------------
# Crossing-probability phase
# (mirror of src/ffs.jl:113-184)
# ---------------------------------------------------------------------
function measure_crossing_probability(L::Int, eta::Float64, p_noise::Float64,
                                       update::Symbol,
                                       source_configs::Vector{FFSConfig},
                                       lambda_target::Float64, lambda_fail::Float64,
                                       max_time_per_trial::Int, n_configs::Int)
    success_configs = FFSConfig[]
    total_successes = 0
    total_trials = 0
    total_timeouts = 0
    batch_size = max(Threads.nthreads() * 4, 32)
    t_start = time()
    progress = nothing
    dir = escape_dir(eta)

    while length(success_configs) < n_configs
        batch_successes = Vector{Union{Nothing, FFSConfig}}(nothing, batch_size)
        batch_timeouts = zeros(Int, batch_size)

        Threads.@threads for trial in 1:batch_size
            config = source_configs[rand(1:length(source_configs))]
            state = GKLState(L, eta, p_noise; update)
            load_config!(state, config)

            success = false
            final_step = evolve_and_measure!(state, max_time_per_trial, step -> begin
                m = compute_total_magnetization(state)
                if forward_of(m, lambda_target, dir)
                    success = true
                    return false
                elseif back_of(m, lambda_fail, dir)
                    return false  # returned toward basin A (crossed the failure boundary)
                end
                return true
            end)

            if success
                batch_successes[trial] = copy_config(state)
            end
            if final_step >= max_time_per_trial
                batch_timeouts[trial] = 1
            end
        end

        n_timeouts = sum(batch_timeouts)
        total_trials += batch_size
        total_timeouts += n_timeouts
        if n_timeouts > 0
            @warn "Crossing phase: $total_trials trials done, $n_timeouts timed out at max_time_per_trial = $max_time_per_trial. Marking single FFS run as failed; increase --max_time_per_trial."
        end
        for c in batch_successes
            if c !== nothing
                total_successes += 1
                if length(success_configs) < n_configs
                    push!(success_configs, c)
                end
            end
        end

        elapsed = time() - t_start
        if elapsed > 5.0 && progress === nothing
            progress = Progress(n_configs; desc="    Collecting configs: ", dt=1.0)
        end
        if progress !== nothing
            update!(progress, min(length(success_configs), n_configs))
        end
    end

    if progress !== nothing
        finish!(progress)
    end

    prob = total_successes / total_trials
    return (prob, success_configs[1:n_configs], total_successes, total_trials, total_timeouts > 0)
end

# ---------------------------------------------------------------------
# Adaptive probe of the next interface position
# (mirror of src/ffs.jl:202-232)
# ---------------------------------------------------------------------
function probe_next_interface(L::Int, eta::Float64, p_noise::Float64,
                               update::Symbol,
                               source_configs::Vector{FFSConfig}, lambda_fail::Float64,
                               M_threshold::Float64,
                               max_time::Int, n_probe::Int, target_prob::Float64)
    dir = escape_dir(eta)
    # We track each probe's *most-advanced* magnetization in the escape direction.
    # For dir = +1 that's max(m); for dir = −1 that's min(m). Storing dir*m lets us
    # always sort ascending and pick a single quantile rule.
    advanced_mags = zeros(Float64, n_probe)
    probes_done = Threads.Atomic{Int}(0)
    n_timeouts = Threads.Atomic{Int}(0)

    Threads.@threads for trial in 1:n_probe
        config = source_configs[rand(1:length(source_configs))]
        state = GKLState(L, eta, p_noise; update)
        load_config!(state, config)

        m0 = compute_total_magnetization(state)
        best = dir * m0  # always-rising signed magnetization

        final_step = evolve_and_measure!(state, max_time, step -> begin
            m = compute_total_magnetization(state)
            signed_m = dir * m
            if signed_m > best
                best = signed_m
            end
            if reached_thresh(m, M_threshold, dir)
                return false  # barrier already crossed
            end
            # Continue while still on the metastable side of the failure boundary.
            return !back_of(m, lambda_fail, dir)
        end)

        Threads.atomic_add!(probes_done, 1)
        if final_step >= max_time
            Threads.atomic_add!(n_timeouts, 1)
            @warn "Probe: $(probes_done[]) / $n_probe done, trial timed out at max_time = $max_time. Marking single FFS run as failed; increase --max_time_per_trial."
        end
        advanced_mags[trial] = best
    end

    # Ascending sort on signed-m: lowest "advancement" first; pick the
    # (1 − target_prob) quantile so that target_prob fraction of trials reached
    # beyond the returned threshold. Convert back to m-units via dir.
    sort!(advanced_mags)
    k = clamp(round(Int, (1.0 - target_prob) * n_probe), 1, n_probe)
    return (dir * advanced_mags[k], n_timeouts[] > 0)
end

# ---------------------------------------------------------------------
# Metastable magnetization: brief equilibration sample
# (mirror of src/ffs.jl:241-261)
# ---------------------------------------------------------------------
function measure_metastable_magnetization(L::Int, eta::Float64, p_noise::Float64,
                                           update::Symbol)
    n_runs = 10
    T_equil = 2
    T_sample = 10
    meta_val = metastable_value(eta)  # +1 for η ≤ 0, −1 for η > 0
    all_mags = Float64[]

    for _ in 1:n_runs
        state = GKLState(L, eta, p_noise; update, init_value=meta_val)
        evolve_time!(state, T_equil)
        evolve_and_measure!(state, T_sample, step -> begin
            push!(all_mags, compute_total_magnetization(state))
            return true
        end)
    end

    return mean(all_mags), std(all_mags)
end

# ---------------------------------------------------------------------
# One full FFS estimation
# (mirror of src/ffs.jl:272-347)
# ---------------------------------------------------------------------
function ffs_single_run(L::Int, eta::Float64, p_noise::Float64, update::Symbol,
                         lambda_0::Float64, M_threshold::Float64,
                         n_configs_per_run::Int, max_time_per_trial::Int,
                         target_crossing_prob::Float64;
                         n_interfaces::Int=0, verbose::Bool=false)
    # Escape direction: +1 if m rises toward M_threshold (η > 0), −1 if m falls
    # (η ≤ 0). All "has λ reached/passed M_threshold?" tests must respect this;
    # plain `lambda_next >= M_threshold` only works for dir = +1.
    dir = escape_dir(eta)
    if verbose
        @printf("  Phase 0: measuring initial flux (λ₀ = %.4f)...\n", lambda_0)
    end
    (phi_0, configs) = measure_initial_flux(L, eta, p_noise, update, lambda_0,
                                             M_threshold, n_configs_per_run,
                                             max_time_per_trial; verbose)
    n_crossings = length(configs)
    if verbose
        @printf("  Φ₀ = %.6e (%d configs)\n", phi_0, n_crossings)
    end

    if phi_0 == 0.0 || isempty(configs)
        return (Inf, Inf, phi_0, true)
    end

    log_product = 0.0
    var_log_tau = 1.0 / n_crossings  # Poisson variance from Phase 0
    current_configs = configs
    phase = 0
    lambda_history = [lambda_0]

    # Two modes (see ising.jl for the rationale):
    #   - canonical (n_interfaces > 0): predetermined uniformly-spaced interfaces
    #     between λ_0 and M_threshold; λ_fail = λ_0.
    #   - adaptive (n_interfaces == 0): probe-driven; λ_fail steps back n_lookback.
    # Note: in this script's sign convention m INCREASES toward M_threshold, so
    # `range(λ_0, M_threshold)` is ascending.
    n_lookback = 7
    fixed_lambdas = n_interfaces > 0 ?
        collect(range(lambda_0, M_threshold, length=n_interfaces + 1)) : Float64[]

    while true
        phase += 1
        if phase > 200
            return (Inf, Inf, phi_0, true)
        end

        if n_interfaces > 0
            if phase > n_interfaces
                break
            end
            lambda_fail = lambda_0
            lambda_next = fixed_lambdas[phase + 1]  # phase 1 → fixed_lambdas[2]
        else
            fail_idx = max(1, length(lambda_history) - n_lookback + 1)
            lambda_fail = lambda_history[fail_idx]

            (lambda_next, probe_timed_out) = probe_next_interface(
                L, eta, p_noise, update, current_configs, lambda_fail,
                M_threshold, max_time_per_trial, n_configs_per_run, target_crossing_prob)

            if probe_timed_out
                return (Inf, Inf, phi_0, true)
            end

            if forward_of(lambda_next, M_threshold, dir)
                lambda_next = M_threshold
            end
        end

        (prob, new_configs, n_success, n_trials, cross_timed_out) = measure_crossing_probability(
            L, eta, p_noise, update, current_configs, lambda_next, lambda_fail,
            max_time_per_trial, n_configs_per_run)

        if verbose
            @printf("  Phase %d: λ = %.4f, P_%d = %.4f (%d/%d trials)\n",
                    phase, lambda_next, phase, prob, n_success, n_trials)
        end

        if cross_timed_out || prob == 0.0 || isempty(new_configs)
            return (Inf, Inf, phi_0, true)
        end

        log_product += log10(prob)
        var_log_tau += (1 - prob) / (prob * n_trials)
        current_configs = new_configs
        push!(lambda_history, lambda_next)

        if reached_thresh(lambda_next, M_threshold, dir)
            break
        end
    end

    log_tau = -log10(phi_0) - log_product
    if verbose
        @printf("  %d phases, log₁₀(τ) = %.2f\n", phase, log_tau)
    end
    return (log_tau, var_log_tau, phi_0, false)
end

# ---------------------------------------------------------------------
# FFS top-level: sweep over p_noise (default) or eta (vary_eta=true)
# (mirror of src/ffs.jl:369-572)
# ---------------------------------------------------------------------
function run_ffs_mode(; L::Int, eta::Float64, p_noise::Float64,
                       p_min::Float64, p_max::Float64,
                       eta_min::Float64, eta_max::Float64,
                       n_steps::Int, vary_eta::Bool,
                       n_configs_per_run::Int, n_repeats::Int,
                       M_threshold::Float64,
                       max_time_per_trial::Int,
                       target_crossing_prob::Float64=0.20,
                       n_interfaces::Int=0,
                       adaptive_L::Bool=false, adaptive_factor::Float64=3.0,
                       update::Symbol=:sync,
                       lambda_0_override::Float64=NaN,
                       sweep_values_override::Union{Nothing,Vector{Float64}}=nothing,
                       kwargs...)
    t_start = time()
    eta = abs(eta)  # user convention: eta > 0 ⇒ physical stable = +1
    L_label = adaptive_L ?
        @sprintf("L = %.2f × ℓ_er per sweep point", adaptive_factor) :
        "L = $L"
    if vary_eta
        sweep_values = collect(range(eta_min, eta_max, length=n_steps))
        all(x -> x > 0, sweep_values) || error("All η values must be > 0 for FFS (need a metastable basin).")
        p_fixed = p_noise
        println("=== Noisy GKL: FFS Mode (sweeping η) ===")
        println("$L_label, p_noise = $p_fixed, update = $update")
        println("η values: $sweep_values")
    else
        # Sweep linear in user's chosen parametrization (p, q=1/p, τ=1/√p, or r=(log(1/p))²).
        # CLI dispatch has already converted to p values when q/τ/r was used.
        sweep_values = sweep_values_override === nothing ?
            collect(range(p_min, p_max, length=n_steps)) : sweep_values_override
        # Always run the smallest p_noise (rarest event, longest τ) first so the
        # user sees the worst-case runtime upfront.
        sort!(sweep_values)
        println("=== Noisy GKL: FFS Mode (sweeping p_noise) ===")
        println("$L_label, η = $eta, update = $update")
        println("p_noise values: $sweep_values")
        all(x -> x > 0, sweep_values) || error("All p_noise values must be > 0.")
        eta_fixed = eta  # η = 0 is allowed (unbiased GKL); FFS basin choice falls on the user
    end

    if n_interfaces > 0
        @printf("Canonical FFS: %d fixed interfaces, λ_fail = λ_0, M_threshold = %.4f\n",
                n_interfaces, M_threshold)
    else
        @printf("Adaptive FFS: M_threshold = %.4f, target P = %.2f\n",
                M_threshold, target_crossing_prob)
    end
    @printf("n_configs_per_run = %d, n_repeats = %d (total configs/sweep pt = %d)\n",
            n_configs_per_run, n_repeats, n_configs_per_run * n_repeats)
    println("Threads: $(Threads.nthreads())")

    n_sweep = length(sweep_values)
    mean_mixing_times = zeros(Float64, n_sweep)
    log_mixing_times = zeros(Float64, n_sweep)
    log_mixing_times_std = zeros(Float64, n_sweep)
    flux_rates = zeros(Float64, n_sweep)
    per_run_log_taus = fill(NaN, n_sweep, n_repeats)
    L_values = fill(L, n_sweep)        # per sweep point; constant L unless adaptive_L
    lc_values = fill(0, n_sweep)        # per sweep point; 0 means not measured

    # Adaptive-L: warm-start the erosion search from min; subsequent sweep points
    # start near the previous lc.
    adaptive_ler_n_trials = 1000
    adaptive_ler_thresh_prob = 0.75
    adaptive_ler_min = 2
    prev_lc = adaptive_ler_min

    for (idx, val) in enumerate(sweep_values)
        if vary_eta
            eta_cur = val
            p_cur = p_fixed
            @printf("\n[%d/%d] η = %.4f\n", idx, n_sweep, eta_cur)
        else
            p_cur = val
            eta_cur = eta_fixed
            @printf("\n[%d/%d] p_noise = %.4f\n", idx, n_sweep, p_cur)
        end

        # Adaptive L per sweep point
        if adaptive_L
            lc = find_critical_length(eta_cur, p_cur, update, adaptive_ler_n_trials,
                                       prev_lc, adaptive_ler_thresh_prob, adaptive_ler_min)
            L_cur = max(round(Int, adaptive_factor * lc), adaptive_ler_min + 2)
            L_cur = max(L_cur, 7)  # GKLState constructor minimum
            prev_lc = lc
            lc_values[idx] = lc
            L_values[idx] = L_cur
            @printf("  Adaptive L: ℓ_er = %d, L = %d\n", lc, L_cur)
        else
            L_cur = L
        end

        dir_cur = escape_dir(eta_cur)  # +1 (m rises) or −1 (m falls)
        if !isnan(lambda_0_override)
            lambda_0 = lambda_0_override
            @printf("  λ₀ = %.4f (user override)\n", lambda_0)
        else
            (m_meta, m_std) = measure_metastable_magnetization(L_cur, eta_cur, p_cur, update)
            # Place λ₀ 2.5σ past the metastable mean in the escape direction.
            lambda_0 = m_meta + dir_cur * 2.5 * m_std
            @printf("  Metastable: m* = %.4f ± %.4f, λ₀ = %.4f\n", m_meta, m_std, lambda_0)
        end

        # λ₀ should still be on the metastable side of M_threshold; if it has already
        # crossed, the escape isn't a rare event.
        if reached_thresh(lambda_0, M_threshold, dir_cur)
            @warn "λ₀ ($(@sprintf("%.4f", lambda_0))) has crossed M_threshold ($(@sprintf("%.4f", M_threshold))) in escape direction — transition is not a rare event"
            mean_mixing_times[idx] = NaN
            log_mixing_times[idx] = NaN
            log_mixing_times_std[idx] = NaN
            continue
        end

        log_taus = Float64[]
        var_log_taus = Float64[]
        per_run_log_tau = fill(NaN, n_repeats)
        phi_0_sum = 0.0

        function _record_run!(rep::Int, log_tau, var_log_tau, phi_0, failed)
            phi_0_sum += phi_0
            if !failed && isfinite(log_tau)
                push!(log_taus, log_tau)
                push!(var_log_taus, var_log_tau)
                per_run_log_tau[rep] = log_tau
            end
        end

        println("  First run:")
        (log_tau, var_log_tau, phi_0, failed) = ffs_single_run(
            L_cur, eta_cur, p_cur, update, lambda_0, M_threshold,
            n_configs_per_run, max_time_per_trial, target_crossing_prob;
            n_interfaces, verbose=true)
        _record_run!(1, log_tau, var_log_tau, phi_0, failed)

        if n_repeats > 1
            progress = Progress(n_repeats - 1; desc="  Remaining runs: ", dt=1.0)
            for rep in 2:n_repeats
                (log_tau, var_log_tau, phi_0, failed) = ffs_single_run(
                    L_cur, eta_cur, p_cur, update, lambda_0, M_threshold,
                    n_configs_per_run, max_time_per_trial, target_crossing_prob;
                    n_interfaces, verbose=false)
                _record_run!(rep, log_tau, var_log_tau, phi_0, failed)
                update!(progress, rep - 1)
            end
            finish!(progress)
        end

        flux_rates[idx] = phi_0_sum / n_repeats
        mean_inv_phi = flux_rates[idx] > 0 ? 1.0 / flux_rates[idx] : Inf
        @printf("  1/Φ₀ = %.1f (averaged over %d runs)\n", mean_inv_phi, n_repeats)

        let parts = String[]
            for (i, lt) in enumerate(per_run_log_tau)
                push!(parts, isnan(lt) ? @sprintf("[%d: fail]", i) :
                                          @sprintf("[%d: %.2f]", i, lt))
            end
            println("  Per-run log₁₀(τ): ", join(parts, " "))
        end
        per_run_log_taus[idx, :] = per_run_log_tau

        if isempty(log_taus)
            mean_mixing_times[idx] = Inf
            log_mixing_times[idx] = Inf
            log_mixing_times_std[idx] = NaN
            @printf("  All %d runs failed — mixing time set to Inf\n", n_repeats)
        else
            n_ok = length(log_taus)
            mean_log = mean(log_taus)
            mean_var = mean(var_log_taus)
            binomial_stderr = sqrt(mean_var / n_ok) / log(10)
            empirical_stderr = n_ok > 1 ? std(log_taus) / sqrt(n_ok) : 0.0
            std_log = max(binomial_stderr, empirical_stderr)

            mean_mixing_times[idx] = 10.0^mean_log
            log_mixing_times[idx] = mean_log
            log_mixing_times_std[idx] = std_log
            @printf("  %d/%d runs succeeded: log₁₀(τ) = %.2f ± %.2f (τ ≈ %.4e)\n",
                    n_ok, n_repeats, mean_log, std_log, 10.0^mean_log)
        end
    end

    elapsed = time() - t_start
    h_part = Int(div(elapsed, 3600))
    m_part = Int(div(rem(elapsed, 3600), 60))
    s_part = rem(elapsed, 60)
    @printf("\nDone! Elapsed: %dh %02dm %05.2fs (%.1f s total)\n",
            h_part, m_part, s_part, elapsed)

    results = Dict{String, Any}(
        "mean_mixing_times" => mean_mixing_times,
        "log_mixing_times" => log_mixing_times,
        "log_mixing_times_std" => log_mixing_times_std,
        "per_run_log_taus" => per_run_log_taus,
        "flux_rates" => flux_rates,
        "L" => L,
        "n_configs_per_run" => n_configs_per_run, "n_repeats" => n_repeats,
        "M_threshold" => M_threshold,
        "target_crossing_prob" => target_crossing_prob,
        "n_interfaces" => n_interfaces,
        "adaptive_L" => adaptive_L,
        "adaptive_factor" => adaptive_factor,
        "L_values" => L_values,
        "lc_values" => lc_values,
        "max_time_per_trial" => max_time_per_trial,
        "update" => String(update),
        "dynamics" => "gkl",
        "elapsed_seconds" => elapsed,
    )
    if vary_eta
        # plotter takes the "vary_v" branch by detecting `vs`
        results["vs"] = sweep_values
        results["p_noise"] = p_fixed
        results["p"] = p_fixed  # routing field for the plotter
    else
        # plotter takes the "vary_p" branch by detecting `p_values`
        results["p_values"] = sweep_values
        results["eta"] = eta_fixed
        results["v"] = 0.0      # routing field; legend label is overridden via Patch E
    end
    return results
end

### =====================================================================
### §5  ler mode (erosion length)
###
### Init convention (nucleation-style): the bulk is initialized in the
### METASTABLE phase (anti-aligned with eta, so e.g. eta>0 ⇒ bulk = -1),
### and the minority cluster is ALIGNED with eta (so eta>0 ⇒ minority = +1).
### The minority is therefore the "incipient nucleus" of the stable phase.
### Under deterministic GKL the minority erodes (sign-symmetric rule); the
### biased noise pressures the bulk to nucleate toward +1, so the test asks
### whether a cluster of size l survives long enough to grow.
### This regime is only meaningful at LOW p_noise (the metastable bulk must
### survive over the trial time T_evolve = factor*l).
### =====================================================================

# ---------------------------------------------------------------------
# Single-cluster shrink probability at domain size l
# (single-chain analog of src/erosion_test.jl:262-298)
# ---------------------------------------------------------------------
function measure_shrink_prob(l::Int, eta::Float64, p_noise::Float64, update::Symbol,
                              n_trials::Int;
                              t_evolve_factor::Float64=2.0,
                              L_sys_factor::Float64=1.0)
    L_sys = max(l + 2, round(Int, L_sys_factor * 4 * l))
    L_sys >= 7 || (L_sys = 7)  # GKLState constructor minimum
    T_evolve = max(1, round(Int, t_evolve_factor * l))
    # Initial minority count is ℓ (single chain); "shrunk" means lost ≥ half the
    # eroder. (The sliding-Ising analog uses 1.5ℓ because it has 2ℓ initial
    # minority spins across two chains — same "≥25% loss" semantics. Here we use
    # the cleaner "≥50% loss" definition.)
    threshold = 0.5 * l
    bulk = eta >= 0 ? Int8(-1) : Int8(1)        # anti-aligned w/ bias = metastable
    minority = -bulk                              # aligned w/ bias

    shrink_flags = zeros(Int, n_trials)
    Threads.@threads for trial in 1:n_trials
        state = GKLState(L_sys, eta, p_noise; update)
        fill!(state.sigma, bulk)
        @inbounds for i in 1:l
            state.sigma[i] = minority
        end
        recompute_mag_sum!(state)

        evolve_time!(state, T_evolve)

        n_minority = count(==(minority), state.sigma)
        if n_minority < threshold
            shrink_flags[trial] = 1
        end
    end
    return sum(shrink_flags) / n_trials
end

# ---------------------------------------------------------------------
# Standard error of lc from cached bracket
# (verbatim from src/erosion_test.jl:407-419)
# ---------------------------------------------------------------------
function lc_stderr_from_cache(lc::Int, cache::Dict{Int,Float64}, n_trials::Int,
                              thresh_prob::Float64)
    haskey(cache, lc) && haskey(cache, lc - 1) || return NaN
    p1 = cache[lc - 1]
    p2 = cache[lc]
    delta = p1 - p2
    isfinite(delta) && delta > 0 || return NaN
    sigma_p1 = sqrt(max(p1 * (1 - p1), 0.0) / n_trials)
    sigma_p2 = sqrt(max(p2 * (1 - p2), 0.0) / n_trials)
    a = thresh_prob - p2
    b = p1 - thresh_prob
    return sqrt(a^2 * sigma_p1^2 + b^2 * sigma_p2^2) / delta^2
end

# ---------------------------------------------------------------------
# Adaptive critical-length search (3-phase)
# (mirror of src/erosion_test.jl:430-528, simplified for single chain)
# ---------------------------------------------------------------------
function find_critical_length(eta::Float64, p_noise::Float64, update::Symbol,
                              n_trials::Int, l_start::Int, thresh_prob::Float64,
                              min_erosion_length::Int=2;
                              t_evolve_factor::Float64=2.0,
                              L_sys_factor::Float64=1.0,
                              cache::Dict{Int,Float64}=Dict{Int,Float64}())
    coarse_trials = min(n_trials, max(200, n_trials ÷ 10))

    function measure_and_cache(l, n)
        sp = measure_shrink_prob(l, eta, p_noise, update, n; t_evolve_factor, L_sys_factor)
        if n == n_trials
            cache[l] = sp
        end
        return sp
    end

    l_lo = max(min_erosion_length, l_start - 2)

    # Phase 1: bracket lc
    sp = measure_and_cache(l_lo, coarse_trials)
    if sp < thresh_prob
        while l_lo > min_erosion_length
            l_lo = max(min_erosion_length, l_lo - 2)
            sp = measure_and_cache(l_lo, coarse_trials)
            @printf("    coarse ↓: l = %d, p_shrink = %.3f (%d trials)\n", l_lo, sp, coarse_trials)
            if sp >= thresh_prob
                break
            end
        end
        if sp < thresh_prob
            sp_full = measure_and_cache(min_erosion_length, n_trials)
            @printf("    verify: l = %d, p_shrink = %.3f (%d trials)\n", min_erosion_length, sp_full, n_trials)
            if sp_full < thresh_prob
                return min_erosion_length
            end
            l_lo = min_erosion_length
            sp = sp_full
        end
    end

    # Cap the upward search so we don't spin forever in regimes where the
    # eroder always wins (sp stays >= thresh_prob for every l we try).
    l_search_max = max(500, 20 * max(l_start, 1))
    step = 1
    l_hi = l_lo
    while true
        l_hi = l_lo + step
        if l_hi > l_search_max
            @warn "ler search: l_hi exceeded cap $l_search_max with p_shrink still >= $thresh_prob. " *
                  "Returning $l_search_max as 'lc' but the system is likely in the eroder-dominated " *
                  "phase where lc is effectively infinite. Increase --p_noise or raise the cap manually."
            return l_search_max
        end
        sp = measure_and_cache(l_hi, coarse_trials)
        @printf("    coarse ↑: l = %d, p_shrink = %.3f (%d trials)\n", l_hi, sp, coarse_trials)
        if sp < thresh_prob
            break
        end
        l_lo = l_hi
        step = min(step * 2, 16)
    end

    # Phase 2: bisection
    while l_hi - l_lo > 1
        l_mid = (l_lo + l_hi) ÷ 2
        sp = measure_and_cache(l_mid, coarse_trials)
        @printf("    bisect: l = %d, p_shrink = %.3f (%d trials)\n", l_mid, sp, coarse_trials)
        if sp >= thresh_prob
            l_lo = l_mid
        else
            l_hi = l_mid
        end
    end

    # Phase 3: verify
    l = l_hi
    while true
        sp_at_l = measure_and_cache(l, n_trials)
        @printf("    verify: l = %d, p_shrink = %.3f (%d trials)\n", l, sp_at_l, n_trials)
        if sp_at_l < thresh_prob
            if l <= min_erosion_length
                return l
            end
            sp_below = measure_and_cache(l - 1, n_trials)
            @printf("    verify: l = %d, p_shrink = %.3f (%d trials)\n", l - 1, sp_below, n_trials)
            if sp_below >= thresh_prob
                return l
            else
                l -= 1
            end
        else
            l += 1
        end
    end
end

# ---------------------------------------------------------------------
# Erode-vs-l: survival probability over l ∈ [lc/5, lc]
# (mirror of src/erosion_test.jl:74-112)
# ---------------------------------------------------------------------
function plot_erode_vs_l(lc::Int, eta::Float64, p_noise::Float64, update::Symbol,
                          n_samples::Int;
                          t_evolve_factor::Float64=2.0,
                          L_sys_factor::Float64=1.0,
                          cache::Dict{Int,Float64}=Dict{Int,Float64}())
    l_min = max(4, round(Int, lc / 5))
    l_max = lc
    l_values = collect(unique(round.(Int, range(l_min, l_max, length=20))))
    println("  Measuring shrink prob vs l: l ∈ [$l_min, $l_max], $(length(l_values)) points, $n_samples samples each")

    probs = Float64[]
    for l in l_values
        if haskey(cache, l)
            sp = cache[l]
            @printf("    l = %d, p_shrink = %.3f (cached)\n", l, sp)
        else
            sp = measure_shrink_prob(l, eta, p_noise, update, n_samples;
                                     t_evolve_factor, L_sys_factor)
            cache[l] = sp
            @printf("    l = %d, p_shrink = %.3f\n", l, sp)
        end
        push!(probs, sp)
    end

    return (l_values, collect(probs))
end

# ---------------------------------------------------------------------
# Spacetime history for a single erosion trial (used by show_histories)
# (mirror of src/erosion_test.jl:7-27)
# ---------------------------------------------------------------------
function run_erosion_history(l::Int, eta::Float64, p_noise::Float64, update::Symbol;
                              t_evolve_factor::Float64=2.0,
                              L_sys_factor::Float64=1.0)
    L_sys = max(l + 2, round(Int, L_sys_factor * 4 * l))
    L_sys >= 7 || (L_sys = 7)
    T_evolve = max(1, round(Int, t_evolve_factor * l))

    bulk = eta >= 0 ? Int8(-1) : Int8(1)
    minority = -bulk

    state = GKLState(L_sys, eta, p_noise; update)
    fill!(state.sigma, bulk)
    @inbounds for i in 1:l
        state.sigma[i] = minority
    end
    recompute_mag_sum!(state)

    history = zeros(Int8, T_evolve + 1, L_sys)
    history[1, :] = state.sigma
    evolve_and_measure!(state, T_evolve, step -> begin
        history[step + 1, :] = state.sigma
        true
    end)
    return history
end

function show_erosion_histories(lc::Int, eta::Float64, p_noise::Float64, update::Symbol;
                                 t_evolve_factor::Float64=2.0,
                                 L_sys_factor::Float64=1.0)
    @eval Main using Plots
    for (scale, label) in [(1.0, "lc"), (0.75, "0.75·lc"), (1.25, "1.25·lc")]
        l = max(2, round(Int, scale * lc))
        println("  Showing 3 histories for l = $l ($label)")
        plots = []
        for trial in 1:3
            hist = run_erosion_history(l, eta, p_noise, update; t_evolve_factor, L_sys_factor)
            p = Base.invokelatest(Plots.heatmap, hist;
                                   c=:RdBu, clims=(-1.5, 1.5),
                                   xlabel="Site", ylabel="t",
                                   title="$label, trial $trial",
                                   aspect_ratio=:auto, size=(300, 600))
            push!(plots, p)
        end
        fig = Base.invokelatest(Plots.plot, plots...; layout=(1, 3), size=(900, 600))
        Base.invokelatest(display, fig)
        println("  Close the plot window to continue...")
        readline()
    end
end

# ---------------------------------------------------------------------
# ler mode top-level
# (mirror of src/erosion_test.jl:114-245, simplified)
# ---------------------------------------------------------------------
function run_ler_mode(; eta::Float64, p_noise::Float64,
                       p_min::Float64, p_max::Float64,
                       eta_min::Float64, eta_max::Float64,
                       n_steps::Int, vary_eta::Bool,
                       thresh_prob::Float64,
                       erosion_num_trials::Int,
                       min_erosion_length::Int=2,
                       update::Symbol=:sync,
                       t_evolve_factor::Float64=2.0,
                       L_sys_factor::Float64=1.0,
                       erode_vs_l::Bool=false,
                       show_histories::Bool=false,
                       sweep_values_override::Union{Nothing,Vector{Float64}}=nothing,
                       kwargs...)
    eta = abs(eta)
    if vary_eta
        sweep_values = collect(range(eta_min, eta_max, length=n_steps))
        all(x -> x > 0, sweep_values) || error("All η values must be > 0 for ler mode.")
        p_fixed = p_noise
        println("=== Noisy GKL: ler Mode (sweeping η) ===")
        println("p_noise = $p_fixed, update = $update")
        println("η values: $sweep_values")
    else
        sweep_values = sweep_values_override === nothing ?
            collect(range(p_min, p_max, length=n_steps)) : sweep_values_override
        println("=== Noisy GKL: ler Mode (sweeping p_noise) ===")
        println("η = $eta, update = $update")
        println("p_noise values: $sweep_values")
        all(x -> x > 0, sweep_values) || error("All p_noise values must be > 0.")
        eta_fixed = eta  # η = 0 is allowed (unbiased GKL); erosion is a local property
    end
    println("Trials per domain size: $erosion_num_trials, threads: $(Threads.nthreads())")

    lc_values = Int[]
    lc_stderrs = Float64[]
    prev_lc = min_erosion_length
    erode_l_data = Vector{Vector{Int}}()
    erode_prob_data = Vector{Vector{Float64}}()

    for (idx, val) in enumerate(sweep_values)
        if vary_eta
            eta_cur = val
            p_cur = p_fixed
            @printf("\n[%d/%d] η = %.4f\n", idx, length(sweep_values), eta_cur)
        else
            p_cur = val
            eta_cur = eta_fixed
            @printf("\n[%d/%d] p_noise = %.4f\n", idx, length(sweep_values), p_cur)
        end

        sp_cache = Dict{Int,Float64}()
        lc = find_critical_length(eta_cur, p_cur, update, erosion_num_trials,
                                  prev_lc, thresh_prob, min_erosion_length;
                                  t_evolve_factor, L_sys_factor, cache=sp_cache)
        lc_se = lc_stderr_from_cache(lc, sp_cache, erosion_num_trials, thresh_prob)
        @printf("  lc = %d ± %.3f\n", lc, lc_se)
        push!(lc_values, lc)
        push!(lc_stderrs, lc_se)
        prev_lc = lc

        if show_histories
            show_erosion_histories(lc, eta_cur, p_cur, update;
                                   t_evolve_factor, L_sys_factor)
        end
        if erode_vs_l
            (ls, ps) = plot_erode_vs_l(lc, eta_cur, p_cur, update,
                                       erosion_num_trials;
                                       t_evolve_factor, L_sys_factor,
                                       cache=sp_cache)
            push!(erode_l_data, ls)
            push!(erode_prob_data, ps)
        end
    end

    println("\nDone!")

    results = Dict{String, Any}(
        "lc_values" => lc_values,
        "lc_stderrs" => lc_stderrs,
        "num_trials" => erosion_num_trials,
        "vary_eta" => vary_eta,
        "t_evolve_factor" => t_evolve_factor,
        "L_sys_factor" => L_sys_factor,
        "thresh_prob" => thresh_prob,
        "update" => String(update),
        "dynamics" => "gkl",
    )
    if vary_eta
        results["vs"] = sweep_values
        results["p_noise"] = p_fixed
        results["p"] = p_fixed   # plotter routing
    else
        results["p_values"] = sweep_values
        results["eta"] = eta_fixed
        results["v"] = 0.0       # plotter routing
    end
    if erode_vs_l
        max_len = maximum(length.(erode_l_data))
        n_curves = length(erode_l_data)
        l_matrix = zeros(Int, max_len, n_curves)
        prob_matrix = fill(NaN, max_len, n_curves)
        for i in 1:n_curves
            n = length(erode_l_data[i])
            l_matrix[1:n, i] = erode_l_data[i]
            prob_matrix[1:n, i] = erode_prob_data[i]
        end
        results["erode_l_values"] = l_matrix
        results["erode_probs"] = prob_matrix
    end
    return results
end

# ---------------------------------------------------------------------
# Domain-wall diffusion constant
# ---------------------------------------------------------------------

"""
    _fit_diffusion(ts, msd) -> (D, D_stderr)

OLS fit of MSD(t) = 2 D t + c (dropping t=0). Returns the diffusion constant
and the standard error of the slope. Uses the analytic OLS covariance
σ² (XᵀX)⁻¹ with σ² = (residual sum of squares) / (n − 2).
"""
function _fit_diffusion(ts::Vector{Int}, msd::Vector{Float64})
    keep = ts .> 0
    x = float.(ts[keep])
    y = msd[keep]
    n = length(x)
    n >= 2 || return (NaN, NaN)
    X = hcat(x, ones(n))
    β = X \ y           # [slope, intercept]
    resid = y .- X * β
    σ² = sum(abs2, resid) / max(n - 2, 1)
    XtX_inv = inv(X' * X)
    var_slope = σ² * XtX_inv[1, 1]
    slope = β[1]
    return (slope / 2, sqrt(max(var_slope, 0.0)) / 2)
end

"""
    measure_diffusion_at_p(L, eta, p_noise, update, init_mag,
                           T_thermalize, T_track, n_trials)
        -> (D, D_stderr, msd_curve)

Run `n_trials` parallel domain-wall trajectories. Each trial:
  1. Initialize σ_i = init_mag for i ≤ L÷2 and -init_mag otherwise.
  2. Thermalize `T_thermalize` GKL steps.
  3. Record x_dw(t) = (m(t) + init_mag)/2 for t = 0..T_track.
Across trials, compute MSD(t) = ⟨(Δx_dw − ⟨Δx_dw⟩)²⟩ (drift subtracted so the
estimator is unbiased even at η ≠ 0), then OLS-fit MSD = 2 D t + c.
"""
function measure_diffusion_at_p(L::Int, eta::Float64, p_noise::Float64,
                                  update::Symbol, init_mag::Int,
                                  T_thermalize::Int, T_track::Int, n_trials::Int)
    init_val = Int8(init_mag)
    xdw = zeros(Float64, n_trials, T_track + 1)

    Threads.@threads for trial in 1:n_trials
        state = GKLState(L, eta, p_noise; update, init_value=init_val)
        @inbounds for i in (L ÷ 2 + 1):L
            state.sigma[i] = -init_val
        end
        recompute_mag_sum!(state)
        evolve_time!(state, T_thermalize)
        # Wall position in lattice-site units: W = Σ (σ_i + init_mag)/2
        # = L · (m + init_mag)/2. Keeping it extensive makes D L-independent
        # in the L → ∞ limit (current bug was that W/L gives D ∝ 1/L²).
        xdw[trial, 1] = L * (compute_total_magnetization(state) + init_mag) / 2
        evolve_and_measure!(state, T_track, step -> begin
            xdw[trial, step + 1] = L * (compute_total_magnetization(state) + init_mag) / 2
            true
        end)
    end

    # Drift-subtracted ensemble MSD.
    dx = xdw .- xdw[:, 1]
    mean_dx = vec(mean(dx, dims=1))  # length T_track+1
    msd = vec(mean((dx .- mean_dx').^2, dims=1))
    ts = collect(0:T_track)
    (D, D_stderr) = _fit_diffusion(ts, msd)
    return (D, D_stderr, msd)
end

function run_diffusion_mode(; L::Int, eta::Float64, p_noise::Float64,
                              p_min::Float64, p_max::Float64,
                              eta_min::Float64, eta_max::Float64,
                              n_steps::Int, vary_eta::Bool,
                              init_mag::Int,
                              T_thermalize::Int, T_track::Int, n_trials::Int,
                              update::Symbol=:sync,
                              sweep_values_override::Union{Nothing,Vector{Float64}}=nothing,
                              kwargs...)
    abs(init_mag) == 1 || error("init_mag must be ±1 (got $init_mag).")
    if vary_eta
        sweep_values = collect(range(eta_min, eta_max, length=n_steps))
        p_fixed = p_noise
        sweep_mode = "eta"
        println("=== Noisy GKL: diffusion Mode (sweeping η) ===")
        println("L = $L, p_noise = $p_fixed, update = $update, init_mag = $init_mag")
        println("T_thermalize = $T_thermalize, T_track = $T_track, n_trials = $n_trials")
        println("η values: $sweep_values")
    else
        sweep_values = sweep_values_override === nothing ?
            collect(range(p_min, p_max, length=n_steps)) : sweep_values_override
        sweep_mode = "p"
        println("=== Noisy GKL: diffusion Mode (sweeping p_noise) ===")
        println("L = $L, η = $eta, update = $update, init_mag = $init_mag")
        println("T_thermalize = $T_thermalize, T_track = $T_track, n_trials = $n_trials")
        println("p_noise values: $sweep_values")
        all(x -> x >= 0, sweep_values) || error("All p_noise values must be ≥ 0.")
        eta_fixed = eta
    end

    n_sweep = length(sweep_values)
    D_values = fill(NaN, n_sweep)
    D_stderrs = fill(NaN, n_sweep)
    msd_curve_first = Float64[]  # populated from the first sweep point as a diagnostic

    for (idx, val) in enumerate(sweep_values)
        if vary_eta
            eta_cur = val
            p_cur = p_fixed
            @printf("\n[%d/%d] η = %.4f\n", idx, n_sweep, eta_cur)
        else
            p_cur = val
            eta_cur = eta_fixed
            @printf("\n[%d/%d] p_noise = %.4f\n", idx, n_sweep, p_cur)
        end
        t0 = time()
        (D, D_stderr, msd) = measure_diffusion_at_p(L, eta_cur, p_cur, update,
                                                    init_mag, T_thermalize, T_track,
                                                    n_trials)
        elapsed = time() - t0
        D_values[idx] = D
        D_stderrs[idx] = D_stderr
        @printf("  D = %.4e ± %.4e  (%d trials, %.1f s wall)\n",
                D, D_stderr, n_trials, elapsed)
        if idx == 1
            msd_curve_first = msd
        end
    end

    println("\nDone!")

    results = Dict{String, Any}(
        "D_values" => D_values,
        "D_stderrs" => D_stderrs,
        "L" => L,
        "update" => String(update),
        "init_mag" => Int(init_mag),
        "T_thermalize" => T_thermalize,
        "T_track" => T_track,
        "n_trials" => n_trials,
        "msd_curve_first" => msd_curve_first,
        "sweep_mode" => sweep_mode,  # "p", "tau", or "eta" — drives plotter labels
    )
    if vary_eta
        results["vs"] = sweep_values
        results["p_noise"] = p_fixed
        results["p"] = p_fixed  # plotter routing
    else
        results["p_values"] = sweep_values
        results["eta"] = eta_fixed
    end
    return results
end

### =====================================================================
### §6  save_results / build_filename / dispatch
### =====================================================================

function build_filename(mode::String, results::Dict{String,Any})
    upd = get(results, "update", "sync")
    if mode == "history"
        L = results["L"]
        eta = results["eta"]
        p = results["p_noise"]
        return @sprintf("data/gkl_history_L%d_eta%.3f_p%.3f_%s.jld2", L, eta, p, upd)
    elseif mode == "ffs"
        L = results["L"]
        # Omit L from filename in adaptive_L mode (L varies per sweep point and
        # the default value carried in `results["L"]` is meaningless there).
        is_adaptive = get(results, "adaptive_L", false)
        L_token = is_adaptive ? "" : @sprintf("L%d_", L)
        adaptive_suffix = is_adaptive ?
            @sprintf("_adaptiveLx%g", get(results, "adaptive_factor", 3.0)) : ""
        if haskey(results, "vs")
            # sweeping eta
            p = results["p_noise"]
            etas = results["vs"]
            return @sprintf("data/gkl_ffs_%sp%.3f_eta%.3fto%.3f%s_%s.jld2",
                            L_token, p, minimum(etas), maximum(etas), adaptive_suffix, upd)
        else
            eta = results["eta"]
            ps = results["p_values"]
            return @sprintf("data/gkl_ffs_%seta%.3f_p%.3fto%.3f%s_%s.jld2",
                            L_token, eta, minimum(ps), maximum(ps), adaptive_suffix, upd)
        end
    elseif mode == "ler"
        if get(results, "vary_eta", false)
            p = results["p_noise"]
            etas = results["vs"]
            return @sprintf("data/gkl_ler_p%.3f_eta%.3fto%.3f_%s.jld2",
                            p, minimum(etas), maximum(etas), upd)
        else
            eta = results["eta"]
            ps = results["p_values"]
            return @sprintf("data/gkl_ler_eta%.3f_p%.3fto%.3f_%s.jld2",
                            eta, minimum(ps), maximum(ps), upd)
        end
    elseif mode == "diffusion"
        L = results["L"]
        im = results["init_mag"]
        if haskey(results, "vs")
            p = results["p_noise"]
            etas = results["vs"]
            return @sprintf("data/gkl_diffusion_L%d_p%.3f_eta%.3fto%.3f_im%+d_%s.jld2",
                            L, p, minimum(etas), maximum(etas), im, upd)
        else
            eta = results["eta"]
            ps = results["p_values"]
            return @sprintf("data/gkl_diffusion_L%d_eta%.3f_p%.3fto%.3f_im%+d_%s.jld2",
                            L, eta, minimum(ps), maximum(ps), im, upd)
        end
    else
        return "data/gkl_$(mode).jld2"
    end
end

function save_results(mode::String, results::Dict{String,Any}; adj::String="")
    if !isdir("data")
        mkdir("data")
    end
    filename = build_filename(mode, results)
    if adj != ""
        filename = replace(filename, ".jld2" => "$(adj).jld2")
    end
    jldsave(filename; (Symbol(k) => v for (k, v) in results)...)
    println("Results saved to $filename")
    clipboard(filename)
end

function run_simulation(mode::String; save::Bool=true, adj::String="", kwargs...)
    kw = Dict(kwargs)
    results = if mode == "history"
        run_history_mode(; kw...)
    elseif mode == "ffs"
        run_ffs_mode(; kw...)
    elseif mode == "ler"
        run_ler_mode(; kw...)
    elseif mode == "diffusion"
        run_diffusion_mode(; kw...)
    else
        error("Unknown mode: $mode. Use 'history', 'ffs', 'ler', or 'diffusion'.")
    end
    if save
        save_results(mode, results; adj)
    end
    return results
end

### =====================================================================
### §7  top-level: defaults per mode, CLI parse, dispatch
### =====================================================================

# p/q/τ/r parametrization helpers for GKL.
# Convention: p = bare per-cell noise rate ∈ (0, 1]; q = 1/p; τ = 1/√p ⇒ p = 1/τ²;
# r = (log(1/p))² = (β J)² ⇒ p = exp(-√r).
# Sweeps are taken to be linear in *whichever parameter the user specifies*
# (p, q, τ, or r). See _resolve_pqtau_gkl below.
_p_from_tau_gkl(t::Float64) = (t > 0 || error("--tau must be > 0 (got τ=$t)."); 1.0 / (t * t))
_p_from_r_gkl(r::Float64) = (r > 0 || error("--r must be > 0 (got r=$r)."); exp(-sqrt(r)))

"""
Resolve --p_min/--p_max/--p, --q_min/--q_max/--q, --tau_min/--tau_max/--tau,
--r_min/--r_max/--r into (p_min, p_max, p, sweep_override) for a GKL run.
The sweep is linear in the user's chosen parametrization; priority on the
fixed value is r > τ > q > p. `sweep_override` is `nothing` when the user
only used --p_min/--p_max (the mode function then does its own linspace).
"""
function _resolve_pqtau_gkl(p_min_default::Float64, p_max_default::Float64,
                            p_default::Float64, n_steps::Int)
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
        sweep_override = [_p_from_r_gkl(rv) for rv in r_values]
        p_min, p_max = minimum(sweep_override), maximum(sweep_override)
    elseif !isnan(tau_min) && !isnan(tau_max)
        tau_values = collect(range(tau_min, tau_max, length=n_steps))
        sweep_override = [_p_from_tau_gkl(t) for t in tau_values]
        p_min, p_max = minimum(sweep_override), maximum(sweep_override)
    elseif !isnan(q_min) && !isnan(q_max)
        q_values = collect(range(q_min, q_max, length=n_steps))
        sweep_override = [1.0 / qv for qv in q_values]
        p_min, p_max = minimum(sweep_override), maximum(sweep_override)
    else
        !isnan(q_min) && (p_max = 1.0 / q_min)
        !isnan(q_max) && (p_min = 1.0 / q_max)
        !isnan(tau_min) && (p_max = _p_from_tau_gkl(tau_min))
        !isnan(tau_max) && (p_min = _p_from_tau_gkl(tau_max))
        !isnan(r_min) && (p_max = _p_from_r_gkl(r_min))
        !isnan(r_max) && (p_min = _p_from_r_gkl(r_max))
    end
    if !isnan(r)
        p = _p_from_r_gkl(r)
    elseif !isnan(tau)
        p = _p_from_tau_gkl(tau)
    elseif !isnan(q)
        p = 1.0 / q
    end
    return p_min, p_max, p, sweep_override
end

mode = String(parse_arg("mode", "history"))

# Shared defaults
eta = 0.0  # default to unbiased GKL noise; >0 biases the +1 phase
update_str = "sync"

if mode == "history"
    L = 1000
    T_steps = 500
    p_noise = 0.05
    init = "domain"
    domain_start = 3L ÷ 8
    domain_end = 5L ÷ 8
elseif mode == "ffs"
    L = 500
    p_noise = 0.10
    p_min = 0.02
    p_max = 0.20
    tau_min = NaN  # if set together with tau_max, sweep uniform in τ=1/√p
    tau_max = NaN
    eta_min = 0.01
    eta_max = 0.20
    n_steps = 6
    vary_eta = false
    n_configs_per_run = 100
    n_repeats = 4
    target_crossing_prob = 0.20
    n_interfaces = 0   # 0 = adaptive (default). >0 = canonical FFS with that many fixed interfaces and λ_fail = λ_0.
    adaptive_L = false  # set L = adaptive_factor × ℓ_er per sweep point.
    adaptive_factor = 3.0
    M_threshold = 0.4
    max_time_per_trial = 1_000_000
    lambda_0_override = NaN
elseif mode == "ler"
    p_noise = 0.10
    p_min = 0.02
    p_max = 0.20
    tau_min = NaN  # if set together with tau_max, sweep uniform in τ=1/√p
    tau_max = NaN
    eta_min = 0.01
    eta_max = 0.20
    n_steps = 6
    vary_eta = false
    thresh_prob = 0.75
    erosion_num_trials = 2000
    min_erosion_length = 2
    t_evolve_factor = 2.0
    L_sys_factor = 1.0
    erode_vs_l = false
    show_histories = false
elseif mode == "diffusion"
    L = 500
    p_noise = 0.05
    p_min = 0.02
    p_max = 0.20
    tau_min = NaN
    tau_max = NaN
    eta_min = 0.01
    eta_max = 0.20
    n_steps = 6
    vary_eta = false
    init_mag = 1            # ±1
    T_thermalize = -1        # -1 → L÷2
    T_track = -1             # -1 → 5L
    n_trials = 100
else
    error("Unknown mode: $mode. Use 'history', 'ffs', 'ler', or 'diffusion'.")
end

# CLI overrides
save = parse_arg("save", true)
adj = String(parse_arg("adj", ""))
eta = parse_arg("eta", eta)
update_str = String(parse_arg("update", update_str))
update_str in ("sync", "async") || error("--update must be 'sync' or 'async', got '$update_str'")
update_sym = Symbol(update_str)

if mode == "history"
    L = parse_arg("L", L)
    T_steps = parse_arg("T_steps", T_steps)
    p_noise = parse_arg("p_noise", p_noise)
    init = String(parse_arg("init", init))
    domain_start = parse_arg("domain_start", 3L ÷ 8)
    domain_end = parse_arg("domain_end", 5L ÷ 8)
elseif mode == "ffs"
    L = parse_arg("L", L)
    p_noise = parse_arg("p_noise", p_noise)
    p_min = parse_arg("p_min", p_min)
    p_max = parse_arg("p_max", p_max)
    tau_min = parse_arg("tau_min", tau_min)
    tau_max = parse_arg("tau_max", tau_max)
    eta_min = parse_arg("eta_min", eta_min)
    eta_max = parse_arg("eta_max", eta_max)
    n_steps = parse_arg("n_steps", n_steps)
    vary_eta = parse_arg("vary_eta", vary_eta)
    n_configs_per_run = parse_arg("n_configs_per_run", n_configs_per_run)
    n_repeats = parse_arg("n_repeats", n_repeats)
    target_crossing_prob = parse_arg("target_crossing_prob", target_crossing_prob)
    n_interfaces = parse_arg("n_interfaces", n_interfaces)
    adaptive_L = parse_arg("adaptive_L", adaptive_L)
    adaptive_factor = parse_arg("adaptive_factor", adaptive_factor)
    M_threshold = parse_arg("M_threshold", M_threshold)
    max_time_per_trial = parse_arg("max_time_per_trial", max_time_per_trial)
    lambda_0_override = parse_arg("lambda_0", lambda_0_override)
    (p_min, p_max, p_noise, sweep_values_override) = _resolve_pqtau_gkl(p_min, p_max, p_noise, n_steps)
elseif mode == "ler"
    p_noise = parse_arg("p_noise", p_noise)
    p_min = parse_arg("p_min", p_min)
    p_max = parse_arg("p_max", p_max)
    tau_min = parse_arg("tau_min", tau_min)
    tau_max = parse_arg("tau_max", tau_max)
    eta_min = parse_arg("eta_min", eta_min)
    eta_max = parse_arg("eta_max", eta_max)
    n_steps = parse_arg("n_steps", n_steps)
    vary_eta = parse_arg("vary_eta", vary_eta)
    thresh_prob = parse_arg("thresh_prob", thresh_prob)
    erosion_num_trials = parse_arg("erosion_num_trials", erosion_num_trials)
    min_erosion_length = parse_arg("min_erosion_length", min_erosion_length)
    t_evolve_factor = parse_arg("t_evolve_factor", t_evolve_factor)
    L_sys_factor = parse_arg("L_sys_factor", L_sys_factor)
    erode_vs_l = parse_arg("erode_vs_l", erode_vs_l)
    show_histories = parse_arg("show_histories", show_histories)
    (p_min, p_max, p_noise, sweep_values_override) = _resolve_pqtau_gkl(p_min, p_max, p_noise, n_steps)
elseif mode == "diffusion"
    L = parse_arg("L", L)
    p_noise = parse_arg("p_noise", p_noise)
    p_min = parse_arg("p_min", p_min)
    p_max = parse_arg("p_max", p_max)
    tau_min = parse_arg("tau_min", tau_min)
    tau_max = parse_arg("tau_max", tau_max)
    eta_min = parse_arg("eta_min", eta_min)
    eta_max = parse_arg("eta_max", eta_max)
    n_steps = parse_arg("n_steps", n_steps)
    vary_eta = parse_arg("vary_eta", vary_eta)
    init_mag = parse_arg("init_mag", init_mag)
    T_thermalize = parse_arg("T_thermalize", T_thermalize)
    T_track = parse_arg("T_track", T_track)
    n_trials = parse_arg("n_trials", n_trials)
    # Materialize the -1 sentinels (we needed L to be parsed first).
    T_thermalize = T_thermalize < 0 ? L ÷ 2 : T_thermalize
    T_track = T_track < 0 ? 5 * L : T_track
    (p_min, p_max, p_noise, sweep_values_override) = _resolve_pqtau_gkl(p_min, p_max, p_noise, n_steps)
end

# Build kwargs and dispatch
kwargs = Dict{Symbol, Any}()
kwargs[:update] = update_sym
kwargs[:eta] = eta

if mode == "history"
    kwargs[:L] = L
    kwargs[:T_steps] = T_steps
    kwargs[:p_noise] = p_noise
    kwargs[:init] = init
    kwargs[:domain_start] = domain_start
    kwargs[:domain_end] = domain_end
elseif mode == "ffs"
    kwargs[:L] = L
    kwargs[:p_noise] = p_noise
    kwargs[:p_min] = p_min
    kwargs[:p_max] = p_max
    kwargs[:sweep_values_override] = sweep_values_override
    kwargs[:eta_min] = eta_min
    kwargs[:eta_max] = eta_max
    kwargs[:n_steps] = n_steps
    kwargs[:vary_eta] = vary_eta
    kwargs[:n_configs_per_run] = n_configs_per_run
    kwargs[:n_repeats] = n_repeats
    kwargs[:target_crossing_prob] = target_crossing_prob
    kwargs[:n_interfaces] = n_interfaces
    kwargs[:adaptive_L] = adaptive_L
    kwargs[:adaptive_factor] = adaptive_factor
    kwargs[:M_threshold] = M_threshold
    kwargs[:max_time_per_trial] = max_time_per_trial
    kwargs[:lambda_0_override] = lambda_0_override
elseif mode == "ler"
    kwargs[:p_noise] = p_noise
    kwargs[:p_min] = p_min
    kwargs[:p_max] = p_max
    kwargs[:sweep_values_override] = sweep_values_override
    kwargs[:eta_min] = eta_min
    kwargs[:eta_max] = eta_max
    kwargs[:n_steps] = n_steps
    kwargs[:vary_eta] = vary_eta
    kwargs[:thresh_prob] = thresh_prob
    kwargs[:erosion_num_trials] = erosion_num_trials
    kwargs[:min_erosion_length] = min_erosion_length
    kwargs[:t_evolve_factor] = t_evolve_factor
    kwargs[:L_sys_factor] = L_sys_factor
    kwargs[:erode_vs_l] = erode_vs_l
    kwargs[:show_histories] = show_histories
elseif mode == "diffusion"
    kwargs[:L] = L
    kwargs[:p_noise] = p_noise
    kwargs[:p_min] = p_min
    kwargs[:p_max] = p_max
    kwargs[:sweep_values_override] = sweep_values_override
    kwargs[:eta_min] = eta_min
    kwargs[:eta_max] = eta_max
    kwargs[:n_steps] = n_steps
    kwargs[:vary_eta] = vary_eta
    kwargs[:init_mag] = init_mag
    kwargs[:T_thermalize] = T_thermalize
    kwargs[:T_track] = T_track
    kwargs[:n_trials] = n_trials
end

n_threads = Threads.nthreads()
if n_threads == 1
    @warn "Running with only 1 thread. For parallel trials, start Julia with: julia -t auto"
end

println("Running gkl: mode=$mode, threads=$n_threads")
run_simulation(mode; save=save, adj=adj, kwargs...)
println("Simulation complete!")
try run(`afplay /System/Library/Sounds/Glass.aiff`) catch end


