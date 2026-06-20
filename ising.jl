#!/usr/bin/env julia
#=
ising.jl — Glauber-dynamics FFS benchmark on the 1D Ising chain.

Standalone companion to `simulation_driver.jl` and `gkl.jl`; does NOT depend
on the SlidingLayers module.

Model: 1D Ising chain, H = -J Σ s_i s_{i+1}, periodic BCs, s_i ∈ {-1, +1},
h = 0. Glauber (heat-bath) single-spin-flip dynamics:
    P_flip = 1 / (1 + exp(β·dE))

Parametrization: primary is p = exp(-β J) ∈ (0, 1], same convention as the
rest of the codebase (small p = rare-event regime). Equivalent aliases:
  q  = 1/p = exp(β J)
  τ  = exp(4 (β J)²)
You may set ranges via any of {p, q, τ}; the sweep is taken to be linear in
the parameter you specify (CLI dispatch converts to p internally).

Modes:
  --mode=history   Spacetime history of σ(x, t) for sanity.
  --mode=ffs       Forward Flux Sampling for the mixing time τ_mem.
  --mode=mixing    Direct ensemble-average measurement of τ_mem (benchmark for FFS).

Time unit: one sweep = L single-spin updates (matches src/core.jl convention).

Usage:
  julia --project=. ising.jl --mode=history --L=200 --T_steps=300 --p=0.5 --show_plot=true
  julia --project=. -t auto ising.jl --mode=ffs --L=200 --p_min=0.35 --p_max=0.55 --n_steps=3 \
        --n_configs_per_run=50 --n_repeats=2 --M_threshold=0.75 --max_time_per_trial=200000
  julia --project=. -t auto ising.jl --mode=mixing --L=200 --q_min=1.8 --q_max=2.9 --n_steps=3 \
        --n_trials=100 --M_threshold=0.75 --max_time=200000
=#

try
    using Revise
catch
end

using Printf, Random, JLD2, Statistics, ProgressMeter

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
### §1  IsingState + Glauber single-spin dynamics
### =====================================================================

mutable struct IsingState
    L::Int
    beta::Float64
    sigma::Vector{Int8}        # length L
    accept::Matrix{Float64}    # 2×3: (σ_idx, ns_idx)
    mag_sum::Int               # running Σ σ_i; kept in sync by every mutator
end

# --------------------------------------------------------------
# Glauber acceptance table:
#   indices: (σ_idx, ns_idx) ∈ {1,2} × {1,2,3}
#     σ_idx maps {-1 → 1, +1 → 2}      via (σ+1) >> 1 + 1
#     ns_idx maps {-2 → 1, 0 → 2, +2 → 3} via (ns + 4) >> 1
#   formula: P_flip = 1 / (1 + exp(β·dE)) with dE = 2σ·ns at h=0.
# Mirrors src/types.jl:35-43 but with Glauber instead of Metropolis.
# --------------------------------------------------------------
function build_glauber_table(beta::Float64)
    table = zeros(Float64, 2, 3)
    for (s_idx, σ) in enumerate((-1, 1))
        for (n_idx, ns) in enumerate((-2, 0, 2))
            dE = 2.0 * σ * ns
            table[s_idx, n_idx] = 1.0 / (1.0 + exp(beta * dE))
        end
    end
    return table
end

function IsingState(L::Int, beta::Float64; init_value::Int8=Int8(1))
    L >= 3 || error("IsingState requires L >= 3; got L=$L.")
    sigma = fill(init_value, L)
    accept = build_glauber_table(beta)
    return IsingState(L, beta, sigma, accept, Int(init_value) * L)
end

@inline function recompute_mag_sum!(state::IsingState)
    s = 0
    @inbounds for i in 1:state.L
        s += state.sigma[i]
    end
    state.mag_sum = s
end

@inline copy_config(state::IsingState) = copy(state.sigma)

function load_config!(state::IsingState, cfg::Vector{Int8})
    length(cfg) == state.L || error("Config length $(length(cfg)) ≠ L=$(state.L).")
    copyto!(state.sigma, cfg)
    recompute_mag_sum!(state)
end

@inline compute_total_magnetization(state::IsingState) =
    state.mag_sum / state.L  # in [-1, 1]

# --------------------------------------------------------------
# Single-spin Glauber update (mirrors src/core.jl single-chain branch
# at lines 13-26, with the table populated by build_glauber_table).
# --------------------------------------------------------------
@inline function glauber_step!(state::IsingState, n_updates::Int)
    L = state.L
    @inbounds for _ in 1:n_updates
        i = rand(1:L)
        σ_old = state.sigma[i]
        left  = state.sigma[i == 1 ? L : i - 1]
        right = state.sigma[i == L ? 1 : i + 1]
        ns = Int(left) + Int(right)         # ∈ {-2, 0, +2}
        s_idx = (Int(σ_old) + 1) >> 1 + 1   # -1→1, +1→2
        n_idx = (ns + 4) >> 1               # -2→1, 0→2, +2→3
        if rand() < state.accept[s_idx, n_idx]
            σ_new = -σ_old
            state.sigma[i] = σ_new
            state.mag_sum += Int(σ_new) - Int(σ_old)
        end
    end
    return nothing
end

# --------------------------------------------------------------
# Time-evolution helpers. Contract matches src/core.jl:144-155 for v=0.
# 1 time unit = L single-spin updates.
# --------------------------------------------------------------
function evolve_time!(state::IsingState, T::Int)
    L = state.L
    @inbounds for _ in 1:T
        glauber_step!(state, L)
    end
    return T
end

function evolve_and_measure!(state::IsingState, T_max::Int, on_measure::Function)::Int
    L = state.L
    for step in 1:T_max
        glauber_step!(state, L)
        if !on_measure(step)
            return step
        end
    end
    return T_max
end

# --------------------------------------------------------------
# Init helper for history mode.
# --------------------------------------------------------------
function init_state!(state::IsingState, init::String)
    L = state.L
    if init == "all_plus"
        fill!(state.sigma, Int8(1))
    elseif init == "all_minus"
        fill!(state.sigma, Int8(-1))
    elseif init == "random"
        @inbounds for i in 1:L
            state.sigma[i] = rand() < 0.5 ? Int8(-1) : Int8(1)
        end
    else
        error("Unknown init: $init. Use 'all_plus', 'all_minus', or 'random'.")
    end
    recompute_mag_sum!(state)
    return nothing
end

### =====================================================================
### §2  History mode
### =====================================================================

function run_history_mode(; L::Int, p::Float64, T_steps::Int=500,
                            init::String="all_plus",
                            show_plot::Bool=false, kwargs...)
    beta = -log(p)
    q = 1.0 / p
    println("=== 1D Ising Glauber: History Mode ===")
    println("L = $L, p = $p (q = $(@sprintf("%.3f", q)), β = $(@sprintf("%.4f", beta))), T = $T_steps, init = $init")

    state = IsingState(L, beta)
    init_state!(state, init)

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

    if show_plot
        @eval Main using Plots
        fig = Base.invokelatest(Plots.heatmap, history;
                                c=:RdBu, clims=(-1.5, 1.5),
                                xlabel="site", ylabel="t",
                                title=@sprintf("Glauber 1D Ising L=%d, p=%.3f", L, p),
                                size=(400, 700))
        Base.invokelatest(display, fig)
        println("Close the plot window (or press Enter) to exit...")
        try readline() catch end
    end

    return Dict{String, Any}(
        "magnetization_history" => history,
        "L" => L, "T_steps" => T_steps,
        "p" => p, "q" => q, "beta" => beta, "init" => init,
        "dynamics" => "ising_glauber",
        # Plotter-compat shims
        "v" => 0.0, "h" => 0.0,
    )
end

### =====================================================================
### §3  FFS mode (port of src/ffs.jl, single-chain Glauber)
###
### State starts at all +1 (m_meta ≈ +1), m DECREASES as the trajectory
### escapes, lambdas form a DECREASING sequence ending at M_threshold.
### All inequalities match src/ffs.jl verbatim.
### =====================================================================

const FFSConfig = Vector{Int8}

# ---------------------------------------------------------------------
# Phase 0: initial flux  (mirror of src/ffs.jl:24-99)
# ---------------------------------------------------------------------
function measure_initial_flux(L::Int, beta::Float64, lambda_0::Float64,
                               M_threshold::Float64,
                               n_configs::Int, max_time::Int;
                               verbose::Bool=false)
    n_threads = Threads.nthreads()
    configs_per_thread = cld(n_configs, n_threads)

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
            state = IsingState(L, beta)  # init = all +1
            above_lambda0 = true
            n_trajectories += 1

            final_step = evolve_and_measure!(state, max_time, step -> begin
                m = compute_total_magnetization(state)
                if above_lambda0 && m < lambda_0
                    push!(my_configs, copy_config(state))
                    Threads.atomic_add!(collected_total, 1)
                    if progress !== nothing
                        update!(progress, collected_total[])
                    end
                    above_lambda0 = false
                    if length(my_configs) >= my_target
                        return false
                    end
                elseif !above_lambda0 && m >= lambda_0
                    above_lambda0 = true
                end
                if !above_lambda0 && m <= M_threshold
                    return false  # left basin A
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
# Crossing-probability phase  (mirror of src/ffs.jl:113-184)
# ---------------------------------------------------------------------
function measure_crossing_probability(L::Int, beta::Float64,
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

    while length(success_configs) < n_configs
        batch_successes = Vector{Union{Nothing, FFSConfig}}(nothing, batch_size)
        batch_timeouts = zeros(Int, batch_size)

        Threads.@threads for trial in 1:batch_size
            config = source_configs[rand(1:length(source_configs))]
            state = IsingState(L, beta)
            load_config!(state, config)

            success = false
            final_step = evolve_and_measure!(state, max_time_per_trial, step -> begin
                m = compute_total_magnetization(state)
                if m < lambda_target
                    success = true
                    return false
                elseif m >= lambda_fail
                    return false
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
# Adaptive probe of next interface  (mirror of src/ffs.jl:202-232)
# ---------------------------------------------------------------------
function probe_next_interface(L::Int, beta::Float64,
                               source_configs::Vector{FFSConfig}, lambda_fail::Float64,
                               M_threshold::Float64,
                               max_time::Int, n_probe::Int, target_prob::Float64)
    min_mags = zeros(Float64, n_probe)
    probes_done = Threads.Atomic{Int}(0)
    n_timeouts = Threads.Atomic{Int}(0)

    Threads.@threads for trial in 1:n_probe
        config = source_configs[rand(1:length(source_configs))]
        state = IsingState(L, beta)
        load_config!(state, config)

        min_m = compute_total_magnetization(state)

        final_step = evolve_and_measure!(state, max_time, step -> begin
            m = compute_total_magnetization(state)
            if m < min_m
                min_m = m
            end
            if m <= M_threshold
                return false
            end
            return m < lambda_fail
        end)

        Threads.atomic_add!(probes_done, 1)
        if final_step >= max_time
            Threads.atomic_add!(n_timeouts, 1)
            @warn "Probe: $(probes_done[]) / $n_probe done, trial timed out at max_time = $max_time. Marking single FFS run as failed; increase --max_time_per_trial."
        end
        min_mags[trial] = min_m
    end

    sort!(min_mags)
    k = clamp(round(Int, target_prob * n_probe), 1, n_probe)
    return (min_mags[k], n_timeouts[] > 0)
end

# ---------------------------------------------------------------------
# Metastable magnetization  (mirror of src/ffs.jl:241-261)
# ---------------------------------------------------------------------
function measure_metastable_magnetization(L::Int, beta::Float64)
    n_runs = 10
    T_equil = 2
    T_sample = 10
    all_mags = Float64[]
    for _ in 1:n_runs
        state = IsingState(L, beta)
        evolve_time!(state, T_equil)
        evolve_and_measure!(state, T_sample, step -> begin
            push!(all_mags, compute_total_magnetization(state))
            return true
        end)
    end
    return mean(all_mags), std(all_mags)
end

# ---------------------------------------------------------------------
# One full FFS estimation  (mirror of src/ffs.jl:272-347)
# ---------------------------------------------------------------------
function ffs_single_run(L::Int, beta::Float64,
                         lambda_0::Float64, M_threshold::Float64,
                         n_configs_per_run::Int, max_time_per_trial::Int,
                         target_crossing_prob::Float64;
                         n_interfaces::Int=0, verbose::Bool=false)
    if verbose
        @printf("  Phase 0: measuring initial flux (λ₀ = %.4f)...\n", lambda_0)
    end
    (phi_0, configs) = measure_initial_flux(L, beta, lambda_0, M_threshold,
                                             n_configs_per_run, max_time_per_trial; verbose)
    n_crossings = length(configs)
    if verbose
        @printf("  Φ₀ = %.6e (%d configs)\n", phi_0, n_crossings)
    end

    if phi_0 == 0.0 || isempty(configs)
        return (Inf, Inf, phi_0, true)
    end

    log_product = 0.0
    var_log_tau = 1.0 / n_crossings
    current_configs = configs
    phase = 0
    lambda_history = [lambda_0]

    # Two modes:
    #   - canonical (n_interfaces > 0): predetermined uniformly-spaced interfaces
    #     between λ_0 and M_threshold; lambda_fail = lambda_0 (canonical choice
    #     used in the original FFS literature). Use this when there's no
    #     well-defined metastable basin (e.g. 1D Ising at h=0): P_i can grow
    #     close to 1 and FFS reduces to ~1/Φ_0 in the barrier-less limit.
    #   - adaptive (n_interfaces == 0, default): aim each P_i ≈
    #     target_crossing_prob via `probe_next_interface`; lambda_fail steps back
    #     `n_lookback` interfaces. Designed for genuine rare events.
    n_lookback = 7

    fixed_lambdas = if n_interfaces > 0
        # n_interfaces+1 boundaries from λ_0 to M_threshold; we'll iterate over
        # the *intermediate* + endpoint values (i.e. lambdas[2..end]).
        collect(range(lambda_0, M_threshold, length=n_interfaces + 1))
    else
        Float64[]
    end

    while true
        phase += 1
        if phase > 200
            return (Inf, Inf, phi_0, true)
        end

        if n_interfaces > 0
            # Canonical: fixed interfaces, fail = λ_0
            if phase > n_interfaces
                break  # already completed all interfaces
            end
            lambda_fail = lambda_0
            lambda_next = fixed_lambdas[phase + 1]  # phase 1 → fixed_lambdas[2]
        else
            # Adaptive
            fail_idx = max(1, length(lambda_history) - n_lookback + 1)
            lambda_fail = lambda_history[fail_idx]

            (lambda_next, probe_timed_out) = probe_next_interface(
                L, beta, current_configs, lambda_fail,
                M_threshold, max_time_per_trial, n_configs_per_run, target_crossing_prob)

            if probe_timed_out
                return (Inf, Inf, phi_0, true)
            end

            if lambda_next <= M_threshold
                lambda_next = M_threshold
            end
        end

        (prob, new_configs, n_success, n_trials, cross_timed_out) = measure_crossing_probability(
            L, beta, current_configs, lambda_next, lambda_fail,
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

        if lambda_next <= M_threshold
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
# FFS top-level: sweep over p = exp(-β J)  (mirror of src/ffs.jl:369-572)
# ---------------------------------------------------------------------
function run_ffs_mode(; L::Int, p_min::Float64, p_max::Float64, n_steps::Int,
                       n_configs_per_run::Int, n_repeats::Int,
                       M_threshold::Float64, max_time_per_trial::Int,
                       target_crossing_prob::Float64=0.15,
                       n_interfaces::Int=0,
                       lambda_0_override::Float64=NaN,
                       sweep_values_override::Union{Nothing,Vector{Float64}}=nothing,
                       kwargs...)
    sweep_values = sweep_values_override === nothing ?
        collect(range(p_min, p_max, length=n_steps)) : sweep_values_override
    all(x -> 0 < x <= 1.0, sweep_values) || @warn "Some p values are outside (0, 1] — βJ ≤ 0 has no barrier."

    println("=== 1D Ising Glauber: FFS Mode (sweeping p = e^{-β J}) ===")
    println("L = $L")
    println("p values: $sweep_values")
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

    for (idx, p) in enumerate(sweep_values)
        beta = -log(p)
        @printf("\n[%d/%d] p = %.4f (q = %.3f, β = %.4f)\n", idx, n_sweep, p, 1.0/p, beta)

        if !isnan(lambda_0_override)
            lambda_0 = lambda_0_override
            @printf("  λ₀ = %.4f (user override)\n", lambda_0)
        else
            (m_meta, m_std) = measure_metastable_magnetization(L, beta)
            lambda_0 = m_meta - 2.5 * m_std
            @printf("  Metastable: m* = %.4f ± %.4f, λ₀ = %.4f\n", m_meta, m_std, lambda_0)
        end

        if lambda_0 <= M_threshold
            @warn "λ₀ ($(@sprintf("%.4f", lambda_0))) <= M_threshold ($(@sprintf("%.4f", M_threshold))) — transition is not a rare event"
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
            L, beta, lambda_0, M_threshold,
            n_configs_per_run, max_time_per_trial, target_crossing_prob;
            n_interfaces, verbose=true)
        _record_run!(1, log_tau, var_log_tau, phi_0, failed)

        if n_repeats > 1
            progress = Progress(n_repeats - 1; desc="  Remaining runs: ", dt=1.0)
            for rep in 2:n_repeats
                (log_tau, var_log_tau, phi_0, failed) = ffs_single_run(
                    L, beta, lambda_0, M_threshold,
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

    println("\nDone!")

    return Dict{String, Any}(
        "mean_mixing_times" => mean_mixing_times,
        "log_mixing_times" => log_mixing_times,
        "log_mixing_times_std" => log_mixing_times_std,
        "per_run_log_taus" => per_run_log_taus,
        "flux_rates" => flux_rates,
        "L" => L, "h" => 0.0,
        "n_configs_per_run" => n_configs_per_run, "n_repeats" => n_repeats,
        "M_threshold" => M_threshold,
        "target_crossing_prob" => target_crossing_prob,
        "n_interfaces" => n_interfaces,
        "max_time_per_trial" => max_time_per_trial,
        "dynamics" => "ising_glauber",
        "p_values" => sweep_values,
        "q_values" => 1.0 ./ sweep_values,
        "v" => 0.0,
    )
end

### =====================================================================
### §4  Mixing mode (direct ensemble-averaged memory time)
### Mirror of src/mixing.jl:8-110, simplified for single-chain Glauber.
### =====================================================================

function measure_mixing_time(L::Int, beta::Float64, n_trials::Int,
                              M_threshold::Float64, max_time::Int)
    trial_times = zeros(Int, n_trials)
    Threads.@threads for trial in 1:n_trials
        state = IsingState(L, beta)
        final_step = evolve_and_measure!(state, max_time, step -> begin
            m = compute_total_magnetization(state)
            m > M_threshold
        end)
        trial_times[trial] = final_step
    end
    n_timed_out = count(==(max_time), trial_times)
    if n_timed_out > 0
        @warn @sprintf("Mixing measurement: %d / %d trials hit max_time = %d at β = %.4f — estimate is unreliable. Increase --max_time.",
                       n_timed_out, n_trials, max_time, beta)
    end
    return mean(trial_times), trial_times
end

function run_mixing_mode(; L::Int, p_min::Float64, p_max::Float64, n_steps::Int,
                          n_trials::Int, M_threshold::Float64, max_time::Int,
                          sweep_values_override::Union{Nothing,Vector{Float64}}=nothing,
                          kwargs...)
    sweep_values = sweep_values_override === nothing ?
        collect(range(p_min, p_max, length=n_steps)) : sweep_values_override
    println("=== 1D Ising Glauber: Mixing Mode (sweeping p = e^{-β J}) ===")
    println("L = $L")
    println("p values: $sweep_values")
    println("Trials per value: $n_trials, threads: $(Threads.nthreads())")
    println("Magnetization threshold: $M_threshold")

    n_sweep = length(sweep_values)
    mean_mixing_times = zeros(Float64, n_sweep)
    mixing_time_stds = zeros(Float64, n_sweep)
    per_trial_times = fill(NaN, n_sweep, n_trials)

    for (idx, p) in enumerate(sweep_values)
        beta = -log(p)
        @printf("\n[%d/%d] p = %.4f (q = %.3f, β = %.4f)\n", idx, n_sweep, p, 1.0/p, beta)

        (mean_tau, trial_times) = measure_mixing_time(L, beta, n_trials, M_threshold, max_time)
        mean_mixing_times[idx] = mean_tau
        mixing_time_stds[idx] = n_trials > 1 ? std(trial_times) / sqrt(n_trials) : 0.0
        per_trial_times[idx, :] = trial_times
        @printf("  Mean mixing time: %.2f ± %.2f (SoM)\n", mean_tau, mixing_time_stds[idx])
    end

    println("\nDone!")

    return Dict{String, Any}(
        "mean_mixing_times" => mean_mixing_times,
        "mixing_time_stds" => mixing_time_stds,   # SoM across trials
        "per_trial_times" => per_trial_times,
        "L" => L, "h" => 0.0,
        "n_trials" => n_trials, "M_threshold" => M_threshold, "max_time" => max_time,
        "dynamics" => "ising_glauber",
        "p_values" => sweep_values,
        "q_values" => 1.0 ./ sweep_values,
        "v" => 0.0,
    )
end

### =====================================================================
### §5  save_results / build_filename / dispatch
### =====================================================================

function build_filename(mode::String, results::Dict{String,Any})
    if mode == "history"
        L = results["L"]
        p = results["p"]
        return @sprintf("data/ising_glauber_history_L%d_p%.3f.jld2", L, p)
    elseif mode == "ffs"
        L = results["L"]
        ps = results["p_values"]
        return @sprintf("data/ising_glauber_ffs_L%d_p%.3fto%.3f.jld2",
                        L, minimum(ps), maximum(ps))
    elseif mode == "mixing"
        L = results["L"]
        ps = results["p_values"]
        return @sprintf("data/ising_glauber_mixing_L%d_p%.3fto%.3f.jld2",
                        L, minimum(ps), maximum(ps))
    else
        return "data/ising_glauber_$(mode).jld2"
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
    try clipboard(filename) catch end
end

function run_simulation(mode::String; save::Bool=true, adj::String="", kwargs...)
    kw = Dict(kwargs)
    results = if mode == "history"
        run_history_mode(; kw...)
    elseif mode == "ffs"
        run_ffs_mode(; kw...)
    elseif mode == "mixing"
        run_mixing_mode(; kw...)
    else
        error("Unknown mode: $mode. Use 'history', 'ffs', or 'mixing'.")
    end
    if save
        save_results(mode, results; adj)
    end
    return results
end

### =====================================================================
### §6  top-level: defaults per mode, CLI parse, dispatch
### =====================================================================

mode = String(parse_arg("mode", "ffs"))

# Shared
L = 1000

# p/q/τ helpers for the 1D Ising chain (same conventions as simulation_driver.jl):
# p = exp(-β J) ∈ (0, 1]; q = 1/p; τ = exp(4 (β J)²).
_p_from_tau_ising_local(t::Float64) = begin
    lt = log(t)
    lt >= 0 || error("--tau must be ≥ 1 (got τ=$t).")
    exp(-sqrt(lt / 4))
end
function _resolve_pqtau_ising_local(p_min_default, p_max_default, p_default, n_steps)
    p_min = parse_arg("p_min", p_min_default)
    p_max = parse_arg("p_max", p_max_default)
    p = parse_arg("p", p_default)
    q_min = parse_arg("q_min", NaN)
    q_max = parse_arg("q_max", NaN)
    q = parse_arg("q", NaN)
    tau_min = parse_arg("tau_min", NaN)
    tau_max = parse_arg("tau_max", NaN)
    tau = parse_arg("tau", NaN)
    sweep_override::Union{Nothing,Vector{Float64}} = nothing
    if !isnan(tau_min) && !isnan(tau_max)
        tau_values = collect(range(tau_min, tau_max, length=n_steps))
        sweep_override = [_p_from_tau_ising_local(t) for t in tau_values]
        p_min, p_max = minimum(sweep_override), maximum(sweep_override)
    elseif !isnan(q_min) && !isnan(q_max)
        q_values = collect(range(q_min, q_max, length=n_steps))
        sweep_override = [1.0 / qv for qv in q_values]
        p_min, p_max = minimum(sweep_override), maximum(sweep_override)
    else
        !isnan(q_min) && (p_max = 1.0 / q_min)
        !isnan(q_max) && (p_min = 1.0 / q_max)
        !isnan(tau_min) && (p_max = _p_from_tau_ising_local(tau_min))
        !isnan(tau_max) && (p_min = _p_from_tau_ising_local(tau_max))
    end
    if !isnan(tau)
        p = _p_from_tau_ising_local(tau)
    elseif !isnan(q)
        p = 1.0 / q
    end
    return p_min, p_max, p, sweep_override
end

if mode == "history"
    L = 200
    p = 0.5  # corresponds to βJ ≈ 0.69; equivalently q ≈ 2, τ ≈ exp(1.9) ≈ 6.7
    T_steps = 500
    init = "all_plus"
    show_plot = false
elseif mode == "ffs"
    p_min = 0.33     # βJ ≈ 1.10 (rare); old q_min=50 ↦ p≈0.36 under exp(4βJ), new q_min=3.0
    p_max = 0.67     # βJ ≈ 0.40 (warm); old q_max=5 ↦ p≈0.67
    p = 0.5
    n_steps = 6
    n_configs_per_run = 200
    n_repeats = 10
    target_crossing_prob = 0.15
    n_interfaces = 0
    M_threshold = 0.75
    max_time_per_trial = 100_000_000
    lambda_0_override = NaN
elseif mode == "mixing"
    p_min = 0.33
    p_max = 0.67
    p = 0.5
    n_steps = 6
    n_trials = 1000
    M_threshold = 0.75
    max_time = 100_000_000
else
    error("Unknown mode: $mode. Use 'history', 'ffs', or 'mixing'.")
end

# CLI overrides
save = parse_arg("save", true)
adj = String(parse_arg("adj", ""))

if mode == "history"
    L = parse_arg("L", L)
    p = parse_arg("p", p)
    # q / τ aliases for a single fixed p:
    q_cli = parse_arg("q", NaN)
    tau_cli = parse_arg("tau", NaN)
    if !isnan(tau_cli)
        p = _p_from_tau_ising_local(tau_cli)
    elseif !isnan(q_cli)
        p = 1.0 / q_cli
    end
    T_steps = parse_arg("T_steps", T_steps)
    init = String(parse_arg("init", init))
    show_plot = parse_arg("show_plot", show_plot)
elseif mode == "ffs"
    L = parse_arg("L", L)
    n_steps = parse_arg("n_steps", n_steps)
    n_configs_per_run = parse_arg("n_configs_per_run", n_configs_per_run)
    n_repeats = parse_arg("n_repeats", n_repeats)
    target_crossing_prob = parse_arg("target_crossing_prob", target_crossing_prob)
    n_interfaces = parse_arg("n_interfaces", n_interfaces)
    M_threshold = parse_arg("M_threshold", M_threshold)
    max_time_per_trial = parse_arg("max_time_per_trial", max_time_per_trial)
    lambda_0_override = parse_arg("lambda_0", lambda_0_override)
    (p_min, p_max, p, sweep_values_override) =
        _resolve_pqtau_ising_local(p_min, p_max, p, n_steps)
elseif mode == "mixing"
    L = parse_arg("L", L)
    n_steps = parse_arg("n_steps", n_steps)
    n_trials = parse_arg("n_trials", n_trials)
    M_threshold = parse_arg("M_threshold", M_threshold)
    max_time = parse_arg("max_time", max_time)
    (p_min, p_max, p, sweep_values_override) =
        _resolve_pqtau_ising_local(p_min, p_max, p, n_steps)
end

# Build kwargs and dispatch
kwargs = Dict{Symbol, Any}()
if mode == "history"
    kwargs[:L] = L
    kwargs[:p] = p
    kwargs[:T_steps] = T_steps
    kwargs[:init] = init
    kwargs[:show_plot] = show_plot
elseif mode == "ffs"
    kwargs[:L] = L
    kwargs[:p_min] = p_min
    kwargs[:p_max] = p_max
    kwargs[:n_steps] = n_steps
    kwargs[:n_configs_per_run] = n_configs_per_run
    kwargs[:n_repeats] = n_repeats
    kwargs[:target_crossing_prob] = target_crossing_prob
    kwargs[:n_interfaces] = n_interfaces
    kwargs[:M_threshold] = M_threshold
    kwargs[:max_time_per_trial] = max_time_per_trial
    kwargs[:lambda_0_override] = lambda_0_override
    kwargs[:sweep_values_override] = sweep_values_override
elseif mode == "mixing"
    kwargs[:L] = L
    kwargs[:p_min] = p_min
    kwargs[:p_max] = p_max
    kwargs[:n_steps] = n_steps
    kwargs[:n_trials] = n_trials
    kwargs[:M_threshold] = M_threshold
    kwargs[:max_time] = max_time
    kwargs[:sweep_values_override] = sweep_values_override
end

n_threads = Threads.nthreads()
if n_threads == 1
    @warn "Running with only 1 thread. For parallel trials, start Julia with: julia -t auto"
end

println("Running ising: mode=$mode, threads=$n_threads")
run_simulation(mode; save=save, adj=adj, kwargs...)
println("Simulation complete!")
try run(`afplay /System/Library/Sounds/Glass.aiff`) catch end
