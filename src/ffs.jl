const FFSConfig = Tuple{Vector{Int8}, Vector{Int8}, Int}

function copy_config(state::SimulationState)::FFSConfig
    return (copy(state.top), copy(state.bottom), state.top_offset)
end

function load_config!(state::SimulationState, config::FFSConfig)
    copyto!(state.top, config[1])
    copyto!(state.bottom, config[2])
    state.top_offset = config[3]
    recompute_mag_sum!(state)
end

"""
    measure_initial_flux(L, beta, h, v, lambda_0, M_threshold, n_configs, max_time)

Standard FFS Phase 0: run long trajectories in basin A, collecting configs each time
magnetization crosses below lambda_0 from above. After a crossing, wait for m to
return above lambda_0 before counting the next one. Each thread runs independent
trajectories until its config quota is met. Trajectories terminate early if m drops
below M_threshold (system has left the metastable basin).
Φ₀ = total_crossings / total_simulation_time.
Returns (flux_rate, configs).
"""
function measure_initial_flux(L::Int, beta::Float64, h::Float64, v::Float64,
                               lambda_0::Float64, M_threshold::Float64,
                               n_configs::Int, max_time::Int;
                               single_layer::Bool=false, verbose::Bool=false,
                               seed_droplet_size::Int=0,
                               randshift::Bool=false)
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
            state = single_layer ? SimulationState(L, beta, h) : SimulationState(L, beta, h, v)
            above_lambda0 = true
            n_trajectories += 1

            final_step = evolve_and_measure!(state, max_time, step -> begin
                m = compute_total_magnetization(state)
                if above_lambda0 && m < lambda_0
                    # Forward crossing of λ₀ — store config
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
                    # Returned to basin A — ready for next crossing
                    above_lambda0 = true
                end
                # If below λ₀ and dropped past M_threshold, system has left
                # the metastable basin — no point continuing this trajectory
                if !above_lambda0 && m <= M_threshold
                    return false
                end
                return true
            end; seed_droplet_size, randshift)

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

"""
    measure_crossing_probability(L, beta, h, v, source_configs, lambda_target,
                                  lambda_fail, max_time_per_trial, n_configs)

Fire trials from source_configs until n_configs successes are collected.
Each trial picks a random source config and runs until m < lambda_target (success)
or m >= lambda_fail (failure/returned toward basin A).
Returns (probability, success_configs, total_successes, total_trials, timed_out).
`timed_out` is true if any trial hit max_time_per_trial — caller should treat
this run as failed (the crossing probability is biased low by such trials).
Shows a progress bar if the phase takes more than 5 seconds.
"""
function measure_crossing_probability(L::Int, beta::Float64, h::Float64, v::Float64,
                                       source_configs::Vector{FFSConfig},
                                       lambda_target::Float64, lambda_fail::Float64,
                                       max_time_per_trial::Int, n_configs::Int;
                                       single_layer::Bool=false,
                                       randshift::Bool=false)
    success_configs = FFSConfig[]
    total_successes = 0
    total_trials = 0
    total_timeouts = 0
    batch_size = max(Threads.nthreads() * 4, 32)
    t_start = time()
    progress = nothing

    while length(success_configs) < n_configs
        # Fire a batch of trials
        batch_successes = Vector{Union{Nothing, FFSConfig}}(nothing, batch_size)

        batch_timeouts = zeros(Int, batch_size)

        Threads.@threads for trial in 1:batch_size
            config = source_configs[rand(1:length(source_configs))]
            state = single_layer ? SimulationState(L, beta, h) : SimulationState(L, beta, h, v)
            load_config!(state, config)

            success = false
            final_step = evolve_and_measure!(state, max_time_per_trial, step -> begin
                m = compute_total_magnetization(state)
                if m < lambda_target
                    success = true
                    return false
                elseif m >= lambda_fail
                    return false  # returned toward basin A
                end
                return true
            end; randshift)

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

        # Show progress bar after 5 seconds
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

"""
    probe_next_interface(L, beta, h, v, source_configs, lambda_fail,
                          M_threshold, max_time, n_probe, target_prob)

Fire n_probe trials from source_configs, tracking the minimum magnetization
reached by each before returning above lambda_fail (failure boundary).
Trials also stop early if m drops below M_threshold (barrier already crossed).
Returns (lambda_next, timed_out): lambda_next is the target_prob quantile of
minimum magnetizations (the next adaptive interface position), and timed_out
is true if any trial hit max_time — caller should treat this run as failed
(timed-out probes give upward-biased min magnetizations).
"""
function probe_next_interface(L::Int, beta::Float64, h::Float64, v::Float64,
                               source_configs::Vector{FFSConfig}, lambda_fail::Float64,
                               M_threshold::Float64,
                               max_time::Int, n_probe::Int, target_prob::Float64;
                               single_layer::Bool=false,
                               randshift::Bool=false)
    min_mags = zeros(Float64, n_probe)
    probes_done = Threads.Atomic{Int}(0)
    n_timeouts = Threads.Atomic{Int}(0)

    Threads.@threads for trial in 1:n_probe
        config = source_configs[rand(1:length(source_configs))]
        state = single_layer ? SimulationState(L, beta, h) : SimulationState(L, beta, h, v)
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
            return m < lambda_fail  # continue while below failure boundary
        end; randshift)

        Threads.atomic_add!(probes_done, 1)
        if final_step >= max_time
            Threads.atomic_add!(n_timeouts, 1)
            @warn "Probe: $(probes_done[]) / $n_probe done, trial timed out at max_time = $max_time. Marking single FFS run as failed; increase --max_time_per_trial."
        end
        min_mags[trial] = min_m
    end

    sort!(min_mags)  # ascending: deepest penetration first
    k = clamp(round(Int, target_prob * n_probe), 1, n_probe)
    return (min_mags[k], n_timeouts[] > 0)
end

"""
    measure_metastable_magnetization(L, beta, h, v; single_layer, seed_droplet_size)

Start from all +1 and briefly equilibrate, then sample magnetization. Returns
(mean, std) of the magnetization in the metastable state under the dynamics
that will be used by the rest of FFS.

If `seed_droplet_size > 0` the modified dynamics is used (droplet seeded
instantly whenever the state returns to all-+); this makes the metastable
distribution have a non-degenerate σ even at very large βJ, which is what we
need so that λ₀ = ⟨m⟩_A − 2.5σ_A lands well below 1.

T_sample is scaled as exp(2βJ) (clamped to a sensible minimum), which is the
relaxation time of a flipped spin sitting next to a + neighbour in the
seed-droplet boundary — i.e. enough time for the seeded droplet to be felt by
the dynamics before we sample.
"""
function measure_metastable_magnetization(L::Int, beta::Float64, h::Float64, v::Float64;
                                           single_layer::Bool=false,
                                           seed_droplet_size::Int=0,
                                           randshift::Bool=false)
    # T_sample scaling: at low T the droplet relaxation time grows like exp(2βJ).
    # Clamp at 10 (the historical hardcoded value) on the low end and at 10⁵ on
    # the high end so we don't blow up wall-clock on a diagnostic measurement.
    n_runs = 10
    T_equil = 2
    T_sample = clamp(round(Int, exp(2 * beta)), 10, 100_000)
    all_mags = Float64[]

    for _ in 1:n_runs
        state = single_layer ? SimulationState(L, beta, h) : SimulationState(L, beta, h, v)
        # If seeding is on, kick the state out of all-+ before equilibrating so
        # the modified dynamics has something to relax around. Otherwise the
        # first sweep just produces another all-+ → no-op for equilibration.
        if seed_droplet_size > 0
            seed_droplet!(state, seed_droplet_size)
        end
        evolve_time!(state, T_equil; seed_droplet_size, randshift)
        evolve_and_measure!(state, T_sample, step -> begin
            push!(all_mags, compute_total_magnetization(state))
            return true
        end; seed_droplet_size, randshift)
    end

    return mean(all_mags), std(all_mags)
end

"""
    ffs_single_run(L, beta, h, v, lambda_0, M_threshold, n_configs_per_run,
                    max_time_per_trial, target_crossing_prob; single_layer, verbose)

Run a single FFS estimation: Phase 0 (initial flux) + adaptive interface phases.
Returns (log_tau, var_log_tau, phi_0, failed) where:
- log_tau is log₁₀ of the estimated mixing time
- var_log_tau is the binomial/Poisson variance estimate of ln(τ) (divide by ln(10)² for log₁₀)
"""
function ffs_single_run(L::Int, beta::Float64, h::Float64, v::Float64,
                         lambda_0::Float64, M_threshold::Float64,
                         n_configs_per_run::Int, max_time_per_trial::Int,
                         target_crossing_prob::Float64;
                         n_interfaces::Int=0,
                         single_layer::Bool=false, verbose::Bool=false,
                         seed_droplet_size::Int=0,
                         randshift::Bool=false)
    # Phase 0: Initial flux. Seeding only meaningfully affects Phase 0 (and the
    # metastable λ₀ measurement upstream); the per-phase trial trajectories at
    # k > 0 start from saved non-all-+ configs and terminate at m_fail well
    # before they could ever return to all-+, so we don't pass seed_droplet_size
    # into measure_crossing_probability / probe_next_interface.
    if verbose
        @printf("  Phase 0: measuring initial flux (λ₀ = %.4f)...\n", lambda_0)
    end
    (phi_0, configs) = measure_initial_flux(L, beta, h, v, lambda_0, M_threshold,
                                             n_configs_per_run, max_time_per_trial;
                                             single_layer, verbose, seed_droplet_size, randshift)
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
    lambda_cur = lambda_0
    lambda_history = [lambda_0]

    # Two modes (see ising.jl for the rationale):
    #   - canonical (n_interfaces > 0): predetermined uniformly-spaced interfaces
    #     between λ_0 and M_threshold; λ_fail = λ_0. Use when there's no genuine
    #     metastable basin — lets each P_i grow toward 1 in the barrier-less
    #     regime so τ_FFS doesn't pick up spurious 1/∏P_i factors.
    #   - adaptive (n_interfaces == 0, default): aim P_i ≈ target_crossing_prob
    #     via probe_next_interface; λ_fail steps back n_lookback interfaces.
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
                L, beta, h, v, current_configs, lambda_fail,
                M_threshold, max_time_per_trial, n_configs_per_run, target_crossing_prob;
                single_layer, randshift)

            if probe_timed_out
                return (Inf, Inf, phi_0, true)
            end

            if lambda_next <= M_threshold
                lambda_next = M_threshold
            end
        end

        (prob, new_configs, n_success, n_trials, cross_timed_out) = measure_crossing_probability(
            L, beta, h, v, current_configs, lambda_next, lambda_fail,
            max_time_per_trial, n_configs_per_run; single_layer, randshift)

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
        lambda_cur = lambda_next
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

"""
    run_ffs_mode(; L, v, h, p, p_min, p_max, v_min, v_max, n_steps, vary_v,
                   n_configs_per_run, n_repeats, M_threshold, max_time_per_trial,
                   target_crossing_prob, kwargs...)

Run Forward Flux Sampling with adaptive interface placement to estimate mixing times.
Each sweep point runs `n_repeats` independent FFS estimations, each with
`n_configs_per_run` source configurations collected at every interface, and reports
the mean and std of log₁₀(τ) across repeats.

`n_configs_per_run` controls per-run bias (adaptive-placement bias and Jensen bias
on log P_k both shrink as 1/n_configs_per_run). `n_repeats` controls variance of
the across-runs mean (shrinks as 1/n_repeats). Total compute is n_configs_per_run *
n_repeats per sweep point; for fixed total budget, MSE is minimized by pushing
n_configs_per_run as large as practical (a small n_repeats ≥ 4 is enough to get a
sanity-check empirical standard error).
"""
function run_ffs_mode(; L::Int, v::Float64, h::Float64, p::Float64,
                       p_min::Float64, p_max::Float64,
                       v_min::Float64, v_max::Float64,
                       n_steps::Int, vary_v::Bool,
                       n_configs_per_run::Int, n_repeats::Int=8,
                       M_threshold::Float64,
                       max_time_per_trial::Int,
                       target_crossing_prob::Float64=0.25,
                       n_interfaces::Int=0,
                       adaptive_L::Bool=false, adaptive_factor::Float64=3.0,
                       single_layer::Bool=false,
                       lambda_0_override::Float64=NaN,
                       seed_droplet_size::Int=0,
                       randshift::Bool=false,
                       sweep_values_override::Union{Nothing,Vector{Float64}}=nothing,
                       kwargs...)
    t_start = time()
    h = -abs(h)
    n_configs_per_run >= 1 || error("n_configs_per_run must be ≥ 1")
    n_repeats >= 1 || error("n_repeats must be ≥ 1")
    !(adaptive_L && single_layer) || error("--adaptive_L=true requires v > 0 (single-chain erosion not supported).")

    # If a sweep_values_override is provided (CLI dispatch already converted
    # --q_min/--q_max or --tau_min/--tau_max into p values), use it directly.
    sweep_p = sweep_values_override
    if single_layer
        sweep_values = sweep_p === nothing ?
            collect(range(p_min, p_max, length=n_steps)) : sweep_p
        # Always run the smallest p (rarest event, longest τ) first so the user
        # sees the worst-case runtime upfront.
        sort!(sweep_values)
        println("=== Single-Chain Ising: FFS Mode (sweeping p) ===")
        println("L = $L, h = $h")
        println("p values: $sweep_values")
    elseif vary_v
        sweep_values = collect(range(v_min, v_max, length=n_steps))
        all(x -> x >= 0, sweep_values) || error("All v values must be ≥ 0 for FFS.")
        # Always run the largest v (slowest sweep point) first so the user sees
        # the worst-case runtime upfront.
        sort!(sweep_values; rev=true)
        p_fixed = p
        println("=== Sliding Ising Chain: FFS Mode (sweeping v) ===")
        if adaptive_L
            @printf("Adaptive L: L = %.2f × ℓ_er per sweep point, p = %.3f (β = %.4f), h = %.3f\n",
                    adaptive_factor, p_fixed, -log(p_fixed), h)
        else
            println("L = $L, p = $p_fixed (β = $(@sprintf("%.4f", -log(p_fixed)))), h = $h")
        end
        println("v values: $sweep_values")
    else
        sweep_values = sweep_p === nothing ?
            collect(range(p_min, p_max, length=n_steps)) : sweep_p
        # Always run the smallest p (rarest event, longest τ) first so the user
        # sees the worst-case runtime upfront.
        sort!(sweep_values)
        println("=== Sliding Ising Chain: FFS Mode (sweeping p) ===")
        if adaptive_L
            @printf("Adaptive L: L = %.2f × ℓ_er per sweep point, v = %.3f, h = %.3f\n",
                    adaptive_factor, v, h)
        else
            println("L = $L, v = $v, h = $h")
        end
        println("p values: $sweep_values")
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
    if seed_droplet_size > 0
        @printf("Droplet seeding: instantly inject k=%d-spin minority droplet on both chains whenever state hits all-+ (modified clock — see SM)\n",
                seed_droplet_size)
    end
    if randshift
        println("Random shifts: Poisson process at rate v (Exp(L/v) inter-shift gaps); removes L/v-commensurability artifacts.")
    end
    println("Threads: $(Threads.nthreads())")

    n_sweep = length(sweep_values)
    mean_mixing_times = zeros(Float64, n_sweep)
    log_mixing_times = zeros(Float64, n_sweep)
    log_mixing_times_std = zeros(Float64, n_sweep)
    flux_rates = zeros(Float64, n_sweep)
    per_run_log_taus = fill(NaN, n_sweep, n_repeats)  # NaN entries = failed runs
    L_values = fill(L, n_sweep)        # per sweep point; constant L unless adaptive_L
    lc_values = fill(0, n_sweep)        # per sweep point; 0 means not measured

    # Adaptive-L: warm-start the erosion search from min_erosion_length;
    # subsequent sweep points start near the previous lc.
    adaptive_ler_n_trials = 3000
    adaptive_ler_thresh_prob = 0.75
    adaptive_ler_min = 2
    prev_lc = adaptive_ler_min

    for (idx, val) in enumerate(sweep_values)
        # Convention: p = exp(-β J), so 0 < p ≤ 1 (rare-event regime at small p).
        if single_layer
            v_cur = 0.0
            beta = -log(val)
            @printf("\n[%d/%d] p = %.3f (β = %.4f)\n", idx, n_sweep, val, beta)
        elseif vary_v
            v_cur = val
            beta = -log(p_fixed)
            @printf("\n[%d/%d] v = %.3f\n", idx, n_sweep, v_cur)
        else
            v_cur = v
            beta = -log(val)
            @printf("\n[%d/%d] p = %.3f (β = %.4f)\n", idx, n_sweep, val, beta)
        end

        # Adaptive L per sweep point
        if adaptive_L
            lc = find_critical_length(beta, h, v_cur, adaptive_ler_n_trials,
                                       prev_lc, adaptive_ler_thresh_prob, adaptive_ler_min;
                                       doublon_mode=false)
            L_cur = max(round(Int, adaptive_factor * lc), adaptive_ler_min + 2)
            prev_lc = lc
            lc_values[idx] = lc
            L_values[idx] = L_cur
            @printf("  Adaptive L: ℓ_er = %d, L = %d\n", lc, L_cur)
        else
            L_cur = L
        end

        # Set λ₀
        if !isnan(lambda_0_override)
            lambda_0 = lambda_0_override
            @printf("  λ₀ = %.4f (user override)\n", lambda_0)
        else
            (m_meta, m_std) = measure_metastable_magnetization(L_cur, beta, h, v_cur;
                                                                 single_layer, seed_droplet_size, randshift)
            lambda_0 = m_meta - 2.5 * m_std
            @printf("  Metastable: m* = %.4f ± %.4f, λ₀ = %.4f\n", m_meta, m_std, lambda_0)
        end

        if lambda_0 <= M_threshold
            @warn "λ₀ ($(@sprintf("%.4f", lambda_0))) <= M_threshold ($(@sprintf("%.4f", M_threshold))) — " *
                  "transition is not a rare event at this temperature"
            mean_mixing_times[idx] = NaN
            log_mixing_times[idx] = NaN
            log_mixing_times_std[idx] = NaN
            continue
        end

        # Run n_repeats independent FFS estimations
        # First run is verbose (shows phase-by-phase details), rest are silent.
        # Per-run log_tau values are kept (NaN for failed runs) so we can print
        # the across-run scatter for diagnostic purposes.
        log_taus = Float64[]              # successful runs only (used for stats)
        var_log_taus = Float64[]          # successful runs only (binomial variance)
        per_run_log_tau = fill(NaN, n_repeats)  # all runs, NaN if failed
        phi_0_sum = 0.0

        function _record_run!(rep::Int, log_tau, var_log_tau, phi_0, failed)
            phi_0_sum += phi_0
            if !failed && isfinite(log_tau)
                push!(log_taus, log_tau)
                push!(var_log_taus, var_log_tau)
                per_run_log_tau[rep] = log_tau
            end
        end

        # First run: verbose
        println("  First run:")
        (log_tau, var_log_tau, phi_0, failed) = ffs_single_run(
            L_cur, beta, h, v_cur, lambda_0, M_threshold,
            n_configs_per_run, max_time_per_trial, target_crossing_prob;
            n_interfaces, single_layer, verbose=true, seed_droplet_size, randshift)
        _record_run!(1, log_tau, var_log_tau, phi_0, failed)

        # Remaining runs: silent with progress bar
        if n_repeats > 1
            progress = Progress(n_repeats - 1; desc="  Remaining runs: ", dt=1.0)
            for rep in 2:n_repeats
                (log_tau, var_log_tau, phi_0, failed) = ffs_single_run(
                    L_cur, beta, h, v_cur, lambda_0, M_threshold,
                    n_configs_per_run, max_time_per_trial, target_crossing_prob;
                    n_interfaces, single_layer, verbose=false, seed_droplet_size, randshift)

                _record_run!(rep, log_tau, var_log_tau, phi_0, failed)
                update!(progress, rep - 1)
            end
            finish!(progress)
        end

        flux_rates[idx] = phi_0_sum / n_repeats
        mean_inv_phi = flux_rates[idx] > 0 ? 1.0 / flux_rates[idx] : Inf
        @printf("  1/Φ₀ = %.1f (averaged over %d runs)\n", mean_inv_phi, n_repeats)

        # Per-run log₁₀(τ) values, so the user can see across-run scatter
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

            # Error bar from two sources, take the larger:
            # 1) Binomial/Poisson variance: average the per-run variances,
            #    then standard error of the mean = sqrt(mean_var / n_ok)
            mean_var = mean(var_log_taus)
            binomial_stderr = sqrt(mean_var / n_ok) / log(10)  # convert ln → log₁₀

            # 2) Empirical std across independent runs (if n_ok > 1)
            if n_ok > 1
                empirical_stderr = std(log_taus) / sqrt(n_ok)
            else
                empirical_stderr = 0.0
            end

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
        "per_run_log_taus" => per_run_log_taus,  # (n_sweep × n_repeats); NaN = failed run
        "flux_rates" => flux_rates,
        "L" => L, "h" => h,
        "n_configs_per_run" => n_configs_per_run, "n_repeats" => n_repeats,
        "M_threshold" => M_threshold,
        "target_crossing_prob" => target_crossing_prob,
        "n_interfaces" => n_interfaces,
        "adaptive_L" => adaptive_L,
        "adaptive_factor" => adaptive_factor,
        "seed_droplet_size" => seed_droplet_size,
        "randshift" => randshift,
        "L_values" => L_values,
        "lc_values" => lc_values,
        "elapsed_seconds" => elapsed,
    )
    if single_layer
        results["p_values"] = sweep_values
        results["single_layer"] = true
    elseif vary_v
        results["vs"] = sweep_values
        results["p"] = p_fixed
    else
        results["p_values"] = sweep_values
        results["v"] = v
    end
    return results
end
