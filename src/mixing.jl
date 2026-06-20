"""
    measure_mixing_time(L, beta, h, v, n_trials, M_threshold, max_time; single_layer=false)

Run `n_trials` trials from the all-+1 state at fixed (β, h, v) and return the mean
number of time units before the magnetization first drops to or below `M_threshold`.
Trials are parallelized over threads. Trials that hit `max_time` emit a warning.
"""
function measure_mixing_time(L::Int, beta::Float64, h::Float64, v::Float64,
                             n_trials::Int, M_threshold::Float64, max_time::Int;
                             single_layer::Bool=false)
    trial_times = zeros(Int, n_trials)
    Threads.@threads for trial in 1:n_trials
        state = single_layer ? SimulationState(L, beta, h) : SimulationState(L, beta, h, v)
        final_step = evolve_and_measure!(state, max_time, step -> begin
            m = compute_total_magnetization(state)
            m > M_threshold
        end)
        trial_times[trial] = final_step
    end
    n_timed_out = count(==(max_time), trial_times)
    if n_timed_out > 0
        @warn @sprintf("Mixing measurement: %d / %d trials hit max_time = %d at (p=%.3f, v=%.3f) — estimate is unreliable. Increase --max_time.",
                       n_timed_out, n_trials, max_time, exp(-beta), v)
    end
    return mean(trial_times)
end

"""
    run_mixing_mode(; L, v, h, p, p_min, p_max, v_min, v_max, n_steps, vary_v,
                      n_trials, M_threshold, max_time, kwargs...)

Run simulation in mixing mode:
- If vary_v=false (default): fix v, sweep p = exp(-β J) over linspace(p_min, p_max, n_steps)
  Convention: 0 < p ≤ 1; rare-event regime at small p.
- If vary_v=true: fix p, sweep v over linspace(v_min, v_max, n_steps)
- For each parameter value, run n_trials trials starting from all +1
- Record time (in time units) to reach magnetization threshold
- Trials are parallelized over threads
- Return results dict for saving
"""
function run_mixing_mode(; L::Int, v::Float64, h::Float64, p::Float64,
                          p_min::Float64, p_max::Float64,
                          v_min::Float64, v_max::Float64,
                          n_steps::Int, vary_v::Bool,
                          n_trials::Int,
                          M_threshold::Float64, max_time::Int,
                          single_layer::Bool=false,
                          sweep_values_override::Union{Nothing,Vector{Float64}}=nothing,
                          kwargs...)
    sweep_p = sweep_values_override
    if single_layer
        sweep_values = sweep_p === nothing ?
            collect(range(p_min, p_max, length=n_steps)) : sweep_p
        println("=== Single-Chain Ising: Mixing Mode (sweeping p) ===")
        println("L = $L, h = $h")
        println("p values: $sweep_values")
    elseif vary_v
        sweep_values = collect(range(v_min, v_max, length=n_steps))
        all(x -> x > 0, sweep_values) || error("All v values must be > 0.")
        p_fixed = p
        println("=== Sliding Ising Chain: Mixing Mode (sweeping v) ===")
        println("L = $L, p = $p_fixed (β = $(@sprintf("%.4f", -log(p_fixed))))")
        println("v values: $sweep_values")
    else
        sweep_values = sweep_p === nothing ?
            collect(range(p_min, p_max, length=n_steps)) : sweep_p
        println("=== Sliding Ising Chain: Mixing Mode (sweeping p) ===")
        println("L = $L, v = $v")
        println("p values: $sweep_values")
    end
    println("Trials per value: $n_trials, threads: $(Threads.nthreads())")
    println("Magnetization threshold: $M_threshold")

    n_sweep = length(sweep_values)
    mean_mixing_times = zeros(Float64, n_sweep)

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

        mean_tau = measure_mixing_time(L, beta, h, v_cur, n_trials, M_threshold, max_time;
                                        single_layer=single_layer)
        mean_mixing_times[idx] = mean_tau
        @printf("  Mean mixing time: %.2f\n", mean_tau)
    end

    println("\nDone!")

    results = Dict{String, Any}(
        "mean_mixing_times" => mean_mixing_times,
        "L" => L, "h" => h,
        "n_trials" => n_trials, "M_threshold" => M_threshold, "max_time" => max_time,
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
