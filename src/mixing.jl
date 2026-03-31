"""
    run_mixing_mode(; L, v, h, p, p_min, p_max, v_min, v_max, n_steps, vary_v,
                      n_trials, M_threshold, max_time, kwargs...)

Run simulation in mixing mode:
- If vary_v=false (default): fix v, sweep p = exp(β) over linspace(p_min, p_max, n_steps)
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
                          single_layer::Bool=false, kwargs...)
    if single_layer
        sweep_values = collect(range(p_min, p_max, length=n_steps))
        println("=== Single-Chain Ising: Mixing Mode (sweeping p) ===")
        println("L = $L, h = $h")
        println("p values: $sweep_values")
    elseif vary_v
        sweep_values = collect(range(v_min, v_max, length=n_steps))
        all(x -> x > 0, sweep_values) || error("All v values must be > 0.")
        p_fixed = p
        println("=== Sliding Ising Chain: Mixing Mode (sweeping v) ===")
        println("L = $L, p = $p_fixed (β = $(@sprintf("%.4f", log(p_fixed))))")
        println("v values: $sweep_values")
    else
        sweep_values = collect(range(p_min, p_max, length=n_steps))
        println("=== Sliding Ising Chain: Mixing Mode (sweeping p) ===")
        println("L = $L, v = $v")
        println("p values: $sweep_values")
    end
    println("Trials per value: $n_trials, threads: $(Threads.nthreads())")
    println("Magnetization threshold: $M_threshold")

    n_sweep = length(sweep_values)
    mean_mixing_times = zeros(Float64, n_sweep)

    for (idx, val) in enumerate(sweep_values)
        if single_layer
            v_cur = 0.0
            beta = log(val)
            @printf("\n[%d/%d] p = %.3f (β = %.4f)\n", idx, n_sweep, val, beta)
        elseif vary_v
            v_cur = val
            beta = log(p_fixed)
            @printf("\n[%d/%d] v = %.3f\n", idx, n_sweep, v_cur)
        else
            v_cur = v
            beta = log(val)
            @printf("\n[%d/%d] p = %.3f (β = %.4f)\n", idx, n_sweep, val, beta)
        end

        trial_times = zeros(Int, n_trials)

        Threads.@threads for trial in 1:n_trials
            state = single_layer ? SimulationState(L, beta, h) : SimulationState(L, beta, h, v_cur)

            final_step = evolve_and_measure!(state, max_time, step -> begin
                m = compute_total_magnetization(state)
                m > M_threshold  # continue while above threshold
            end)
            trial_times[trial] = final_step
        end

        n_timed_out = count(==(max_time), trial_times)
        if n_timed_out > 0
            label = vary_v ? @sprintf("v = %.3f", val) : (single_layer ? @sprintf("T = %.4f", val) : @sprintf("p = %.3f", val))
            @warn "$label: $n_timed_out / $n_trials trials hit max_time = $max_time — mixing time estimate is unreliable. Increase --max_time."
        end
        mean_tau = mean(trial_times)
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
