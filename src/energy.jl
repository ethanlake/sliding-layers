"""
    run_energy_mode(; L, v, h, p, p_min, p_max, v_min, v_max, n_steps, vary_v,
                      T_equil, T_sample, kwargs...)

Run simulation in energy mode:
- If vary_v=false (default): fix v, sweep p = exp(β) over linspace(p_min, p_max, n_steps)
- If vary_v=true: fix p, sweep v over linspace(v_min, v_max, n_steps)
- For each parameter value, run one long simulation and estimate steady-state energy and heat flow
- Parallelized over sweep values
- T_equil and T_sample are in time units (each = L MC updates)
- Return results dict for saving
"""
function run_energy_mode(; L::Int, v::Float64, h::Float64, p::Float64,
                          p_min::Float64, p_max::Float64,
                          v_min::Float64, v_max::Float64,
                          n_steps::Int, vary_v::Bool,
                          T_equil::Int, T_sample::Int, kwargs...)
    if vary_v
        sweep_values = collect(range(v_min, v_max, length=n_steps))
        p_fixed = p
        beta_fixed = log(p_fixed)
        println("=== Sliding Ising Chain: Energy Mode (sweeping v) ===")
        println("L = $L, p = $p_fixed (β = $(@sprintf("%.4f", beta_fixed)))")
        println("v values: $sweep_values")
    else
        sweep_values = collect(range(p_min, p_max, length=n_steps))
        println("=== Sliding Ising Chain: Energy Mode (sweeping p) ===")
        println("L = $L, v = $v")
        println("p values: $sweep_values")
    end
    println("Equilibration: $T_equil time units, Sampling: $T_sample time units")
    println("Threads: $(Threads.nthreads())")

    n_sweep = length(sweep_values)
    mean_energies = zeros(Float64, n_sweep)
    mean_heat_flows = zeros(Float64, n_sweep)

    Threads.@threads for idx in 1:n_sweep
        val = sweep_values[idx]
        tid = Threads.threadid()

        if vary_v
            v_cur = val
            beta_val = beta_fixed
            @printf("  [Thread %d] v = %.3f: equilibrating (%d time units)...\n", tid, v_cur, T_equil)
        else
            v_cur = v
            beta_val = log(val)
            @printf("  [Thread %d] p = %.3f (β = %.4f): equilibrating (%d time units)...\n", tid, val, beta_val, T_equil)
        end

        state = SimulationState(L, beta_val, h, v_cur)
        # Random initial condition
        state.top .= rand(Int8[-1, 1], L)
        state.bottom .= rand(Int8[-1, 1], L)
        recompute_mag_sum!(state)

        evolve_time!(state, T_equil)

        if vary_v
            @printf("  [Thread %d] v = %.3f: sampling (%d time units)...\n", tid, v_cur, T_sample)
        else
            @printf("  [Thread %d] p = %.3f: sampling (%d time units)...\n", tid, val, T_sample)
        end

        E_sum = 0.0
        Q_sum = 0.0
        evolve_and_measure_heat!(state, T_sample, (step, heat) -> begin
            E_sum += compute_energy(state)
            Q_sum += heat
            true
        end)

        mean_energies[idx] = E_sum / T_sample
        mean_heat_flows[idx] = Q_sum / T_sample
        @printf("  [Thread %d] done: mean energy = %.6f, mean heat flow = %.6f\n",
                tid, mean_energies[idx], mean_heat_flows[idx])
    end

    println("\nDone!")

    results = Dict{String, Any}(
        "mean_energies" => mean_energies,
        "mean_heat_flows" => mean_heat_flows,
        "L" => L, "h" => h,
        "T_equil" => T_equil, "T_sample" => T_sample,
    )
    if vary_v
        results["vs"] = sweep_values
        results["p"] = p_fixed
    else
        results["p_values"] = sweep_values
        results["v"] = v
    end
    return results
end
