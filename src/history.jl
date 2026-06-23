"""
    run_history_mode(; L, v, beta, h, T_steps, init, domain_start, domain_end, kwargs...)

Run simulation in history mode:
- Initialize chains according to `init` ("domain" or "random")
- Run T_steps time units, recording magnetization at each integer time
- Return results dict for saving
"""
function run_history_mode(; L::Int, v::Float64, beta::Float64, h::Float64,
                           T_steps::Int, init::String="domain",
                           domain_start::Int=3L÷8, domain_end::Int=5L÷8,
                           kwargs...)
    println("=== Sliding Ising Chain: History Mode ===")
    println("L = $L, v = $v, β = $beta, T = $T_steps, init = $init")

    state = SimulationState(L, beta, h, v)

    if init == "domain"
        println("Minority domain: sites $domain_start to $domain_end")
        for i in domain_start:domain_end
            state.top[i] = Int8(-1)
            state.bottom[i] = Int8(-1)
        end
        recompute_mag_sum!(state)
    elseif init == "random"
        state.top .= rand(Int8[-1, 1], L)
        state.bottom .= rand(Int8[-1, 1], L)
        recompute_mag_sum!(state)
    else
        error("Unknown init: $init. Use 'domain' or 'random'.")
    end

    magnetization_history = zeros(Int8, T_steps + 1, L)
    magnetization_history[1, :] = compute_local_magnetization(state)

    println("Running simulation...")
    @time evolve_and_measure!(state, T_steps, step -> begin
        magnetization_history[step + 1, :] = compute_local_magnetization(state)
        if step % 200 == 0
            m = compute_total_magnetization(state)
            @printf("  Step %d/%d, m = %.4f\n", step, T_steps, m)
        end
        true
    end)

    println("Done!")

    return Dict{String, Any}(
        "magnetization_history" => magnetization_history,
        "L" => L, "v" => v, "beta" => beta, "h" => h,
        "T_steps" => T_steps, "init" => init,
        "domain_start" => domain_start, "domain_end" => domain_end,
    )
end
