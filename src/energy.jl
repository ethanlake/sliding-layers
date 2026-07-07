"""
    run_energy_mode(; L, v, h, p, p_min, p_max, v_min, v_max, n_steps, vary_v,
                      T_equil, T_sample, kwargs...)

Run simulation in energy mode:
- If vary_v=false (default): fix v, sweep p = exp(-β J) over linspace(p_min, p_max, n_steps)
  Convention: 0 < p ≤ 1.
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
                          T_equil::Int, T_sample::Int,
                          sweep_values_override::Union{Nothing,Vector{Float64}}=nothing,
                          _checkpoint_adj::Union{Nothing,String}=nothing,
                          randshift::Bool=true,
                          randstart::Bool=false,
                          kwargs...)
    sweep_p = sweep_values_override
    if vary_v
        sweep_values = collect(range(v_min, v_max, length=n_steps))
        p_fixed = p
        beta_fixed = -log(p_fixed)
        println("=== Sliding Ising Chain: Energy Mode (sweeping v) ===")
        println("L = $L, p = $p_fixed (β = $(@sprintf("%.4f", beta_fixed)))")
        println("v values: $sweep_values")
    else
        sweep_values = sweep_p === nothing ?
            collect(range(p_min, p_max, length=n_steps)) : sweep_p
        println("=== Sliding Ising Chain: Energy Mode (sweeping p) ===")
        println("L = $L, v = $v")
        println("p values: $sweep_values")
    end
    println("Equilibration: $T_equil time units, Sampling: $T_sample time units")
    println("Threads: $(Threads.nthreads())")

    n_sweep = length(sweep_values)
    # NaN-initialised so the plotter can mask incomplete sweep points after a
    # crash. The arrays are pre-allocated so that incremental writes after each
    # sweep-point completion mutate the SAME dict we serialise to disk.
    mean_energies = fill(NaN, n_sweep)
    mean_heat_flows = fill(NaN, n_sweep)
    mean_energies_std = fill(NaN, n_sweep)
    mean_heat_flows_std = fill(NaN, n_sweep)
    # Block-averaging for unbiased stderrs on autocorrelated MC samples. We
    # carve the T_sample sampling window into n_blocks blocks (default 20);
    # each block's mean is one "independent" observation if block_size is
    # much larger than the integrated autocorrelation time. Stderr of the
    # overall mean is std(block_means) / sqrt(n_blocks).
    n_blocks_target = 20
    block_size = max(1, T_sample ÷ n_blocks_target)
    n_blocks = T_sample ÷ block_size

    # Build the full results dict NOW (with NaN placeholders for unfilled
    # arrays). After each sweep point completes we'll dump it to disk under a
    # lock — a crash mid-sweep then loses at most one in-flight sweep point.
    results = Dict{String, Any}(
        "mean_energies" => mean_energies,
        "mean_heat_flows" => mean_heat_flows,
        "mean_energies_std" => mean_energies_std,
        "mean_heat_flows_std" => mean_heat_flows_std,
        "n_blocks" => n_blocks,
        "block_size" => block_size,
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

    # Compute the checkpoint path now if checkpointing is enabled. _checkpoint_adj
    # is non-nothing when run_simulation passes save=true; the string carries
    # the (possibly empty) --adj suffix so the partial JLD2 lands at the same
    # path the final save_results would write. We also dump an immediate
    # skeleton (all-NaN arrays + full metadata) so the user has a recognisable
    # output file even if no sweep point finishes before a crash.
    ckpt_path = nothing
    if _checkpoint_adj !== nothing
        if !isdir("data"); mkdir("data"); end
        ckpt_path = build_filename("energy", results)
        if !isempty(_checkpoint_adj)
            ckpt_path = replace(ckpt_path, ".jld2" => "$(_checkpoint_adj).jld2")
        end
        @printf("Checkpointing to %s after each sweep point.\n", ckpt_path)
        jldsave(ckpt_path; (Symbol(k) => v for (k, v) in results)...)
    end
    ckpt_lock = ReentrantLock()

    Threads.@threads for idx in 1:n_sweep
        val = sweep_values[idx]
        tid = Threads.threadid()

        if vary_v
            v_cur = val
            beta_val = beta_fixed
            @printf("  [Thread %d] v = %.3f: equilibrating (%d time units)...\n", tid, v_cur, T_equil)
        else
            v_cur = v
            beta_val = -log(val)
            @printf("  [Thread %d] p = %.3f (β = %.4f): equilibrating (%d time units)...\n", tid, val, beta_val, T_equil)
        end

        state = SimulationState(L, beta_val, h, v_cur)
        # Default: polarized (all-+1 from constructor). `--randstart=true`
        # instead flips each spin i.i.d. on {-1, +1}. Steady-state observables
        # (⟨E⟩, ⟨Q̇⟩) are invariant to the init after T_equil burn-in; the
        # choice just changes the transient the burn-in has to erase.
        if randstart
            state.top .= rand(Int8[-1, 1], L)
            state.bottom .= rand(Int8[-1, 1], L)
            recompute_mag_sum!(state)
        end

        evolve_time!(state, T_equil; randshift)

        if vary_v
            @printf("  [Thread %d] v = %.3f: sampling (%d time units, %d blocks of %d)...\n",
                    tid, v_cur, T_sample, n_blocks, block_size)
        else
            @printf("  [Thread %d] p = %.3f: sampling (%d time units, %d blocks of %d)...\n",
                    tid, val, T_sample, n_blocks, block_size)
        end

        E_blocks = zeros(Float64, n_blocks)
        Q_blocks = zeros(Float64, n_blocks)
        block_idx = 1
        E_in_block = 0.0
        Q_in_block = 0.0
        samples_in_block = 0
        evolve_and_measure_heat!(state, T_sample, (step, heat) -> begin
            # Past the last full block? Discard leftover samples (≤ block_size-1).
            block_idx > n_blocks && return true
            E_in_block += compute_energy(state)
            Q_in_block += heat
            samples_in_block += 1
            if samples_in_block == block_size
                E_blocks[block_idx] = E_in_block / block_size
                Q_blocks[block_idx] = Q_in_block / block_size
                block_idx += 1
                E_in_block = 0.0
                Q_in_block = 0.0
                samples_in_block = 0
            end
            true
        end; randshift)

        mean_energies[idx] = mean(E_blocks)
        mean_heat_flows[idx] = mean(Q_blocks)
        # Stderr of the mean from block averaging; falls back to 0 if only one
        # block (Statistics.std would error on n=1).
        mean_energies_std[idx]   = n_blocks > 1 ? std(E_blocks) / sqrt(n_blocks) : 0.0
        mean_heat_flows_std[idx] = n_blocks > 1 ? std(Q_blocks) / sqrt(n_blocks) : 0.0
        @printf("  [Thread %d] done: ⟨E⟩ = %.6f ± %.6f, ⟨Q̇⟩ = %.6f ± %.6f\n",
                tid, mean_energies[idx], mean_energies_std[idx],
                mean_heat_flows[idx], mean_heat_flows_std[idx])

        # Periodic checkpoint. Lock the dump so concurrent sweep-point threads
        # don't interleave writes to the same JLD2.
        if ckpt_path !== nothing
            lock(ckpt_lock) do
                jldsave(ckpt_path; (Symbol(k) => v for (k, v) in results)...)
            end
        end
    end

    println("\nDone!")
    return results
end
