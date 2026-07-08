# ── Demon-based T_eff helpers ──

"""
    demon_probe(state, E_d, h) -> E_d_new

Probe the demon: pick a random site, compute the energy change for a hypothetical
spin flip (without flipping). If the demon can absorb the cost (E_d - dE >= 0),
update E_d. Returns the new E_d.
"""
function demon_probe(state::SimulationState, E_d::Float64, h::Float64)
    L = state.L
    top = state.top
    bottom = state.bottom

    if state.single_layer
        i = rand(1:L)
        ip = i == L ? 1 : i + 1
        im = i == 1 ? L : i - 1
        σ = top[i]
        ns = top[im] + top[ip]
    else
        chain = rand(1:2)
        i = rand(1:L)
        ip = i == L ? 1 : i + 1
        im = i == 1 ? L : i - 1
        if chain == 1
            σ = top[i]
            ns = top[im] + top[ip] + bottom[i]
        else
            σ = bottom[i]
            ns = bottom[im] + bottom[ip] + top[i]
        end
    end

    dE = 2.0 * σ * (ns + h)

    if E_d - dE >= 0
        E_d -= dE
    end

    return E_d
end

"""
    fit_teff_demon(bins, counts) -> Float64

Fit the demon energy histogram to P(E_d) ~ exp(-E_d / T_eff).
Returns T_eff from linear regression of log(P) vs E_d.
"""
function fit_teff_demon(bins::Vector{Int}, counts::Vector{Int})
    total = sum(counts)
    valid = counts .> 0
    x = Float64.(bins[valid])
    y = log.(counts[valid] ./ total)

    n = length(x)
    if n < 2
        return NaN
    end

    sx = sum(x)
    sy = sum(y)
    sxx = sum(x .^ 2)
    sxy = sum(x .* y)
    denom = n * sxx - sx^2
    if abs(denom) < 1e-15
        return NaN
    end
    slope = (n * sxy - sx * sy) / denom
    if slope >= 0
        return NaN
    end
    return -1.0 / slope
end

# ── FDR-based T_eff helpers ──

"""
    compute_autocorrelation(m_series, T_response) -> Vector{Float64}

Compute the autocorrelation C(τ) = ⟨(m(t) - m̄)(m(t+τ) - m̄)⟩ for τ = 0..T_response.
"""
function compute_autocorrelation(m_series::Vector{Float64}, T_response::Int)
    T = length(m_series)
    m_mean = mean(m_series)
    dm = m_series .- m_mean
    C = zeros(Float64, T_response + 1)
    for tau in 0:T_response
        s = 0.0
        n = T - tau
        @inbounds for t in 1:n
            s += dm[t] * dm[t + tau]
        end
        C[tau + 1] = s / n
    end
    return C
end

"""
    compute_dCdtau(C) -> Vector{Float64}

Compute dC/dτ via centered finite differences. Returns array of same length as C.
"""
function compute_dCdtau(C::Vector{Float64})
    n = length(C)
    dC = zeros(Float64, n)
    if n >= 2
        dC[1] = C[2] - C[1]  # forward difference at boundary
    end
    for i in 2:n-1
        dC[i] = (C[i+1] - C[i-1]) / 2.0  # centered
    end
    if n >= 2
        dC[n] = C[n] - C[n-1]  # backward difference at boundary
    end
    return dC
end

"""
    run_perturbation_experiments(state, h_pert, n_perturbations, T_response, T_spacing)

Run n_perturbations impulse-response experiments. For each:
1. Evolve T_spacing steps to decorrelate
2. Record baseline magnetization
3. Apply field h_pert for one time step (swapping acceptance table), then restore
4. Evolve T_response steps recording the magnetization response

Returns a matrix (n_perturbations × T_response) of Δm values.
"""
function run_perturbation_experiments(state::SimulationState, h_pert::Float64,
                                      n_perturbations::Int, T_response::Int, T_spacing::Int)
    L = state.L
    delta_m = zeros(Float64, n_perturbations, T_response)

    for k in 1:n_perturbations
        # Decorrelate
        evolve_time!(state, T_spacing)

        # Baseline
        m_before = compute_total_magnetization(state)

        # Apply perturbation for one time step
        table_saved = copy(state.acceptance_table)
        h_orig = state.h
        state.h = h_orig + h_pert
        if state.single_layer
            state.acceptance_table = build_acceptance_table_single(state.beta, state.h)
        else
            state.acceptance_table = build_acceptance_table(state.beta, state.h)
        end
        evolve_and_measure!(state, 1, _ -> true)
        state.h = h_orig
        state.acceptance_table = table_saved

        # Record response
        evolve_and_measure!(state, T_response, step -> begin
            delta_m[k, step] = compute_total_magnetization(state) - m_before
            true
        end)
    end

    return delta_m
end

"""
    extract_teff_fdr(C, dC, R, chi) -> (teff_integrated, teff_pointwise)

Extract T_eff from:
- Primary (Eq. 2): teff_pointwise[τ] = dC[τ] / R[τ]
- Cross-check: linear regression of χ vs [C(0) - C(τ)], slope = 1/T_eff
"""
function extract_teff_fdr(C::Vector{Float64}, dC::Vector{Float64},
                           R::Vector{Float64}, chi::Vector{Float64})
    T_response = length(R)

    # Pointwise T_eff = dC/dτ / R(τ)
    teff_pw = zeros(Float64, T_response)
    for tau in 1:T_response
        if abs(R[tau]) > 1e-15
            teff_pw[tau] = dC[tau + 1] / R[tau]  # dC is 1-indexed: dC[tau+1] = dC/dτ at τ
        else
            teff_pw[tau] = NaN
        end
    end

    # Integrated cross-check: χ(τ) vs C(0) - C(τ)
    delta_C = [C[1] - C[tau + 1] for tau in 1:T_response]  # C[1] = C(0)

    # Linear regression through origin: chi = (1/T_eff) * delta_C
    # => T_eff = sum(delta_C .* delta_C) / sum(delta_C .* chi)
    valid = isfinite.(chi) .& isfinite.(delta_C) .& (abs.(chi) .> 1e-15)
    if sum(valid) < 2
        return (NaN, teff_pw)
    end
    x = delta_C[valid]
    y = chi[valid]
    slope = sum(x .* y) / sum(x .* x)
    teff_int = abs(slope) > 1e-15 ? 1.0 / slope : NaN

    return (teff_int, teff_pw)
end

# ── Main entry point ──

"""
    run_teff_mode(; ...)

Run simulation in effective temperature mode.
Default: FDR method (dC/dτ / R(τ) and χ vs ΔC cross-check).
With --demon=true: demon algorithm (histogram of demon energy).
"""
function run_teff_mode(; L::Int, v::Float64, h::Float64, T::Float64,
                        T_min::Float64, T_max::Float64,
                        v_min::Float64, v_max::Float64,
                        n_steps::Int, vary_v::Bool,
                        T_equil::Int, T_sample::Int,
                        demon_interval::Int=200,
                        demon::Bool=false,
                        h_pert::Float64=0.01,
                        n_perturbations::Int=200,
                        T_response::Int=0,
                        n_samples::Int=1,
                        single_layer::Bool=false,
                        randstart::Bool=false, kwargs...)

    # ── Sweep setup (shared by both methods) ──
    if vary_v
        sweep_values = collect(range(v_min, v_max, length=n_steps))
        T_fixed = T
        beta_fixed = 1.0 / T_fixed
    else
        sweep_values = collect(range(T_min, T_max, length=n_steps))
    end

    # ── Demon method ──
    if demon
        if vary_v
            println("=== Sliding Ising Chain: T_eff Mode [demon] (sweeping v) ===")
            println("L = $L, T = $T_fixed (β = $(@sprintf("%.4f", beta_fixed)))")
            println("v values: $sweep_values")
        else
            println("=== Sliding Ising Chain: T_eff Mode [demon] (sweeping T) ===")
            println("L = $L, v = $v")
            println("T values: $sweep_values")
        end
        println("Equilibration: $T_equil, Sampling: $T_sample, Demon interval: $demon_interval MC steps")
        println("Samples per sweep value: $n_samples")
        println("Threads: $(Threads.nthreads())")

        n_sweep = length(sweep_values)
        teff_values = zeros(Float64, n_sweep)
        all_histograms = Vector{Dict{Int,Int}}(undef, n_sweep)
        n_probes_per_step = max(1, L ÷ demon_interval)

        Threads.@threads for idx in 1:n_sweep
            val = sweep_values[idx]
            tid = Threads.threadid()

            if vary_v
                v_cur = val; beta_val = beta_fixed; h_val = h
            else
                v_cur = v; beta_val = 1.0 / val; h_val = h
            end

            histogram = Dict{Int,Int}()

            for s in 1:n_samples
                label = vary_v ? @sprintf("v = %.3f", v_cur) : @sprintf("T = %.4f", val)
                @printf("  [Thread %d] %s, sample %d/%d: running...\n", tid, label, s, n_samples)

                state = single_layer ? SimulationState(L, beta_val, h_val) : SimulationState(L, beta_val, h_val, v_cur)
                if randstart
                    state.top .= rand(Int8[-1, 1], L)
                    state.bottom .= rand(Int8[-1, 1], L)
                    recompute_mag_sum!(state)
                end
                evolve_time!(state, T_equil)

                E_d = 0.0
                evolve_and_measure!(state, T_sample, step -> begin
                    for _ in 1:n_probes_per_step
                        E_d = demon_probe(state, E_d, h_val)
                        E_d_int = round(Int, E_d)
                        histogram[E_d_int] = get(histogram, E_d_int, 0) + 1
                    end
                    true
                end)
            end

            bins = sort(collect(keys(histogram)))
            counts = [histogram[b] for b in bins]
            teff = fit_teff_demon(bins, counts)
            teff_values[idx] = teff
            all_histograms[idx] = histogram
            @printf("  [Thread %d] done: T_eff = %.4f\n", tid, teff)
        end

        println("\nDone!")
        all_bins_set = sort(collect(reduce(union, keys.(all_histograms))))
        n_bins = length(all_bins_set)
        hist_matrix = zeros(Int, n_bins, n_sweep)
        for idx in 1:n_sweep
            for (j, b) in enumerate(all_bins_set)
                hist_matrix[j, idx] = get(all_histograms[idx], b, 0)
            end
        end

        results = Dict{String, Any}(
            "teff_values" => teff_values,
            "demon_bins" => all_bins_set,
            "demon_histograms" => hist_matrix,
            "method" => "demon",
            "L" => L, "h" => h,
            "T_equil" => T_equil, "T_sample" => T_sample,
            "demon_interval" => demon_interval,
        )
        if vary_v
            results["vs"] = sweep_values; results["T"] = T_fixed
        else
            results["T_values"] = sweep_values; results["v"] = v
        end
        return results
    end

    # ── FDR method (default) ──

    T_response_actual = T_response > 0 ? T_response : min(1000, T_sample ÷ 10)
    T_spacing = 2 * T_response_actual

    if vary_v
        println("=== Sliding Ising Chain: T_eff Mode [FDR] (sweeping v) ===")
        println("L = $L, T = $T_fixed (β = $(@sprintf("%.4f", beta_fixed)))")
        println("v values: $sweep_values")
    else
        println("=== Sliding Ising Chain: T_eff Mode [FDR] (sweeping T) ===")
        println("L = $L, v = $v")
        println("T values: $sweep_values")
    end
    @printf("h_pert = %.4f, n_perturbations = %d, T_response = %d\n", h_pert, n_perturbations, T_response_actual)
    println("Equilibration: $T_equil, Sampling: $T_sample")
    println("Threads: $(Threads.nthreads())")

    n_sweep = length(sweep_values)
    teff_values = zeros(Float64, n_sweep)
    all_C = Vector{Vector{Float64}}(undef, n_sweep)
    all_dC = Vector{Vector{Float64}}(undef, n_sweep)
    all_R = Vector{Vector{Float64}}(undef, n_sweep)
    all_chi = Vector{Vector{Float64}}(undef, n_sweep)
    all_teff_pw = Vector{Vector{Float64}}(undef, n_sweep)

    Threads.@threads for idx in 1:n_sweep
        val = sweep_values[idx]
        tid = Threads.threadid()

        if vary_v
            v_cur = val; beta_val = beta_fixed; h_val = h
            @printf("  [Thread %d] v = %.3f: equilibrating...\n", tid, v_cur)
        else
            v_cur = v; beta_val = 1.0 / val; h_val = h
            @printf("  [Thread %d] T = %.4f (β = %.4f): equilibrating...\n", tid, val, beta_val)
        end

        state = single_layer ? SimulationState(L, beta_val, h_val) : SimulationState(L, beta_val, h_val, v_cur)
        if randstart
            state.top .= rand(Int8[-1, 1], L)
            state.bottom .= rand(Int8[-1, 1], L)
            recompute_mag_sum!(state)
        end
        evolve_time!(state, T_equil)

        # Phase 1: Collect m(t) time series for autocorrelation
        label = vary_v ? @sprintf("v = %.3f", v_cur) : @sprintf("T = %.4f", val)
        @printf("  [Thread %d] %s: Phase 1 — correlation (%d steps)...\n", tid, label, T_sample)

        m_series = zeros(Float64, T_sample)
        evolve_and_measure!(state, T_sample, step -> begin
            m_series[step] = compute_total_magnetization(state)
            true
        end)

        C = compute_autocorrelation(m_series, T_response_actual)
        dC = compute_dCdtau(C)

        # Phase 2: Perturbation experiments for response
        @printf("  [Thread %d] %s: Phase 2 — response (%d perturbations)...\n", tid, label, n_perturbations)

        delta_m = run_perturbation_experiments(state, h_pert, n_perturbations,
                                               T_response_actual, T_spacing)

        R = vec(mean(delta_m, dims=1)) ./ h_pert
        chi = cumsum(R)

        # Extract T_eff
        (teff_int, teff_pw) = extract_teff_fdr(C, dC, R, chi)
        teff_values[idx] = teff_int
        all_C[idx] = C
        all_dC[idx] = dC
        all_R[idx] = R
        all_chi[idx] = chi
        all_teff_pw[idx] = teff_pw

        @printf("  [Thread %d] %s: T_eff = %.4f (integrated)\n", tid, label, teff_int)
    end

    # Build results
    println("\nDone!")

    # Store arrays as padded matrices
    C_matrix = zeros(Float64, T_response_actual + 1, n_sweep)
    dC_matrix = zeros(Float64, T_response_actual + 1, n_sweep)
    R_matrix = zeros(Float64, T_response_actual, n_sweep)
    chi_matrix = zeros(Float64, T_response_actual, n_sweep)
    teff_pw_matrix = zeros(Float64, T_response_actual, n_sweep)
    for idx in 1:n_sweep
        C_matrix[:, idx] = all_C[idx]
        dC_matrix[:, idx] = all_dC[idx]
        R_matrix[:, idx] = all_R[idx]
        chi_matrix[:, idx] = all_chi[idx]
        teff_pw_matrix[:, idx] = all_teff_pw[idx]
    end

    results = Dict{String, Any}(
        "teff_values" => teff_values,
        "C_arrays" => C_matrix,
        "dC_arrays" => dC_matrix,
        "R_arrays" => R_matrix,
        "chi_arrays" => chi_matrix,
        "teff_pointwise" => teff_pw_matrix,
        "method" => "fdr",
        "T_response" => T_response_actual,
        "h_pert" => h_pert,
        "n_perturbations" => n_perturbations,
        "L" => L, "h" => h,
        "T_equil" => T_equil, "T_sample" => T_sample,
    )
    if vary_v
        results["vs"] = sweep_values; results["T"] = T_fixed
    else
        results["T_values"] = sweep_values; results["v"] = v
    end
    return results
end
