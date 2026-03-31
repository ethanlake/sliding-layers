"""
    run_erosion_history(l, beta, h, v, T_evolve) -> Matrix{Int8}

Run a single erosion trial for domain size l and return the spacetime
magnetization history (T_evolve+1 × L_sys), matching the history mode format.
"""
function run_erosion_history(l::Int, beta::Float64, h::Float64, v::Float64)
    L_sys = round(Int, 2 * l * (v + 1))
    T_evolve = 2 * l
    state = SimulationState(L_sys, beta, h, v)

    for i in 1:l
        state.top[i] = Int8(-1)
        state.bottom[i] = Int8(-1)
    end
    recompute_mag_sum!(state)

    history = zeros(Int8, T_evolve + 1, L_sys)
    history[1, :] = compute_local_magnetization(state)

    evolve_and_measure!(state, T_evolve, step -> begin
        history[step + 1, :] = compute_local_magnetization(state)
        true
    end)

    return history
end

"""
    show_erosion_histories(lc, beta, h, v)

Display spacetime histories for domain sizes lc, 0.75*lc, and 1.25*lc
(3 trials each), using the same heatmap style as sliding_plotter.
Blocks until the user closes each figure before showing the next set.
"""
function show_erosion_histories(lc::Int, beta::Float64, h::Float64, v::Float64)
    for (scale, label) in [(1.0, "lc"), (0.75, "0.75·lc"), (1.25, "1.25·lc")]
        l = max(2, round(Int, scale * lc))
        L_sys = round(Int, 2 * l * (v + 1))
        T_evolve = 2 * l
        println("  Showing 3 histories for l = $l ($label), L_sys = $L_sys, T = $T_evolve")

        plots = []
        for trial in 1:3
            hist = run_erosion_history(l, beta, h, v)
            # Shift into co-moving frame at v/2
            for t in 0:T_evolve
                shift = round(Int, v * t / 2) % L_sys
                if shift > 0
                    hist[t+1, :] = circshift(hist[t+1, :], -shift)
                end
            end
            p = Plots.heatmap(hist',
                       c=:RdBu, clims=(-2, 2),
                       xlabel="Site", ylabel="t",
                       title="$label, trial $trial",
                       aspect_ratio=:auto, size=(300, 600))
            push!(plots, p)
        end
        fig = Plots.plot(plots..., layout=(1, 3), size=(900, 600))
        display(fig)
        println("  Close the plot window to continue...")
        readline()
    end
end

"""
    plot_erode_vs_l(lc, beta, h, v, thresh_prob; doublon_mode)

Plot shrinkage probability vs domain size l, for l in [lc/2, 3lc/2] with
10 points and 200 samples each. Draws a horizontal line at thresh_prob.
Blocks until the user closes the plot.
"""
function plot_erode_vs_l(lc::Int, beta::Float64, h::Float64, v::Float64,
                          thresh_prob::Float64, n_samples::Int;
                          doublon_mode::Bool=false, show_plots::Bool=false,
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
            sp = measure_shrink_prob(l, beta, h, v, n_samples; doublon_mode)
            cache[l] = sp
            @printf("    l = %d, p_shrink = %.3f\n", l, sp)
        end
        push!(probs, sp)
    end

    if show_plots
        fig = Plots.plot(l_values, probs, seriestype=:scatter,
                         xlabel="l", ylabel="shrink probability",
                         title=@sprintf("lc = %d, β = %.3f, v = %.1f", lc, beta, v),
                         legend=false, markersize=5, size=(500, 400))
        Plots.hline!(fig, [thresh_prob], linestyle=:dash, color=:red, linewidth=1.5)
        Plots.vline!(fig, [lc], linestyle=:dash, color=:gray, linewidth=1.0)
        display(fig)
        println("  Close the plot window to continue...")
        readline()
    end

    return (l_values, collect(probs))
end

function run_erosion_test_mode(; v::Float64, h::Float64, p::Float64,
                                p_min::Float64, p_max::Float64,
                                v_min::Float64, v_max::Float64,
                                n_steps::Int, vary_v::Bool,
                                thresh_prob::Float64,
                                erosion_num_trials::Int,
                                lmin::Int=4, show_histories::Bool=false,
                                doublon_mode::Bool=false,
                                erode_vs_l::Bool=false,
                                show_plots::Bool=false, kwargs...)
    h = -abs(h)
    if vary_v
        sweep_values = collect(range(v_min, v_max, length=n_steps))
        all(x -> x > 0, sweep_values) || error("All v values must be > 0 for erosion test.")
        p_fixed = p
        println("=== Sliding Ising Chain: Erosion Test Mode (sweeping v) ===")
        println("p = $p_fixed (β = $(@sprintf("%.4f", log(p_fixed))))")
        println("v values: $sweep_values")
    else
        sweep_values = collect(range(p_min, p_max, length=n_steps))
        v == 0.0 && error("Erosion test mode requires v > 0 (sliding is needed for erosion).")
        println("=== Sliding Ising Chain: Erosion Test Mode (sweeping p) ===")
        println("v = $v")
        println("p values: $sweep_values")
    end
    println("Trials per domain size: $erosion_num_trials, threads: $(Threads.nthreads())")

    lc_values = Int[]
    prev_lc = lmin  # starting guess; updated after each sweep value
    erode_l_data = Vector{Vector{Int}}()     # l values for each sweep point
    erode_prob_data = Vector{Vector{Float64}}()  # shrink probs for each sweep point

    for (idx, val) in enumerate(sweep_values)
        if vary_v
            v_cur = val
            beta_val = log(p_fixed)
            @printf("\n[%d/%d] v = %.3f\n", idx, length(sweep_values), v_cur)
        else
            v_cur = v
            beta_val = log(val)
            @printf("\n[%d/%d] p = %.3f (β = %.4f)\n", idx, length(sweep_values), val, beta_val)
        end

        sp_cache = Dict{Int,Float64}()
        lc = find_critical_length(beta_val, h, v_cur, erosion_num_trials, prev_lc, thresh_prob, lmin; doublon_mode, cache=sp_cache)

        if vary_v
            @printf("  lc = %d for v = %.3f\n", lc, v_cur)
        else
            @printf("  lc = %d for p = %.3f\n", lc, val)
        end
        push!(lc_values, lc)
        prev_lc = lc

        if show_histories
            show_erosion_histories(lc, beta_val, h, v_cur)
        end
        if erode_vs_l
            (ls, ps) = plot_erode_vs_l(lc, beta_val, h, v_cur, thresh_prob, erosion_num_trials; doublon_mode, show_plots, cache=sp_cache)
            push!(erode_l_data, ls)
            push!(erode_prob_data, ps)
        end
    end

    println("\nDone!")

    results = Dict{String, Any}(
        "lc_values" => lc_values,
        "h" => h, "num_trials" => erosion_num_trials, "vary_v" => vary_v,
    )
    if vary_v
        results["vs"] = sweep_values
        results["p"] = p_fixed
    else
        results["p_values"] = sweep_values
        results["v"] = v
    end
    if erode_vs_l
        # Save as matrices padded to uniform length (JLD2-friendly)
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
        results["thresh_prob"] = thresh_prob
    end
    return results
end

"""
    measure_shrink_prob(l, beta, h, v, n_trials; doublon_mode=false) -> Float64

Run n_trials erosion simulations for domain size l. Returns the fraction of
trials where the domain shrunk.

If doublon_mode=false (default): shrinkage = total minority spins < 1.5l after 2l time steps.
If doublon_mode=true: shrinkage = number of doublons (sites where both spins are minority) < 0.25l after 4l time steps.
"""
function measure_shrink_prob(l::Int, beta::Float64, h::Float64, v::Float64, n_trials::Int;
                              doublon_mode::Bool=false)
    L_sys = round(Int, 2 * l * (v + 1))
    T_evolve = doublon_mode ? 5 * l : 2 * l
    threshold = doublon_mode ? 0.1 * l : 1.5 * l

    shrink_flags = zeros(Int, n_trials)

    Threads.@threads for trial in 1:n_trials
        state = SimulationState(L_sys, beta, h, v)

        for i in 1:l
            state.top[i] = Int8(-1)
            state.bottom[i] = Int8(-1)
        end
        recompute_mag_sum!(state)

        evolve_time!(state, T_evolve)

        if doublon_mode
            n_doublons = count(i -> state.top[i] == Int8(-1) && state.bottom[i] == Int8(-1), 1:L_sys)
            shrunk = n_doublons <= threshold
        else
            n_flipped = count(==(Int8(-1)), state.top) + count(==(Int8(-1)), state.bottom)
            shrunk = n_flipped < threshold
        end

        if shrunk
            shrink_flags[trial] = 1
        end
    end

    return sum(shrink_flags) / n_trials
end

"""
    find_critical_length(beta, h, v, n_trials, l_start) -> Int

Find the critical domain size lc using adaptive search:
1. Coarse phase: starting from l_start, take geometrically growing steps with
   a small number of trials to quickly bracket lc (find l where shrink_prob < 0.75).
2. Binary search: narrow the bracket to find the exact transition point.
3. Final verification: confirm with the full trial count.
"""
function find_critical_length(beta::Float64, h::Float64, v::Float64,
                              n_trials::Int, l_start::Int, thresh_prob::Float64,
                              lmin::Int=4; doublon_mode::Bool=false,
                              cache::Dict{Int,Float64}=Dict{Int,Float64}())
    coarse_trials = min(n_trials, max(200, n_trials ÷ 10))

    # Helper: measure and cache results from full n_trials runs
    function measure_and_cache(l, n)
        sp = measure_shrink_prob(l, beta, h, v, n; doublon_mode)
        if n == n_trials
            cache[l] = sp
        end
        return sp
    end

    # Start search a bit below the previous lc to handle non-monotonicity
    l_lo = max(lmin, l_start - 2)

    # Phase 1: Coarse search — find a bracket [l_lo, l_hi] containing lc
    # First, ensure l_lo actually shrinks (shrink_prob >= 0.75)
    sp = measure_and_cache(l_lo, coarse_trials)
    if sp < thresh_prob
        # l_lo is already large enough — lc <= l_lo, search downward
        while l_lo > lmin
            l_lo = max(lmin, l_lo - 2)
            sp = measure_and_cache(l_lo, coarse_trials)
            @printf("    coarse ↓: l = %d, p_shrink = %.3f (%d trials)\n", l_lo, sp, coarse_trials)
            if sp >= thresh_prob
                break
            end
        end
        if sp < thresh_prob
            # Even l=lmin doesn't shrink — lc = lmin
            sp_full = measure_and_cache(lmin, n_trials)
            @printf("    verify: l = %d, p_shrink = %.3f (%d trials)\n", lmin, sp_full, n_trials)
            return lmin
        end
    end

    # Now l_lo has shrink_prob >= 0.75. Search upward with growing steps.
    step = 1
    l_hi = l_lo
    while true
        l_hi = l_lo + step
        sp = measure_and_cache(l_hi, coarse_trials)
        @printf("    coarse ↑: l = %d, p_shrink = %.3f (%d trials)\n", l_hi, sp, coarse_trials)
        if sp < thresh_prob
            break  # bracket found: [l_hi - step, l_hi]
        end
        l_lo = l_hi
        step = min(step * 2, 16)  # double step size, cap at 16
    end

    # Phase 2: Binary search within [l_lo, l_hi]
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

    # Phase 3: Verify with full trial count
    # We need both: shrink_prob(l) < thresh AND shrink_prob(l-1) >= thresh
    l = l_hi
    while true
        sp_at_l = measure_and_cache(l, n_trials)
        @printf("    verify: l = %d, p_shrink = %.3f (%d trials)\n", l, sp_at_l, n_trials)
        if sp_at_l < thresh_prob
            # Confirm l-1 is above threshold
            if l <= lmin
                return l
            end
            sp_below = measure_and_cache(l - 1, n_trials)
            @printf("    verify: l = %d, p_shrink = %.3f (%d trials)\n", l - 1, sp_below, n_trials)
            if sp_below >= thresh_prob
                return l  # confirmed: l-1 shrinks, l survives
            else
                # l-1 also survives — transition is lower, search down
                l -= 1
            end
        else
            # l doesn't survive — transition is higher, search up
            l += 1
        end
    end
end
