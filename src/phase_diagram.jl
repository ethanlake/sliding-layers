"""
    detect_v_onset(vs, ys, rel_threshold) -> Float64

Return the smallest v in `vs` such that `ys[i] > (1 + rel_threshold) * minimum(ys)`.
Non-finite entries in `ys` are ignored. Returns NaN if no v crosses the threshold.
"""
function detect_v_onset(vs::Vector{Float64}, ys::Vector{Float64},
                        rel_threshold::Float64)
    valid = isfinite.(ys)
    any(valid) || return NaN
    y_min = minimum(ys[valid])
    cutoff = (1 + rel_threshold) * y_min
    for i in eachindex(vs)
        if valid[i] && ys[i] > cutoff
            return vs[i]
        end
    end
    return NaN
end

"""
    run_phase_diagram_mode(; L, h, p_min, p_max, n_p_steps, v_min, v_max, n_v_steps,
                            observable, onset_threshold, ...)

For each p in linspace(p_min, p_max, n_p_steps), determine the onset velocity v*(p)
by sweeping v in increasing order and stopping at the first v where the observable
exceeds onset_threshold times its baseline. v*(p) is then refined by linearly
interpolating between the bracketing pair (last v below cutoff, first v above) so
it falls where the connecting line crosses the cutoff, not on a grid point.
Remaining entries of y_matrix stay NaN once the sweep stops.

Mixing observable: baseline = t_mix(v=0), cutoff = onset_threshold · t_mix(v=0).
Erosion observable: baseline = lc(v_min) (since v=0 is not allowed), cutoff =
onset_threshold · lc(v_min).

If show_plots is true, display a per-p diagnostic plot of the inner curve with
v* and the cutoff drawn in.
"""
function run_phase_diagram_mode(; L::Int, h::Float64,
                                  p_min::Float64, p_max::Float64, n_p_steps::Int,
                                  v_min::Float64, v_max::Float64, n_v_steps::Int,
                                  observable::String,
                                  onset_threshold::Float64,
                                  n_trials::Int, M_threshold::Float64, max_time::Int,
                                  erosion_num_trials::Int, thresh_prob::Float64,
                                  min_erosion_length::Int, doublon_mode::Bool,
                                  show_plots::Bool, single_layer::Bool,
                                  sweep_values_override::Union{Nothing,Vector{Float64}}=nothing,
                                  kwargs...)
    observable in ("mixing", "erosion") ||
        error("observable must be \"mixing\" or \"erosion\", got \"$observable\"")
    all(v -> v > 0, range(v_min, v_max, length=n_v_steps)) ||
        error("All v values must be > 0")

    p_values = sweep_values_override === nothing ?
        collect(range(p_min, p_max, length=n_p_steps)) : sweep_values_override
    v_values = collect(range(v_min, v_max, length=n_v_steps))
    y_matrix = fill(NaN, n_p_steps, n_v_steps)
    v_onset = fill(NaN, n_p_steps)
    y_baseline = fill(NaN, n_p_steps)  # y(v=0) for mixing, y(v_min) for erosion

    baseline_label = observable == "mixing" ? "t_mix(v=0)" : "lc(v_min)"
    println("=== Phase Diagram Mode ($(observable)) ===")
    if observable == "mixing"
        println("L = $L, h = $h, single_layer = $single_layer")
    else
        println("h = $h (min_erosion_length = $min_erosion_length, doublon_mode = $doublon_mode)")
    end
    println("v* = smallest v with y(v) > $(onset_threshold) · $(baseline_label)")
    println("p values: $p_values")
    println("v values: $v_values")
    println("Threads: $(Threads.nthreads())")

    measure_y = (beta, v) -> begin
        if observable == "mixing"
            return measure_mixing_time(L, beta, h, v, n_trials, M_threshold, max_time;
                                       single_layer=single_layer)
        else
            return Float64(find_critical_length(beta, h, v, erosion_num_trials,
                                                min_erosion_length, thresh_prob,
                                                min_erosion_length;
                                                doublon_mode=doublon_mode))
        end
    end

    for (i, p) in enumerate(p_values)
        # Convention: p = exp(-β J), so 0 < p ≤ 1.
        beta = -log(p)
        @printf("\n[%d/%d] p = %.3f (β = %.4f)\n", i, n_p_steps, p, beta)

        # prev_v / prev_y: the most recent (v, y) pair strictly below cutoff,
        # used as the lower bracket for linear interpolation.
        prev_v = NaN
        prev_y = NaN

        if observable == "mixing"
            y_baseline[i] = measure_y(beta, 0.0)
            cutoff = onset_threshold * y_baseline[i]
            @printf("    t_mix(v=0) = %.4g  →  cutoff = %.4g\n", y_baseline[i], cutoff)
            prev_v = 0.0
            prev_y = y_baseline[i]
        end
        # For erosion, baseline is set from the first sweep point below.

        for (j, v) in enumerate(v_values)
            y = measure_y(beta, v)
            y_matrix[i, j] = y
            if observable == "erosion" && j == 1
                y_baseline[i] = y
                cutoff = onset_threshold * y_baseline[i]
                @printf("    v = %.3f → lc = %.4g  (baseline; cutoff = %.4g)\n",
                        v, y, cutoff)
                prev_v = v
                prev_y = y
                continue
            end
            @printf("    v = %.3f → %s = %.4g\n",
                    v, observable == "mixing" ? "t_mix" : "lc", y)
            cutoff = onset_threshold * y_baseline[i]
            if isfinite(y) && isfinite(y_baseline[i]) && y > cutoff
                # Linear interpolation between (prev_v, prev_y) and (v, y)
                # to find where the curve crosses cutoff.
                if isfinite(prev_v) && isfinite(prev_y) && y != prev_y
                    v_onset[i] = prev_v + (cutoff - prev_y) / (y - prev_y) * (v - prev_v)
                else
                    v_onset[i] = v
                end
                break
            end
            prev_v = v
            prev_y = y
        end

        @printf("  → v* = %.4g\n", v_onset[i])

        if show_plots
            ylab = observable == "mixing" ? "mean mixing time" : "critical erosion length"
            mask = isfinite.(y_matrix[i, :])
            v_plot = collect(v_values[mask])
            y_plot = collect(y_matrix[i, :][mask])
            # For mixing, prepend the (v=0, t_mix(0)) baseline so the plotted
            # line covers the same range used by the interpolation.
            if observable == "mixing" && isfinite(y_baseline[i])
                pushfirst!(v_plot, 0.0)
                pushfirst!(y_plot, y_baseline[i])
            end
            plt = Plots.plot(v_plot, y_plot, marker=:circle,
                             xlabel="v", ylabel=ylab,
                             title=@sprintf("p = %.3f, v* = %.4g", p, v_onset[i]),
                             label="data", legend=:topleft)
            if isfinite(y_baseline[i])
                Plots.hline!(plt, [onset_threshold * y_baseline[i]], linestyle=:dot,
                             label=@sprintf("%.2f·%s", onset_threshold, baseline_label))
            end
            if isfinite(v_onset[i])
                Plots.vline!(plt, [v_onset[i]], linestyle=:dash, label="v*")
            end
            display(plt)
        end
    end

    println("\nDone!")

    return Dict{String, Any}(
        "p_values" => p_values,
        "v_values" => v_values,
        "y_matrix" => y_matrix,
        "v_onset_values" => v_onset,
        "y_baseline" => y_baseline,
        "observable" => observable,
        "onset_threshold" => onset_threshold,
        "L" => L, "h" => h,
        "single_layer" => single_layer,
    )
end
