"""
    run_simulation(mode; save=true, kwargs...)

Dispatch to the appropriate simulation mode and optionally save results to JLD2.
"""
function run_simulation(mode::String; save::Bool=true, adj::String="", kwargs...)
    kwargs_dict = Dict(kwargs)

    results = if mode == "history"
        run_history_mode(; kwargs_dict...)
    elseif mode == "mixing"
        run_mixing_mode(; kwargs_dict...)
    elseif mode == "energy"
        run_energy_mode(; kwargs_dict...)
    elseif mode == "ffs"
        run_ffs_mode(; kwargs_dict...)
    elseif mode == "teff"
        run_teff_mode(; kwargs_dict...)
    elseif mode == "erosion_test"
        run_erosion_test_mode(; kwargs_dict...)
    else
        error("Unknown mode: $mode. Use 'history', 'mixing', 'ffs', 'energy', 'teff', or 'erosion_test'.")
    end

    if save
        save_results(mode, results; adj)
    end

    return results
end

"""
    save_results(mode, results)

Save simulation results to a JLD2 file with auto-numbered filenames.
Key names are chosen for compatibility with sliding_plotter.py.
"""
function save_results(mode::String, results::Dict{String, Any}; adj::String="")
    if !isdir("data")
        mkdir("data")
    end

    # Build filename from mode and key parameters
    filename = build_filename(mode, results)
    if adj != ""
        filename = replace(filename, ".jld2" => "$(adj).jld2")
    end

    # Save all result entries as top-level keys
    jldsave(filename; (Symbol(k) => v for (k, v) in results)...)

    println("Results saved to $filename")
    clipboard(filename)
end

"""
    build_filename(mode, results)

Build a descriptive filename from mode and parameters.
"""
function build_filename(mode::String, results::Dict{String, Any})
    h = get(results, "h", 0.0)
    h_suffix = h != 0.0 ? @sprintf("_h%.2f", h) : ""

    if mode == "history"
        L = results["L"]
        v = results["v"]
        beta = results["beta"]
        return @sprintf("data/ising_sliding_history_L%d_v%.2f_beta%.3f%s.jld2", L, v, beta, h_suffix)
    elseif mode == "mixing"
        L = results["L"]
        if get(results, "single_layer", false)
            return @sprintf("data/ising_single_layer_mixing_L%d%s.jld2", L, h_suffix)
        elseif haskey(results, "vs")
            p = results["p"]
            return @sprintf("data/ising_sliding_mixing_L%d_p%.2f%s.jld2", L, p, h_suffix)
        else
            v = results["v"]
            return @sprintf("data/ising_sliding_mixing_L%d_v%.2f%s.jld2", L, v, h_suffix)
        end
    elseif mode == "energy"
        L = results["L"]
        if haskey(results, "vs")
            p = results["p"]
            return @sprintf("data/ising_sliding_energy_L%d_p%.2f%s.jld2", L, p, h_suffix)
        else
            v = results["v"]
            return @sprintf("data/ising_sliding_energy_L%d_v%.2f%s.jld2", L, v, h_suffix)
        end
    elseif mode == "teff"
        L = results["L"]
        if haskey(results, "vs")
            T_val = results["T"]
            return @sprintf("data/ising_sliding_teff_L%d_T%.2f%s.jld2", L, T_val, h_suffix)
        else
            v = results["v"]
            return @sprintf("data/ising_sliding_teff_L%d_v%.2f%s.jld2", L, v, h_suffix)
        end
    elseif mode == "ffs"
        L = results["L"]
        if get(results, "single_layer", false)
            return @sprintf("data/ising_single_layer_ffs_L%d%s.jld2", L, h_suffix)
        elseif haskey(results, "vs")
            p = results["p"]
            return @sprintf("data/ising_sliding_ffs_L%d_p%.2f%s.jld2", L, p, h_suffix)
        else
            v = results["v"]
            return @sprintf("data/ising_sliding_ffs_L%d_v%.2f%s.jld2", L, v, h_suffix)
        end
    elseif mode == "erosion_test"
        if get(results, "vary_v", false)
            p = results["p"]
            return @sprintf("data/ising_sliding_erosion_p%.2f%s.jld2", p, h_suffix)
        else
            v = results["v"]
            return @sprintf("data/ising_sliding_erosion_v%.2f%s.jld2", v, h_suffix)
        end
    else
        return "data/ising_sliding_$(mode).jld2"
    end
end
