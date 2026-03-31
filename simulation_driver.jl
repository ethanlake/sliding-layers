#!/usr/bin/env julia
#=
Simulation driver for SlidingLayers.

Usage from command line:
    julia --project=. -t auto simulation_driver.jl --mode=mixing --L=1000
    julia --project=. -t 4 simulation_driver.jl --mode=energy --energy_p=3.0
    julia --project=. simulation_driver.jl --mode=history --beta=1.5 --T_steps=2000
    julia --project=. -t auto simulation_driver.jl --mode=erosion_test --v=2.0

Usage from REPL:
    using Pkg
    Pkg.activate(".")
    using Revise
    include("simulation_driver.jl")
=#

try
    using Revise
catch
    @warn "Revise.jl not loaded: restart the REPL to see changes to SlidingLayers module."
end

using SlidingLayers

### parse command-line arguments ###

function parse_arg(key::String, default_value)
    for arg in ARGS
        if startswith(arg, "--$key=")
            value_str = split(arg, "=", limit=2)[2]
            if lowercase(value_str) == "nothing" || value_str == ""
                return nothing
            elseif default_value isa Bool
                return lowercase(value_str) in ["true", "1", "yes"]
            elseif default_value isa Int
                return parse(Int, value_str)
            elseif default_value isa Float64
                return parse(Float64, value_str)
            elseif default_value isa String
                return value_str
            elseif default_value isa Vector{Float64}
                return parse.(Float64, split(value_str, ","))
            else
                return value_str
            end
        end
    end
    return default_value
end

### default parameters ###

mode = "mixing"
mode = String(parse_arg("mode", mode))

# Set mode-dependent defaults
if mode == "history"
    L = 2000
    v = 2.0
    beta = 2.0
    h = 0.0
    T_steps = 1000
    init = "domain"
    domain_start = L ÷ 4
    domain_end = 2L ÷ 4
elseif mode == "mixing"
    L = 2000
    v = 2.0
    h = 0.0
    p = 5.0
    p_min = 2.0
    p_max = 3.5
    v_min = 0.5
    v_max = 5.0
    n_steps = 4
    vary_v = false
    n_trials = 100
    M_threshold = 0.65
    max_time = 1000000
    single_layer = false
elseif mode == "ffs"
    L = 2000
    v = 2.0
    h = 0.0
    p = 5.0
    p_min = 2.0
    p_max = 5.0
    v_min = 0.5
    v_max = 5.0
    n_steps = 6
    vary_v = false
    n_configs = 200
    n_repeats = 5
    target_crossing_prob = 0.25
    M_threshold = 0.75
    max_time_per_trial = 1000000
    single_layer = false
    lambda_0_override = NaN
elseif mode == "energy"
    L = 2000
    v = 2.0
    h = 0.0
    p = 2.6
    p_min = 2.0
    p_max = 5.0
    v_min = 0.0
    v_max = 5.0
    n_steps = 6
    vary_v = false
    T_equil = 125000
    T_sample = 200000
elseif mode == "teff"
    L = 2000
    v = 2.0
    h = 0.0
    T = 0.5
    T_min = 0.3
    T_max = 1.0
    v_min = 0.0
    v_max = 5.0
    n_steps = 6
    vary_v = false
    T_equil = 125000
    T_sample = 200000
    demon_interval = 200
    demon = false
    h_pert = 0.01
    n_perturbations = 200
    T_response = 0
    n_samples = 1
    single_layer = false
    show_plots = false
elseif mode == "erosion_test"
    v = 2.0
    h = 0.0
    p = 5.0
    p_min = 2.5
    p_max = 15.0
    v_min = 0.5
    v_max = 5.0
    n_steps = 6
    vary_v = false
    thresh_prob = 0.75
    erosion_num_trials = 5000
    lmin = 4
    show_histories = false
    doublon_mode = false
    erode_vs_l = false
    show_plots = false
else
    error("Unknown mode: $mode. Use 'history', 'mixing', 'ffs', 'energy', 'teff', or 'erosion_test'.")
end

# Parse command-line overrides
save = parse_arg("save", true)
adj = String(parse_arg("adj", ""))

if mode == "history"
    L = parse_arg("L", L)
    v = parse_arg("v", v)
    beta = parse_arg("beta", beta)
    h = parse_arg("h", h)
    T_steps = parse_arg("T_steps", T_steps)
    init = String(parse_arg("init", init))
    domain_start = parse_arg("domain_start", L ÷ 4)
    domain_end = parse_arg("domain_end", 2L ÷ 4)
elseif mode == "mixing"
    L = parse_arg("L", L)
    v = parse_arg("v", v)
    h = parse_arg("h", h)
    p = parse_arg("p", p)
    p_min = parse_arg("p_min", p_min)
    p_max = parse_arg("p_max", p_max)
    v_min = parse_arg("v_min", v_min)
    v_max = parse_arg("v_max", v_max)
    n_steps = parse_arg("n_steps", n_steps)
    vary_v = parse_arg("vary_v", vary_v)
    n_trials = parse_arg("n_trials", n_trials)
    M_threshold = parse_arg("M_threshold", M_threshold)
    max_time = parse_arg("max_time", max_time)
    single_layer = parse_arg("single_layer", single_layer)
elseif mode == "ffs"
    L = parse_arg("L", L)
    v = parse_arg("v", v)
    h = parse_arg("h", h)
    p = parse_arg("p", p)
    p_min = parse_arg("p_min", p_min)
    p_max = parse_arg("p_max", p_max)
    v_min = parse_arg("v_min", v_min)
    v_max = parse_arg("v_max", v_max)
    n_steps = parse_arg("n_steps", n_steps)
    vary_v = parse_arg("vary_v", vary_v)
    n_configs = parse_arg("n_configs", n_configs)
    n_repeats = parse_arg("n_repeats", n_repeats)
    target_crossing_prob = parse_arg("target_crossing_prob", target_crossing_prob)
    M_threshold = parse_arg("M_threshold", M_threshold)
    max_time_per_trial = parse_arg("max_time_per_trial", max_time_per_trial)
    single_layer = parse_arg("single_layer", single_layer)
    lambda_0_override = parse_arg("lambda_0", lambda_0_override)
elseif mode == "energy"
    L = parse_arg("L", L)
    v = parse_arg("v", v)
    h = parse_arg("h", h)
    p = parse_arg("p", p)
    p_min = parse_arg("p_min", p_min)
    p_max = parse_arg("p_max", p_max)
    v_min = parse_arg("v_min", v_min)
    v_max = parse_arg("v_max", v_max)
    n_steps = parse_arg("n_steps", n_steps)
    vary_v = parse_arg("vary_v", vary_v)
    T_equil = parse_arg("T_equil", T_equil)
    T_sample = parse_arg("T_sample", T_sample)
elseif mode == "teff"
    L = parse_arg("L", L)
    v = parse_arg("v", v)
    h = parse_arg("h", h)
    T = parse_arg("T", T)
    T_min = parse_arg("T_min", T_min)
    T_max = parse_arg("T_max", T_max)
    v_min = parse_arg("v_min", v_min)
    v_max = parse_arg("v_max", v_max)
    n_steps = parse_arg("n_steps", n_steps)
    vary_v = parse_arg("vary_v", vary_v)
    T_equil = parse_arg("T_equil", T_equil)
    T_sample = parse_arg("T_sample", T_sample)
    demon_interval = parse_arg("demon_interval", demon_interval)
    demon = parse_arg("demon", demon)
    h_pert = parse_arg("h_pert", h_pert)
    n_perturbations = parse_arg("n_perturbations", n_perturbations)
    T_response = parse_arg("T_response", T_response)
    n_samples = parse_arg("n_samples", n_samples)
    single_layer = parse_arg("single_layer", single_layer)
    show_plots = parse_arg("show_plots", show_plots)
elseif mode == "erosion_test"
    v = parse_arg("v", v)
    h = parse_arg("h", h)
    p = parse_arg("p", p)
    p_min = parse_arg("p_min", p_min)
    p_max = parse_arg("p_max", p_max)
    v_min = parse_arg("v_min", v_min)
    v_max = parse_arg("v_max", v_max)
    n_steps = parse_arg("n_steps", n_steps)
    vary_v = parse_arg("vary_v", vary_v)
    thresh_prob = parse_arg("thresh_prob", thresh_prob)
    erosion_num_trials = parse_arg("erosion_num_trials", erosion_num_trials)
    lmin = parse_arg("lmin", lmin)
    show_histories = parse_arg("show_histories", show_histories)
    doublon_mode = parse_arg("doublon_mode", doublon_mode)
    erode_vs_l = parse_arg("erode_vs_l", erode_vs_l)
    show_plots = parse_arg("show_plots", show_plots)
end

### check thread count ###

n_threads = Threads.nthreads()
if n_threads == 1
    @warn "Running with only 1 thread. For parallel trials, start Julia with: julia -t auto\n  Or set JULIA_NUM_THREADS=auto in your shell profile."
end

### build kwargs and run ###

kwargs = Dict{Symbol, Any}()

if mode == "history"
    kwargs[:L] = L
    kwargs[:v] = v
    kwargs[:beta] = beta
    kwargs[:h] = h
    kwargs[:T_steps] = T_steps
    kwargs[:init] = init
    kwargs[:domain_start] = domain_start
    kwargs[:domain_end] = domain_end
elseif mode == "mixing"
    kwargs[:L] = L
    kwargs[:v] = v
    kwargs[:h] = h
    kwargs[:p] = p
    kwargs[:p_min] = p_min
    kwargs[:p_max] = p_max
    kwargs[:v_min] = v_min
    kwargs[:v_max] = v_max
    kwargs[:n_steps] = n_steps
    kwargs[:vary_v] = vary_v
    kwargs[:n_trials] = n_trials
    kwargs[:M_threshold] = M_threshold
    kwargs[:max_time] = max_time
    kwargs[:single_layer] = single_layer
elseif mode == "ffs"
    kwargs[:L] = L
    kwargs[:v] = v
    kwargs[:h] = h
    kwargs[:p] = p
    kwargs[:p_min] = p_min
    kwargs[:p_max] = p_max
    kwargs[:v_min] = v_min
    kwargs[:v_max] = v_max
    kwargs[:n_steps] = n_steps
    kwargs[:vary_v] = vary_v
    kwargs[:n_configs] = n_configs
    kwargs[:n_repeats] = n_repeats
    kwargs[:target_crossing_prob] = target_crossing_prob
    kwargs[:M_threshold] = M_threshold
    kwargs[:max_time_per_trial] = max_time_per_trial
    kwargs[:single_layer] = single_layer
    kwargs[:lambda_0_override] = lambda_0_override
elseif mode == "energy"
    kwargs[:L] = L
    kwargs[:v] = v
    kwargs[:h] = h
    kwargs[:p] = p
    kwargs[:p_min] = p_min
    kwargs[:p_max] = p_max
    kwargs[:v_min] = v_min
    kwargs[:v_max] = v_max
    kwargs[:n_steps] = n_steps
    kwargs[:vary_v] = vary_v
    kwargs[:T_equil] = T_equil
    kwargs[:T_sample] = T_sample
elseif mode == "teff"
    kwargs[:L] = L
    kwargs[:v] = v
    kwargs[:h] = h
    kwargs[:T] = T
    kwargs[:T_min] = T_min
    kwargs[:T_max] = T_max
    kwargs[:v_min] = v_min
    kwargs[:v_max] = v_max
    kwargs[:n_steps] = n_steps
    kwargs[:vary_v] = vary_v
    kwargs[:T_equil] = T_equil
    kwargs[:T_sample] = T_sample
    kwargs[:demon_interval] = demon_interval
    kwargs[:demon] = demon
    kwargs[:h_pert] = h_pert
    kwargs[:n_perturbations] = n_perturbations
    kwargs[:T_response] = T_response
    kwargs[:n_samples] = n_samples
    kwargs[:single_layer] = single_layer
    kwargs[:show_plots] = show_plots
elseif mode == "erosion_test"
    kwargs[:v] = v
    kwargs[:h] = h
    kwargs[:p] = p
    kwargs[:p_min] = p_min
    kwargs[:p_max] = p_max
    kwargs[:v_min] = v_min
    kwargs[:v_max] = v_max
    kwargs[:n_steps] = n_steps
    kwargs[:vary_v] = vary_v
    kwargs[:thresh_prob] = thresh_prob
    kwargs[:erosion_num_trials] = erosion_num_trials
    kwargs[:lmin] = lmin
    kwargs[:show_histories] = show_histories
    kwargs[:doublon_mode] = doublon_mode
    kwargs[:erode_vs_l] = erode_vs_l
    kwargs[:show_plots] = show_plots
end

println("Running simulation: mode=$mode, threads=$n_threads")
run_simulation(mode; save=save, adj=adj, kwargs...)
println("Simulation complete!")
try run(`afplay /System/Library/Sounds/Glass.aiff`) catch end
