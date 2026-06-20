module SlidingLayers

__precompile__(true)

using Printf, Random, JLD2, Statistics, InteractiveUtils, ProgressMeter, Plots

include("types.jl")
include("core.jl")
include("history.jl")
include("mixing.jl")
include("ffs.jl")
include("energy.jl")
include("teff.jl")
include("erosion_test.jl")
include("phase_diagram.jl")
include("simulation.jl")

export SimulationState, run_simulation

end  # module SlidingLayers
