# Simulation Driver

## Usage

```bash
julia --project=. -t auto simulation_driver.jl --mode=mixing --L=1000 --n_steps=10
```

The driver parses command-line arguments in `--key=value` format, sets mode-dependent defaults, and calls `run_simulation(mode; kwargs...)`.

## Threading

Use `-t auto` or `-t N` to enable multithreading. The driver warns if running with 1 thread. All modes parallelize internally:
- **mixing/ffs**: trials parallelized per sweep value
- **energy**: sweep values parallelized across threads
- **erosion_test**: trials parallelized per domain size

## Argument Parsing

`parse_arg(key, default)` scans `ARGS` for `--key=value` and type-coerces based on the default:
- `Bool`: accepts `true/false/1/0/yes`
- `Int`, `Float64`: parsed numerically
- `Vector{Float64}`: comma-separated (e.g., `--p_values=2.0,3.0,4.0`)
- `String`: returned as-is

## Sweep Parameters

Most modes support sweeping over temperature ($p$) or velocity ($v$):
- `p_min`, `p_max`, `n_steps`: linearly spaced $p = e^\beta$ values
- `v_min`, `v_max`: velocity range (when `vary_v=true`)
- `p`: fixed $p$ value (when sweeping $v$)
- `v`: fixed velocity (when sweeping $p$)

## Output

Results are saved to `data/` as JLD2 files with descriptive filenames. The filename is automatically copied to the system clipboard for convenience.

## REPL Usage

```julia
using Pkg; Pkg.activate(".")
using Revise
include("simulation_driver.jl")
```

With Revise loaded, changes to the `SlidingLayers` module are automatically picked up without restarting.
