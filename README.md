# sliding-layers
creating long-lived memories with friction

## Running simulations

Requires Julia. First install dependencies:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Run a simulation from the command line with `simulation_driver.jl`. Use `-t auto` to enable multithreading:

```bash
julia --project=. -t auto simulation_driver.jl --mode=mixing --L=1000 --n_trials=50
```

### Modes

- **history** — spacetime evolution of a domain wall
- **mixing** — mixing time vs temperature (parallelized over trials)
- **ffs** — mixing time via forward flux sampling for rare-event regimes (parallelized over trials)
- **energy** — steady-state energy and heat flow vs temperature or velocity (parallelized over sweep values)
- **teff** — effective temperature via demon algorithm (parallelized over sweep values)
- **erosion_test** — critical domain size vs temperature (parallelized over trials)

### Examples

```bash
# History mode
julia --project=. simulation_driver.jl --mode=history --L=2000 --beta=2.0 --T_steps=1000

# Mixing mode with 4 threads
julia --project=. -t 4 simulation_driver.jl --mode=mixing --L=2000 --n_trials=100

# Energy mode (sweep v at fixed p, default)
julia --project=. -t auto simulation_driver.jl --mode=energy --p=2.6 --v_min=0 --v_max=5

# Energy mode (sweep p at fixed v)
julia --project=. -t auto simulation_driver.jl --mode=energy --vary_v=false --v=2.0 --p_min=2 --p_max=5

# FFS mode for rare-event mixing times (sweep p)
julia --project=. -t auto simulation_driver.jl --mode=ffs --L=2000 --p_min=2.0 --p_max=5.0 --n_steps=6

# FFS mode (sweep v at fixed p)
julia --project=. -t auto simulation_driver.jl --mode=ffs --L=2000 --vary_v=true --p=4.0 --v_min=0.5 --v_max=5.0

# T_eff mode (sweep T)
julia --project=. -t auto simulation_driver.jl --mode=teff --v=2.0 --T_min=0.3 --T_max=1.0

# T_eff mode (sweep v at fixed T, with plots)
julia --project=. -t auto simulation_driver.jl --mode=teff --vary_v=true --T=0.5 --v_min=0 --v_max=5 --show_plots=true

# Erosion test (sweep p)
julia --project=. -t auto simulation_driver.jl --mode=erosion_test --v=1.0 --erosion_num_trials=5000

# Erosion test (sweep v at fixed p)
julia --project=. -t auto simulation_driver.jl --mode=erosion_test --vary_v=true --p=5.0 --v_min=0.5 --v_max=5.0
```

All parameters have defaults and can be overridden with `--param=value`. Results are saved to `data/` as JLD2 files.

### Plotting

Plot results with the Python plotter:

```bash
python sliding_plotter.py data/ising_sliding_mixing_L2000_v0.00.jld2
```

### Full parameter reference

All arguments use `--key=value` syntax. List-valued parameters use commas (e.g. `--p_values=2.0,2.5,3.0`).

**Global**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | String | `"mixing"` | Simulation mode: `history`, `mixing`, `ffs`, `energy`, `teff`, `erosion_test` |
| `save` | Bool | `true` | Save results to JLD2 file |

**history**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `L` | Int | `2000` | Chain length |
| `v` | Float | `2.0` | Sliding velocity |
| `beta` | Float | `2.0` | Inverse temperature |
| `h` | Float | `0.0` | Magnetic field |
| `T_steps` | Int | `1000` | Number of time steps |
| `init` | String | `"domain"` | Initial condition: `"domain"` or `"random"` |
| `domain_start` | Int | `L ÷ 4` | Minority domain start site (domain init only) |
| `domain_end` | Int | `L ÷ 2` | Minority domain end site (domain init only) |

**mixing**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `L` | Int | `2000` | Chain length |
| `vary_v` | Bool | `false` | If true, sweep v instead of p |
| `v` | Float | `2.0` | Fixed velocity (when `vary_v=false`) |
| `p` | Float | `5.0` | Fixed p = exp(β) (when `vary_v=true`) |
| `h` | Float | `0.0` | Magnetic field |
| `p_min` | Float | `2.0` | Min p value (when `vary_v=false`) |
| `p_max` | Float | `3.5` | Max p value (when `vary_v=false`) |
| `v_min` | Float | `0.5` | Min v value (when `vary_v=true`) |
| `v_max` | Float | `5.0` | Max v value (when `vary_v=true`) |
| `n_steps` | Int | `4` | Number of sweep values (linearly spaced) |
| `n_trials` | Int | `100` | Trials per sweep value |
| `M_threshold` | Float | `0.65` | Magnetization threshold for mixing |
| `max_time` | Int | `100000` | Max steps before timeout |
| `single_layer` | Bool | `false` | If true, simulate a single chain (no inter-chain coupling, no sliding) |

**energy**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `L` | Int | `2000` | Chain length |
| `vary_v` | Bool | `false` | If true, sweep v instead of p |
| `v` | Float | `2.0` | Fixed velocity (when `vary_v=false`) |
| `p` | Float | `2.6` | Fixed p = exp(β) (when `vary_v=true`) |
| `h` | Float | `0.0` | Magnetic field |
| `p_min` | Float | `2.0` | Min p value (when `vary_v=false`) |
| `p_max` | Float | `5.0` | Max p value (when `vary_v=false`) |
| `v_min` | Float | `0.0` | Min v value (when `vary_v=true`) |
| `v_max` | Float | `5.0` | Max v value (when `vary_v=true`) |
| `n_steps` | Int | `6` | Number of sweep values (linearly spaced) |
| `T_equil` | Int | `125000` | Equilibration time units |
| `T_sample` | Int | `200000` | Sampling time units |

**teff** (effective temperature)

Two methods for measuring the effective temperature of the nonequilibrium steady state:

**Default (FDR method):** Measures the fluctuation-dissipation ratio T_eff = (dC/dτ) / R(τ), where C(τ) is the magnetization autocorrelation and R(τ) is the impulse response to a small field perturbation. Also computes the integrated cross-check χ(τ) vs C(0)-C(τ). The plotter shows the parametric FDR plot and pointwise T_eff(τ) automatically.

**Demon method** (`--demon=true`): Weakly couples a "demon" with energy E_d to the system. Every `demon_interval` MC steps, a hypothetical spin flip is probed; if the demon can absorb the cost, E_d is updated. The histogram P(E_d) is fit to exp(-E_d/T_eff).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `L` | Int | `2000` | Chain length |
| `vary_v` | Bool | `false` | If true, sweep v instead of T |
| `v` | Float | `2.0` | Fixed velocity (when `vary_v=false`) |
| `T` | Float | `0.5` | Fixed temperature (when `vary_v=true`); p = exp(1/T) |
| `h` | Float | `0.0` | Magnetic field |
| `T_min` | Float | `0.3` | Min temperature (when `vary_v=false`) |
| `T_max` | Float | `1.0` | Max temperature (when `vary_v=false`) |
| `v_min` | Float | `0.0` | Min v value (when `vary_v=true`) |
| `v_max` | Float | `5.0` | Max v value (when `vary_v=true`) |
| `n_steps` | Int | `6` | Number of sweep values (linearly spaced) |
| `T_equil` | Int | `125000` | Equilibration time units |
| `T_sample` | Int | `200000` | Sampling time units |
| `demon` | Bool | `false` | If true, use demon method instead of FDR |
| `demon_interval` | Int | `200` | Demon probe interval in MC steps (demon method only) |
| `h_pert` | Float | `0.01` | Perturbation field strength (FDR method only) |
| `n_perturbations` | Int | `200` | Number of perturbation experiments (FDR method only) |
| `T_response` | Int | `0` | Max time lag for correlation/response; 0 = auto (FDR method only) |
| `show_plots` | Bool | `false` | If true, display diagnostic plots after each sweep value |

Note: teff mode parameterizes by temperature T directly (not p). The coupling p = exp(1/T) is computed internally.

**ffs** (forward flux sampling with adaptive interfaces)

Uses adaptive interface placement to estimate mixing times. λ₀ is set automatically per sweep point by briefly equilibrating from all +1 and placing it 2.5σ below the metastable mean magnetization. Subsequent interfaces are positioned adaptively by probing where `target_crossing_prob` fraction of trials penetrate before returning to basin. This concentrates interfaces where the free-energy barrier is steepest.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `L` | Int | `2000` | Chain length |
| `vary_v` | Bool | `false` | If true, sweep v instead of p |
| `v` | Float | `2.0` | Fixed velocity (when `vary_v=false`) |
| `p` | Float | `5.0` | Fixed p = exp(β) (when `vary_v=true`) |
| `h` | Float | `0.0` | Magnetic field (sign forced negative to oppose initial +1 state) |
| `p_min` | Float | `2.0` | Min p = exp(β) value (when `vary_v=false`) |
| `p_max` | Float | `5.0` | Max p = exp(β) value (when `vary_v=false`) |
| `v_min` | Float | `0.5` | Min v value (when `vary_v=true`) |
| `v_max` | Float | `5.0` | Max v value (when `vary_v=true`) |
| `n_steps` | Int | `6` | Number of sweep values (linearly spaced) |
| `n_configs` | Int | `200` | Configs collected (and probe trials) per interface |
| `target_crossing_prob` | Float | `0.25` | Target crossing probability per interface (quantile for adaptive placement) |
| `M_threshold` | Float | `0.75` | Target magnetization |
| `max_time_per_trial` | Int | `100000` | Timeout per trial |
| `single_layer` | Bool | `false` | If true, simulate a single chain (no inter-chain coupling, no sliding) |

**erosion_test**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vary_v` | Bool | `false` | If true, sweep v instead of p |
| `v` | Float | `2.0` | Fixed velocity (when `vary_v=false`; must be > 0) |
| `p` | Float | `5.0` | Fixed p = exp(β) (when `vary_v=true`) |
| `h` | Float | `0.0` | Magnetic field |
| `p_min` | Float | `2.5` | Min p value (when `vary_v=false`) |
| `p_max` | Float | `15.0` | Max p value (when `vary_v=false`) |
| `v_min` | Float | `0.5` | Min v value (when `vary_v=true`) |
| `v_max` | Float | `5.0` | Max v value (when `vary_v=true`) |
| `n_steps` | Int | `6` | Number of sweep values (linearly spaced) |
| `thresh_prob` | Float | `0.75` | Shrink probability threshold for critical size |
| `erosion_num_trials` | Int | `5000` | Trials per domain size per sweep value |
| `lmin` | Int | `4` | Minimum domain size for search |
| `doublon_mode` | Bool | `false` | If true, use doublon criterion: shrinkage = doublons ≤ 0.1l after 5l steps (default: minority spins < 1.5l after 2l steps) |
| `show_histories` | Bool | `false` | If true, display spacetime heatmaps at lc, 0.75·lc, 1.25·lc after each sweep value |
| `erode_vs_l` | Bool | `false` | If true, plot and save shrink probability vs l for each sweep value |
