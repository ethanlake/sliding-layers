# sliding-layers
creating long-lived memories with friction

This repository contains three independent simulation drivers, each a self-contained Julia script invoked from the command line:

| Driver | Model | Notes |
|---|---|---|
| [`simulation_driver.jl`](simulation_driver.jl) | Sliding two-chain Ising | The original driver; backed by the `SlidingLayers` module under [`src/`](src/). |
| [`gkl.jl`](gkl.jl) | Noisy 1D GKL cellular automaton | Standalone, no module dependency. |
| [`ising.jl`](ising.jl) | 1D Glauber Ising chain (h=0) | Standalone; primarily for FFS benchmarks. |

Plotting for all three is unified through [`sliding_plotter.py`](sliding_plotter.py). The dynamics is auto-detected from a `dynamics` tag inside each JLD2 file.

**Conventions for `p`, `q`, `τ`, `r`** (across all sliding-Ising, GKL, and Glauber-Ising modes):

| | Ising chain (sliding & `ising.jl`) | GKL automaton |
|---|---|---|
| `p` | `exp(-β J)` | per-cell noise rate |
| `q` | `1/p = exp(β J)` | `1/p` |
| `τ` | `exp(4 (β J)²)` | `1/√p` |
| `r` | `(β J)² = (log(1/p))²` | `(log(1/p))²` |

In both models `0 < p ≤ 1`, **small p = rare-event regime** (low temperature for Ising, low noise for GKL). In any mode that takes a min/max sweep over noise strength / temperature, you may set the range via any one of `--p_min/--p_max`, `--q_min/--q_max`, `--tau_min/--tau_max`, or `--r_min/--r_max`. The sweep is taken **linear in whichever parameter you specify**, and the values are converted to `p` internally (`p = exp(−√r)` for the `r` parametrization). Same for the fixed single value (`--p`, `--q`, `--tau`, `--r`); priority `r > tau > q > p`.

Earlier history: old data files saved with `p = exp(+β J)` (i.e. p > 1) are under the original sign convention; new runs use `exp(-β J)`. `ising.jl`'s `q` previously meant `exp(4 β J)` — it now means `exp(β J) = 1/p`, consistent with the rest of the codebase. Saved arrays in `ising.jl` outputs are now `p_values` (primary), with `q_values = 1/p_values` saved as a convenience.

## Setup

Requires Julia (≥1.10). One-time dependency install:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Run any of the three drivers with `-t auto` to enable multithreading. Results are saved to `data/` as JLD2 files. All parameters accept `--key=value` overrides on the command line.

---

## `simulation_driver.jl` — sliding two-chain Ising

### Modes

- **history** — spacetime evolution of a domain wall in the co-moving frame
- **mixing** — direct ensemble-averaged mixing time vs temperature / velocity
- **ffs** — mixing time via forward flux sampling (with optional canonical-mode interfaces and adaptive L)
- **energy** — steady-state energy and heat flow vs temperature or velocity
- **teff** — effective temperature via FDR or demon algorithm
- **erosion_test** — critical erosion length ℓ_er vs temperature or velocity
- **phase_diagram** — onset velocity v\*(p) above which mixing time / erosion length grows with v

### Examples

```bash
# History mode (minority domain centered at L/2 by default)
julia --project=. simulation_driver.jl --mode=history --L=2000 --beta=2.0 --T_steps=1000

# Mixing mode with 4 threads
julia --project=. -t 4 simulation_driver.jl --mode=mixing --L=2000 --n_trials=100

# Energy mode (sweep v at fixed p, default)
julia --project=. -t auto simulation_driver.jl --mode=energy --p=0.385 --v_min=0 --v_max=5

# FFS mode for rare-event mixing times (sweep p; small p = strong barrier)
julia --project=. -t auto simulation_driver.jl --mode=ffs --L=2000 --p_min=0.2 --p_max=0.5 \
    --n_steps=6 --n_configs_per_run=200 --n_repeats=10

# FFS with canonical (fixed) interfaces — recommended in barrier-less regimes
julia --project=. -t auto simulation_driver.jl --mode=ffs --L=2000 --p_min=0.2 --p_max=0.5 \
    --n_steps=6 --n_configs_per_run=200 --n_repeats=10 --n_interfaces=10

# FFS with adaptive system size: L = adaptive_factor × ℓ_er per sweep point
julia --project=. -t auto simulation_driver.jl --mode=ffs --v=2.0 --p_min=0.2 --p_max=0.5 \
    --n_steps=6 --n_configs_per_run=200 --n_repeats=10 --adaptive_L=true --adaptive_factor=3.0

# T_eff mode (FDR method, default)
julia --project=. -t auto simulation_driver.jl --mode=teff --v=2.0 --T_min=0.3 --T_max=1.0

# Erosion test (sweep p)
julia --project=. -t auto simulation_driver.jl --mode=erosion_test --v=1.0 --erosion_num_trials=5000

# Phase diagram (mixing-time onset)
julia --project=. -t auto simulation_driver.jl --mode=phase_diagram --observable=mixing \
    --L=500 --p_min=0.25 --p_max=0.5 --n_p_steps=5 --v_min=0.5 --v_max=5.0 --n_v_steps=10
```

### Global parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | String | `"mixing"` | One of `history`, `mixing`, `ffs`, `energy`, `teff`, `erosion_test`, `phase_diagram` |
| `save` | Bool | `true` | Save results to JLD2 file under `data/` |
| `adj` | String | `""` | Suffix appended to the saved filename (before `.jld2`) |
| `randshift` | Bool | `true` | Top-chain shift schedule. `true` (default) → Poisson process at rate `v` (inter-shift gaps ~ Exp(`L/v`)); `false` → legacy deterministic Bresenham schedule (shift `i` at MC-update count `round(i·L/v)`). The Poisson schedule removes spurious integer-vs-half-integer-`v` artifacts that arise when `L/v` is rational with a small denominator. Applies to every mode that drives MC dynamics (`history`, `mixing`, `ffs`, `energy`, `teff`, `erosion_test`, `phase_diagram`). |

### history mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `L` | Int | `2000` | Chain length |
| `v` | Float | `2.0` | Sliding velocity |
| `beta` | Float | `2.0` | Inverse temperature |
| `h` | Float | `0.0` | Magnetic field |
| `T_steps` | Int | `1000` | Number of time steps |
| `init` | String | `"domain"` | Initial condition: `"domain"` or `"random"` |
| `domain_start` | Int | `3L÷8` | Minority-domain start (centered at L/2 by default) |
| `domain_end` | Int | `5L÷8` | Minority-domain end |

### mixing mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `L` | Int | `2000` | Chain length |
| `vary_v` | Bool | `false` | If true, sweep v instead of p |
| `v` | Float | `2.0` | Fixed velocity (when `vary_v=false`) |
| `p` | Float | `0.2` | Fixed p = exp(−β J) (when `vary_v=true`) |
| `h` | Float | `0.0` | Magnetic field |
| `p_min`, `p_max` | Float | `0.29`, `0.5` | p sweep range (0 < p ≤ 1) |
| `v_min`, `v_max` | Float | `0.5`, `5.0` | v sweep range |
| `n_steps` | Int | `4` | Number of sweep values |
| `n_trials` | Int | `100` | Trials per sweep value |
| `M_threshold` | Float | `0.65` | Magnetization escape threshold |
| `max_time` | Int | `1000000` | Per-trial timeout |
| `single_layer` | Bool | `false` | Simulate a single chain (no inter-chain coupling, no sliding) |

### ffs mode (forward flux sampling)

By default uses **adaptive interface placement**: each interface λᵢ is chosen so that ≈`target_crossing_prob` of probe trials cross it, with `λ_fail` set `n_lookback=7` interfaces back. Adaptive mode is calibrated for genuine rare events. For barrier-less regimes (where each P_i would naturally be close to 1), use `--n_interfaces=N` to switch to **canonical FFS**: N uniformly-spaced fixed interfaces between λ_0 and `M_threshold`, with `λ_fail = λ_0` always.

For nucleation-style runs, `--adaptive_L=true` sets `L = adaptive_factor × ℓ_er` per sweep point — the erosion length is found at each (β, v) by the same routine as `erosion_test` mode and the system is sized to fit a critical droplet.

Phase-0 / probe / crossing-phase timeouts propagate as `failed=true` for the single FFS run, so partial timeouts visibly reduce the `n_ok/n_repeats` count rather than silently biasing the estimate.

**Modified-clock seeding** (`--seed_droplet_size=k`, `k > 0`): whenever the dynamics returns to the all-`+` basin floor during Phase 0 / probes / crossings, instantly inject a `k`-spin minority droplet aligned on both chains. The skipped all-`+` sojourn time is not accounted for in `τ_mem`, so the resulting estimator is a **modified-clock memory time** `t̃_mem` rather than the bare `τ_mem` — useful when the Phase-0 ensemble is bimodal (single-spin fluctuations vs. genuine droplets) and direct sampling produces an erratic source distribution. See SM for the formal modified-clock convention; the plotter switches the y-label to `$\widetilde t_{\sf mem}$` automatically when this flag is present in the JLD2.

**Periodic checkpointing**: when `save=true`, both FFS and energy modes pre-allocate the result arrays as `NaN`, write a skeleton JLD2 immediately, and re-serialize the full results dict after each completed sweep point (under a `ReentrantLock` to coordinate threads). A crash mid-sweep loses at most one in-flight sweep point; the plotter masks `NaN` entries automatically.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `L` | Int | `2000` | Chain length (ignored when `adaptive_L=true`) |
| `vary_v` | Bool | `false` | If true, sweep v instead of p |
| `v` | Float | `2.0` | Fixed velocity (when `vary_v=false`) |
| `p` | Float | `0.2` | Fixed p = exp(−β J) (when `vary_v=true`) |
| `h` | Float | `0.0` | Sign forced negative (h ↦ −\|h\|) to oppose initial +1 state |
| `p_min`, `p_max` | Float | `0.2`, `0.5` | p sweep range (0 < p ≤ 1; small p = rare-event regime) |
| `v_min`, `v_max` | Float | `0.5`, `5.0` | v sweep range |
| `n_steps` | Int | `6` | Number of sweep values |
| `n_configs_per_run` | Int | `100` | Configs collected per interface, per FFS run; controls per-run bias (∝ 1/`n_configs_per_run`) |
| `n_repeats` | Int | `8` | Independent FFS runs averaged at each sweep point; controls variance of the mean |
| `target_crossing_prob` | Float | `0.15` | Target P_i quantile for adaptive placement (ignored when `n_interfaces > 0`) |
| `n_interfaces` | Int | `0` | `>0` switches to canonical FFS with that many fixed uniformly-spaced interfaces and `λ_fail = λ_0` |
| `adaptive_L` | Bool | `false` | Per sweep point, find ℓ_er and set `L = adaptive_factor × ℓ_er` (requires `single_layer=false`) |
| `adaptive_factor` | Float | `3.0` | Multiplier on ℓ_er when `adaptive_L=true` |
| `M_threshold` | Float | `0.75` | Target magnetization (escape threshold) |
| `max_time_per_trial` | Int | `200000000` | Per-trial timeout (very generous; see timeout-propagation note above) |
| `lambda_0` | Float | `NaN` | If set, overrides the auto-calibrated λ_0 |
| `single_layer` | Bool | `false` | Simulate a single chain |
| `seed_droplet_size` | Int | `0` | `>0` enables modified-clock seeding: inject a `k`-spin droplet on both chains whenever the state returns to all-`+`. Produces `t̃_mem` instead of `τ_mem` (see paragraph above). |

The driver also auto-detects a `v`-sweep: passing both `--v_min` and `--v_max` (even without `--vary_v=true`) flips `vary_v` on. Explicit `--vary_v=true|false` still wins. (Same auto-detect in `energy` mode.)

### energy mode

Steady-state energy `⟨E⟩` and heat flow `⟨Q̇⟩` are estimated by **block averaging** the `T_sample` sampling window into 20 blocks: each block's mean is one quasi-independent observation, and the stderr-of-the-mean is `std(block_means) / √n_blocks`. Saved arrays `mean_energies_std` and `mean_heat_flows_std` are read by the plotter as errorbars. Periodic checkpointing (above) applies — same NaN-skeleton + per-sweep-point dump contract as FFS. The `v`-sweep auto-detect (passing `--v_min` and `--v_max`) also works here.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `L` | Int | `2000` | Chain length |
| `vary_v` | Bool | `false` | If true, sweep v instead of p |
| `v` | Float | `2.0` | Fixed velocity |
| `p` | Float | `0.385` | Fixed p = exp(−β J) |
| `h` | Float | `0.0` | Magnetic field |
| `p_min`, `p_max` | Float | `0.2`, `0.5` | p sweep range (0 < p ≤ 1) |
| `v_min`, `v_max` | Float | `0.0`, `5.0` | v sweep range |
| `n_steps` | Int | `6` | Number of sweep values |
| `T_equil` | Int | `125000` | Equilibration time units |
| `T_sample` | Int | `200000` | Sampling time units |

### teff mode (effective temperature)

Two estimators are available:

- **FDR method** (default): measures T_eff = (∂C/∂τ) / R(τ), where C(τ) is the magnetization autocorrelation and R(τ) is the impulse response to a small field perturbation. Also computes the parametric χ(τ) vs C(0)−C(τ) cross-check.
- **Demon method** (`--demon=true`): weakly couples an Einstein-demon with energy E_d. Histogram P(E_d) is fit to exp(−E_d/T_eff).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `L` | Int | `2000` | Chain length |
| `vary_v` | Bool | `false` | If true, sweep v instead of T |
| `v` | Float | `2.0` | Fixed velocity |
| `T` | Float | `0.5` | Fixed temperature |
| `h` | Float | `0.0` | Magnetic field |
| `T_min`, `T_max` | Float | `0.3`, `1.0` | T sweep range |
| `v_min`, `v_max` | Float | `0.0`, `5.0` | v sweep range |
| `n_steps` | Int | `6` | Number of sweep values |
| `T_equil` | Int | `125000` | Equilibration time units |
| `T_sample` | Int | `200000` | Sampling time units |
| `demon` | Bool | `false` | Use demon method instead of FDR |
| `demon_interval` | Int | `200` | Demon probe interval (demon method) |
| `h_pert` | Float | `0.01` | Perturbation field strength (FDR method) |
| `n_perturbations` | Int | `200` | Number of perturbation experiments (FDR method) |
| `T_response` | Int | `0` | Max correlation/response time lag; 0 = auto (FDR method) |
| `single_layer` | Bool | `false` | Single chain |

### erosion_test mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vary_v` | Bool | `false` | If true, sweep v instead of p |
| `v` | Float | `2.0` | Fixed velocity (must be > 0 for sliding) |
| `p` | Float | `0.2` | Fixed p = exp(−β J) |
| `h` | Float | `0.0` | Magnetic field |
| `p_min`, `p_max` | Float | `0.0667`, `0.4` | p sweep range (0 < p ≤ 1) |
| `v_min`, `v_max` | Float | `0.5`, `5.0` | v sweep range |
| `n_steps` | Int | `6` | Number of sweep values |
| `thresh_prob` | Float | `0.75` | Shrink-probability threshold defining ℓ_er |
| `erosion_num_trials` | Int | `5000` | Trials per domain size at the verification stage |
| `min_erosion_length` | Int | `2` | Lower bound for ℓ_er search |
| `doublon_mode` | Bool | `false` | Use doublon shrink criterion: doublons ≤ 0.1·l after 5l steps (else minority spins < 1.5·l after 2l steps) |
| `t_evolve_factor` | Float | `nothing` | Override evolution factor (`nothing` → 2.0 non-doublon, 5.0 doublon) |
| `L_sys_factor` | Float | `1.0` | Multiplier on the per-trial system size L_sys = 2·l·(v+1) |
| `first_passage_mode` | Bool | `false` | Variable-time first-passage escape measurement instead of fixed-time shrinkage |
| `min_doublons` | Int | `10` | Lower boundary for the first-passage erode condition |
| `erode_vs_l` | Bool | `false` | Also save and (optionally) plot shrink probability vs l for each sweep value |
| `show_histories` | Bool | `false` | Display spacetime heatmaps at ℓ_er, 0.75·ℓ_er, 1.25·ℓ_er after each sweep value (requires Plots.jl) |

### phase_diagram mode

For each p in the outer sweep, finds the onset velocity v\* by sweeping v and stopping at the first v where the observable exceeds `onset_threshold · y_baseline`. Baseline is `t_mix(v=0)` for the mixing observable, `lc(v_min)` for erosion (since v > 0 is required for erosion).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `L` | Int | `2000` | Chain length (mixing observable only) |
| `h` | Float | `0.0` | Magnetic field |
| `observable` | String | `"mixing"` | `"mixing"` or `"erosion"` |
| `p_min`, `p_max` | Float | `0.2`, `0.5` | Outer p sweep range (0 < p ≤ 1) |
| `n_p_steps` | Int | `6` | Outer p sweep size |
| `v_min`, `v_max` | Float | `0.5`, `5.0` | Inner v sweep range |
| `n_v_steps` | Int | `10` | Inner v sweep size |
| `onset_threshold` | Float | `10.0` | Multiplier on baseline used to detect v\* |
| `n_trials` | Int | `100` | (mixing) per (p, v) point |
| `M_threshold` | Float | `0.65` | (mixing) |
| `max_time` | Int | `1000000` | (mixing) per-trial timeout |
| `erosion_num_trials` | Int | `5000` | (erosion) |
| `thresh_prob` | Float | `0.75` | (erosion) |
| `min_erosion_length` | Int | `2` | (erosion) |
| `doublon_mode` | Bool | `false` | (erosion) |
| `single_layer` | Bool | `false` | (mixing) |

---

## `gkl.jl` — noisy 1D GKL cellular automaton

Canonical Gács-Kurdyumov-Levin rule with biased noise: at each site, with probability `p_noise`, the state is overwritten to +1 with probability (1+η)/2, else −1. Both **synchronous** and **asynchronous** update modes are supported via `--update={sync,async}`.

### Modes

- **history** — spacetime σ(x, t)
- **ffs** — memory time τ via FFS (same algorithmic structure as sliding-Ising)
- **ler** — erosion length ℓ_er (single-chain analog of `erosion_test`)
- **diffusion** — magnetization-MSD-based diffusion constant D

### Convention (FFS / ler)

With η > 0 (forced via `eta = abs(eta)`), the noise drives the chain toward +1; the **metastable** phase (where FFS starts) is anti-aligned with η, and the ler-mode minority cluster is aligned with η. FFS interfaces form an *increasing* sequence λ_0 < λ_1 < … < `M_threshold`.

### Examples

```bash
# History (sync, domain init)
julia --project=. gkl.jl --mode=history --L=400 --T_steps=400 --eta=0.10 --p_noise=0.05

# FFS sweep over p_noise
julia --project=. -t auto gkl.jl --mode=ffs --L=500 --eta=0.05 \
    --p_min=0.05 --p_max=0.15 --n_steps=6 --n_configs_per_run=200 --n_repeats=4

# FFS with canonical interfaces and adaptive L
julia --project=. -t auto gkl.jl --mode=ffs --eta=0.05 \
    --p_min=0.05 --p_max=0.15 --n_steps=6 --n_configs_per_run=200 --n_repeats=4 \
    --n_interfaces=10 --adaptive_L=true

# Erosion length sweep
julia --project=. -t auto gkl.jl --mode=ler --eta=0.10 \
    --p_min=0.05 --p_max=0.20 --n_steps=6 --erosion_num_trials=2000

# Diffusion constant sweep
julia --project=. -t auto gkl.jl --mode=diffusion --L=500 --eta=0.0 \
    --p_min=0.05 --p_max=0.20 --n_steps=5 --n_trials=200
```

### Shared parameters (any mode)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | String | `"history"` | `history`, `ffs`, `ler`, `diffusion` |
| `eta` | Float | `0.0` | Noise bias; +1 phase favored when η > 0 (FFS / ler force `abs(eta)`) |
| `update` | String | `"sync"` | `"sync"` or `"async"` (one async step = L random-with-replacement single-cell updates) |
| `save` | Bool | `true` | |
| `adj` | String | `""` | Filename suffix |

### history mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `L` | Int | `1000` | |
| `T_steps` | Int | `500` | |
| `p_noise` | Float | `0.05` | |
| `init` | String | `"domain"` | `"domain"` / `"random"` / `"all_plus"` / `"all_minus"` |
| `domain_start`, `domain_end` | Int | `3L÷8`, `5L÷8` | |

### ffs mode

Mirrors `simulation_driver.jl`'s FFS structure (same `n_interfaces`, `adaptive_L`, timeout-propagation contract). The sweep variable is either `p_noise` (default) or `eta` (`--vary_eta=true`). An alternative `--tau_min` / `--tau_max` lets you sweep uniformly in τ = 1/√p instead of in p.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `L` | Int | `500` | Ignored if `adaptive_L=true` |
| `p_noise` | Float | `0.10` | Fixed p_noise (when `vary_eta=true`) |
| `p_min`, `p_max` | Float | `0.02`, `0.20` | p_noise sweep range |
| `tau_min`, `tau_max` | Float | `NaN`, `NaN` | If both finite, sweep uniformly in τ=1/√p instead |
| `r_min`, `r_max` | Float | `NaN`, `NaN` | If both finite, sweep uniformly in `r = (log(1/p))²` instead (`p = exp(−√r)`) |
| `eta_min`, `eta_max` | Float | `0.01`, `0.20` | η sweep range (when `vary_eta=true`) |
| `n_steps` | Int | `6` | |
| `vary_eta` | Bool | `false` | If true, sweep η at fixed p_noise |
| `n_configs_per_run` | Int | `100` | |
| `n_repeats` | Int | `4` | |
| `target_crossing_prob` | Float | `0.20` | Adaptive-mode interface target |
| `n_interfaces` | Int | `0` | `>0` switches to canonical FFS |
| `adaptive_L` | Bool | `false` | Set `L = adaptive_factor × ℓ_er` per sweep point |
| `adaptive_factor` | Float | `3.0` | |
| `M_threshold` | Float | `0.4` | |
| `max_time_per_trial` | Int | `1_000_000` | |
| `lambda_0` | Float | `NaN` | Override auto-λ_0 |

### ler mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `p_noise` | Float | `0.10` | |
| `p_min`, `p_max` | Float | `0.02`, `0.20` | |
| `tau_min`, `tau_max` | Float | `NaN`, `NaN` | Uniform-in-τ sweep |
| `r_min`, `r_max` | Float | `NaN`, `NaN` | Uniform-in-`r` sweep, `r = (log(1/p))²`, `p = exp(−√r)` |
| `eta_min`, `eta_max` | Float | `0.01`, `0.20` | |
| `n_steps` | Int | `6` | |
| `vary_eta` | Bool | `false` | |
| `thresh_prob` | Float | `0.75` | |
| `erosion_num_trials` | Int | `2000` | |
| `min_erosion_length` | Int | `2` | |
| `t_evolve_factor` | Float | `2.0` | |
| `L_sys_factor` | Float | `1.0` | |
| `erode_vs_l` | Bool | `false` | |
| `show_histories` | Bool | `false` | Display spacetime heatmaps (requires Plots.jl) |

### diffusion mode

Measures the magnetization-MSD diffusion constant D by tracking deviations of total magnetization across `n_trials` thermalized trajectories.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `L` | Int | `500` | |
| `p_noise`, `p_min`, `p_max`, `tau_min`, `tau_max`, `eta_min`, `eta_max`, `n_steps`, `vary_eta` | — | as in ffs/ler | |
| `init_mag` | Int | `1` | ±1 starting magnetization |
| `T_thermalize` | Int | `-1` | `-1` → L÷2 |
| `T_track` | Int | `-1` | `-1` → 5L |
| `n_trials` | Int | `100` | |

---

## `ising.jl` — 1D Glauber Ising chain at h=0

Primary sweep parameter is **p = exp(−β J) ∈ (0, 1]** (same as everywhere else in the codebase). You may equivalently specify `--q_min/--q_max` (where `q = 1/p = exp(β J)`) or `--tau_min/--tau_max` (where `τ = exp(4 (β J)²)`); the sweep is taken to be linear in whichever you specify. Glauber dynamics: `P_flip = 1 / (1 + exp(β·ΔE))` on a single chain with periodic BCs. Built primarily as a controlled benchmark for the FFS pipeline against direct ensemble-averaged measurements.

### Modes

- **history** — spacetime sanity at a single p
- **ffs** — FFS for the mixing time τ_mem
- **mixing** — direct ensemble-averaged measurement of τ_mem (for benchmarking)

### Examples

```bash
# Tiny FFS run (in p)
julia --project=. -t auto ising.jl --mode=ffs --L=200 \
    --p_min=0.35 --p_max=0.55 --n_steps=3 --n_configs_per_run=50 --n_repeats=2 --M_threshold=0.75

# Equivalent FFS run via q (= 1/p)
julia --project=. -t auto ising.jl --mode=ffs --L=200 \
    --q_min=1.8 --q_max=2.9 --n_steps=3 --n_configs_per_run=50 --n_repeats=2 --M_threshold=0.75

# Matching direct mixing-time measurement (same M_threshold)
julia --project=. -t auto ising.jl --mode=mixing --L=200 \
    --p_min=0.35 --p_max=0.55 --n_steps=3 --n_trials=200 --M_threshold=0.75

# Canonical-mode FFS (recommended for h=0 Glauber, which is barrier-less)
julia --project=. -t auto ising.jl --mode=ffs --L=200 \
    --p_min=0.35 --p_max=0.55 --n_steps=3 --n_configs_per_run=200 --n_repeats=10 \
    --n_interfaces=10
```

### Shared parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | String | `"ffs"` | `history`, `ffs`, `mixing` |
| `save`, `adj` | | | As elsewhere |

### history mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `L` | Int | `200` | |
| `p` | Float | `0.5` | `β = -log(p)`. May also be set via `--q` or `--tau`. |
| `T_steps` | Int | `500` | |
| `init` | String | `"all_plus"` | `"all_plus"` / `"all_minus"` / `"random"` |

### ffs mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `L` | Int | `1000` | |
| `p_min`, `p_max` | Float | `0.33`, `0.67` | p sweep range; equivalent `q ∈ [1.5, 3.0]`. May also be set via `--q_min/--q_max` or `--tau_min/--tau_max`. |
| `n_steps` | Int | `6` | |
| `n_configs_per_run` | Int | `200` | |
| `n_repeats` | Int | `10` | |
| `target_crossing_prob` | Float | `0.15` | |
| `n_interfaces` | Int | `0` | `>0` = canonical FFS |
| `M_threshold` | Float | `0.75` | |
| `max_time_per_trial` | Int | `100_000_000` | |
| `lambda_0` | Float | `NaN` | |

### mixing mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `L` | Int | `1000` | |
| `p_min`, `p_max` | Float | `0.33`, `0.67` | Same conventions as ffs. May also be set via `--q_min/--q_max` or `--tau_min/--tau_max`. |
| `n_steps` | Int | `6` | |
| `n_trials` | Int | `1000` | |
| `M_threshold` | Float | `0.75` | |
| `max_time` | Int | `100_000_000` | |

---

## Plotting

All output JLD2 files (from any driver) are read by [`sliding_plotter.py`](sliding_plotter.py). Mode is auto-detected from the file contents. Dynamics-aware label switching is enabled via a `dynamics` tag inside the file: GKL files get `rainbow` colors and labels `p`/`η`; ising-glauber files get axis label `q` and linear y-scale.

```bash
# Single file (auto-detect mode)
python3 sliding_plotter.py data/ising_sliding_mixing_L2000_v0.00.jld2

# FFS curve overlaid with direct-mixing scatter (benchmark mode)
python3 sliding_plotter.py \
    data/ising_glauber_ffs_L500_q200.00to2000.00.jld2 \
    data/ising_glauber_mixing_L500_q200.00to2000.00.jld2

# Same overlay, but rescale each FFS curve to match direct at smallest x
python3 sliding_plotter.py \
    data/ising_glauber_ffs_L500_q200.00to2000.00.jld2 \
    data/ising_glauber_mixing_L500_q200.00to2000.00.jld2\
    --anchor_normalize
```

### Useful flags

| Flag | Modes | Description |
|---|---|---|
| `--mode` | all | Override auto-detected mode |
| `--inset` | ffs / mixing | Add a small inset showing exponential-fit coefficients (multi-file required) |
| `--fit_inset` | erosion_test / ffs / mixing | Add an inset showing the per-file linear-fit slope |
| `--ploglog` | ffs | log-log plot of log₁₀(τ) vs p with per-file fit log τ ~ p^a |
| `--a=A` | ffs | Plot τ against x^A instead of x (test log τ ~ p^A); also overrides the exponent in the `--xr` transform |
| `--alpha=Α` | ffs / mixing | When sweeping p, plot τ against (β·J)^α instead of p |
| `--xp` | ffs (GKL p-sweep) | Plot `t_mem` against `p` (explicit default; useful to override an implicit transform) |
| `--xq` | ffs (GKL p-sweep) | Plot `t_mem` against `1/p` |
| `--xlogsqq` | ffs (GKL p-sweep) | Plot `t_mem` against `(log(1/p))²` |
| `--xr` | ffs (sliding-Ising p-sweep) | Plot `t_mem` against `(β J)² = (log(1/p))²` (same transform as `--xlogsqq` but labeled with the Ising temperature variable); combine with `--a=A` to use `(β J)^A` |
| `--anchor_normalize` | ffs+mixing overlay | Rescale each FFS curve so it matches the direct value at the smallest x |
| `--logy` | erosion_test | Log y-axis |
| `--raw` | erosion_test | P_shrink vs raw l instead of l / ℓ_c |
| `--small_stats` | erosion_test | (1 − l/ℓ_c)² vs 1 − P on log y |
| `--heat` | energy | Heat flow instead of energy |
| `--t_max=T` | history | Truncate time axis to [0, T] |
| `--hide_ticks` | history | Hide axis ticks/labels |
| `--xscale`, `--yscale` | any | Force `linear` or `log` on the corresponding axis of every figure produced this run |
| `--cmap=NAME` | any | Override the per-file colormap with the named matplotlib colormap (e.g. `viridis`, `Greens`). Replaces the default `coolwarm_r` / rainbow / `Oranges` / `Blues` choices in every plot mode that colors multiple files |
| `--legloc=LOC` | any | Force the legend location on every figure (e.g. `"upper left"`, `"lower right"`); overrides each plot mode's hard-coded default (typically `"best"`) |
| `--fitrange=F` | ffs / mixing / energy | Per-file linear fit of `ln(y)` vs the *plotted* x-coordinate over the upper `(1−F)` portion of x (e.g. `--fitrange=0.2` → fit over `x ≥ x_min + 0.2·(x_max − x_min)`, i.e. the last 80%). Honors `--a`/`--alpha`/`--xr`/`--xq`/`--xp`/`--xlogsqq`, so e.g. `--xr --a=2 --fitrange=0.2` fits `ln(y) ~ (βJ)²`. Prints slope, intercept, their stderrs, R², and n per file; overlays each fit as a black dashed line. Ignored with `--ploglog` (where the y-axis is already `log10`). |
| `--residuals` | ffs / mixing / energy (with `--fitrange`) | Companion diagnostic: swap the figure for a 3:1 two-panel gridspec with the data + fit lines on top and per-file residuals `ln(y) − (slope·x + intercept)` on the bottom (matching colors, shared x-axis). Residuals are shown for **all** plotted points — including those outside the fit window — so systematic curvature outside the fit anchors a "wrong-exponent" diagnostic. No-op without `--fitrange`. |
| `--pmin`, `--pmax`, `--vmin`, `--vmax` | phase_diagram | Crop axes |

The plotter also auto-switches the FFS y-axis label between `$t_{\sf mem}$` and `$\widetilde t_{\sf mem}$` depending on whether the JLD2 records a non-zero `seed_droplet_size` (the modified-clock estimator described in the FFS section above).
