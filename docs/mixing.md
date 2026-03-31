# Mixing Mode

## Purpose

Measures the mixing time — the time for the system to escape from the fully magnetized metastable state ($m = 1$, all spins $+1$) and reach a threshold magnetization $M_\text{threshold}$. This is done by direct simulation: run the dynamics and record when $m$ first drops below the threshold.

## Physics

At low temperatures (high $p = e^\beta$), the ferromagnetic state is metastable: the system sits near $m \approx 1$ for an exponentially long time before a rare fluctuation nucleates a domain of opposite magnetization large enough to grow. The mixing time $\tau$ grows rapidly with $\beta$, and at some point direct simulation becomes infeasible (see the FFS mode for rare-event sampling).

With sliding ($v > 0$), the relative motion between chains disrupts inter-chain correlations. This can either stabilize or destabilize the metastable state depending on the velocity, creating a rich dependence $\tau(v, \beta)$.

## Sweep Modes

- **Sweep p** (default): fix $v$, vary $p = e^\beta$ over a linearly spaced grid from `p_min` to `p_max`.
- **Sweep v** (`vary_v=true`): fix $p$, vary $v$ over a grid from `v_min` to `v_max`.
- **Single layer** (`single_layer=true`): 1D Ising chain (no inter-chain coupling, no sliding), sweep $p$.

## Parameters

- `n_trials`: independent trials per sweep value (parallelized over threads).
- `M_threshold`: magnetization threshold (default 0.65).
- `max_time`: timeout per trial.
- `n_steps`: number of sweep values.

## Output

An array `mean_mixing_times` of length `n_steps`, plus the sweep parameter array (`p_values` or `vs`). Plotted on a log scale by `sliding_plotter.py`.

## Implementation

See `src/mixing.jl`. Each trial calls `evolve_and_measure!` with a callback that returns `false` (stop) when $m \leq M_\text{threshold}$. The returned step count is the mixing time for that trial.
