# History Mode

## Purpose

Records the full spacetime evolution of the local magnetization $m_i(t) = \sigma_i^{\text{top}} + \sigma_i^{\text{bottom}}$, producing a 2D heatmap of the spin configuration over time. This is useful for visualizing domain wall dynamics, domain erosion, and the interplay between thermal fluctuations and sliding.

## Physics

When initialized with a minority domain (a contiguous block of $-1$ spins in a sea of $+1$ spins), the system exhibits competition between thermal diffusion (which tends to erode the domain) and the ferromagnetic coupling (which resists erosion). With sliding ($v > 0$), the domain drifts through the system and its lifetime depends on the velocity — a key observable in the study of friction-induced memory.

## Parameters

- `init`: `"domain"` places a block of $-1$ spins from `domain_start` to `domain_end`; `"random"` initializes both chains uniformly at random.
- `T_steps`: number of time units to simulate (measurements at each integer time).
- Standard physical parameters: `L`, `v`, `beta`, `h`.

## Output

A matrix `magnetization_history` of shape `(T_steps+1, L)` with values in $\{-2, 0, +2\}$, saved to JLD2. The Python plotter renders this as a heatmap with time running upward and site index on the horizontal axis.

## Implementation

See `src/history.jl`. Uses `evolve_and_measure!` with a callback that records `compute_local_magnetization(state)` at each time step.
