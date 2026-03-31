# Energy Mode

## Purpose

Measures the steady-state energy per spin (and optionally the heat flow) as a function of temperature or sliding velocity. This characterizes how sliding drives the system out of equilibrium.

## Physics

At equilibrium ($v = 0$), the energy is determined by the Boltzmann distribution. With sliding ($v > 0$), the system is driven out of equilibrium: the relative motion of the chains continuously disrupts inter-chain bonds, injecting energy. The system reaches a nonequilibrium steady state where the energy pumped in by sliding is balanced by dissipation through the thermal bath.

The energy is normalized so that the ferromagnetic ground state (all spins aligned) has $E = 0$. The raw energy $E_\text{raw} = -\sum_{\langle i,j \rangle} \sigma_i \sigma_j$ has ground state $-3L$ (3 bonds per site), so the reported energy per spin is $E_\text{raw}/(2L) + 3/2$.

The heat flow $Q$ measures the net energy exchanged with the thermal bath per time unit, computed by tracking the energy change $\Delta E$ of each accepted Metropolis flip.

## Sweep Modes

- **Sweep p** (default): fix $v$, vary $p = e^\beta$.
- **Sweep v** (`vary_v=true`): fix $p$, vary $v$.

## Parameters

- `T_equil`: equilibration time (in time units, each = $L$ MC updates). Discarded.
- `T_sample`: sampling time. Energy and heat flow are averaged over this window.
- Parallelized over sweep values (each velocity/temperature runs independently on a separate thread).

## Output

- `mean_energies`: steady-state energy per spin at each sweep value.
- `mean_heat_flows`: average heat flow per time unit.

## Implementation

See `src/energy.jl`. Each sweep value initializes a random state, equilibrates with `evolve_time!`, then samples using `evolve_and_measure_heat!` which tracks both energy and heat flow per time step.
