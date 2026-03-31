# The Sliding Ising Ladder

## Physical System

The model consists of two coupled Ising chains ("ladder") of length L with periodic boundary conditions. Each site i on each chain carries a spin $\sigma_i = \pm 1$. The Hamiltonian is

$$H = -J_\parallel \sum_{\langle i,j \rangle_\parallel} \sigma_i \sigma_j - J_\perp \sum_i \sigma_i^{\text{top}} \sigma_i^{\text{bottom}} - h \sum_i \sigma_i$$

where the first sum runs over nearest-neighbor pairs within each chain (in-chain coupling $J_\parallel$), the second sum couples vertically aligned sites between chains (inter-chain coupling $J_\perp$), and $h$ is an external magnetic field. We set $J_\parallel = J_\perp = 1$.

Each spin has exactly 3 neighbors: 2 in-chain (left, right with periodic boundaries) and 1 inter-chain (the spin directly above/below).

## Sliding Dynamics

The key feature is that one chain slides relative to the other at velocity $v$. Concretely, the top chain is cyclically shifted right by one lattice site every $\text{round}(L/v)$ Metropolis updates. Between shifts, single-spin Metropolis updates are applied to randomly selected spins on either chain.

The time convention is: **1 time unit = L single-spin MC updates**, independent of $v$. In one time unit, $v$ shifts occur, so each spin slides past $v$ opposing sites between consecutive updates. When $v = 0$, no shifts occur and the system reduces to a standard two-chain Ising model.

## Metropolis Updates

A single update selects a random chain and site, computes the energy change $\Delta E = 2\sigma_i(\sum_{\text{neighbors}} \sigma_j + h)$, and accepts the flip with probability $\min(1, e^{-\beta \Delta E})$. Since $\sigma \in \{-1, +1\}$ and the neighbor sum takes only values in $\{-3, -1, 1, 3\}$, there are only 8 possible $(\sigma, \text{neighbor sum})$ combinations. The acceptance probabilities are precomputed into a $2 \times 4$ lookup table at construction time, eliminating expensive `exp()` calls from the inner loop.

## Single-Chain Mode

The code also supports a single-chain Ising model (no inter-chain coupling, no sliding). In this mode, each spin has 2 neighbors (left, right), and the acceptance table is $2 \times 3$ for neighbor sums in $\{-2, 0, 2\}$.

## Implementation

The simulation state is stored in a `SimulationState` struct containing the two spin arrays (`Vector{Int8}`), physical parameters ($L, \beta, h, v$), the precomputed number of updates per shift (`n_sub`), and the acceptance table. See `src/types.jl` and `src/core.jl`.
