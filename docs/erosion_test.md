# Erosion Test Mode

## Purpose

Finds the critical domain size $\ell_c$ — the smallest domain that can survive against thermal erosion — as a function of temperature or sliding velocity. Domains smaller than $\ell_c$ rapidly decay; domains larger than $\ell_c$ persist.

## Physics

A minority domain is a contiguous block of $-1$ spins in a background of $+1$ spins. At finite temperature, thermal fluctuations erode the domain boundaries. Without sliding, a domain of size $\ell$ survives if the surface tension energy ($\propto$ boundary length) exceeds the thermal energy cost of fluctuations. With sliding, the chains move relative to each other, misaligning the inter-chain correlations within the domain and potentially accelerating erosion.

The critical domain size $\ell_c$ sets the fundamental length scale for memory formation: information stored in domains smaller than $\ell_c$ is rapidly erased, while larger domains can persist for long times.

## Algorithm

For each parameter point:

1. Start with domain size $\ell = 4$.
2. Create a system of size $L_\text{sys} = \text{round}(2\ell(v+1))$ — large enough to prevent the domain from wrapping around.
3. Place a minority domain of size $\ell$ (both chains set to $-1$ for the first $\ell$ sites).
4. Evolve for $T = 2\ell$ time units.
5. Check if the domain has "shrunk" (see criteria below).
6. Repeat for many trials; compute $P_\text{shrink}$.
7. If $P_\text{shrink} < \text{thresh\_prob}$ (default 0.75), the domain survives: $\ell_c = \ell$.
8. Otherwise, increment $\ell$ and repeat.

### Shrinkage Criteria

**Default mode**: The domain has shrunk if the total number of minority spins ($-1$ on either chain) drops below $1.5\ell$ (started at $2\ell$).

**Doublon mode** (`doublon_mode=true`): The domain has shrunk if the number of "doublons" — sites where *both* top and bottom spins are $-1$ — drops below $0.1\ell$ (started at $\ell$). This is a stricter criterion that specifically probes inter-chain correlations. Evolution time is extended to $5\ell$.

### Adaptive Search

Rather than testing every integer $\ell$ from 4 upward, the code uses a three-phase adaptive search:

1. **Coarse bracketing**: Starting from the previous sweep point's $\ell_c$, take geometrically growing steps (1, 2, 4, 8, 16) with a small number of trials (1/10 of the full count) to quickly bracket $\ell_c$.
2. **Binary search**: Narrow the bracket to adjacent integers.
3. **Verification**: Confirm with the full trial count that $P_\text{shrink}(\ell_c) < \text{thresh}$ AND $P_\text{shrink}(\ell_c - 1) \geq \text{thresh}$.

The previous sweep point's $\ell_c$ is used as the starting guess for the next, exploiting the smooth dependence of $\ell_c$ on the sweep parameter.

## Sweep Modes

- **Sweep p** (default): fix $v$, vary $p = e^\beta$.
- **Sweep v** (`vary_v=true`): fix $p$, vary $v$.

## Diagnostic Options

- `show_histories=true`: After finding $\ell_c$, display spacetime heatmaps for domain sizes $0.75\ell_c$, $\ell_c$, and $1.25\ell_c$ (3 trials each), shifted into the co-moving frame at $v/2$.
- `erode_vs_l=true`: Plot $P_\text{shrink}$ vs $\ell$ for a range of domain sizes around $\ell_c$.

## Output

- `lc_values`: critical domain size at each sweep value.
- Sweep parameter array and metadata.
- Optionally: shrinkage probability curves (`erode_l_values`, `erode_probs`).

## Implementation

See `src/erosion_test.jl`. Key functions: `measure_shrink_prob`, `find_critical_length`, `run_erosion_test_mode`, and diagnostic functions `show_erosion_histories` and `plot_erode_vs_l`.
