# Forward Flux Sampling (FFS) Mode

## Purpose

Estimates mixing times that are too long for direct simulation using forward flux sampling, a rare-event technique. Where the mixing mode is limited to $\tau \lesssim 10^5$ (hours of runtime), FFS can estimate $\tau \sim 10^{15}$ or more.

## Physics Background

The system starts in the metastable ferromagnetic state ($m \approx 1$). Escaping to the disordered state ($m \approx 0$) requires crossing a free-energy barrier: a rare nucleation event must create a domain large enough to grow rather than shrink. The probability of this event decreases exponentially with $\beta$, making the mixing time $\tau \propto e^{\Delta F / k_B T}$ where $\Delta F$ is the barrier height.

FFS decomposes this exponentially unlikely transition into a product of moderately unlikely steps. Instead of waiting for one trajectory to cross the entire barrier, we measure the conditional probability of progressing from one magnetization level to the next, and multiply them together.

## Algorithm Overview

### Interfaces

FFS defines a series of magnetization levels (interfaces) between the metastable state and the target:

$$1 \approx \lambda_0 > \lambda_1 > \lambda_2 > \cdots > \lambda_n \approx M_\text{threshold}$$

The mixing time is estimated as:

$$\tau \approx \frac{1}{\Phi_0 \prod_{i=1}^n P_i}$$

where $\Phi_0$ is the rate of first crossings of $\lambda_0$ from the metastable basin, and $P_i$ is the conditional probability of reaching $\lambda_i$ given that you started at $\lambda_{i-1}$ (before returning to the basin $m \geq \lambda_0$).

### Phase 0: Initial Flux

Run long trajectories starting from all $+1$. Each time the magnetization first crosses below $\lambda_0$ (going downward), record the spin configuration and the crossing time. The flux rate is:

$$\Phi_0 = \frac{N_\text{crossings}}{\sum_i t_i}$$

where $t_i$ are the first-passage times of individual trajectories. This is parallelized across threads, with each thread running independent chains until its quota is met.

### Phase $i$: Interface Crossing

From the stored configurations at $\lambda_{i-1}$, launch trial trajectories. Each trial:
1. Picks a random stored configuration from the previous interface.
2. Runs until either $m < \lambda_i$ (**success** — store configuration) or $m \geq \lambda_{i-1}$ (**failure** — returned above the previous interface).

The crossing probability is $P_i = N_\text{success} / N_\text{trials}$. Trials are run in batches until `n_configs` successes are collected, ensuring good configuration diversity for the next interface.

## Computational Optimizations

### 1. Adaptive Interface Placement

Rather than using fixed, evenly-spaced interfaces, the code places each interface adaptively based on **probe trials**. Before collecting configurations at an interface, `n_configs` probe trials are fired from the current interface. Each probe runs until it either returns to basin ($m \geq \lambda_0$) or reaches $M_\text{threshold}$, tracking the minimum magnetization reached. The next interface is placed at the `target_crossing_prob` quantile (default 0.25) of these minimum magnetizations.

This concentrates interfaces where the barrier is steepest (small steps where crossing is hard) and takes large steps where the free energy is flat (easy crossings). It avoids wasting computational effort on regions that contribute negligibly to the product $\prod P_i$.

### 2. Log-Space Accumulation

The product $\prod P_i$ can be astronomically small (e.g., $10^{-40}$). To avoid floating-point underflow, the code accumulates $\sum \log_{10}(P_i)$ and only exponentiates at the end.

### 3. Barrier Crossing Detection

Once past the free-energy barrier, the system rolls downhill freely and will never return to the metastable basin. The FFS framework becomes unnecessary in this regime — remaining crossing probabilities are all $\approx 1$. The code detects this by checking the probed interface spacing: if a single step covers more than 50% of the remaining distance from $\lambda_\text{cur}$ to $M_\text{threshold}$, the barrier has been crossed and FFS terminates.

Without this, post-barrier probes would time out (trajectories never return to $\lambda_0$), wasting compute time on an already-determined outcome.

### 4. Early Termination in Probes

Probe trials stop early if the magnetization drops below $M_\text{threshold}$. This prevents post-barrier probes from running to the timeout limit — once a trajectory has crossed the target, there is no need to wait for it to return to basin.

### 5. Adaptive $\lambda_0$ from Metastable State

The first interface $\lambda_0$ is set automatically per sweep point by briefly equilibrating from all $+1$ and measuring the mean and standard deviation of the metastable magnetization: $\lambda_0 = m^* - 2.5\sigma$. This ensures $\lambda_0$ is close enough to $m = 1$ that crossings are frequent (fast Phase 0), while being far enough in the tail that each crossing represents a genuine fluctuation away from equilibrium.

### 6. Guaranteed Configuration Collection

Both Phase 0 and the crossing phases run until exactly `n_configs` successful configurations are collected (not a fixed number of trials). This ensures consistent statistical quality at every interface, regardless of the crossing probability. The crossing probability is computed as $P_i = n_\text{configs} / n_\text{total trials}$.

### 7. Opposing Magnetic Field

When $h \neq 0$, the sign of $h$ is forced negative (opposing the initial $+1$ magnetization). This destabilizes the metastable state, reducing mixing times and making FFS converge faster, which is useful for testing and benchmarking.

## Error Estimation

The variance of $\log_{10}(\tau)$ is estimated by propagating the statistical uncertainties from each phase:

$$\text{Var}[\ln \tau] \approx \frac{1}{N_\text{crossings}} + \sum_i \frac{1 - P_i}{P_i \cdot N_{\text{trials},i}}$$

The first term is the Poisson variance from the initial flux measurement (Phase 0). Each subsequent term is the binomial variance from interface $i$. The reported $\pm$ uncertainty is $\sqrt{\text{Var}} / \ln 10$ converted to $\log_{10}$ units.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_configs` | 200 | Configurations collected per interface |
| `target_crossing_prob` | 0.25 | Target quantile for adaptive interface placement |
| `M_threshold` | 0.75 | Target magnetization (safety floor) |
| `max_time_per_trial` | 100000 | Timeout per trial |
| `vary_v` | false | If true, sweep $v$ instead of $p$ |

## Output

- `mean_mixing_times`: estimated $\tau$ at each sweep point
- `log_mixing_times`: $\log_{10}(\tau)$
- `log_mixing_times_std`: uncertainty in $\log_{10}(\tau)$
- `flux_rates`: $\Phi_0$ at each sweep point
- Sweep parameter array (`p_values` or `vs`)

## Implementation

See `src/ffs.jl`. Key functions:
- `measure_initial_flux`: Phase 0 (parallelized across threads)
- `probe_next_interface`: adaptive interface placement
- `measure_crossing_probability`: interface crossing with guaranteed config collection
- `measure_metastable_magnetization`: sets $\lambda_0$ automatically
- `run_ffs_mode`: main loop with barrier crossing detection
