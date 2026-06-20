"""
    metropolis_step!(state, n_updates)

Perform n_updates Metropolis single-spin updates on the two-chain Ising system.
Each update randomly selects a chain and site, then attempts a spin flip.
Uses precomputed acceptance table to avoid exp() calls in the inner loop.
"""
function metropolis_step!(state::SimulationState, n_updates::Int)
    top = state.top
    L = state.L
    table = state.acceptance_table
    mag_sum = state.mag_sum
    if state.single_layer
        @inbounds for _ in 1:n_updates
            i = rand(1:L)
            ip = i == L ? 1 : i + 1
            im = i == 1 ? L : i - 1
            σ = top[i]
            neighbors_sum = top[im] + top[ip]
            σ_idx = (σ + Int8(1)) >> 1 + 1        # -1→1, +1→2
            ns_idx = (neighbors_sum + 4) >> 1      # -2→1, 0→2, 2→3
            if rand() < table[σ_idx, ns_idx]
                top[i] = -σ
                mag_sum -= 2 * σ
            end
        end
    else
        bottom = state.bottom
        # We sample the site `i` in the LEGACY frame (the frame that would exist
        # if shift_chain! had physically rotated the top array). The top-array
        # storage index for legacy slot i is `mod1(i − top_offset, L)`; the
        # bottom-array index for legacy slot i is just i (bottom never shifts).
        # This mirrors the legacy RNG / move-set: same i = same physical bond.
        top_offset = state.top_offset
        @inbounds for _ in 1:n_updates
            chain = rand(1:2)
            i = rand(1:L)
            i_top = i - top_offset
            i_top = i_top < 1 ? i_top + L : i_top  # mod1 without the function call (hot path)
            ip_bot = i == L ? 1 : i + 1
            im_bot = i == 1 ? L : i - 1
            ip_top = i_top == L ? 1 : i_top + 1
            im_top = i_top == 1 ? L : i_top - 1
            if chain == 1
                σ = top[i_top]
                neighbors_sum = top[im_top] + top[ip_top] + bottom[i]
            else
                σ = bottom[i]
                neighbors_sum = bottom[im_bot] + bottom[ip_bot] + top[i_top]
            end
            σ_idx = (σ + Int8(1)) >> 1 + 1        # -1→1, +1→2
            ns_idx = (neighbors_sum + 5) >> 1      # -3→1, -1→2, 1→3, 3→4
            if rand() < table[σ_idx, ns_idx]
                if chain == 1
                    top[i_top] = -σ
                else
                    bottom[i] = -σ
                end
                mag_sum -= 2 * σ
            end
        end
    end
    state.mag_sum = mag_sum
end

"""
    metropolis_step_heat!(state, n_updates) -> Float64

Same as metropolis_step!, but returns the total energy change from accepted flips.
"""
function metropolis_step_heat!(state::SimulationState, n_updates::Int)
    top = state.top
    L = state.L
    table = state.acceptance_table
    h = state.h
    mag_sum = state.mag_sum
    total_dE = 0.0
    if state.single_layer
        @inbounds for _ in 1:n_updates
            i = rand(1:L)
            ip = i == L ? 1 : i + 1
            im = i == 1 ? L : i - 1
            σ = top[i]
            neighbors_sum = top[im] + top[ip]
            σ_idx = (σ + Int8(1)) >> 1 + 1
            ns_idx = (neighbors_sum + 4) >> 1
            if rand() < table[σ_idx, ns_idx]
                top[i] = -σ
                mag_sum -= 2 * σ
                total_dE += 2 * σ * (neighbors_sum + h)
            end
        end
    else
        bottom = state.bottom
        top_offset = state.top_offset
        @inbounds for _ in 1:n_updates
            chain = rand(1:2)
            i = rand(1:L)
            i_top = i - top_offset
            i_top = i_top < 1 ? i_top + L : i_top
            ip_bot = i == L ? 1 : i + 1
            im_bot = i == 1 ? L : i - 1
            ip_top = i_top == L ? 1 : i_top + 1
            im_top = i_top == 1 ? L : i_top - 1
            if chain == 1
                σ = top[i_top]
                neighbors_sum = top[im_top] + top[ip_top] + bottom[i]
            else
                σ = bottom[i]
                neighbors_sum = bottom[im_bot] + bottom[ip_bot] + top[i_top]
            end
            σ_idx = (σ + Int8(1)) >> 1 + 1
            ns_idx = (neighbors_sum + 5) >> 1
            if rand() < table[σ_idx, ns_idx]
                if chain == 1
                    top[i_top] = -σ
                else
                    bottom[i] = -σ
                end
                mag_sum -= 2 * σ
                total_dE += 2 * σ * (neighbors_sum + h)
            end
        end
    end
    state.mag_sum = mag_sum
    return total_dE
end

"""
    shift_chain!(state)

Cyclically shift the top chain right by 1 *legacy slot*. We never actually
rotate the storage array — instead, we increment `state.top_offset`, which
the Metropolis kernel and energy-evaluator use to compute the correct
top↔bottom partner per legacy slot. This makes the operation O(1) rather
than the previous O(L) memcpy.

Convention: legacy slot i ↔ top-storage index `mod1(i − top_offset, L)`.
A right-shift in the legacy frame increments top_offset by 1, so the spin
that was at legacy slot i now sits at legacy slot i+1 (its storage index is
unchanged).
"""
function shift_chain!(state::SimulationState)
    state.top_offset = mod(state.top_offset + 1, state.L)
    return nothing
end

"""
    seed_droplet!(state, k)

In-place: flip k consecutive spins on both chains (or just the top chain in
single-layer mode) starting at a uniformly-random position (with periodic
wrap-around). Used by FFS to deterministically inject a minority droplet into
an all-+ configuration, removing the exp(6βJ) waiting time for a thermal
fluctuation in the all-+ sojourn. Caller is responsible for ensuring the
pre-state really is all-+; otherwise mag_sum is recomputed defensively.

Assumes spins to be flipped were +1 (the typical use is right after detecting
the all-+ state). With k flips on each chain in a two-layer setup that's 2k
spins total, so mag_sum decreases by 4k. In single_layer mode it's k spins
and mag_sum decreases by 2k.
"""
function seed_droplet!(state::SimulationState, k::Int)
    k > 0 || return
    L = state.L
    k <= L || error("seed_droplet_size $k must be ≤ L=$L.")
    start = rand(1:L)
    top = state.top
    bottom = state.bottom
    if state.single_layer
        @inbounds for j in 0:(k-1)
            i = mod1(start + j, L)
            top[i] = -top[i]
        end
    else
        # The droplet must be vertically aligned in the LEGACY frame (i.e., the
        # flipped top and bottom spins share a cross bond). With offset != 0,
        # the storage indices that share a bond at legacy slot s are
        # (top: mod1(s − off, L), bottom: s); we iterate `start..start+k-1` in
        # legacy-slot space.
        off = state.top_offset
        @inbounds for j in 0:(k-1)
            s = mod1(start + j, L)         # legacy slot
            i_top = mod1(s - off, L)
            top[i_top] = -top[i_top]
            bottom[s] = -bottom[s]
        end
    end
    recompute_mag_sum!(state)
    return nothing
end

# Internal: did the state just hit the all-+ basin floor?
@inline _is_all_plus(state::SimulationState) =
    state.mag_sum == (state.single_layer ? state.L : 2 * state.L)

"""
    evolve_and_measure!(state, T_steps, on_measure; seed_droplet_size=0, randshift=false) -> Int

Run the simulation for up to T_steps time units, calling `on_measure(step)` at each
integer time. `on_measure` should return `true` to continue or `false` to stop early.
Returns the number of completed measurement steps.

Time convention:
- 1 time unit = L single-spin MC updates (independent of v)
- For v > 0, the top chain is shifted with average rate v per time unit. If
  `randshift=false` (default), shifts are placed deterministically (Bresenham:
  shift i at MC-update count `round(Int, i·L/v)`). If `randshift=true`, the
  inter-shift gap (in MC updates) is sampled from Exp(L/v) — a Poisson shift
  process with rate v, removing all commensurate-vs-incommensurate `L/v`
  effects (see paper SM for the integer/half-integer artifact this addresses).
- Measurements occur at every integer time (every L MC updates).
"""
function evolve_and_measure!(state::SimulationState, T_steps::Int, on_measure::Function;
                              seed_droplet_size::Int=0, randshift::Bool=false)
    v = state.v
    L = state.L

    if v == 0.0
        for step in 1:T_steps
            metropolis_step!(state, L)
            # Modified-clock droplet seeding: if dynamics just returned to the
            # all-+ basin floor, instantly inject a minority droplet (FFS only;
            # caller signals opt-in via seed_droplet_size > 0). The skipped
            # all-+ sojourn time is not accounted for — see SM section on
            # modified-clock convention.
            if seed_droplet_size > 0 && _is_all_plus(state)
                seed_droplet!(state, seed_droplet_size)
            end
            if !on_measure(step)
                return step
            end
        end
        return T_steps
    end

    # Event-driven schedule. Shifts are either deterministic (Bresenham at
    # `round(Int, i·L/v)`) or stochastic (inter-shift gaps ~ Exp(L/v)) under
    # `randshift=true`. Measurements always at `j·L`.
    L_over_v = L / v
    measurement_count = 0
    update_count = 0          # total MC updates done so far
    # Match the legacy "shift first" ordering: fire shift i=0 at
    # update_count = 0 below before any MC, then subsequent shifts at
    # `round(i·L/v)` (deterministic) or `update_count + Exp(L/v)` (stochastic).
    next_shift_idx = 0
    next_shift_at = 0
    next_measure_at = L

    while measurement_count < T_steps
        next_event_at = min(next_shift_at, next_measure_at)
        delta = next_event_at - update_count
        if delta > 0
            metropolis_step!(state, delta)
            update_count = next_event_at
        end

        # Order matters at coincident events (round(i·L/v) == j·L, which
        # happens at every measurement for integer v): MEASURE FIRST, THEN
        # SHIFT. Reasons:
        #   (1) the loop-exit check on the final measurement runs before the
        #       would-be shift, so the trajectory ends with T_steps·v shifts
        #       (not T_steps·v + 1);
        #   (2) at mid-trajectory coincidences, the measurement sees the
        #       PRE-shift config — matching the legacy cycle, where shifts
        #       fired at the START of each outer iter (i.e., after the
        #       previous iter's measure).
        # Shift-first order produces a systematic top-chain rotation offset
        # in saved Phase-0 configs, which biases the FFS estimator at integer
        # v but not at half-integer v (where coincidences only hit every
        # second measurement). Don't flip the order back.
        if next_measure_at == next_event_at
            measurement_count += 1
            if seed_droplet_size > 0 && _is_all_plus(state)
                seed_droplet!(state, seed_droplet_size)
            end
            next_measure_at = (measurement_count + 1) * L
            if !on_measure(measurement_count) || measurement_count >= T_steps
                return measurement_count
            end
        end
        if next_shift_at == next_event_at
            shift_chain!(state)
            next_shift_idx += 1
            if randshift
                # Poisson shift process at rate v: inter-shift gap ~ Exp(L/v)
                # in MC-update units. randexp() draws a unit-mean exponential.
                # Round to integer update count; clamp gap ≥ 1 so the schedule
                # always advances and we don't busy-loop on a degenerate zero.
                gap = max(1, round(Int, randexp() * L_over_v))
                next_shift_at = update_count + gap
            else
                next_shift_at = round(Int, next_shift_idx * L_over_v)
            end
        end
    end
    return measurement_count
end

"""
    evolve_and_measure_heat!(state, T_steps, on_measure) -> Int

Same as evolve_and_measure!, but tracks heat flow from MC updates.
Calls on_measure(step, heat) where heat is the total energy change from
accepted MC updates during that time step (L updates), normalized by 2L.
"""
function evolve_and_measure_heat!(state::SimulationState, T_steps::Int, on_measure::Function;
                                   randshift::Bool=false)
    v = state.v
    L = state.L

    if v == 0.0
        for step in 1:T_steps
            dE = metropolis_step_heat!(state, L)
            if !on_measure(step, dE / (2 * L))
                return step
            end
        end
        return T_steps
    end

    # Same event schedule as evolve_and_measure! — see comment there.
    L_over_v = L / v
    measurement_count = 0
    update_count = 0
    next_shift_idx = 0
    next_shift_at = 0
    next_measure_at = L
    heat_this_step = 0.0

    while measurement_count < T_steps
        next_event_at = min(next_shift_at, next_measure_at)
        delta = next_event_at - update_count
        if delta > 0
            heat_this_step += metropolis_step_heat!(state, delta)
            update_count = next_event_at
        end

        # MEASURE FIRST, THEN SHIFT at coincident events — see comment in
        # evolve_and_measure! for the rationale (avoids one extra shift at
        # the trajectory end and a top-chain-rotation offset bias at every
        # coincident event for integer v).
        if next_measure_at == next_event_at
            measurement_count += 1
            next_measure_at = (measurement_count + 1) * L
            if !on_measure(measurement_count, heat_this_step / (2 * L)) || measurement_count >= T_steps
                return measurement_count
            end
            heat_this_step = 0.0
        end
        if next_shift_at == next_event_at
            shift_chain!(state)
            next_shift_idx += 1
            if randshift
                gap = max(1, round(Int, randexp() * L_over_v))
                next_shift_at = update_count + gap
            else
                next_shift_at = round(Int, next_shift_idx * L_over_v)
            end
        end
    end
    return measurement_count
end

"""
    evolve_time!(state, T)

Advance the simulation by T time units without taking measurements.
Same time convention as `evolve_and_measure!`.
"""
function evolve_time!(state::SimulationState, T::Int; seed_droplet_size::Int=0,
                       randshift::Bool=false)
    v = state.v
    L = state.L

    if v == 0.0
        # When seeding is enabled we need to check periodically (otherwise the
        # state could spend the whole T·L attempts in all-+). Granularity = L.
        if seed_droplet_size > 0
            for _ in 1:T
                metropolis_step!(state, L)
                if _is_all_plus(state)
                    seed_droplet!(state, seed_droplet_size)
                end
            end
        else
            metropolis_step!(state, T * L)
        end
        return
    end

    # Deterministic Bresenham schedule (default), or Poisson process with rate
    # v if `randshift=true`. Both run for exactly T*L MC updates.
    L_over_v = L / v
    total_updates = T * L
    update_count = 0
    next_shift_idx = 0
    next_shift_at = 0

    while update_count < total_updates
        next_event_at = min(next_shift_at, total_updates)
        delta = next_event_at - update_count
        if delta > 0
            metropolis_step!(state, delta)
            update_count = next_event_at
        end
        if next_shift_at <= update_count && update_count < total_updates
            shift_chain!(state)
            next_shift_idx += 1
            if randshift
                gap = max(1, round(Int, randexp() * L_over_v))
                next_shift_at = update_count + gap
            else
                next_shift_at = round(Int, next_shift_idx * L_over_v)
            end
            if seed_droplet_size > 0 && _is_all_plus(state)
                seed_droplet!(state, seed_droplet_size)
            end
        end
    end
end

"""
    compute_total_magnetization(state)

Total magnetization per spin: m = (1/2L) Σ σ_i. Returns value in [-1, 1].
"""
function compute_total_magnetization(state::SimulationState)
    if state.single_layer
        return state.mag_sum / state.L
    else
        return state.mag_sum / (2 * state.L)
    end
end

"""
    compute_local_magnetization(state)

Site-by-site magnetization: m_i = σ_top[i] + σ_bottom[i]. Returns vector with values in {-2, 0, +2}.
"""
function compute_local_magnetization(state::SimulationState)
    if state.single_layer || state.top_offset == 0
        return state.top .+ state.bottom
    end
    # Offset-aware: at legacy slot i, top is stored at mod1(i − top_offset, L);
    # bottom is at i. Return the per-legacy-slot vertical pair sum.
    L = state.L
    off = state.top_offset
    top = state.top
    bottom = state.bottom
    out = Vector{Int8}(undef, L)
    @inbounds for i in 1:L
        j = i - off
        j = j < 1 ? j + L : j
        out[i] = top[j] + bottom[i]
    end
    return out
end

"""
    compute_energy(state)

Total energy per spin of the two-chain system, normalized so ferromagnetic ground state = 0.
H = -J Σ⟨i,j⟩ σ_i σ_j with J=1. Each bond counted once.
"""
function compute_energy(state::SimulationState)
    top = state.top
    bottom = state.bottom
    L = state.L
    off = state.top_offset
    E = 0
    @inbounds for i in 1:L
        ip = i == L ? 1 : i + 1
        E -= top[i] * top[ip]
        E -= bottom[i] * bottom[ip]
        # Cross bond at legacy slot i: top spin stored at mod1(i − off, L)
        # pairs with bottom[i]. Iterating i over storage indices and using the
        # SAME storage index for top while shifting bottom by +off is the
        # equivalent formulation (each bond counted exactly once).
        j = i + off
        j = j > L ? j - L : j
        E -= top[i] * bottom[j]
    end
    return E / (2 * L) + 1.5
end
