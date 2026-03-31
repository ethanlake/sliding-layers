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
        @inbounds for _ in 1:n_updates
            chain = rand(1:2)
            i = rand(1:L)
            ip = i == L ? 1 : i + 1
            im = i == 1 ? L : i - 1
            if chain == 1
                σ = top[i]
                neighbors_sum = top[im] + top[ip] + bottom[i]
            else
                σ = bottom[i]
                neighbors_sum = bottom[im] + bottom[ip] + top[i]
            end
            σ_idx = (σ + Int8(1)) >> 1 + 1        # -1→1, +1→2
            ns_idx = (neighbors_sum + 5) >> 1      # -3→1, -1→2, 1→3, 3→4
            if rand() < table[σ_idx, ns_idx]
                if chain == 1
                    top[i] = -σ
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
        @inbounds for _ in 1:n_updates
            chain = rand(1:2)
            i = rand(1:L)
            ip = i == L ? 1 : i + 1
            im = i == 1 ? L : i - 1
            if chain == 1
                σ = top[i]
                neighbors_sum = top[im] + top[ip] + bottom[i]
            else
                σ = bottom[i]
                neighbors_sum = bottom[im] + bottom[ip] + top[i]
            end
            σ_idx = (σ + Int8(1)) >> 1 + 1
            ns_idx = (neighbors_sum + 5) >> 1
            if rand() < table[σ_idx, ns_idx]
                if chain == 1
                    top[i] = -σ
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

Cyclically shift the top chain right by 1 position, in-place (zero allocations).
"""
function shift_chain!(state::SimulationState)
    top = state.top
    L = state.L
    @inbounds begin
        last = top[L]
        for i in L:-1:2
            top[i] = top[i-1]
        end
        top[1] = last
    end
end

"""
    evolve_and_measure!(state, T_steps, on_measure) -> Int

Run the simulation for up to T_steps time units, calling `on_measure(step)` at each
integer time. `on_measure` should return `true` to continue or `false` to stop early.
Returns the number of completed measurement steps.

Time convention:
- 1 time unit = L single-spin MC updates (independent of v)
- For v > 0, one chain is shifted by 1 lattice site every round(Int, L/v) MC updates,
  so there are v shifts per time unit and each site moves v times between updates.
- Measurements occur at every integer time (every L MC updates).
"""
function evolve_and_measure!(state::SimulationState, T_steps::Int, on_measure::Function)
    v = state.v
    L = state.L

    if v == 0.0
        for step in 1:T_steps
            metropolis_step!(state, L)
            if !on_measure(step)
                return step
            end
        end
        return T_steps
    end

    updates_per_shift = state.n_sub  # round(Int, L/v)
    measurement_count = 0
    updates_since_last_measure = 0

    while measurement_count < T_steps
        shift_chain!(state)

        remaining = updates_per_shift
        while remaining > 0
            to_next_measure = L - updates_since_last_measure
            batch = min(remaining, to_next_measure)
            metropolis_step!(state, batch)
            updates_since_last_measure += batch
            remaining -= batch

            if updates_since_last_measure >= L
                measurement_count += 1
                updates_since_last_measure = 0
                if !on_measure(measurement_count) || measurement_count >= T_steps
                    return measurement_count
                end
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
function evolve_and_measure_heat!(state::SimulationState, T_steps::Int, on_measure::Function)
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

    updates_per_shift = state.n_sub
    measurement_count = 0
    updates_since_last_measure = 0
    heat_this_step = 0.0

    while measurement_count < T_steps
        shift_chain!(state)

        remaining = updates_per_shift
        while remaining > 0
            to_next_measure = L - updates_since_last_measure
            batch = min(remaining, to_next_measure)
            heat_this_step += metropolis_step_heat!(state, batch)
            updates_since_last_measure += batch
            remaining -= batch

            if updates_since_last_measure >= L
                measurement_count += 1
                updates_since_last_measure = 0
                if !on_measure(measurement_count, heat_this_step / (2 * L)) || measurement_count >= T_steps
                    return measurement_count
                end
                heat_this_step = 0.0
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
function evolve_time!(state::SimulationState, T::Int)
    v = state.v
    L = state.L

    if v == 0.0
        metropolis_step!(state, T * L)
        return
    end

    updates_per_shift = state.n_sub
    n_shifts = round(Int, T * v)

    for _ in 1:n_shifts
        shift_chain!(state)
        metropolis_step!(state, updates_per_shift)
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
    return state.top .+ state.bottom
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
    E = 0
    @inbounds for i in 1:L
        ip = i == L ? 1 : i + 1
        E -= top[i] * top[ip]
        E -= bottom[i] * bottom[ip]
        E -= top[i] * bottom[i]
    end
    return E / (2 * L) + 1.5
end
