"""
    build_acceptance_table(beta, h)

Precompute Metropolis acceptance probabilities for all possible (σ, neighbors_sum) combinations.

For the two-chain Ising model, each spin has exactly 3 neighbors (2 in-chain + 1 inter-chain),
each in {-1, +1}. So neighbors_sum ∈ {-3, -1, 1, 3} and σ ∈ {-1, +1}, giving 8 possible
values of dE = 2σ(neighbors_sum + h). We precompute min(1.0, exp(-β·dE)) for each.

Index mapping:
- σ_idx:  -1 → 1, +1 → 2   via (σ + 1) >> 1 + 1
- ns_idx: -3 → 1, -1 → 2, 1 → 3, 3 → 4   via (neighbors_sum + 5) >> 1
"""
function build_acceptance_table(beta::Float64, h::Float64)
    table = Matrix{Float64}(undef, 2, 4)
    for (s_idx, σ) in enumerate(Int8[-1, 1])
        for (n_idx, ns) in enumerate([-3, -1, 1, 3])
            dE = 2 * σ * (ns + h)
            table[s_idx, n_idx] = dE <= 0 ? 1.0 : exp(-beta * dE)
        end
    end
    return table
end

"""
    build_acceptance_table_single(beta, h)

Precompute Metropolis acceptance probabilities for a single-chain Ising model.
Each spin has 2 neighbors (left + right), so neighbors_sum ∈ {-2, 0, 2}.

Index mapping:
- σ_idx:  -1 → 1, +1 → 2   via (σ + 1) >> 1 + 1
- ns_idx: -2 → 1,  0 → 2, 2 → 3   via (neighbors_sum + 4) >> 1
"""
function build_acceptance_table_single(beta::Float64, h::Float64)
    table = Matrix{Float64}(undef, 2, 3)
    for (s_idx, σ) in enumerate(Int8[-1, 1])
        for (n_idx, ns) in enumerate([-2, 0, 2])
            dE = 2 * σ * (ns + h)
            table[s_idx, n_idx] = dE <= 0 ? 1.0 : exp(-beta * dE)
        end
    end
    return table
end

"""
    SimulationState

Holds all mutable state for an Ising simulation, including precomputed
quantities for performance (acceptance table, n_sub).
When single_layer=true, only the top chain is used (1D Ising model).
"""
mutable struct SimulationState
    top::Vector{Int8}
    bottom::Vector{Int8}
    L::Int
    beta::Float64
    h::Float64
    v::Float64
    n_sub::Int
    acceptance_table::Matrix{Float64}
    single_layer::Bool
    mag_sum::Int  # running sum of all spins (top + bottom), for O(1) magnetization
end

"""
    SimulationState(L, beta, h, v)

Construct a two-chain SimulationState with all spins initialized to +1.
"""
function SimulationState(L::Int, beta::Float64, h::Float64, v::Float64)
    top = ones(Int8, L)
    bottom = ones(Int8, L)
    n_sub = v == 0.0 ? L : round(Int, L / v)
    acceptance_table = build_acceptance_table(beta, h)
    return SimulationState(top, bottom, L, beta, h, v, n_sub, acceptance_table, false, 2L)
end

"""
    SimulationState(L, beta, h; single_layer=true)

Construct a single-chain SimulationState with all spins initialized to +1.
No sliding (v=0), bottom chain unused.
"""
function SimulationState(L::Int, beta::Float64, h::Float64)
    top = ones(Int8, L)
    bottom = ones(Int8, L)
    acceptance_table = build_acceptance_table_single(beta, h)
    return SimulationState(top, bottom, L, beta, h, 0.0, L, acceptance_table, true, L)
end

"""
    recompute_mag_sum!(state)

Recompute the running magnetization sum from the spin arrays.
Call this after any direct modification of state.top or state.bottom.
"""
function recompute_mag_sum!(state::SimulationState)
    if state.single_layer
        state.mag_sum = sum(state.top)
    else
        state.mag_sum = sum(state.top) + sum(state.bottom)
    end
end
