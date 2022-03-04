"""
    weibull_rng(λ, β, size=1; seed=nothing)

Generate a `size` element array of random variables from a Weibull(`λ`, `β`) distribution. Optionally you can set a specific seed.

# Examples

```julia-repl
julia> weibull_rng(2, 2, seed=42)
1-element Vector{Float64}:
 0.31445725834527055

julia> weibull_rng(2, 2, 2)
2-element Vector{Float64}:
 0.39561285703154575
 0.6021921673483441

julia> weibull_rng(2, 2, (2,2))
2×2 Matrix{Float64}:
 0.428896  0.109897
 0.812854  0.427906
 
```
"""
function weibull_rng(λ::Real, β::Real, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    U = get_std_uniform(size, seed=seed)
    X = (1/λ) .* (-log.(1 .- U)) .^ (1/β)
    return X
end
