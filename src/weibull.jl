"""
    weibull_rng(λ, β, shape=1; seed=nothing)

Generate a `shape` element array of random variables from a Weibull(`λ`, `β`) distribution. Optionally you can set a specific seed.

# Notes

The pdf of an Weibull(λ, β) distribution is given as:

``f(x, λ, β) = λ e^{-λx^β} \\quad x ≥ 0``

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

# References

C. Alexopoulos, D. Goldsman. Random variate generation. 2020.

Law, A. Simulation modeling and analysis, 5th Ed. McGraw Hill Education, Tuscon, 2013.
"""
function weibull_rng(λ::Real, β::Real, shape::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    U = get_std_uniform(shape, seed=seed)
    X = (1/λ) .* (-log.(1 .- U)) .^ (1/β)
    return X
end
