"""
    expon_rng(λ, size=1; seed=nothing)

Generate a `size` element array of random variables from a Exponential(`λ`) distribution. Optionally you can set a specific seed.

# Notes

The pdf of an Exponential(λ) distribution is given as:

``f(x, λ) = λe^{-λx} \\quad x ≥ 0``

# Examples

```julia-repl
julia> expon_rng(3)
1-element Vector{Float64}:
 0.07033135663980515

julia> expon_rng(1.2, seed=42)
1-element Vector{Float64}:
 0.3296112244200808

julia> expon_rng(1.2, (2, 2))
2×2 Matrix{Float64}:
 1.9327    0.134739
 0.746861  0.155614
```

# References

D.P. Kroese, T. Taimre, Z.I. Botev. Handbook of Monte Carlo Methods. 
  Wiley Series in Probability and Statistics, John Wiley & Sons, New York, 2011.
"""
function expon_rng(λ::Real, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    U = get_std_uniform(size, seed=seed)
    X = (-1/λ) .* log.(1 .- U)  # could also use just U
    return X
end