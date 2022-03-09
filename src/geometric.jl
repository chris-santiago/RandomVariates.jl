"""
    geometric_rng(p, shape=1; seed=nothing)

Generate a `shape` element array of random variables from a Geometric(`p`) distribution. Optionally you can set a specific seed.

# Notes

The Geometric distribution is given:

``f(x,p) = (1-p)^{x-1}p) \\quad x = 1,2,3,…``

where ``0 ≤ p ≤ 1``

# Examples

```julia-repl
julia> geometric_rng(.8)
1-element Vector{Int64}:
 1

julia> geometric_rng(.8, 5)
5-element Vector{Int64}:
 2
 3
 1
 1
 1

julia> geometric_rng(.8, (2,2), seed=45)
2×2 Matrix{Int64}:
 1  1
 1  1
 
```

# References

D.P. Kroese, T. Taimre, Z.I. Botev. Handbook of Monte Carlo Methods. 
  Wiley Series in Probability and Statistics, John Wiley & Sons, New York, 2011.

C. Alexopoulos, D. Goldsman. Random variate generation. 2020.
"""
function geometric_rng(p::Real, shape::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    check_p(p)
    c = 1 - p
    U = get_std_uniform(shape, seed=seed)
    X = ceil.(Int, log.(1 .- U) ./ log(c))
    return X
end
