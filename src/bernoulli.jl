"""
    bernoulli_rng(p, size=1; seed=nothing)

Generate a `size` element array of random variables from a Bernoulli(`p`) distribution. Optionally you can set a specific seed.

# Notes

The pdf of the Bernoulli Distribution is given:

``f(x, p) = p^x (1-p)^{1-x} \\quad x \\in [0,1]``

# Examples

```julia-repl
julia> bernoulli_rng(.34)
1-element BitVector:
 0

julia> bernoulli_rng(.34, 5)
5-element BitVector:
 0
 0
 1
 0
 1

julia> bernoulli_rng(.8, (2,2), seed=42)
2×2 BitMatrix:
 1  0
 0  1
```

# References

D.P. Kroese, T. Taimre, Z.I. Botev. Handbook of Monte Carlo Methods. 
  Wiley Series in Probability and Statistics, John Wiley & Sons, New York, 2011.
"""
function bernoulli_rng(p::Real, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    check_p(p)
    U = get_std_uniform(size, seed=seed)
    X = U .< p
    return X
end
