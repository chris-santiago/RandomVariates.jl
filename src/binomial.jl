"""
    binomial_rng(p, n, size=1; seed=nothing)

Generate a `size` element array of random variables from a Binomial(`p`, `n`) distribution. Optionally you can set a specific seed.

# Notes

The Binomial(x, n, p) distribution describes the total number of successes in a sequence of n Bernoulli(p) trials.

The pdf is given:

``f(x,n,p) = \\binom{n}{x} p^x (1-p)^{n-x} \\quad x = 0,1,\\dots, n``

# Examples

```julia-repl
julia> binomial_rng(.3, 10)
1×1 Matrix{Int64}:
 3

julia> binomial_rng(.3, 10, (2,2))
2×2×1 Array{Int64, 3}:
[:, :, 1] =
 2  1
 2  2
```

# References

D.P. Kroese, T. Taimre, Z.I. Botev. Handbook of Monte Carlo Methods. 
  Wiley Series in Probability and Statistics, John Wiley & Sons, New York, 2011.
"""
function binomial_rng(p::Real, n::Int, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    check_p(p)
    U = bernoulli_rng(p, (size..., n), seed=seed)
    X = sum(U, dims=ndims(U))  # want sum over final or `n` dimension
    return X
end
