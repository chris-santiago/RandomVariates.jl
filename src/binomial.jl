"""
    binomial_rng(p, n, size=1, seed=nothing)

Generate a `size` element array of random variables from a Binomial(`p`, `n`) distribution.

# Examples

```julia-repl
julia> binomial_rng(.3, 10)
1×1 Matrix{Int64}:
 3
```

```julia-repl
julia> binomial_rng(.3, 10, (2,2))
2×2×1 Array{Int64, 3}:
[:, :, 1] =
 2  1
 2  2
```
"""
function binomial_rng(p::Real, n::Int, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    check_p(p)
    U = bernoulli_rng(p, (size..., n), seed=seed)
    X = sum(U, dims=ndims(U))  # want sum over final or `n` dimension
    return X
end
