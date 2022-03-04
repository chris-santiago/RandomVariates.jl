"""
    get_neg_binomial_prn(α, β, size=1, seed=nothing)

Generate a random variable from a Negative Binomial(`p`, `r`) distribution.

# Examples

```julia-repl
julia> get_neg_binomial_prn(.2, 2)
16
```

```julia-repl
julia> get_neg_binomial_prn(.2, 2, seed=42)
6
```
"""
function get_neg_binomial_prn(p::Real, r::Int; seed::Union{Int, Nothing}=nothing)
    check_p(p)
    U = bernoulli_rng(p, 1000, seed=seed)
    X = sum(cumsum(U, dims=1) .< r) + 1
    return X
end


"""
    neg_binomial_rng(α, β, size=1, seed=nothing)

Generate a `size` element array of random variables from a Negative Binomial(`p`, `r`) distribution.

# Examples

```julia-repl
julia> neg_binomial_rng(.5, 2)
1-element Vector{Float64}:
 3.0
```

```julia-repl
julia> neg_binomial_rng(.5, 5, 5)
5-element Vector{Float64}:
  8.0
 10.0
  8.0
 13.0
 10.0
```

```julia-repl
julia> neg_binomial_rng(.5, 2, (2,2))
2×2 Matrix{Float64}:
 3.0  4.0
 4.0  2.0
```
"""
function neg_binomial_rng(p::Real, r::Int, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    check_p(p)
    X = zeros(size)
    X .= get_neg_binomial_prn.(p, r, seed=seed)
    return X
end