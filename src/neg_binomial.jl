"""
    get_neg_binomial_prn(p, r, shape=1; seed=nothing)

Generate a random variable from a Negative Binomial(`p`, `r`) distribution. Optionally you can set a specific seed.

# Examples

```julia-repl
julia> get_neg_binomial_prn(.2, 2)
 16

julia> get_neg_binomial_prn(.2, 2, seed=42)
 6

```

# References

Walk, C. Handbook on statistical distributions for experimentalists. 2007.
"""
function get_neg_binomial_prn(p::Real, r::Int; seed::Union{Int, Nothing}=nothing)
    check_p(p)
    n = convert(Int, r/p * 10)
    U = bernoulli_rng(p, n, seed=seed)
    X = sum(cumsum(U, dims=1) .< r) + 1   # TODO CHECK THIS AGAINST ANOTHER METHOD (e.g. convolution)
    return X
end


"""
    neg_binomial_rng(p, r, shape=1; seed=nothing)

Generate a `shape` element array of random variables from a Negative Binomial(`p`, `r`) distribution. Optionally you can set a specific seed.

# Notes

The Negative Binomial distribution is given:

``f(x,p,r) = \\binom{x-1}{r-1} (1-p)^{x-r} p^r \\quad x = 0,1,\\dots, n``

# Examples

```julia-repl
julia> neg_binomial_rng(.5, 2)
1-element Vector{Float64}:
 3.0

julia> neg_binomial_rng(.5, 5, 5)
5-element Vector{Float64}:
  8.0
 10.0
  8.0
 13.0
 10.0

julia> neg_binomial_rng(.5, 2, (2,2))
2×2 Matrix{Float64}:
 3.0  4.0
 4.0  2.0
 
```

# References

Walk, C. Handbook on statistical distributions for experimentalists. 2007.
"""
function neg_binomial_rng(p::Real, r::Int, shape::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    check_p(p)
    X = zeros(shape)
    X .= get_neg_binomial_prn.(p, r, seed=seed)
    return X
end


"""
    conv_neg_binomial_rng(p, r, shape=1; seed=nothing)

Generate a `shape` element array of random variables from a Negative Binomial(`p`, `r`) distribution. Optionally you can set a specific seed.

# Notes

The Negative Binomial distribution is given:

``f(x,p,r) = \\binom{x-1}{r-1} (1-p)^{x-r} p^r \\quad x = 0,1,\\dots, n``

Uses a convolution algorithm to generate random variables, which is slightly slower than [`neg_binomial_rng`](@ref).

# Examples

```julia-repl
julia> conv_neg_binomial_rng(.4, 5, 1)
1×1 Matrix{Int64}:
 8

julia> conv_neg_binomial_rng(.4, 5, 5)
5×1 Matrix{Int64}:
 11
 14 
 8
 10
 13

julia> conv_neg_binomial_rng(.4, 5, (2,2))
2×2×1 Array{Int64, 3}:
[:, :, 1] =
 20  11
 7  10
```

# References

Law, A. Simulation modeling and analysis, 5th Ed. McGraw Hill Education, Tuscon, 2013.
"""
function conv_neg_binomial_rng(p::Real, r::Int, shape::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    check_p(p)
    Y = geometric_rng(p, (shape..., r), seed=seed)
    X = sum(Y, dims=ndims(Y))
    return X
end
