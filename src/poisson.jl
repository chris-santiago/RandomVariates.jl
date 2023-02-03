"""
    poisson_rng(p, n, shape=1; seed=nothing)

Generate a `shape` element array of random variables from a Poisson(`λ`) distribution. Optionally you can set a specific seed.

# Notes

The Poisson distribution is given by:

``f(x, λ) = \\frac{λ^x}{x!} e^λ \\quad x = 0,1,2,…``

# Examples

```julia-repl
julia> poisson_rng(3)
1×1 Matrix{Int64}:
 7

julia> poisson_rng(10, 5)
5×1 Matrix{Int64}:
 13
 11
 10
  8
 15

julia> poisson_rng(10, (5,5))
5×5×1 Array{Int64, 3}:
[:, :, 1] =
 11  15   9  11   9
  8  15  13  10   9
 11  12   4  10   6
  7   9  13  11   7
 13   7  10  10  14
 
```

# References

R. Larson, A. Odoni. Urban operations research. Prentice-Hall, New Jersey, 1981.

G. Last, M. Penrose. Lectures on the poisson process. Cambridge University Press, 2017.
"""
function poisson_rng(λ::Real, shape::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    n = ceil(Int, λ*5)  # ensure n is integer
    U = expon_rng(λ, (shape..., n), seed=seed)
    X = sum(cumsum(U, dims=ndims(U)) .< 1, dims=ndims(U))
    return X
end
