"""
    beta_rng(α, β, size=1; seed=nothing)

Generate a `size` element array of random variables from a Beta(`α`, `β`) distribution. Optionally you can set a specific seed.

# Examples

```julia-repl
julia> beta_rng(1,2)
1-element Vector{Float64}:
 0.44456674672905633

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
"""
function beta_rng(α::Real, β::Real, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    seed_setter(seed)
    Y₁ = gamma_rng(α, 1, size, seed=seed)
    Y₂ = gamma_rng(β, 1, size, seed=seed)
    X = Y₁ ./ (Y₁ .+ Y₂)
    return X
end
