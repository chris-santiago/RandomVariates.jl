"""
    beta_rng(α, β, shape=1; seed=nothing)

Generate a `shape` element array of random variables from a Beta(`α`, `β`) distribution. Optionally you can set a specific seed.

# Notes

The Beta distribution is given:

``f(x,α,β) = \\frac{x^{α-1 (1-x)^{β-1}}}{Β(α,β)} \\quad x \\in [0,1]``

where ``Β(α,β) = \\frac{Γ(α)Γ(β)}{Γ(α+β)}``

# Examples

```julia-repl
julia> beta_rng(1,2)
1-element Vector{Float64}:
 0.44456674672905633

julia> beta_rng(1, 2, (2,2))
2×2 Matrix{Float64}:
 0.0792132  0.595657
 0.737615   0.649721

```

# References

D.P. Kroese, T. Taimre, Z.I. Botev. Handbook of Monte Carlo Methods. 
  Wiley Series in Probability and Statistics, John Wiley & Sons, New York, 2011.

Law, A. Simulation modeling and analysis, 5th Ed. McGraw Hill Education, Tuscon, 2013.
"""
function beta_rng(α::Real, β::Real, shape::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    seed_setter(seed)
    Y₁ = gamma_rng(α, 1, shape, seed=seed)
    Y₂ = gamma_rng(β, 1, shape, seed=seed)
    X = Y₁ ./ (Y₁ .+ Y₂)
    return X
end
