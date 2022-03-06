"""
    get_gamma_prn(α, β; seed=nothing)

Generate a random variable from a Gamma(`α`, `β`) distribution. Optionally you can set a specific seed.

# Notes

This method cannot handle values of α < 1. Users should use [`gamma_rng`](@ref), instead.

# Examples

```julia-repl
julia> get_gamma_prn(1)
 0.11697493392617847

julia> get_gamma_prn(1, 2)
 0.09261426424752814
```

# References

D.P. Kroese, T. Taimre, Z.I. Botev. Handbook of Monte Carlo Methods. 
  Wiley Series in Probability and Statistics, John Wiley & Sons, New York, 2011.
"""
function get_gamma_prn(α::Real, β::Real=1; seed::Union{Int, Nothing}=nothing)
    seed_setter(seed)
    d = α - (1/3)
    c = 1 / sqrt(9*d)
    while true
        z = get_std_normal()[1]
        if z > -1/c
            v = (1 + c * z) ^ 3
            u = get_std_uniform()[1]
            if log(u) <= (0.5 * z^2 + d - (d * v + d * log(v)))
                return d * v / β
            end
        end
    end
end


"""
    gamma_rng(α, β, size=1; seed=nothing)

Generate a `size` element array of random variables from a Gamma(`α`, `β`) distribution. Optionally you can set a specific seed.

# Notes

The Gamma distribution is given:

``f(x,α,β) = \\frac{β^α x^{α-1} e^{-βx}}{Γ(α)} \\quad x ≥ 0``

# Examples

```julia-repl
julia> gamma_rng(1,1)
1-element Vector{Float64}:
 0.5190236735858542

julia> gamma_rng(1,1,4)
4-element Vector{Float64}:
 0.3035517926878862
 0.5765419737109622
 0.44121996206333797
 0.7325887616559309

julia> gamma_rng(1,1,(2,2))
2×2 Matrix{Float64}:
 0.228818  0.88849
 0.665729  1.01668
 
```

# References

D.P. Kroese, T. Taimre, Z.I. Botev. Handbook of Monte Carlo Methods. 
  Wiley Series in Probability and Statistics, John Wiley & Sons, New York, 2011.
"""
function gamma_rng(α::Real, β::Real=1, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    seed_setter(seed)
    if α < 1
        α += 1
        X = zeros(size)
        X .= get_gamma_prn.(α, β)
        U = get_std_uniform(size)
        X .*= U.^(1/α)
        return X
    end
    X = zeros(size)
    X .= get_gamma_prn.(α, β)
    return X
end
