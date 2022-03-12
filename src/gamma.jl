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
    gamma_rng(α, β, shape=1; seed=nothing)

Generate a `shape` element array of random variables from a Gamma(`α`, `β`) distribution. Optionally you can set a specific seed.

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

Law, A. Simulation modeling and analysis, 5th Ed. McGraw Hill Education, Tuscon, 2013.
"""
function gamma_rng(α::Real, β::Real=1, shape::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    seed_setter(seed)
    X = β .* get_gamma_rv(α, shape)
end


function get_gamma_rv(α::Real, shape::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    seed_setter(seed)
    if α < 0
        throw(ArgumentError("Not implemented for α < 0."))
    elseif α == 1
        return expon_rng(1, shape)
    elseif α < 1
        X = zeros(shape)
        X .= get_gamma_ad.(α)
        return X
    else 
        X = zeros(shape)
        X .= get_gamma_gb.(α)
        return X
    end
end


function get_gamma_ad(α::Real; max_iter=300)
    b = (ℯ + α)/ℯ
    i = 0
    while i < max_iter
        U₁ = get_std_uniform()[1]
        P = b*U₁
        if P > 1
            Y = -log((b-P)/α)
            U₂ = get_std_uniform()[1]
            if U₂ ≤ Y^(α-1)
                return Y
            end
        else
            Y = P^(1/α)
            U₂ = get_std_uniform()[1]
            if U₂ ≤ ℯ^(-Y)
                return Y
            end
        end
        i +=1
    end
end


function get_gamma_gb(α::Real; max_iter=300)
    a = 1 / sqrt(2*α - 1)
    b = α - log(4)
    q = α + 1/a
    θ = 4.5
    d = 1 + log(θ)
    i = 0
    while i < max_iter
        U₁, U₂ = get_std_uniform(2)
        V = a * log(U₁/(1-U₁))
        Y = α * ℯ^V
        Z = U₁^2 * U₂
        W = b + q * V - Y
        if W + d - θ * Z ≥ 0
            return Y
        elseif W ≥ log(Z)
            return Y
        end
        i += 1
    end
end
