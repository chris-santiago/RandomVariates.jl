module RandomVariates

export SEED, A, MOD, uniform_rng, expon_rng, erlang_rng, bernoulli_rng, 
binomial_rng, poisson_rng, normal_rng, gamma_rng, weibull_rng, geometric_rng

using Dates

global SEED = Dates.value(Dates.now())  # Use current epoch time as default seed

# using POSIX params for LCG
# https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use
const A = 25214903917
const C = 11
const MOD = 2^48

"""
    set_seed(seed::Int)

Set the global `SEED` variable.
"""
function set_seed(seed::Int)
    global SEED = seed
end


"""
    set_user_seed(seed::Int)

Set a user-defined seed as global `SEED` variable.
"""
function set_user_seed(seed::Int)
    global SEED = seed * 7856209
end


"""
    get_seed()

Get the global `SEED` variable.
"""
function get_seed()
	return SEED
end


"""
    gen_prn()

Generate a pseudorandom number.

# Notes

Uses a linear congruential generator (LCG) with [POSIX parameters](https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use):

``X_n = 25214903917 X_{n-1} + 11 \\quad mod \\quad 2^{48}``
"""
function gen_prn()
    seed = get_seed()
	prn = mod(A * seed + C, MOD)
    set_seed(prn)
    return prn
end


"""
    get_std_normal(size=1, seed=nothing)

Generate a `size` element array of random variables from a Uniform(0,1) distribution.

# Examples

```julia-repl
julia> get_std_uniform()
1-element Vector{Float64}:
 0.42443098343863284
```

```julia-repl
julia> get_std_uniform(seed=43)
1-element Vector{Float64}:
 0.09636209187468836
```

```julia-repl
julia> get_std_uniform(5)
5-element Vector{Float64}:
 0.6584669595802204
 0.33437978955868886
 0.509019330923099
 0.12156905126458639
 0.917393216014684
```
"""
function get_std_uniform(size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    if !isnothing(seed)
        set_user_seed(seed)
    end
    U = zeros(size)  # preallocate array
    U .= gen_prn.()  # vectorize assignment for efficiency
    U = U./MOD
	return U
end


"""
    uniform_rng(a, b, size=1, seed=nothing)

Generate a `size` element array of random variables from a Uniform(`a`, `b`) distribution.

# Examples

```julia-repl
julia> uniform_rng(1, 6)
1-element Vector{Float64}:
 2.638331960912094
```

```julia-repl
julia> uniform_rng(1, 6, seed=42)
1-element Vector{Float64}:
 2.6333962626438314
```

```julia-repl
julia> uniform_rng(0, 1, (4,4))
4×4 Matrix{Float64}:
 0.640603   0.757195  0.325722  0.645452
 0.955188   0.155203  0.953206  0.0046541
 0.0923526  0.490721  0.451705  0.516445
 0.661619   0.527063  0.212847  0.832298
 ```
"""
function uniform_rng(a::Real=0, b::Real=1, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    U = get_std_uniform(size, seed=seed)
    X = a .+ (b-a) .* U
    return X
end


"""
    expon_rng(λ, size=1, seed=nothing)

Generate a `size` element array of random variables from a Exponential(`λ`) distribution.

# Examples

```julia-repl
julia> expon_rng(3)
1-element Vector{Float64}:
 0.07033135663980515
```

```julia-repl
julia> expon_rng(1.2, seed=42)
1-element Vector{Float64}:
 0.3296112244200808
```

```julia-repl
julia> expon_rng(1.2, (2, 2))
2×2 Matrix{Float64}:
 1.9327    0.134739
 0.746861  0.155614
```
"""
function expon_rng(λ::Real, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    U = get_std_uniform(size, seed=seed)
    X = (-1/λ) .* log.(1 .- U)  # could also use just U
    return X
end


"""
    erlang_rng(k, λ, size=1, seed=nothing)

Generate a `size` element array of random variables from a Erlang_{`k`}(`λ`) distribution.

# Examples

```julia-repl
julia> erlang_rng(5, .5)
1-element Vector{Float64}:
 10.803989701023117
```

```julia-repl
julia> erlang_rng(3, 1, (2,2))
2×2×1 Array{Float64, 3}:
[:, :, 1] =
 2.19956  4.18505
 5.46892  2.5633
```
"""
function erlang_rng(k::Int, λ::Real, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    U = get_std_uniform((size..., k), seed=seed)
    k_dim = ndims(U)  # k is final dimension
    X = (-1/λ) .* log.(prod(U, dims=k_dim))  # Here (-1/λ) represents mean
    return X
end


"""
    weibull_rng(λ, β, size=1, seed=nothing)

Generate a `size` element array of random variables from a Weibull(`λ`, `β`) distribution.

# Examples

```julia-repl
julia> weibull_rng(2, 2, seed=42)
1-element Vector{Float64}:
 0.31445725834527055
```

```julia-repl
julia> weibull_rng(2, 2, 2)
2-element Vector{Float64}:
 0.39561285703154575
 0.6021921673483441
```

```julia-repl
julia> weibull_rng(2, 2, (2,2))
2×2 Matrix{Float64}:
 0.428896  0.109897
 0.812854  0.427906
```

"""
function weibull_rng(λ::Real, β::Real, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    U = get_std_uniform(size, seed=seed)
    X = (1/λ) .* (-log.(1 .- U)) .^ (1/β)
    return X
end


"""
    bernoulli_rng(p, size=1, seed=nothing)

Generate a `size` element array of random variables from a Bernoulli(`p`) distribution.

# Examples

```julia-repl
julia> bernoulli_rng(.34)
1-element BitVector:
 0
```

```julia-repl
julia> bernoulli_rng(.34, 5)
5-element BitVector:
 0
 0
 1
 0
 1
```

```julia-repl
julia> bernoulli_rng(.8, (2,2), seed=42)
2×2 BitMatrix:
 1  0
 0  1
```

"""
function bernoulli_rng(p::AbstractFloat, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    U = get_std_uniform(size, seed=seed)
    X = (1 - p) .<= U
    return X
end


"""
    geometric_rng(p, size=1, seed=nothing)

Generate a `size` element array of random variables from a Geometric(`p`) distribution.

# Examples

```julia-repl
julia> geometric_rng(.8)
1-element Vector{Int64}:
 1
```

```julia-repl
julia> geometric_rng(.8, 5)
5-element Vector{Int64}:
 2
 3
 1
 1
 1
```

```julia-repl
julia> geometric_rng(.8, (2,2), seed=45)
2×2 Matrix{Int64}:
 1  1
 1  1
```
"""
function geometric_rng(p::AbstractFloat, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    U = get_std_uniform(size, seed=seed)
    X = ceil.(Int, log.(1 .- U) ./ log(1 - p))
    return X
end


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
function binomial_rng(p::AbstractFloat, n::Int, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    U = bernoulli_rng(p, (size..., n), seed=seed)
    X = sum(U, dims=ndims(U))  # want sum over final or `n` dimension
    return X
end


function poisson_rng(λ::Real, size::Int=1; seed::Union{Int, Nothing}=nothing)
    X = zeros(Int, size)
    for i in 1:size
        X[i] = sum(cumsum(expon_rng(λ, λ*1e2, seed=seed)) .< 1)
    end
    return X
end


function get_std_normal(size::Int=1; seed::Union{Int, Nothing}=nothing)
    a = sqrt.(-2 .* log.(get_std_uniform(size, seed=seed)))
    b = 2 * π .* get_std_uniform(size, seed=seed)
    A = a .* sin.(b)
    B = a .* cos.(b)
    X = collect(Iterators.flatten(zip(A, B)))[1:size]
    return X
end


function normal_rng(μ::Real=0, σ²::Real=1, size::Int=1; seed::Union{Int, Nothing}=nothing)
    X = get_std_normal(size, seed=seed) .* σ² .+ μ
    return X
end


function get_gamma_prn(α::Real, β=1; seed::Union{Int, Nothing}=nothing)
    if α < 1
        u = get_std_uniform(seed=seed)[1]
        x = get_gamma_prn(α+1, β)
        x *= u^(1/α)
        return x
    end
    d = α - (1/3)
    c = 1 / sqrt(9*d)
    z = get_std_normal(seed=seed)[1]
    while true
        v = 0
        while v <= 0
            v = (1 + c * z) ^ 3
        end
        u = get_std_uniform(seed=seed)[1]
        if u < 1 - 0.331 * z^4
            return d * v / β
        end
        if log(u) < 0.5 * z^2 + d * (1 - v + log(v))
            return d * v / β
        end
    end
end


function gamma_rng(α::Real, β=1, size::Int=1; seed::Union{Int, Nothing}=nothing)
    X = zeros(size)
    X .= get_gamma_prn.(α, β, seed=seed)
    return X
end


function neg_binomial_rng()
    throw(error("Not Implemented"))
end


function beta_rng()
    throw(error("Not Implemented"))
end

# End of Module
end
