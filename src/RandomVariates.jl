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
    seed_setter(seed::Union{Int, Nothing}=nothing)

Set a user defined seed, if given.
"""
function seed_setter(seed::Union{Int, Nothing}=nothing)
    if !isnothing(seed)
        set_user_seed(seed)  # only set user seed once so we get new seed in subsequent calls
    end
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


function check_p(p::Real)
    if (p > 1) || (p < 0)
        throw(ArgumentError("Parameter `p` must fall between 0 and 1."))
    end
end

"""
    get_std_normal(size=1, seed=nothing)

Generate a `size` element array of random variables from a standard Uniform(0,1) distribution.

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
    seed_setter(seed)
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
function bernoulli_rng(p::Real, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    check_p(p)
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
function geometric_rng(p::Real, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    check_p(p)
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
function binomial_rng(p::Real, n::Int, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    check_p(p)
    U = bernoulli_rng(p, (size..., n), seed=seed)
    X = sum(U, dims=ndims(U))  # want sum over final or `n` dimension
    return X
end


"""
    poisson_rng(p, n, size=1, seed=nothing)

Generate a `size` element array of random variables from a Poisson(`λ`) distribution.

# Examples

```julia-repl
julia> poisson_rng(3)
1×1 Matrix{Int64}:
 7
```

```julia-repl
julia> poisson_rng(10, 5)
5×1 Matrix{Int64}:
 13
 11
 10
  8
 15
```

```julia-repl
julia> poisson_rng(10, (5,5))
5×5×1 Array{Int64, 3}:
[:, :, 1] =
 11  15   9  11   9
  8  15  13  10   9
 11  12   4  10   6
  7   9  13  11   7
 13   7  10  10  14
```
"""
function poisson_rng(λ::Real, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    n = ceil(Int, λ*1e2)  # ensure n is integer
    U = expon_rng(λ, (size..., n), seed=seed)
    # X = sum(cumsum(U, dims=ndims(U)) .< 1, dims=length(size))
    X = sum(cumsum(U, dims=ndims(U)) .< 1, dims=ndims(U))
    return X
end


"""
    get_std_normal(size=1, seed=nothing)

Generate a `size` element array of random variables from a standard Normal(0, 1) distribution.

# Examples

```julia-repl
julia> get_std_normal()
1-element Vector{Float64}:
 0.6315076033452351
```

```julia-repl
julia> get_std_normal(5, seed=43)
5-element Vector{Float64}:
  1.2311463458421277
  1.7786409025309897
 -0.4178415161339713
  0.3518755172644067
 -0.16742990320047046
```

```julia-repl
julia> get_std_normal((2,2))
2×2 Matrix{Float64}:
 -0.900365   -0.432759
 -0.0350299   1.55754
```
"""
function get_std_normal(size::Int=1; seed::Union{Int, Nothing}=nothing)
    seed_setter(seed)
    a = sqrt.(-2 .* log.(get_std_uniform(size)))
    b = 2 * π .* get_std_uniform(size)
    A = a .* sin.(b)
    B = a .* cos.(b)
    X = collect(Iterators.flatten(zip(A, B)))[1:size]
    return X
end


function get_std_normal(size::Tuple{Vararg{Int}}; seed::Union{Int, Nothing}=nothing)
    X = get_std_normal(reduce(*, size), seed=seed)
    return reshape(X, size)
end


"""
    normal_rng(μ, σ², size=1, seed=nothing)

Generate a `size` element array of random variables from a Normal(`μ`, `σ²`) distribution.

# Examples

```julia-repl
julia> normal_rng()
1-element Vector{Float64}:
 0.03130435813519526
```

```julia-repl
julia> normal_rng(3, 9, 2)
2-element Vector{Float64}:
  7.362935421449054
 -1.0173543995738399
```

```julia-repl
julia> normal_rng(0,1,(2,2))
2×2 Matrix{Float64}:
 -0.640505   0.30303
 -0.0556832  0.714122
```
"""
function normal_rng(μ::Real=0, σ²::Real=1, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    X = get_std_normal(size, seed=seed) .* σ² .+ μ
    return X
end


"""
    get_gamma_prn(α, β, seed=nothing)

Generate a random variable from a Gamma(`α`, `β`) distribution.

# Notes

This method cannot handle values of α < 1. Users should use [`gamma_rng`](@ref), instead.

# Examples

```julia-repl
julia> get_gamma_prn(1)
0.11697493392617847
```

```julia-repl
julia> get_gamma_prn(1, 2)
0.09261426424752814
```
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
    gamma_rng(α, β, size=1, seed=nothing)

Generate a `size` element array of random variables from a Gamma(`α`, `β`) distribution.

# Examples

```julia-repl
julia> gamma_rng(1,1)
1-element Vector{Float64}:
 0.5190236735858542
```

```julia-repl
julia> gamma_rng(1,1,4)
4-element Vector{Float64}:
 0.3035517926878862
 0.5765419737109622
 0.44121996206333797
 0.7325887616559309
```

```julia-repl
julia> gamma_rng(1,1,(2,2))
2×2 Matrix{Float64}:
 0.228818  0.88849
 0.665729  1.01668
```
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


"""
    beta_rng(α, β, size=1, seed=nothing)

Generate a `size` element array of random variables from a Beta(`α`, `β`) distribution.

# Examples

```julia-repl
julia> beta_rng(1,2)
1-element Vector{Float64}:
 0.44456674672905633
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
function beta_rng(α::Real, β::Real, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    seed_setter(seed)
    Y₁ = gamma_rng(α, 1, size, seed=seed)
    Y₂ = gamma_rng(β, 1, size, seed=seed)
    X = Y₁ ./ (Y₁ .+ Y₂)
    return X
end

# End of Module
end
