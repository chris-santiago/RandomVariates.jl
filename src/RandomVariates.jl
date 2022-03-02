module RandomVariates

export SEED, A, MOD, uniform_rng, expon_rng, erlang_rng, bernoulli_rng, binomial_rng, poisson_rng, normal_rng

using Dates

SEED = Dates.value(Dates.now())  # Use current epoch time as default seed

# using POSIX params for LCG
# https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use
const A = 25214903917
const C = 11
const MOD = 2^48


function set_seed(seed::Int)
    global SEED = seed
end


function set_user_seed(seed::Int)
    global SEED = seed * 7856209
end


function get_seed()
	return SEED
end


function gen_prn()
    seed = get_seed()
	prn = mod(A * seed + C, MOD)
    set_seed(prn)
    return prn
end


function get_std_uniform(size=1; seed=nothing)
    if !isnothing(seed)
        set_user_seed(seed)
    end
    U = zeros(size)  # preallocate array
    U .= gen_prn.()  # vectorize assignment for efficiency
    U = U./MOD
	return U
end


function uniform_rng(a, b, size=1; seed=nothing)
    U = get_std_uniform(size, seed=seed)
    X = a .+ (b-a) .* U
    return X
end


function expon_rng(λ, size=1; seed=nothing)
    U = get_std_uniform(size, seed=seed)
    X = (-1/λ) .* log.(1 .- U)  # could also use just U
    return X
end


function erlang_rng(k, λ, size=1; seed=nothing)
    U = zeros(k, size)
    for i in 1:k
        U[i, :] = get_std_uniform(size, seed=seed)
    end
    # X = (-λ/k) .* log.(prod(U, dims=1))
    X = (-1/λ) .* log.(prod(U, dims=1))  # Here (-1/λ) represents mean
    return X
end


function bernoulli_rng(p, size=1; seed=nothing)
    U = get_std_uniform(size, seed=seed)
    X = (1 - p) .<= U
    return X
end


function binomial_rng(p, n, size=1; seed=nothing)
    U = zeros(Int, size, n)
    for i in 1:size
        U[i, :] = bernoulli_rng(p, n, seed=seed)
    end
    X = sum(U, dims=2)
    return X
end


function poisson_rng(λ, size=1; seed=nothing)
    X = zeros(Int, size)
    for i in 1:size
        X[i] = sum(cumsum(expon_rng(λ, λ*1e2, seed=seed)) .< 1)
    end
    return X
end


function get_std_normal(size=1; seed=nothing)
    a = sqrt.(-2 .* log.(get_std_uniform(size, seed=seed)))
    b = 2 * π .* get_std_uniform(size, seed=seed)
    A = a .* sin.(b)
    B = a .* cos.(b)
    X = collect(Iterators.flatten(zip(A, B)))[1:size]
    return X
end


function normal_rng(μ=0, σ²=1, size=1; seed=nothing)
    X = get_std_normal(size, seed=seed) .* σ² .+ μ
    return X
end


# End of Module
end
