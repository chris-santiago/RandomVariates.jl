module RandomVariates

export uniform_rng, expon_rng, erlang_rng, bernoulli_rng, binomial_rng

using Dates

ENV["JULIA_SEED"] = Dates.value(Dates.now())

A = 16807
MOD = 2^31 - 1


function set_seed(seed::Int)
    ENV["JULIA_SEED"] = seed
end


function get_seed()
	return parse(Int, ENV["JULIA_SEED"])
end


function gen_prn(seed=nothing)
    if isnothing(seed)
        seed = get_seed()
    else
        seed *= 73965
    end
	prn = mod(A * seed, MOD)
    set_seed(prn)
    return prn
end


function get_uniform(n, seed=nothing)
    U = [gen_prn(seed) for i in 1:n]  # this doesn't work for user-set seeds
    U = U./MOD
	# U = zeros(n)
	# x = gen_prn(seed)
	# U[1] = x/MOD
	# for i in 2:n
	# 	x = gen_prn(x)
	# 	U[i] = x/MOD
	# end
	return U
end


function uniform_rng(a, b, size=1, seed=nothing)
    U = get_uniform(size, seed)
    X = a .+ (b-a) .* U
    return X
end


function expon_rng(λ, size=1, seed=nothing)
    U = get_uniform(size, seed)
    X = -λ .* log.(1 .- U)  # could also use just U
    return X
end


function erlang_rng(k, λ, size=1, seed=nothing)
    U = zeros(k, size)
    for i in 1:k
        U[i, :] = get_uniform(size, seed)
    end
    # X = (-λ/k) .* log.(prod(U, dims=1))
    X = (-1/λ) .* log.(prod(U, dims=1))  # Here (-1/λ) represents mean
    return X
end


function bernoulli_rng(p, size=1, seed=nothing)
    U = get_uniform(size, seed)
    X = (1 - p) .<= U
    return X
end


function binomial_rng(p, n, size=1, seed=nothing)
    U = zeros(size, n)
    for i in 1:size
        U[i, :] = bernoulli_rng(p, n, seed)
    end
    X = sum(U, dims=2)
    return X
end

# use Base.Iterators.takewhile
# function poisson_rng(λ, size=1, seed=nothing)
#     X = 0
#     while cumsum(X) <= 1


# End of Module
end
