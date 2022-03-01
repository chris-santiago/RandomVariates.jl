module RandomVariates

export uniform_rng

using Dates

A = 16807
MOD = 2^31 - 1


function get_seed(seed::Nothing)
	return Dates.value(Dates.now())
end


function get_seed(seed::Int)
	return seed
end


function gen_prn(seed=nothing)
	seed = get_seed(seed)
	return mod(A * seed, MOD)
end


function get_uniform(n, seed=nothing)
	U = zeros(n)
	x = gen_prn(seed)
	U[1] = x/MOD
	for i in 2:n
		x = gen_prn(x)
		U[i] = x/MOD
	end
	return U
end


function uniform_rng(a, b, n=1, seed=nothing)
    U = get_uniform(n, seed)
    X = a .+ (b-a) .* U
    return X
end


function expon_rng(λ, n=1, seed=nothing)
    U = get_uniform(n, seed)
    X = -λ .* log.(1 .- U)  # could also use just U
    return X
end


function erlang_rng(k, λ, n=1, seed=nothing)
    U = zeros(k, n)
    for i in 1:k
        U[i, :] = get_uniform(n, seed)
    end
    # X = (-λ/k) .* log.(prod(U, dims=1))
    X = (-1/λ) .* log.(prod(U, dims=1))  # Here (-1/λ) represents mean
    return X
end


function bernoulli_rng(p, n=1, seed=nothing)
    U = get_uniform(n, seed)
    X = U .< p
    return X
end

u = bernoulli_rng(.2342, 10000)
sum(u)/10000

# End of Module
end
