module RandomVariates

export uniform_rng

# Write your package code here.
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


function uniform_rng(n, seed=nothing)
	U = zeros(n)
	x = gen_prn(seed)
	U[1] = x/MOD
	for i in 2:n
		x = gen_prn(x)
		U[i] = x/MOD
	end
	return U
end


end
