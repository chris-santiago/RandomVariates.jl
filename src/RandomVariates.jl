module RandomVariates

export
# SEED, A, C, MOD,
bernoulli_rng,
beta_rng,
binomial_rng,
erlang_rng,
expon_rng,
gamma_rng,
geometric_rng,
neg_binomial_rng,
conv_neg_binomial_rng,
normal_rng,
lognormal_rng,
mv_normal_rng,
poisson_rng,
uniform_rng,
weibull_rng,
tausworthe_rng,
triag_rng

include("uniform.jl")
include("exponential.jl")
include("erlang.jl")
include("weibull.jl")
include("bernoulli.jl")
include("geometric.jl")
include("poisson.jl")
include("binomial.jl")
include("neg_binomial.jl")
include("normal.jl")
include("gamma.jl")
include("beta.jl")
include("tausworthe.jl")
include("triangular.jl")

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
    check_p(p::Real)
Check that parameter `p` falls between 0 and 1.
"""
function check_p(p::Real)
    if (p > 1) || (p < 0)
        throw(ArgumentError("Parameter `p` must fall between 0 and 1."))
    end
end


# End of Module
end
