using RandomVariates
using Statistics
using Distributions


u = RandomVariates.binomial_rng(.2342, 1000, 1000)
sum(u)/1000

d = Distributions.Binomial(1000, .2342)
z = rand(d, 1000)
sum(z)/1000

e = RandomVariates.expon_rng(λ=1, size=10)