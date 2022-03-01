using RandomVariates
using Statistics
using Distributions


x = RandomVariates.normal_rng(0,1,10000)
Statistics.mean(x)
Statistics.std(x)

u = RandomVariates.binomial_rng(.2342, 1000, 1000)
sum(u)/1000

d = Distributions.Binomial(1000, .2342)
z = rand(d, 1000)
sum(z)/1000

e = RandomVariates.expon_rng(λ=1, size=10)