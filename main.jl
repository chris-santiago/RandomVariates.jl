using RandomVariates
using Statistics
using Distributions
using Gadfly
import Cairo, Fontconfig


x = RandomVariates.normal_rng(0, 1, 10000)
Statistics.mean(x)
Statistics.std(x)
plot(x=x, Geom.histogram(density=true))

u = RandomVariates.binomial_rng(.2342, 1000, 1000)
sum(u)/1000

d = Distributions.Binomial(1000, .2342)
z = rand(d, 1000)
sum(z)/1000

e = RandomVariates.expon_rng(1, 10000)
exp_plot = plot(x=e, Geom.histogram(density=true))
exp_plot |> PNG("expon.png", 4inch, 4inch)

x = RandomVariates.gamma_rng(1/2, 2, 1000)
plot(x=x, Geom.histogram(density=false))