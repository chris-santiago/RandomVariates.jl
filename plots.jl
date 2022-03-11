using RandomVariates
using Gadfly
import Cairo, Fontconfig


N = 1000000
A = [1 .8; .8 1]

struct Dist
    title
    x
    xmin
    xmax
end


function plot_dist(d::Dist)
    plot(x=d.x, Stat.density, Geom.polygon(fill=true, preserve_order=true), Guide.title(d.title), Coord.cartesian(xmin=d.xmin, xmax=d.xmax))
end

function plot_hist(d::Dist)
    plot(x=d.x, Geom.histogram(density=true), Guide.title(d.title), Coord.cartesian(xmin=d.xmin, xmax=d.xmax))
end


uniform = Dist("Standard Uniform", uniform_rng(0,1,N), 0, 1)
normal = Dist("Standard Normal", normal_rng(0,1,N), nothing, nothing)
bernoulli = Dist("Bernoulli(.34)", bernoulli_rng(.34, N), nothing, nothing)
beta = Dist("Beta(2,8)", beta_rng(2,8, N), 0, 1)
binomial = Dist("Binomial(.43, 10)", binomial_rng(.43, 10, N), nothing, nothing)
erlang = Dist("Erlang(7, 4)", erlang_rng(7, 4, N), 0, nothing)
expon = Dist("Exponential(4)", expon_rng(4, N), 0, nothing)
gamma = Dist("Gamma(6,4)", gamma_rng(6,4,N), 0, nothing)
geom = Dist("Geometric(.2)", geometric_rng(.2, N), 0, nothing)
neg_binom = Dist("Negative Binomial(.25, 20)", conv_neg_binomial_rng(.25, 20, N), 0, nothing)
poisson = Dist("Poission(10)", poisson_rng(10, N), 0, nothing)
triang = Dist("Triangular(1,5,2)", triag_rng(1,5,2,N), nothing, nothing)
weibull = Dist("Weibull(1,5)", weibull_rng(1,5,N), 0, nothing)
mv_norm = Dist("Multivariate Normal([0,2], Σ)", mv_normal_rng([0,2], A, Int(N/1000)), nothing, nothing)

for d in [uniform, normal, expon, erlang, gamma, triang, weibull, beta]
    out = plot_dist(d)
    out |> SVG("images/$(d.title).svg", 5inch, 5inch)
end


for d in [bernoulli, binomial, geom, neg_binom, poisson]
    out = plot_hist(d)
    out |> SVG("images/$(d.title).svg", 5inch, 5inch)
end


out = plot(x=mv_norm.x[:, 1], y=mv_norm.x[:, 2], Geom.density2d(levels=10), Geom.point, alpha=[0.4], Guide.title(mv_norm.title))
out |> SVG("images/$(mv_norm.title).svg", 5inch, 5inch)