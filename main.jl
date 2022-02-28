using RandomVariates
using Statistics

u = RandomVariates.uniform_rng(100000)
print(Statistics.mean(u))
