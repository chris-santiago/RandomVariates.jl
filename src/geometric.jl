"""
    geometric_rng(p, size=1; seed=nothing)

Generate a `size` element array of random variables from a Geometric(`p`) distribution. Optionally you can set a specific seed.

# Examples

```julia-repl
julia> geometric_rng(.8)
1-element Vector{Int64}:
 1

julia> geometric_rng(.8, 5)
5-element Vector{Int64}:
 2
 3
 1
 1
 1

julia> geometric_rng(.8, (2,2), seed=45)
2×2 Matrix{Int64}:
 1  1
 1  1
 
```
"""
function geometric_rng(p::Real, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    check_p(p)
    U = get_std_uniform(size, seed=seed)
    X = ceil.(Int, log.(1 .- U) ./ log(1 - p))
    return X
end
