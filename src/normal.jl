"""
    get_std_normal(size=1; seed=nothing)

Generate a `size` element array of random variables from a standard Normal(0, 1) distribution. Optionally you can set a specific seed.

# Examples

```julia-repl
julia> get_std_normal()
1-element Vector{Float64}:
 0.6315076033452351

julia> get_std_normal(5, seed=43)
5-element Vector{Float64}:
  1.2311463458421277
  1.7786409025309897
 -0.4178415161339713
  0.3518755172644067
 -0.16742990320047046

julia> get_std_normal((2,2))
2×2 Matrix{Float64}:
 -0.900365   -0.432759
 -0.0350299   1.55754

```
"""
function get_std_normal(size::Int=1; seed::Union{Int, Nothing}=nothing)
    seed_setter(seed)
    a = sqrt.(-2 .* log.(get_std_uniform(size)))
    b = 2 * π .* get_std_uniform(size)
    A = a .* sin.(b)
    B = a .* cos.(b)
    X = collect(Iterators.flatten(zip(A, B)))[1:size]
    return X
end


function get_std_normal(size::Tuple{Vararg{Int}}; seed::Union{Int, Nothing}=nothing)
    X = get_std_normal(reduce(*, size), seed=seed)
    return reshape(X, size)
end


"""
    normal_rng(μ, σ², size=1; seed=nothing)

Generate a `size` element array of random variables from a Normal(`μ`, `σ²`) distribution. Optionally you can set a specific seed.

# Notes

The Normal distribution is given by:

``f(x, μ, σ²) = \\frac{1}{{σ\\sqrt{2π}} e^{ - \\frac{1}{2} (\\frac{x-μ}{σ})^2}``

# Examples

```julia-repl
julia> normal_rng()
1-element Vector{Float64}:
 0.03130435813519526

julia> normal_rng(3, 9, 2)
2-element Vector{Float64}:
  7.362935421449054
 -1.0173543995738399

julia> normal_rng(0,1,(2,2))
2×2 Matrix{Float64}:
 -0.640505   0.30303
 -0.0556832  0.714122
 
```

# References

Walk, C. Handbook on statistical distributions for experimentalists. 2007.
"""
function normal_rng(μ::Real=0, σ²::Real=1, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    X = get_std_normal(size, seed=seed) .* σ² .+ μ
    return X
end

# NOT DOCUMENTED, NOT EXPORTED!
function lognormal_rng(μ::Real=0, σ²::Real=1, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    Y = normal_rng(μ, σ², size, seed=seed)
    X = exp.(Y)
    return X
end