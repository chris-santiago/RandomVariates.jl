using LinearAlgebra

"""
    get_std_normal(shape=1; seed=nothing)

Generate a `shape` element array of random variables from a standard Normal(0, 1) distribution. Optionally you can set a specific seed.

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
function get_std_normal(shape::Int=1; seed::Union{Int, Nothing}=nothing)
    seed_setter(seed)
    a = sqrt.(-2 .* log.(get_std_uniform(shape)))
    b = 2 * π .* get_std_uniform(shape)
    A = a .* sin.(b)
    B = a .* cos.(b)
    X = collect(Iterators.flatten(zip(A, B)))[1:shape]
    return X
end


function get_std_normal(shape::Tuple{Vararg{Int}}; seed::Union{Int, Nothing}=nothing)
    X = get_std_normal(reduce(*, shape), seed=seed)
    return reshape(X, shape)
end


"""
    normal_rng(μ, σ², shape=1; seed=nothing)

Generate a `shape` element array of random variables from a Normal(`μ`, `σ²`) distribution. Optionally you can set a specific seed.

# Notes

The Normal distribution is given by:

``f(x, μ, σ²) = \\frac{1}{σ\\sqrt{2π}} e^{ - \\frac{1}{2} (\\frac{x-μ}{σ})^2}``

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
function normal_rng(μ::Real=0, σ²::Real=1, shape::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    σ = sqrt(σ²)
    X = get_std_normal(shape, seed=seed) .* σ .+ μ
    return X
end

# NOT DOCUMENTED, NOT EXPORTED!
function lognormal_rng(μ::Real=0, σ²::Real=1, shape::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    Y = normal_rng(μ, σ², shape, seed=seed)
    X = exp.(Y)
    return X
end


"""
    get_mv_std_normal(μ, Σ; seed=nothing)

Generate an array of random variables from a Multivariate Normal(`μ`, `Σ`) distribution. Optionally you can set a specific seed.

# Notes

The Normal distribution is given by:

``f(\\textbf{x, μ}, Σ) = \\frac{1}{2π^{k/2} |Σ|^{1/2}} exp \\{ - \\frac{(\\textbf{x} - \\textbf{μ})^T Σ^{-1} (\\textbf{x} - \\textbf{μ})}{2} \\} \\quad \\textbf{x} \\in \\mathbb{R}^k``

with mean vector ``\\textbf{μ} = (μ_1, μ_2, …, μ_k)`` and covariance matrix ``Σ``.

# Examples

```julia-repl
julia> A
2×2 Matrix{Float64}:
 1.0  0.9
 0.9  1.0

julia> get_mv_std_normal([2,2], A)
1×2 transpose(::Vector{Float64}) with eltype Float64:
 2.60139  1.76837

julia> B
3×3 Matrix{Float64}:
 1.0  0.8  0.3
 0.8  1.0  0.6
 0.3  0.6  1.0

julia> get_mv_std_normal([0,1,2], B)
1×3 transpose(::Vector{Float64}) with eltype Float64:
 -1.24861  0.110397  0.609328
 
```

# References

Law, A. Simulation modeling and analysis, 5th Ed. McGraw Hill Education, Tuscon, 2013.
"""
function get_mv_std_normal(μ, Σ; seed=nothing)
    Z = get_std_normal(length(μ), seed=seed)
    C = cholesky(Σ)
    X = transpose(μ) + transpose(Z) * C.U
    return X
end


"""
    mv_normal_rng(μ, Σ, shape=1; seed=nothing)

Generate a `shape` element array of random variables from a Multivariate Normal(`μ`, `Σ`) distribution. Optionally you can set a specific seed.

# Notes

The Normal distribution is given by:

``f(\\textbf{x, μ}, Σ) = \\frac{1}{2π^{k/2} |Σ|^{1/2}} exp \\{ - \\frac{(\\textbf{x} - \\textbf{μ})^T Σ^{-1} (\\textbf{x} - \\textbf{μ})}{2} \\} \\quad \\textbf{x} \\in \\mathbb{R}^k``

with mean vector ``\\textbf{μ} = (μ_1, μ_2, …, μ_k)`` and covariance matrix ``Σ``.

# Examples

```julia-repl
julia> A
2×2 Matrix{Float64}:
 1.0  0.9
 0.9  1.0

julia> mv_normal_rng([2,2], A)
1×2 Matrix{Float64}:
 0.398043  0.651268

julia> mv_normal_rng([2,2], A, 4)
4×2 Matrix{Float64}:
 2.12004  1.9172
 4.81543  5.02956
 2.97096  2.2597
 3.27012  3.07918
 
```

# References

Law, A. Simulation modeling and analysis, 5th Ed. McGraw Hill Education, Tuscon, 2013.
"""
function mv_normal_rng(μ::Array, Σ::Matrix, shape::Int=1; seed=nothing)
    X = zeros(shape, length(μ))
    for i in 1:shape(X, 1)
        X[i, :] += transpose(get_mv_std_normal(μ, Σ; seed=seed))
    end
    return X
end