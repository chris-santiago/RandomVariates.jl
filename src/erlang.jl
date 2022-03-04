"""
    erlang_rng(k, λ, size=1; seed=nothing)

Generate a `size` element array of random variables from a Erlang_{`k`}(`λ`) distribution. Optionally you can set a specific seed.

# Examples

```julia-repl
julia> erlang_rng(5, .5)
1-element Vector{Float64}:
 10.803989701023117

julia> erlang_rng(3, 1, (2,2))
2×2×1 Array{Float64, 3}:
[:, :, 1] =
 2.19956  4.18505
 5.46892  2.5633
```
"""
function erlang_rng(k::Int, λ::Real, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    U = get_std_uniform((size..., k), seed=seed)
    k_dim = ndims(U)  # k is final dimension
    X = (-1/λ) .* log.(prod(U, dims=k_dim))  # Here (-1/λ) represents mean
    return X
end
