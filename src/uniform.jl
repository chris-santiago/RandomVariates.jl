"""
    gen_prn()

Generate a pseudorandom number.

# Notes

Uses a linear congruential generator (LCG) with [POSIX parameters](https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use):

``X_n = 25214903917 X_{n-1} + 11 \\quad mod \\quad 2^{48}``

# Examples

```julia-repl
julia> gen_prn()
 156750217634815

julia> gen_prn()
 63914890472862
```
"""
function gen_prn()
    seed = get_seed()
	prn = mod(A * seed + C, MOD)
    set_seed(prn)
    return prn
end


"""
    get_std_uniform(size=1; seed=nothing)

Generate a `size` element array of random variables from a standard Uniform(0,1) distribution. Optionally you can set a specific seed.

# Examples

```julia-repl
julia> get_std_uniform()
1-element Vector{Float64}:
 0.42443098343863284

julia> get_std_uniform(seed=43)
1-element Vector{Float64}:
 0.09636209187468836

julia> get_std_uniform(5)
5-element Vector{Float64}:
 0.6584669595802204
 0.33437978955868886
 0.509019330923099
 0.12156905126458639
 0.917393216014684

```
"""
function get_std_uniform(size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    seed_setter(seed)
    U = zeros(size)  # preallocate array
    U .= gen_prn.()  # vectorize assignment for efficiency
    U = U./MOD
	return U
end


"""
    uniform_rng(a, b, size=1; seed=nothing)

Generate a `size` element array of random variables from a Uniform(`a`, `b`) distribution. Optionally you can set a specific seed.

# Notes

The Uniform distribution is given by:

``f(x, a, b) = \\frac{``}{b-a} \\quad for a ≤ x ≤ ≤ b``

# Examples

```julia-repl
julia> uniform_rng(1, 6)
1-element Vector{Float64}:
 2.638331960912094

julia> uniform_rng(1, 6, seed=42)
1-element Vector{Float64}:
 2.6333962626438314

julia> uniform_rng(0, 1, (4,4))
4×4 Matrix{Float64}:
 0.640603   0.757195  0.325722  0.645452
 0.955188   0.155203  0.953206  0.0046541
 0.0923526  0.490721  0.451705  0.516445
 0.661619   0.527063  0.212847  0.832298
 
```

# References

Walk, C. Handbook on statistical distributions for experimentalists. 2007.
"""
function uniform_rng(a::Real=0, b::Real=1, size::Union{Int, Tuple{Vararg{Int}}}=1; seed::Union{Int, Nothing}=nothing)
    U = get_std_uniform(size, seed=seed)
    X = a .+ (b-a) .* U
    return X
end
