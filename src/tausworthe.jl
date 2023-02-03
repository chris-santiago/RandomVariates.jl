using Random: bitrand, MersenneTwister

"""
    tausworthe_rng(shape::Int=1; r::Int=3, q::Int=128)

Generate a `shape` element array of random variables from a standard Uniform(0,1) distribution using a Tausworthe RNG.

# Notes

Implementation:

``B_i = B_{i-r} \\quad XOR \\quad B_{i-q}``

# Examples

```julia-repl
julia> U = tausworthe_rng(1)
1-element Vector{Float64}:
 0.5462285033427179

julia> U = tausworthe_rng((2,2))
2×2 Matrix{Float64}:
 0.782613  0.365878
 0.176636  0.0413817

```

# References

Shu Tezuka and Pierre L'Ecuyer. 1991. Efficient and portable combined Tausworthe random number generators. ACM Trans. Model. Comput. Simul. 1, 2 (April 1991), 99–112. DOI:https://doi.org/10.1145/116890.116892

Law, A. Simulation modeling and analysis, 5th Ed. McGraw Hill Education, Tuscon, 2013.
"""
function tausworthe_rng(shape::Int=1; r::Int=3, q::Int=128, seed::Union{Int, Nothing}=nothing)
    n_bits = 32
    shape *= n_bits
    seed = bitrand(MersenneTwister(seed), q)
    B = cat(seed, zeros(Bool, shape), dims=1)

    i = q + 1
    while i < length(B)
        B[i] += (B[i-r] != B[i-q])
        i +=1
    end

    B = convert.(Int, B[q+1:end])
    bit_string = join.(collect(Iterators.partition(B, n_bits)))
    U = parse.(Int, bit_string, base=2) ./ 2^n_bits
    return U
end


function tausworthe_rng(shape::Tuple{Vararg{Int}}; r::Int=3, q::Int=128, seed::Union{Int, Nothing}=nothing)
    U = tausworthe_rng(reduce(*, shape), r=r, q=q, seed=seed)
    return reshape(U, shape)
end
