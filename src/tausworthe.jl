using Random: bitrand

"""
    tausworthe_rng(size::Int=1; r::Int=3, q::Int=128)

Generate a `size` element array of random variables from a standard Uniform(0,1) distribution using a Tausworthe RNG.

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
"""
function tausworthe_rng(size::Int=1; r::Int=3, q::Int=128)
    n_bits = 32
    size *= n_bits
    seed = bitrand(q)
    B = cat(seed, zeros(Bool, size), dims=1)

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


function tausworthe_rng(size::Tuple{Vararg{Int}}; r::Int=3, q::Int=128)
    U = tausworthe_rng(reduce(*, size), r=r, q=q)
    return reshape(U, size)
end
