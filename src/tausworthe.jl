using Random: bitrand


function tausworthe_rng(size::Int; r::Int=3, q::Int=128)
    if size < q
        throw(ArgumentError("Size must be > 128"))
    end

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
