@doc raw"""
    triag_rng(a, b, m, shape=1; seed=nothing)

Generate a `shape` element array of random variables from a Triangular(`a`, `b`, `m`) distribution. Optionally you can set a specific seed.

# Notes

The Triangular distribution is given by:

```math
f(x, a, b, m) = \begin{cases}
\frac{2(x-a)}{(m-a)(b-a)}  & \text{if  } a < x ≤ m, \\
\frac{2(b-x)}{(b-m)(b-a)}  & \text{if  } m < x ≤ c, \\
0 & \text{otherwise}
\end{cases}
```

# Examples

```julia-repl
julia> triag_rng()
1-element Vector{Float64}:
 0.3559088458688944

julia> triag_rng(0,7,2,5)
5-element Vector{Float64}:
 1.3758072115332673
 6.049452463477447
 6.042781317411027
 2.914959243260448
 4.454707036522528
```

# References

W. Stein and M. Keblis. A new method to simulate the triangular distribution. Mathematical and Computer Modelling, Volume 49, Issues 5–6, 2009.
"""
function triag_rng(a=0,b=1,m=.5,shape=1)
    u = get_std_uniform(shape)
    v = get_std_uniform(shape)
    t_min = min.(u,v)
    t_max = max.(u,v)
    c = (m-a)/(b-a)
    t = ((1-c) .* t_min) .+ (c .* t_max)
    X = a .+ (b-a) .*t
    return X
end