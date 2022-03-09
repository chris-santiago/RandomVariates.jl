# William E. Stein, Matthew F. Keblis,
# A new method to simulate the triangular distribution,
# Mathematical and Computer Modelling,
# Volume 49, Issues 5–6,
# 2009,
# Pages 1143-1147,
# ISSN 0895-7177,
# https://doi.org/10.1016/j.mcm.2008.06.013.

function triag_rng(a=0,b=1,m=.5,shape=1000)
    u = get_std_uniform(shape)
    v = get_std_uniform(shape)
    t_min = min.(u,v)
    t_max = max.(u,v)
    c = (m-a)/(b-a)
    t = ((1-c) .* t_min) .+ (c .* t_max)
    X = a .+ (b-a) .*t
    return X
end