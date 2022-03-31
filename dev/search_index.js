var documenterSearchIndex = {"docs":
[{"location":"uniform/#Uniform-Random-Variables","page":"Uniform","title":"Uniform Random Variables","text":"","category":"section"},{"location":"uniform/#Generating-Standard-Uniform","page":"Uniform","title":"Generating Standard Uniform","text":"","category":"section"},{"location":"uniform/","page":"Uniform","title":"Uniform","text":"RandomVariates.get_std_uniform","category":"page"},{"location":"uniform/#RandomVariates.get_std_uniform","page":"Uniform","title":"RandomVariates.get_std_uniform","text":"get_std_uniform(shape=1; seed=nothing)\n\nGenerate a shape element array of random variables from a standard Uniform(0,1) distribution. Optionally you can set a specific seed.\n\nExamples\n\njulia> get_std_uniform()\n1-element Vector{Float64}:\n 0.42443098343863284\n\njulia> get_std_uniform(seed=43)\n1-element Vector{Float64}:\n 0.09636209187468836\n\njulia> get_std_uniform(5)\n5-element Vector{Float64}:\n 0.6584669595802204\n 0.33437978955868886\n 0.509019330923099\n 0.12156905126458639\n 0.917393216014684\n\n\n\n\n\n\n","category":"function"},{"location":"uniform/#Uniform","page":"Uniform","title":"Uniform","text":"","category":"section"},{"location":"uniform/","page":"Uniform","title":"Uniform","text":"RandomVariates.uniform_rng","category":"page"},{"location":"uniform/#RandomVariates.uniform_rng","page":"Uniform","title":"RandomVariates.uniform_rng","text":"uniform_rng(a, b, shape=1; seed=nothing)\n\nGenerate a shape element array of random variables from a Uniform(a, b) distribution. Optionally you can set a specific seed.\n\nNotes\n\nThe Uniform distribution is given by:\n\nf(x a b) = frac1b-a quad textfor quad a  x  b\n\nExamples\n\njulia> uniform_rng(1, 6)\n1-element Vector{Float64}:\n 2.638331960912094\n\njulia> uniform_rng(1, 6, seed=42)\n1-element Vector{Float64}:\n 2.6333962626438314\n\njulia> uniform_rng(0, 1, (4,4))\n4×4 Matrix{Float64}:\n 0.640603   0.757195  0.325722  0.645452\n 0.955188   0.155203  0.953206  0.0046541\n 0.0923526  0.490721  0.451705  0.516445\n 0.661619   0.527063  0.212847  0.832298\n \n\nReferences\n\nWalk, C. Handbook on statistical distributions for experimentalists. 2007.\n\n\n\n\n\n","category":"function"},{"location":"normal/#Normal-Random-Variables","page":"Normal","title":"Normal Random Variables","text":"","category":"section"},{"location":"normal/#Generating-Standard-Normal","page":"Normal","title":"Generating Standard Normal","text":"","category":"section"},{"location":"normal/","page":"Normal","title":"Normal","text":"RandomVariates.get_std_normal","category":"page"},{"location":"normal/#RandomVariates.get_std_normal","page":"Normal","title":"RandomVariates.get_std_normal","text":"get_std_normal(shape=1; seed=nothing)\n\nGenerate a shape element array of random variables from a standard Normal(0, 1) distribution. Optionally you can set a specific seed.\n\nExamples\n\njulia> get_std_normal()\n1-element Vector{Float64}:\n 0.6315076033452351\n\njulia> get_std_normal(5, seed=43)\n5-element Vector{Float64}:\n  1.2311463458421277\n  1.7786409025309897\n -0.4178415161339713\n  0.3518755172644067\n -0.16742990320047046\n\njulia> get_std_normal((2,2))\n2×2 Matrix{Float64}:\n -0.900365   -0.432759\n -0.0350299   1.55754\n\n\n\n\n\n\n","category":"function"},{"location":"normal/#Normal","page":"Normal","title":"Normal","text":"","category":"section"},{"location":"normal/","page":"Normal","title":"Normal","text":"RandomVariates.normal_rng","category":"page"},{"location":"normal/#RandomVariates.normal_rng","page":"Normal","title":"RandomVariates.normal_rng","text":"normal_rng(μ, σ², shape=1; seed=nothing)\n\nGenerate a shape element array of random variables from a Normal(μ, σ²) distribution. Optionally you can set a specific seed.\n\nNotes\n\nThe Normal distribution is given by:\n\nf(x μ σ²) = frac1σsqrt2π e^ - frac12 (fracx-μσ)^2\n\nExamples\n\njulia> normal_rng()\n1-element Vector{Float64}:\n 0.03130435813519526\n\njulia> normal_rng(3, 9, 2)\n2-element Vector{Float64}:\n  7.362935421449054\n -1.0173543995738399\n\njulia> normal_rng(0,1,(2,2))\n2×2 Matrix{Float64}:\n -0.640505   0.30303\n -0.0556832  0.714122\n \n\nReferences\n\nWalk, C. Handbook on statistical distributions for experimentalists. 2007.\n\n\n\n\n\n","category":"function"},{"location":"normal/#Multivariate-Normal","page":"Normal","title":"Multivariate Normal","text":"","category":"section"},{"location":"normal/","page":"Normal","title":"Normal","text":"RandomVariates.get_mv_std_normal\nRandomVariates.mv_normal_rng","category":"page"},{"location":"normal/#RandomVariates.get_mv_std_normal","page":"Normal","title":"RandomVariates.get_mv_std_normal","text":"get_mv_std_normal(μ, Σ; seed=nothing)\n\nGenerate an array of random variables from a Multivariate Normal(μ, Σ) distribution. Optionally you can set a specific seed.\n\nNotes\n\nThe Normal distribution is given by:\n\nf(textbfx μ Σ) = frac12π^k2 Σ^12 exp  - frac(textbfx - textbfμ)^T Σ^-1 (textbfx - textbfμ)2  quad textbfx in mathbbR^k\n\nwith mean vector textbfμ = (μ_1 μ_2  μ_k) and covariance matrix Σ.\n\nExamples\n\njulia> A\n2×2 Matrix{Float64}:\n 1.0  0.9\n 0.9  1.0\n\njulia> get_mv_std_normal([2,2], A)\n1×2 transpose(::Vector{Float64}) with eltype Float64:\n 2.60139  1.76837\n\njulia> B\n3×3 Matrix{Float64}:\n 1.0  0.8  0.3\n 0.8  1.0  0.6\n 0.3  0.6  1.0\n\njulia> get_mv_std_normal([0,1,2], B)\n1×3 transpose(::Vector{Float64}) with eltype Float64:\n -1.24861  0.110397  0.609328\n \n\nReferences\n\nLaw, A. Simulation modeling and analysis, 5th Ed. McGraw Hill Education, Tuscon, 2013.\n\n\n\n\n\n","category":"function"},{"location":"normal/#RandomVariates.mv_normal_rng","page":"Normal","title":"RandomVariates.mv_normal_rng","text":"mv_normal_rng(μ, Σ, shape=1; seed=nothing)\n\nGenerate a shape element array of random variables from a Multivariate Normal(μ, Σ) distribution. Optionally you can set a specific seed.\n\nNotes\n\nThe Normal distribution is given by:\n\nf(textbfx μ Σ) = frac12π^k2 Σ^12 exp  - frac(textbfx - textbfμ)^T Σ^-1 (textbfx - textbfμ)2  quad textbfx in mathbbR^k\n\nwith mean vector textbfμ = (μ_1 μ_2  μ_k) and covariance matrix Σ.\n\nExamples\n\njulia> A\n2×2 Matrix{Float64}:\n 1.0  0.9\n 0.9  1.0\n\njulia> mv_normal_rng([2,2], A)\n1×2 Matrix{Float64}:\n 0.398043  0.651268\n\njulia> mv_normal_rng([2,2], A, 4)\n4×2 Matrix{Float64}:\n 2.12004  1.9172\n 4.81543  5.02956\n 2.97096  2.2597\n 3.27012  3.07918\n \n\nReferences\n\nLaw, A. Simulation modeling and analysis, 5th Ed. McGraw Hill Education, Tuscon, 2013.\n\n\n\n\n\n","category":"function"},{"location":"prn/#Pseudorandom-Numbers","page":"Pseudorandom Numbers","title":"Pseudorandom Numbers","text":"","category":"section"},{"location":"prn/#Using-a-Linear-Congruential-Generator","page":"Pseudorandom Numbers","title":"Using a Linear Congruential Generator","text":"","category":"section"},{"location":"prn/","page":"Pseudorandom Numbers","title":"Pseudorandom Numbers","text":"RandomVariates.gen_prn","category":"page"},{"location":"prn/#RandomVariates.gen_prn","page":"Pseudorandom Numbers","title":"RandomVariates.gen_prn","text":"gen_prn()\n\nGenerate a pseudorandom number.\n\nNotes\n\nUses a linear congruential generator (LCG) with POSIX parameters:\n\nX_n = 25214903917 X_n-1 + 11 quad mod quad 2^48\n\nExamples\n\njulia> gen_prn()\n 156750217634815\n\njulia> gen_prn()\n 63914890472862\n\n\n\n\n\n","category":"function"},{"location":"prn/#Using-a-Tausworthe-Generator","page":"Pseudorandom Numbers","title":"Using a Tausworthe Generator","text":"","category":"section"},{"location":"prn/","page":"Pseudorandom Numbers","title":"Pseudorandom Numbers","text":"RandomVariates.tausworthe_rng","category":"page"},{"location":"prn/#RandomVariates.tausworthe_rng","page":"Pseudorandom Numbers","title":"RandomVariates.tausworthe_rng","text":"tausworthe_rng(shape::Int=1; r::Int=3, q::Int=128)\n\nGenerate a shape element array of random variables from a standard Uniform(0,1) distribution using a Tausworthe RNG.\n\nNotes\n\nImplementation:\n\nB_i = B_i-r quad XOR quad B_i-q\n\nExamples\n\njulia> U = tausworthe_rng(1)\n1-element Vector{Float64}:\n 0.5462285033427179\n\njulia> U = tausworthe_rng((2,2))\n2×2 Matrix{Float64}:\n 0.782613  0.365878\n 0.176636  0.0413817\n\n\nReferences\n\nShu Tezuka and Pierre L'Ecuyer. 1991. Efficient and portable combined Tausworthe random number generators. ACM Trans. Model. Comput. Simul. 1, 2 (April 1991), 99–112. DOI:https://doi.org/10.1145/116890.116892\n\nLaw, A. Simulation modeling and analysis, 5th Ed. McGraw Hill Education, Tuscon, 2013.\n\n\n\n\n\n","category":"function"},{"location":"discrete/#Discrete-Random-Variables","page":"Discrete","title":"Discrete Random Variables","text":"","category":"section"},{"location":"discrete/#Bernoulli","page":"Discrete","title":"Bernoulli","text":"","category":"section"},{"location":"discrete/","page":"Discrete","title":"Discrete","text":"RandomVariates.bernoulli_rng","category":"page"},{"location":"discrete/#RandomVariates.bernoulli_rng","page":"Discrete","title":"RandomVariates.bernoulli_rng","text":"bernoulli_rng(p, shape=1; seed=nothing)\n\nGenerate a shape element array of random variables from a Bernoulli(p) distribution. Optionally you can set a specific seed.\n\nNotes\n\nThe pdf of the Bernoulli Distribution is given:\n\nf(x p) = p^x (1-p)^1-x quad x in 01\n\nExamples\n\njulia> bernoulli_rng(.34)\n1-element BitVector:\n 0\n\njulia> bernoulli_rng(.34, 5)\n5-element BitVector:\n 0\n 0\n 1\n 0\n 1\n\njulia> bernoulli_rng(.8, (2,2), seed=42)\n2×2 BitMatrix:\n 1  0\n 0  1\n\nReferences\n\nD.P. Kroese, T. Taimre, Z.I. Botev. Handbook of Monte Carlo Methods.    Wiley Series in Probability and Statistics, John Wiley & Sons, New York, 2011.\n\n\n\n\n\n","category":"function"},{"location":"discrete/#Binomial","page":"Discrete","title":"Binomial","text":"","category":"section"},{"location":"discrete/","page":"Discrete","title":"Discrete","text":"RandomVariates.binomial_rng","category":"page"},{"location":"discrete/#RandomVariates.binomial_rng","page":"Discrete","title":"RandomVariates.binomial_rng","text":"binomial_rng(p, n, shape=1; seed=nothing)\n\nGenerate a shape element array of random variables from a Binomial(p, n) distribution. Optionally you can set a specific seed.\n\nNotes\n\nThe Binomial(x, n, p) distribution describes the total number of successes in a sequence of n Bernoulli(p) trials.\n\nThe pdf is given:\n\nf(xnp) = binomnx p^x (1-p)^n-x quad x = 01dots n\n\nExamples\n\njulia> binomial_rng(.3, 10)\n1×1 Matrix{Int64}:\n 3\n\njulia> binomial_rng(.3, 10, (2,2))\n2×2×1 Array{Int64, 3}:\n[:, :, 1] =\n 2  1\n 2  2\n\nReferences\n\nD.P. Kroese, T. Taimre, Z.I. Botev. Handbook of Monte Carlo Methods.    Wiley Series in Probability and Statistics, John Wiley & Sons, New York, 2011.\n\n\n\n\n\n","category":"function"},{"location":"discrete/#Geometric","page":"Discrete","title":"Geometric","text":"","category":"section"},{"location":"discrete/","page":"Discrete","title":"Discrete","text":"RandomVariates.geometric_rng","category":"page"},{"location":"discrete/#RandomVariates.geometric_rng","page":"Discrete","title":"RandomVariates.geometric_rng","text":"geometric_rng(p, shape=1; seed=nothing)\n\nGenerate a shape element array of random variables from a Geometric(p) distribution. Optionally you can set a specific seed.\n\nNotes\n\nThe Geometric distribution is given:\n\nf(xp) = (1-p)^x-1p) quad x = 123\n\nwhere 0  p  1\n\nExamples\n\njulia> geometric_rng(.8)\n1-element Vector{Int64}:\n 1\n\njulia> geometric_rng(.8, 5)\n5-element Vector{Int64}:\n 2\n 3\n 1\n 1\n 1\n\njulia> geometric_rng(.8, (2,2), seed=45)\n2×2 Matrix{Int64}:\n 1  1\n 1  1\n \n\nReferences\n\nD.P. Kroese, T. Taimre, Z.I. Botev. Handbook of Monte Carlo Methods.    Wiley Series in Probability and Statistics, John Wiley & Sons, New York, 2011.\n\nC. Alexopoulos, D. Goldsman. Random variate generation. 2020.\n\n\n\n\n\n","category":"function"},{"location":"discrete/#Poisson","page":"Discrete","title":"Poisson","text":"","category":"section"},{"location":"discrete/","page":"Discrete","title":"Discrete","text":"RandomVariates.poisson_rng","category":"page"},{"location":"discrete/#RandomVariates.poisson_rng","page":"Discrete","title":"RandomVariates.poisson_rng","text":"poisson_rng(p, n, shape=1; seed=nothing)\n\nGenerate a shape element array of random variables from a Poisson(λ) distribution. Optionally you can set a specific seed.\n\nNotes\n\nThe Poisson distribution is given by:\n\nf(x λ) = fracλ^xx e^λ quad x = 012\n\nExamples\n\njulia> poisson_rng(3)\n1×1 Matrix{Int64}:\n 7\n\njulia> poisson_rng(10, 5)\n5×1 Matrix{Int64}:\n 13\n 11\n 10\n  8\n 15\n\njulia> poisson_rng(10, (5,5))\n5×5×1 Array{Int64, 3}:\n[:, :, 1] =\n 11  15   9  11   9\n  8  15  13  10   9\n 11  12   4  10   6\n  7   9  13  11   7\n 13   7  10  10  14\n \n\nReferences\n\nR. Larson, A. Odoni. Urban operations research. Prentice-Hall, New Jersey, 1981.\n\nG. Last, M. Penrose. Lectures on the poisson process. Cambridge University Press, 2017.\n\n\n\n\n\n","category":"function"},{"location":"discrete/#Negative-Binomial","page":"Discrete","title":"Negative Binomial","text":"","category":"section"},{"location":"discrete/","page":"Discrete","title":"Discrete","text":"RandomVariates.neg_binomial_rng\nRandomVariates.conv_neg_binomial_rng","category":"page"},{"location":"discrete/#RandomVariates.neg_binomial_rng","page":"Discrete","title":"RandomVariates.neg_binomial_rng","text":"neg_binomial_rng(p, r, shape=1; seed=nothing)\n\nGenerate a shape element array of random variables from a Negative Binomial(p, r) distribution. Optionally you can set a specific seed.\n\nNotes\n\nThe Negative Binomial distribution is given:\n\nf(xpr) = binomx-1r-1 (1-p)^x-r p^r quad x = 01dots n\n\nExamples\n\njulia> neg_binomial_rng(.5, 2)\n1-element Vector{Float64}:\n 3.0\n\njulia> neg_binomial_rng(.5, 5, 5)\n5-element Vector{Float64}:\n  8.0\n 10.0\n  8.0\n 13.0\n 10.0\n\njulia> neg_binomial_rng(.5, 2, (2,2))\n2×2 Matrix{Float64}:\n 3.0  4.0\n 4.0  2.0\n \n\nReferences\n\nWalk, C. Handbook on statistical distributions for experimentalists. 2007.\n\n\n\n\n\n","category":"function"},{"location":"discrete/#RandomVariates.conv_neg_binomial_rng","page":"Discrete","title":"RandomVariates.conv_neg_binomial_rng","text":"neg_binomial_rng(p, r, shape=1; seed=nothing)\n\nGenerate a shape element array of random variables from a Negative Binomial(p, r) distribution. Optionally you can set a specific seed.\n\nNotes\n\nThe Negative Binomial distribution is given:\n\nf(xpr) = binomx-1r-1 (1-p)^x-r p^r quad x = 01dots n\n\nUses a convolution algorithm to generate random variables, which is slightly slower than neg_binomial_rng.\n\nExamples\n\njulia> conv_neg_binomial_rng(.4, 5, 1)\n1×1 Matrix{Int64}:\n 8\n\njulia> conv_neg_binomial_rng(.4, 5, 5)\n5×1 Matrix{Int64}:\n 11\n 14 \n 8\n 10\n 13\n\njulia> conv_neg_binomial_rng(.4, 5, (2,2))\n2×2×1 Array{Int64, 3}:\n[:, :, 1] =\n 20  11\n 7  10\n\nReferences\n\nLaw, A. Simulation modeling and analysis, 5th Ed. McGraw Hill Education, Tuscon, 2013.\n\n\n\n\n\n","category":"function"},{"location":"continuous/#Other-Continuous-Random-Variables","page":"Continuous","title":"Other Continuous Random Variables","text":"","category":"section"},{"location":"continuous/#Exponential","page":"Continuous","title":"Exponential","text":"","category":"section"},{"location":"continuous/","page":"Continuous","title":"Continuous","text":"RandomVariates.expon_rng","category":"page"},{"location":"continuous/#RandomVariates.expon_rng","page":"Continuous","title":"RandomVariates.expon_rng","text":"expon_rng(λ, shape=1; seed=nothing)\n\nGenerate a shape element array of random variables from a Exponential(λ) distribution. Optionally you can set a specific seed.\n\nNotes\n\nThe pdf of an Exponential(λ) distribution is given as:\n\nf(x λ) = λe^-λx quad x  0\n\nnote: Note\nSome Exponential distributions are parameterized by β = frac1λ, where λ is the number of events in an interval.  In such cases, β represents the mean interarrival time; here we use λ to represent mean arrival rate per unit of time.\n\nExamples\n\njulia> expon_rng(3)\n1-element Vector{Float64}:\n 0.07033135663980515\n\njulia> expon_rng(1.2, seed=42)\n1-element Vector{Float64}:\n 0.3296112244200808\n\njulia> expon_rng(1.2, (2, 2))\n2×2 Matrix{Float64}:\n 1.9327    0.134739\n 0.746861  0.155614\n\nReferences\n\nD.P. Kroese, T. Taimre, Z.I. Botev. Handbook of Monte Carlo Methods.    Wiley Series in Probability and Statistics, John Wiley & Sons, New York, 2011.\n\n\n\n\n\n","category":"function"},{"location":"continuous/#Erlang","page":"Continuous","title":"Erlang","text":"","category":"section"},{"location":"continuous/","page":"Continuous","title":"Continuous","text":"RandomVariates.erlang_rng","category":"page"},{"location":"continuous/#RandomVariates.erlang_rng","page":"Continuous","title":"RandomVariates.erlang_rng","text":"erlang_rng(k, λ, shape=1; seed=nothing)\n\nGenerate a shape element array of random variables from a Erlang_{k}(λ) distribution. Optionally you can set a specific seed.\n\nNotes\n\nThe pdf of an Erlang_{k}(λ) distribution is given as:\n\nf(x k λ) = fracλ^k e^-λx x^k-1(k-1) quad x  0\n\nExamples\n\njulia> erlang_rng(5, .5)\n1-element Vector{Float64}:\n 10.803989701023117\n\njulia> erlang_rng(3, 1, (2,2))\n2×2×1 Array{Float64, 3}:\n[:, :, 1] =\n 2.19956  4.18505\n 5.46892  2.5633\n\nReferences\n\nD. Goldsman, P. Goldsman. A first course in probability and statistics. 2021.\n\nL. Martino, D. Luengo. Extremely efficient generation of Gamma random variables for α ≥ 1. 2013.\n\n\n\n\n\n","category":"function"},{"location":"continuous/#Weibull","page":"Continuous","title":"Weibull","text":"","category":"section"},{"location":"continuous/","page":"Continuous","title":"Continuous","text":"RandomVariates.weibull_rng","category":"page"},{"location":"continuous/#RandomVariates.weibull_rng","page":"Continuous","title":"RandomVariates.weibull_rng","text":"weibull_rng(λ, β, shape=1; seed=nothing)\n\nGenerate a shape element array of random variables from a Weibull(λ, β) distribution. Optionally you can set a specific seed.\n\nNotes\n\nThe pdf of an Weibull(λ, β) distribution is given as:\n\nf(x λ β) = λ e^-λx^β quad x  0\n\nExamples\n\njulia> weibull_rng(2, 2, seed=42)\n1-element Vector{Float64}:\n 0.31445725834527055\n\njulia> weibull_rng(2, 2, 2)\n2-element Vector{Float64}:\n 0.39561285703154575\n 0.6021921673483441\n\njulia> weibull_rng(2, 2, (2,2))\n2×2 Matrix{Float64}:\n 0.428896  0.109897\n 0.812854  0.427906\n \n\nReferences\n\nC. Alexopoulos, D. Goldsman. Random variate generation. 2020.\n\nLaw, A. Simulation modeling and analysis, 5th Ed. McGraw Hill Education, Tuscon, 2013.\n\n\n\n\n\n","category":"function"},{"location":"continuous/#Gamma","page":"Continuous","title":"Gamma","text":"","category":"section"},{"location":"continuous/","page":"Continuous","title":"Continuous","text":"RandomVariates.gamma_rng","category":"page"},{"location":"continuous/#RandomVariates.gamma_rng","page":"Continuous","title":"RandomVariates.gamma_rng","text":"gamma_rng(α, β, shape=1; seed=nothing)\n\nGenerate a shape element array of random variables from a Gamma(α, β) distribution. Optionally you can set a specific seed.\n\nNotes\n\nThe Gamma distribution is given:\n\nf(xαβ) = fracβ^α x^α-1 e^-βxΓ(α) quad x  0\n\nExamples\n\njulia> gamma_rng(1,1)\n1-element Vector{Float64}:\n 0.5190236735858542\n\njulia> gamma_rng(1,1,4)\n4-element Vector{Float64}:\n 0.3035517926878862\n 0.5765419737109622\n 0.44121996206333797\n 0.7325887616559309\n\njulia> gamma_rng(1,1,(2,2))\n2×2 Matrix{Float64}:\n 0.228818  0.88849\n 0.665729  1.01668\n \n\nReferences\n\nLaw, A. Simulation modeling and analysis, 5th Ed. McGraw Hill Education, Tuscon, 2013.\n\n\n\n\n\n","category":"function"},{"location":"continuous/#Beta","page":"Continuous","title":"Beta","text":"","category":"section"},{"location":"continuous/","page":"Continuous","title":"Continuous","text":"RandomVariates.beta_rng","category":"page"},{"location":"continuous/#RandomVariates.beta_rng","page":"Continuous","title":"RandomVariates.beta_rng","text":"beta_rng(α, β, shape=1; seed=nothing)\n\nGenerate a shape element array of random variables from a Beta(α, β) distribution. Optionally you can set a specific seed.\n\nNotes\n\nThe Beta distribution is given:\n\nf(xαβ) = fracx^α-1 (1-x)^β-1Β(αβ) quad x in 01\n\nwhere Β(αβ) = fracΓ(α)Γ(β)Γ(α+β)\n\nExamples\n\njulia> beta_rng(1,2)\n1-element Vector{Float64}:\n 0.44456674672905633\n\njulia> beta_rng(1, 2, (2,2))\n2×2 Matrix{Float64}:\n 0.0792132  0.595657\n 0.737615   0.649721\n\n\nReferences\n\nD.P. Kroese, T. Taimre, Z.I. Botev. Handbook of Monte Carlo Methods.    Wiley Series in Probability and Statistics, John Wiley & Sons, New York, 2011.\n\nLaw, A. Simulation modeling and analysis, 5th Ed. McGraw Hill Education, Tuscon, 2013.\n\n\n\n\n\n","category":"function"},{"location":"continuous/#Triangular","page":"Continuous","title":"Triangular","text":"","category":"section"},{"location":"continuous/","page":"Continuous","title":"Continuous","text":"RandomVariates.triag_rng","category":"page"},{"location":"continuous/#RandomVariates.triag_rng","page":"Continuous","title":"RandomVariates.triag_rng","text":"triag_rng(a, b, m, shape=1; seed=nothing)\n\nGenerate a shape element array of random variables from a Triangular(a, b, m) distribution. Optionally you can set a specific seed.\n\nNotes\n\nThe Triangular distribution is given by:\n\nf(x a b m) = begincases\nfrac2(x-a)(m-a)(b-a)   textif   a  x  m \nfrac2(b-x)(b-m)(b-a)   textif   m  x  c \n0  textotherwise\nendcases\n\nExamples\n\njulia> triag_rng()\n1-element Vector{Float64}:\n 0.3559088458688944\n\njulia> triag_rng(0,7,2,5)\n5-element Vector{Float64}:\n 1.3758072115332673\n 6.049452463477447\n 6.042781317411027\n 2.914959243260448\n 4.454707036522528\n\nReferences\n\nW. Stein and M. Keblis. A new method to simulate the triangular distribution. Mathematical and Computer Modelling, Volume 49, Issues 5–6, 2009.\n\n\n\n\n\n","category":"function"},{"location":"#RandomVariates.jl","page":"Home","title":"RandomVariates.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A collection of random variable generators.","category":"page"},{"location":"","page":"Home","title":"Home","text":"note: Note\nThese generators were written as part of a class project, and not designed to maximize efficiency.  Please use Julia's built-in Random module for real-life simulation.","category":"page"},{"location":"#Content","page":"Home","title":"Content","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Pages = [\"prn.md\", \"uniform.md\", \"normal.md\", \"discrete.md\", \"continuous.md\"]","category":"page"},{"location":"#Index","page":"Home","title":"Index","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"}]
}