using Documenter
using RandomVariates

makedocs(
    sitename = "RandomVariates",
    format = Documenter.HTML(),
    modules = [RandomVariates],
    pages = [
        "Home" => "index.md",
        "prn.md",
        "Uniform" => "uniform.md",
        "Normal" => "normal.md",
        "Discrete" => "discrete.md",
        "Continuous" => "continuous.md"
    ]
)

deploydocs(
    repo = "github.com/chris-santiago/RandomVariates.jl.git",
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#


# DocumenterTools.genkeys(user="chris-santiago",
#        repo="git@github.com:chris-santiago/RandomVariates.jl.git")