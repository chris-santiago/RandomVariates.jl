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

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
