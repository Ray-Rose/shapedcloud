using Pkg
Pkg.activate(".")
Pkg.add(["JuMP", "ECOS", "Clarabel", "LinearAlgebra"])
println("--- Verifying ---")
using JuMP, ECOS, Clarabel
println("JuMP version:    ", pkgversion(JuMP))
println("ECOS version:    ", pkgversion(ECOS))
println("Clarabel version: ", pkgversion(Clarabel))
