#=
Solve a dumped *assembled* SCvx subproblem via JuMP/Clarabel — the Julia twin of
`tools/py-oracle/solve_scvx_subproblem.py`. Reads the standard-form matrices
`(c, A, b, G, h, cones)` that `oracle_scvx_subproblem.rs::dump_oracle_fixtures`
writes to `tools/oracle-data/scvx_*.txt`, reconstructs the generic SOCP

    min cᵀx  s.t.  A x = b,  (h − G x)[cone] ∈ SOC^dim,

and solves it. Used to cross-check the CVXPY/Clarabel oracle (two independent
solvers agreeing rules out an oracle-side bug) before baking the reference cost
into the Rust test.

Usage:  julia --project=tools/jl-oracle tools/jl-oracle/solve_scvx_subproblem.jl [name ...]
=#

using Pkg
Pkg.activate(@__DIR__)
using JuMP, Clarabel

data_dir() = normpath(joinpath(@__DIR__, "..", "oracle-data"))

function parse_dump(path)
    lines = filter(l -> !isempty(l) && !startswith(l, "#"), strip.(readlines(path)))
    i = Ref(1)
    function scalar(key)
        parts = split(lines[i[]]); @assert parts[1] == key
        i[] += 1; return parse(Int, parts[2])
    end
    NP = scalar("NP"); NE = scalar("NE"); NCT = scalar("NCT"); NCONES = scalar("NCONES")
    function rvec(key, n)
        @assert lines[i[]] == key "expected $key got $(lines[i[]])"; i[] += 1
        v = [parse(Float64, lines[i[] + k - 1]) for k in 1:n]; i[] += n; return v
    end
    c = rvec("c", NP); b = rvec("b", NE); h = rvec("h", NCT)
    function rmat(key)
        parts = split(lines[i[]]); @assert parts[1] == key
        r = parse(Int, parts[2]); col = parse(Int, parts[3]); i[] += 1
        vals = [parse(Float64, lines[i[] + k - 1]) for k in 1:r*col]; i[] += r * col
        return permutedims(reshape(vals, col, r))  # row-major flat → (r, col)
    end
    A = rmat("A"); G = rmat("G")
    @assert split(lines[i[]])[1] == "cones"; i[] += 1
    cones = Tuple{Int,Int}[]
    for _ in 1:NCONES
        o, d = split(lines[i[]]); push!(cones, (parse(Int, o), parse(Int, d))); i[] += 1
    end
    return (NP=NP, NE=NE, NCT=NCT, NCONES=NCONES, c=c, A=A, b=b, h=h, G=G, cones=cones)
end

function solve_dump(d)
    m = Model(Clarabel.Optimizer); set_silent(m)
    @variable(m, x[1:d.NP])
    s = d.h .- d.G * x
    @constraint(m, d.A * x .== d.b)
    for (off, dim) in d.cones
        if dim == 1
            @constraint(m, s[off + 1] >= 0)         # off is 0-based
        else
            @constraint(m, s[off+1:off+dim] in SecondOrderCone())
        end
    end
    @objective(m, Min, d.c' * x)
    optimize!(m)
    return (status=termination_status(m), cost=objective_value(m))
end

names = isempty(ARGS) ? ["fixedtf", "freetf"] : ARGS
for name in names
    path = isfile(name) ? name : joinpath(data_dir(), "scvx_$(name).txt")
    if !isfile(path)
        println("--- $(name) ---\n  MISSING dump: $(path)\n"); continue
    end
    d = parse_dump(path)
    r = solve_dump(d)
    println("--- $(basename(path)) ---")
    println("  dims   : NP=$(d.NP) NE=$(d.NE) NCT=$(d.NCT) NCONES=$(d.NCONES)")
    println("  solver = Clarabel (via JuMP)")
    println("  status = $(r.status)")
    println("  cost   = ", round(r.cost, digits=6))
    println()
end
