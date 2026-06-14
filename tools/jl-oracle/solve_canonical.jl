using Pkg
Pkg.activate(".")
using JuMP, Clarabel

# Same three canonical problems as solve_canonical.py.

function solve_toy()
    m = Model(Clarabel.Optimizer)
    set_silent(m)
    @variable(m, x[1:3])
    @constraint(m, [x[1], x[2], x[3]] in SecondOrderCone())
    @constraint(m, x[2] + x[3] == 1)
    @objective(m, Min, x[1])
    optimize!(m)
    return (status=termination_status(m), cost=objective_value(m), x=value.(x))
end

function solve_two_cone_mixed()
    m = Model(Clarabel.Optimizer)
    set_silent(m)
    @variable(m, x[1:3])
    @constraint(m, x[1] == 1)
    @constraint(m, x[2] == 1)
    @constraint(m, [x[3], x[1], x[2]] in SecondOrderCone())
    @constraint(m, x[3] >= 2)
    @objective(m, Min, x[3])
    optimize!(m)
    return (status=termination_status(m), cost=objective_value(m), x=value.(x))
end

function solve_socp_4d()
    m = Model(Clarabel.Optimizer)
    set_silent(m)
    @variable(m, x[1:4])
    @constraint(m, [x[1], x[2], x[3], x[4]] in SecondOrderCone())
    @constraint(m, sum(x) == 4)
    @objective(m, Min, x[1])
    optimize!(m)
    return (status=termination_status(m), cost=objective_value(m), x=value.(x))
end

for (name, fn) in [("toy_1cone", solve_toy),
                   ("two_cone_mixed", solve_two_cone_mixed),
                   ("socp_4d", solve_socp_4d)]
    r = fn()
    println("--- $(name) ---")
    println("  solver = Clarabel (via JuMP)")
    println("  status = $(r.status)")
    println("  cost   = ", round(r.cost, digits=12))
    print("  x      = [")
    print(join([round(v, digits=12) for v in r.x], ", "))
    println("]")
    println()
end
