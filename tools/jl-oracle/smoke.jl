using Pkg
Pkg.activate(".")
using JuMP, ECOS, Clarabel
# Same toy SOCP: min x[1] s.t. ||(x[2], x[3])|| <= x[1], x[2]+x[3]=1
m = Model(Clarabel.Optimizer)
set_silent(m)
@variable(m, x[1:3])
@constraint(m, [x[1]; x[2]; x[3]] in SecondOrderCone())
@constraint(m, x[2] + x[3] == 1)
@objective(m, Min, x[1])
optimize!(m)
println("Clarabel status: ", termination_status(m))
println("Clarabel cost:   ", objective_value(m), " (expected ~0.7071)")
println("Clarabel x:      ", value.(x))
