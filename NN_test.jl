oil_model = Model(Gurobi.Optimizer)
set_silent(oil_model)
@objective(oil_model, Max, 0)

@variable(oil_model, p)
@variable(oil_model, out)

include("ICNN_to_LP.jl")
NN_formulate!(oil_model, "models/NN_well_1.json", out, p; U_in=[1.66], L_in=[0.3])

@objective(oil_model, Min, out)
optimize!(oil_model)
objective_value(oil_model)
value(p)


df = CSV.read("data/well_1.csv", DataFrame)

x_range = LinRange{Float32}(minimum(df.PWH), maximum(df.PWH), 50)

plot(x_range, x -> forward_pass_NN!(oil_model, x, out, p), label="NN", title="WELL 1", xlabel="pressure", ylabel="flow")

display(scatter!(df.PWH, df.QOIL, label="Data"))