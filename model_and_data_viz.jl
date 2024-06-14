using CSV
using DataFrames
using Plots
using JuMP
using GLPK # because my Gurobi license is expired
using JSON

include("ICNN_to_LP.jl")

for w in 1:8
    oil_model = Model(GLPK.Optimizer)
    @objective(oil_model, Max, 0)

    @variable(oil_model, p)
    @variable(oil_model, out)

    ICNN_formulate!(oil_model, "models/ICNN_well_$w.json", out, p)

    df = CSV.read("data/well_$w.csv", DataFrame)

    x_range = LinRange{Float32}(minimum(df.PWH), maximum(df.PWH), 50)

    plot(x_range, x -> -forward_pass_ICNN!(oil_model, x, out, p), label="ICNN", title="WELL $w", xlabel="pressure", ylabel="flow")

    display(scatter!(df.PWH, df.QOIL, label="Data"))
end

df = CSV.read("data/flowline_1.csv", DataFrame)
scatter(df.QOIL, df.PDS, markersize=1, label="data", title="FLOWLINE PRESSURE", xlabel="oil flow")
scatter(df.QGAS, df.PDS, markersize=1, label="data", title="FLOWLINE PRESSURE", xlabel="gas flow")
scatter(df.QWAT, df.PDS, markersize=1, label="data", title="FLOWLINE PRESSURE", xlabel="water flow")
scatter(df.PUS, df.PDS, markersize=1, label="data", title="FLOWLINE PRESSURE", xlabel="upstream pressure")
