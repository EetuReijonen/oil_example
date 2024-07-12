using CSV
using DataFrames
using Plots
using JuMP
using Gurobi
using Gogeta

### WELL PRESSURE CURVES ###

for w in 1:8
    for model in ["NN", "ICNN"]
        oil_model = Model(Gurobi.Optimizer)
        set_silent(oil_model)
        @objective(oil_model, Max, 0)

        @variable(oil_model, p)
        @variable(oil_model, out)

        df = CSV.read("data/well_$w.csv", DataFrame)

        if model == "NN"
            NN_formulate!(oil_model, "models/NN_well_$w.json", out, p; U_in=maximum(df.PWH), L_in=minimum(df.PWH))
        elseif model == "ICNN"
            ICNN_formulate!(oil_model, "models/ICNN_well_$w.json", out, p)
        end

        x_range = LinRange{Float32}(minimum(df.PWH), maximum(df.PWH), 50)

        if model == "NN"
            plot(x_range, x -> -forward_pass_NN!(oil_model, x, out, p), label="NN", title="WELL $w", xlabel="pressure", ylabel="flow")
        elseif model == "ICNN"
            plot(x_range, x -> -forward_pass_ICNN!(oil_model, x, out, p), label="ICNN", title="WELL $w", xlabel="pressure", ylabel="flow")
        end

        display(scatter!(df.PWH, df.QOIL, label="Data"))
    end
end

### FLOWLINE PRESSURE ###

for model in ["NN", "ICNN"]
    df = CSV.read("data/flowline_1.csv", DataFrame)

    oil_model = Model(Gurobi.Optimizer)
    @objective(oil_model, Max, 0)
    @variable(oil_model, qoil)
    @variable(oil_model, qgas)
    @variable(oil_model, qwater)
    @variable(oil_model, pus)
    @variable(oil_model, pds)
        
    set_silent(oil_model)

    if model == "ICNN"
        ICNN_formulate!(oil_model, "models/ICNN_flowline_1.json", pds, qoil, qgas, qwater, pus)
        # ICNN_formulate!(oil_model, "models/ICNN_flowline_negated.json", pds, qoil, qgas, qwater, pus)
    elseif model == "NN"
        NN_formulate!(oil_model, "models/NN_flowline_1.json", pds, qoil, qgas, qwater, pus; U_in=[maximum(df.QOIL), maximum(df.QGAS), maximum(df.QWAT), maximum(df.PUS)], L_in=[minimum(df.QOIL), minimum(df.QGAS), minimum(df.QWAT), minimum(df.PUS)])
    end

    pds = if model == "NN" 
        map(row -> forward_pass_NN!(oil_model, collect(row[2:end-1]), pds, [qoil, qgas, qwater, pus]), eachrow(df))
    elseif model == "ICNN"
        map(row -> -forward_pass_ICNN!(oil_model, collect(row[2:end-1]), pds, [qoil, qgas, qwater, pus]), eachrow(df))
    end

    scatter(df.QOIL, df.PDS, markersize=2, markerstrokewidth=0, label="data", title="FLOWLINE", xlabel="oil flow", ylabel="downstream pressure")
    display(scatter!(df.QOIL, pds, markersize=2, markerstrokewidth=0, label=model))

    scatter(df.QGAS, df.PDS, markersize=2, markerstrokewidth=0, label="data", title="FLOWLINE", xlabel="gas flow", ylabel="downstream pressure")
    display(scatter!(df.QGAS, pds, markersize=2, markerstrokewidth=0, label=model))

    scatter(df.QWAT, df.PDS, markersize=2, markerstrokewidth=0, label="data", title="FLOWLINE", xlabel="water flow", ylabel="downstream pressure")
    display(scatter!(df.QWAT, pds, markersize=2, markerstrokewidth=0, label=model))

    scatter(df.PUS, df.PDS, markersize=2, markerstrokewidth=0, label="data", title="FLOWLINE", xlabel="upstream pressure", ylabel="downstream pressure")
    display(scatter!(df.PUS, pds, markersize=2, markerstrokewidth=0, label=model))

end

### ONLY FLOWLINE DATA ###
df = CSV.read("data/flowline_1.csv", DataFrame)

scatter(df.QOIL, -df.PDS, markersize=2, markerstrokewidth=0, label="data", title="FLOWLINE", xlabel="oil flow", ylabel="downstream pressure")

scatter(df.QGAS, -df.PDS, markersize=2, markerstrokewidth=0, label="data", title="FLOWLINE", xlabel="gas flow", ylabel="downstream pressure")

scatter(df.QWAT, -df.PDS, markersize=2, markerstrokewidth=0, label="data", title="FLOWLINE", xlabel="water flow", ylabel="downstream pressure")

scatter(df.PUS, -df.PDS, markersize=2, markerstrokewidth=0, label="data", title="FLOWLINE", xlabel="upstream pressure", ylabel="downstream pressure")