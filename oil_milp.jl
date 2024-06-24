# Oil optimization problem as described in Grimstad and Anderson (2019)

### GRAPH STRUCTURE ###

Nw = [i for i in 1:8]
Nm = [9, 10]
Ns = [11, 12]
N = union(Nw, Nm, Ns)

Ed = vec([(i, j) for i in 1:8, j in [9, 10]])
Er = [(9, 11), (10, 12)]
E = union(Ed, Er)

Ein = Vector{Vector}(undef, 12)
[Ein[i] = [] for i in 1:8]
Ein[9] = [(i, 9) for i in 1:8]
Ein[10] = [(i, 10) for i in 1:8]
Ein[11] = [(9, 11)]
Ein[12] = [(10, 12)]

Eout = Vector{Vector}(undef, 12)
[Eout[i] = [(i, 9), (i, 10)] for i in 1:8]
Eout[9] = [(9, 11)]
Eout[10] = [(10, 12)]
Eout[11] = []
Eout[12] = []

C = [1, 2, 3] # oil, gas, water

### CONSTANTS ###

P_LIMS = [ # pressure limits for each node
    (0.3, 1.65329),
    (0.3, 1.55873),
    (0.3, 1.76193),
    (0.3, 1.51174),
    (0.3, 1.44545),
    (0.3, 1.32290),
    (0.3, 1.62300),
    (0.3, 1.24020),
    (0.299867, 2.09987),
    (0.299867, 2.09987),
    (0.005022, 1.33179),
    (0.005022, 1.33179),
]

Q_OIL = [ # oil flow bounds from the wells
    (0.0, 0.240734),
    (0.0, 0.284504),
    (0.0, 0.190091),
    (0.0, 0.162084),
    (0.0, 0.090637),
    (0.0, 0.232649),
    (0.0, 0.269781),
    (0.0, 0.323648),

]

Q_GAS = (0.0, 10.0) # gas flow bounds from the wells
# totally redundant because the GOR and WOR constraints lock the water and gas flow to the oil flow

Q_WATER = (0.0, 10.0) # water flow bounds from the wells
# totally redundant because the GOR and WOR constraints lock the water and gas flow to the oil flow

GOR = [1053.0, 805.0, 789.0, 790.0, 633.0, 745.0, 742.0, 759.0] .* 0.001 # gas-oil ratio

WCT = [12.7, 23.7, 41.7, 55.4, 54.81, 27.0, 0, 0] * 0.01 # water contents
WOR = map(wct -> wct/(1-wct), WCT) # water-oil ratio

P_SEP = [0.20, 0.30] # separator pressures

### JUMP MODEL ###

using JuMP
using Gurobi
using JSON

include("ICNN_to_LP.jl")

oil_model = Model(Gurobi.Optimizer)
set_silent(oil_model)

@variable(oil_model, p[N])
@variable(oil_model, q[E, C])
@variable(oil_model, y[Ed], Bin) # 14l

# 14a
@objective(oil_model, Max, sum([q[e, 1] for e in Er]))

# 14b
@constraint(oil_model, [c=C, i=Nm], sum([q[e, c] for e in Ein[i]]) == sum([q[e, c] for e in Eout[i]]))

# 14c
@variable(oil_model, g[e=Er])
[ICNN_formulate!(oil_model, "models/ICNN_flowline_negated.json", g[e], q[e, 1], q[e, 2], q[e, 3], p[e[1]]) for e in Er]
# TODO unoptimized code, bounds are recalculated both the same model twice
# [NN_formulate!(oil_model, "models/NN_flowline_1.json", g[e], q[e, 1], q[e, 2], q[e, 3], p[e[1]]; U_in=[1.8, 1.92, 0.96, 2.09987], L_in=[0.0, 0.024, 0.0, 0.299867]) for e in Er] # these bounds are from the dataset that was used to train the flowline models
@constraint(oil_model, [e in Er], p[e[2]] == -g[e])

# 14d
@constraint(oil_model, [e=Ed], p[e[1]] - p[e[2]] <= (P_LIMS[e[1]][2] - P_LIMS[e[2]][1]) * (1 - y[e]))
@constraint(oil_model, [e=Ed], (-P_LIMS[e[2]][2] + P_LIMS[e[1]][1]) * (1 - y[e]) <= p[e[1]] - p[e[2]])

# 14e
@constraint(oil_model, [i=Nw], sum([y[e] for e in Eout[i]]) <= 1)

# 14f
@constraint(oil_model, [e=Ed], y[e] * Q_OIL[e[1]][1] <= q[e, 1])
@constraint(oil_model, [e=Ed], q[e, 1] <= y[e] * Q_OIL[e[1]][2])

# gas
@constraint(oil_model, [e=Ed], y[e] * Q_GAS[1] <= q[e, 2])
@constraint(oil_model, [e=Ed], q[e, 2] <= y[e] * Q_GAS[2])

# water
@constraint(oil_model, [e=Ed], y[e] * Q_WATER[1] <= q[e, 3])
@constraint(oil_model, [e=Ed], q[e, 3] <= y[e] * Q_WATER[2])

# 14g pressure limits at each node
@constraint(oil_model, [i=N], P_LIMS[i][1] <= p[i] <= P_LIMS[i][2])

# 14h
@variable(oil_model, f[i=Nw])
[ICNN_formulate!(oil_model, "models/ICNN_well_$i.json", f[i], p[i]) for i in Nw]
# [NN_formulate!(oil_model, "models/NN_well_$i.json", f[i], p[i]; U_in=P_LIMS[i][2], L_in=P_LIMS[i][1]) for i in Nw]
@constraint(oil_model, [i=Nw], sum([q[e, 1] for e in Eout[i]]) == -f[i])

# 14i
@constraint(oil_model, [i=Nw], sum([q[e, 2] for e in Eout[i]]) == GOR[i] * sum([q[e, 1] for e in Eout[i]]))

# 14j
@constraint(oil_model, [i=Nw], sum([q[e, 3] for e in Eout[i]]) == WOR[i] * sum([q[e, 1] for e in Eout[i]]))

# 14k
@constraint(oil_model, [i=Ns], p[i] == P_SEP[i-10])

### SOLUTION ###

objective_function(oil_model)
unset_silent(oil_model)
optimize!(oil_model)
solution_summary(oil_model)

sum([value(q[e, 1]) for e in Er])