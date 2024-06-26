"""
    function ICNN_formulate!(jump::JuMP.Model, filepath::String, output_var, input_vars...)

Formulates the input convex neural network (ICNN) as an LP into a JuMP model. The model parameters must be contained in a JSON file located in the given filepath. For more information about ICNNs see Amos et al. (2017).

JSON file structure:
- Parameter sets must be named based on the layer type  - either "FCx" or "SKIPx" where 'x' indicates the layer number.
- 'FC' indicates a fully connected layer. These parameter sets contain weights and biases.
- 'SKIP' indicates a skip connection layer. These only contain weights.
- For a clear example on how to export these parameters from Tensorflow, see examples/-folder of the package repository

This function modifies the JuMP model given as input. The input and output variable references that are given as input to this function will be linked to the appropriate variables in the ICNN formulation. No additional names are added to the JuMP model - all variables are added as anonymous. Also the JuMP model objective function is modified to include the ICNN output as a penalty term.

# Arguments

- `jump`: JuMP model where the LP formulation should be saved
- `filepath`: relative path to the JSON file containing the model parameters
- `output_var`: reference to the variable that should be linked to the ICNN output
- `input_vars`: references to the variables that will be used as the ICNN inputs

"""

function ICNN_formulate!(jump::JuMP.Model, filepath::String, output_var, input_vars...)

    @assert objective_sense(jump) in [MAX_SENSE, MIN_SENSE] "Objective sense (Min/Max) must be set."

    W = JSON.parsefile(filepath)
    # Weights: W[layer name][1][column num][row]
    # Biases: W[layer name][2][bias index]
    # names SKIP-2,3,...,k (skip connection)
    #       FC-1,2,...,k (fully connected)
    #       (k is output layer index)

    layers = parse.(Int, last.(filter(key -> occursin("FC", key), keys(W)))) # FC indices
    sort!(layers)

    n_neurons = Dict{Int, Int}() # layer => num. of neurons
    n_neurons[0] = length(W["FC1"][1]) # input length
    for layer in layers
        n_neurons[layer] = length(W["FC"*string(layer)][2])
    end

    z = @variable(jump, [layer=union(0, layers), 1:n_neurons[layer]])
    @constraint(jump, [layer=layers[1:end-1], neuron=1:n_neurons[layer]], z[layer, neuron] >= 0)

    input = [var for var in input_vars]
    @constraint(jump, [neuron=1:n_neurons[0]], z[0, neuron] == input[neuron])

    for layer in layers
        for neuron in 1:n_neurons[layer]

            FC_sum = @expression(jump, sum([W["FC"*string(layer)][1][neuron_last][neuron] * z[layer-1, neuron_last] for neuron_last in 1:n_neurons[layer-1]]))
            bias = @expression(jump, W["FC"*string(layer)][2][neuron])
            
            if layer == 1
                @constraint(jump, z[layer, neuron] >= FC_sum + bias)
            else
                SKIP_sum = @expression(jump, sum([W["SKIP"*string(layer)][1][input][neuron] * z[0, input] for input in 1:n_neurons[0]]))
                @constraint(jump, z[layer, neuron] >= FC_sum + SKIP_sum + bias)
            end
        end
    end

    # reference to output variable
    @constraint(jump, output_var == z[maximum(layers), 1])

    sign = objective_sense(jump) == JuMP.MAX_SENSE ? -1.0 : 1.0

    # modify objective to include penalty term
    @objective(jump, objective_sense(jump), objective_function(jump) + sign * output_var)
end

"""
    function forward_pass_ICNN!(jump, input, output_var, input_vars)

Calculates the forward pass through the ICNN (the output that is produced with the given input). 

# Arguments

- `jump`: JuMP model with the ICNN
- `input`: value to be forwaard passed
- `output_var`: reference to the ICNN output variable
- `input_vars`: references to the ICNN input variables

"""
function forward_pass_ICNN!(jump, input, output_var, input_vars)

    input_var = [var for var in input_vars]
    [fix(input_var[neuron], input[neuron]) for neuron in eachindex(input)]
    
    try
        optimize!(jump)
        return value(output_var)
    catch e
        println("Error with forward pass: $e")
        @warn "Input or output outside of bounds or incorrectly constructed model."
        return NaN
    end
end

function NN_formulate!(jump::JuMP.Model, filepath::String, output_var, input_vars...; U_in, L_in)

    W = JSON.parsefile(filepath)
    # Weights: W[layer name][1][column num][row]
    # Biases: W[layer name][2][bias index]
    # names FC-1,2,...,k (fully connected)
    #       (k is output layer index)

    layers = parse.(Int, last.(filter(key -> occursin("FC", key), keys(W)))) # FC indices
    sort!(layers)

    n_neurons = Dict{Int, Int}() # layer => num. of neurons
    n_neurons[0] = length(W["FC1"][1]) # input length
    for layer in layers
        n_neurons[layer] = length(W["FC"*string(layer)][2])
    end
    neurons(layer) = 1:n_neurons[layer]

    x = @variable(jump, [layer=union(0, layers), 1:n_neurons[layer]])
    s = @variable(jump, [layer=layers[1:end-1], 1:n_neurons[layer]])
    z = @variable(jump, [layer=layers[1:end-1], 1:n_neurons[layer]])
    @constraint(jump, [layer=layers[1:end-1], neuron=1:n_neurons[layer]], x[layer, neuron] >= 0)
    @constraint(jump, [layer=layers[1:end-1], neuron=1:n_neurons[layer]], s[layer, neuron] >= 0)

    input = [var for var in input_vars]
    @constraint(jump, [neuron=1:n_neurons[0]], x[0, neuron] == input[neuron])

    @constraint(jump, [neuron=1:n_neurons[0]], L_in[neuron] <= x[0, neuron] <= U_in[neuron])

    # U_bounds = Vector{Vector}(undef, length(layers))
    # L_bounds = Vector{Vector}(undef, length(layers))

    for layer in layers

        weights = W["FC"*string(layer)][1]
        biases = W["FC"*string(layer)][2]

        for neuron in 1:n_neurons[layer]
            
            FC_sum = @expression(jump, sum([W["FC"*string(layer)][1][neuron_last][neuron] * x[layer-1, neuron_last] for neuron_last in 1:n_neurons[layer-1]]))
            bias = @expression(jump, W["FC"*string(layer)][2][neuron])
            
            U_bound, L_bound = calculate_bounds(jump, layer, neuron, weights, biases, neurons, x)
            
            if layer == maximum(layers)
                @constraint(jump, x[layer, neuron] == FC_sum + bias)
                break
            end

            set_binary(z[layer, neuron])
            @constraint(jump, x[layer, neuron] <= max(0, U_bound) * z[layer, neuron])
            @constraint(jump, s[layer, neuron] <= max(0, -L_bound) * (1-z[layer, neuron]))

            @constraint(jump, x[layer, neuron] - s[layer, neuron] == FC_sum + bias)
        end
    end

    # reference to output variable
    @constraint(jump, output_var == x[maximum(layers), 1])
end

function calculate_bounds(model::JuMP.Model, layer, neuron, W, b, neurons, x)

    og_obj = objective_function(model)
    og_sense = objective_sense(model)

    @objective(model, Max, b[neuron] + sum(W[i][neuron] * x[layer-1, i] for i in neurons(layer-1)))
    optimize!(model)
    
    upper_bound = if termination_status(model) == OPTIMAL
        objective_value(model)
    else
        @warn "Layer $layer, neuron $neuron could not be solved to optimality."
        min(objective_bound(model), 1000.0)
    end

    set_objective_sense(model, MIN_SENSE)
    optimize!(model)
 
    lower_bound = if termination_status(model) == OPTIMAL
        objective_value(model)
    else
        @warn "Layer $layer, neuron $neuron could not be solved to optimality."
        max(objective_bound(model), -1000.0)
    end

    println("Neuron: $neuron, bounds: [$lower_bound, $upper_bound]")

    @objective(model, og_sense, og_obj)

    return upper_bound, lower_bound
end

function forward_pass_NN!(jump, input, output_var, input_vars)

    input_var = [var for var in input_vars]
    [fix(input_var[neuron], input[neuron]) for neuron in eachindex(input)]
    
    try
        optimize!(jump)
        return value(output_var)
    catch e
        println("Error with forward pass: $e")
        @warn "Input or output outside of bounds or incorrectly constructed model."
        return NaN
    end
end