######################################################################  
##                            Pset 6                               ###
##                          Your Name                              ###
#######################################################################

# Import necessary packages
import Pkg
Pkg.add("Optim")         
Pkg.add("HTTP")        
Pkg.add("GLM")           
Pkg.add("LinearAlgebra") 
Pkg.add("Random")        
Pkg.add("Statistics")    
Pkg.add("DataFrames")  
Pkg.add("CSV")          
Pkg.add("FreqTables")
Pkg.add("DataFramesMeta")

# Load the required libraries
using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables, DataFramesMeta

# Struct to encapsulate parameters for likebus function
struct DataParams
    zbin::Int
    xbin::Int
    xtran::Matrix{Float64}
    Zstate::Vector{Int}
    Xstate::Matrix{Int}
    B::Vector{Int}
    Y::Matrix{Int}
    X::Matrix{Float64}
    β::Float64
    N::Int
    T::Int
    xval::Vector{Float64}
end

# Create grids for transition matrices
function create_grids()
    function xgrid(theta, xval)
        N = length(xval)
        xub = vcat(xval[2:N], Inf)
        xtran1 = zeros(N, N)
        xtran1c = zeros(N, N)
        lcdf = zeros(N)
        for i in 1:N
            xtran1[:,i] = (xub[i] .>= xval) .* (1 .- exp.(-theta * (xub[i] .- xval)) .- lcdf)
            lcdf .+= xtran1[:,i]
            xtran1c[:,i] .+= lcdf
        end
        return xtran1, xtran1c
    end

    zval = collect(0.25:0.01:1.25)
    zbin = length(zval)
    xval = collect(0:0.125:25)
    xbin = length(xval)
    xtran = zeros(xbin * zbin, xbin)
    
    for z in 1:zbin
        xtran[(z-1)*xbin+1:z*xbin, :], _ = xgrid(zval[z], xval)
    end

    return zval, zbin, xval, xbin, xtran
end

# Data loading and preprocessing
function load_and_preprocess_data(url)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    df = @transform(df, :bus_id = 1:size(df, 1))
    return df
end

# Reshape the data into long format
function reshape_data(df)
    dfy = @select(df, :bus_id, :Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10, :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20, :RouteUsage, :Branded)
    dfy_long = stack(dfy, Not([:bus_id, :RouteUsage, :Branded]))
    rename!(dfy_long, :value => :Y)
    dfy_long = @transform(dfy_long, :time = kron(collect(1:20), ones(size(df, 1))))
    select!(dfy_long, Not(:variable))

    dfx = @select(df, :bus_id, :Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, :Odo7, :Odo8, :Odo9, :Odo10, :Odo11, :Odo12, :Odo13, :Odo14, :Odo15, :Odo16, :Odo17, :Odo18, :Odo19, :Odo20)
    dfx_long = stack(dfx, Not([:bus_id]))
    rename!(dfx_long, :value => :Odometer)
    dfx_long = @transform(dfx_long, :time = kron(collect(1:20), ones(size(df, 1))))
    select!(dfx_long, Not(:variable))

    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id, :time], makeunique=true)
    sort!(df_long, [:bus_id, :time])
    
    return df_long
end

# Static model estimation
function estimate_static_model(df_long)
    θ̂_glm = glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink())
    return θ̂_glm
end

# Prepare dynamic data for model
function prepare_dynamic_data(df)
    Y = Matrix(df[:, [:Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10, :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20]])
    X = Matrix(df[:, [:Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, :Odo7, :Odo8, :Odo9, :Odo10, :Odo11, :Odo12, :Odo13, :Odo14, :Odo15, :Odo16, :Odo17, :Odo18, :Odo19, :Odo20]])
    Z = Vector(df[:, :RouteUsage])
    B = Vector(df[:, :Branded])
    N = size(Y, 1)
    T = size(Y, 2)
    Xstate = Matrix(df[:, [:Xst1, :Xst2, :Xst3, :Xst4, :Xst5, :Xst6, :Xst7, :Xst8, :Xst9, :Xst10, :Xst11, :Xst12, :Xst13, :Xst14, :Xst15, :Xst16, :Xst17, :Xst18, :Xst19, :Xst20]])
    Zstate = Vector(df[:, :Zst])
    
    return Y, X, Z, B, N, T, Xstate, Zstate
end

# Dynamic model likelihood function
@views @inbounds function likebus(θ, d::DataParams)
    FV = zeros(d.zbin * d.xbin, 2, d.T+1)
    
    for t = d.T:-1:1
        for b in 0:1
            for z in 1:d.zbin
                for x in 1:d.xbin
                    row = x + (z-1) * d.xbin
                    v1 = θ[1] + θ[2] * d.xval[x] + θ[3] * b + d.xtran[row, :] ⋅ FV[(z-1) * d.xbin+1:z * d.xbin, b+1, t+1]
                    v0 = d.xtran[1 + (z-1) * d.xbin, :] ⋅ FV[(z-1) * d.xbin+1:z * d.xbin, b+1, t+1]
                    FV[row, b+1, t] = d.β * log(exp(v1) + exp(v0))
                end
            end
        end
    end

    like = 0
    for i in 1:d.N
        row0 = (d.Zstate[i]-1) * d.xbin + 1
        for t in 1:d.T
            row1 = d.Xstate[i, t] + (d.Zstate[i]-1) * d.xbin
            v1 = θ[1] + θ[2] * d.X[i, t] + θ[3] * d.B[i] + (d.xtran[row1, :] .- d.xtran[row0, :]) ⋅ FV[row0:row0+d.xbin-1, d.B[i]+1, t+1]
            dem = 1 + exp(v1)
            like -= (d.Y[i, t] == 1) * v1 - log(dem)
        end
    end
    return like
end

# Dynamic model estimation using the optimizer
function estimate_dynamic_model(df)
    # Prepare dynamic data
    Y, X, Z, B, N, T, Xstate, Zstate = prepare_dynamic_data(df)

    # Create grids
    zval, zbin, xval, xbin, xtran = create_grids()
    
    # Create the DataParams struct
    data_parms = DataParams(zbin, xbin, xtran, Zstate, Xstate, B, Y, X, 0.9, N, T, xval)

    # True parameters for testing
    θ_true = [2.0, -0.15, 1.0]

    println("Timing evaluation of the likelihood function")
    @time likebus(θ_true, data_parms)

    θ̂_optim = optimize(a -> likebus(a, data_parms), θ_true, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 100_000, show_trace = true))
    return θ̂_optim.minimizer
end

# Main function to run the analysis
function run_analysis()
    url_static = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
    df_static = load_and_preprocess_data(url_static)
    df_long = reshape_data(df_static)
    return df_long
end

# Load the dataset and process it for analysis
df_long = run_analysis()

######################################################################  
##                       Question 2                          ###
#######################################################################
println("\n---------------- Question 2 ----------------")

logit_formula = @formula(Y ~ Odometer + Odometer^2 + RouteUsage + RouteUsage^2 + 
                         Branded + time + time^2 + 
                         Odometer * RouteUsage * Branded * time * time^2)

logit_model = glm(logit_formula, df_long, Binomial(), LogitLink())

println(logit_model)


######################################################################  
##                       Question 3a                          ###
#######################################################################
println("\n---------------- Question 3a ----------------")

zval, zbin, xval, xbin, xtran = create_grids()

######################################################################  
##                       Question 3b                          ###
#######################################################################
######################################################################  
##                       Question 3b                          ###
#######################################################################
println("\n---------------- Question 3b ----------------")

# First, make sure the data is loaded and preprocessed correctly
url_dynamic = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
df_dynamic = load_and_preprocess_data(url_dynamic)

# Define the dynamic data including Y using the prepare_dynamic_data function
Y, X, Z, B, N, T, Xstate, Zstate = prepare_dynamic_data(df_dynamic)

# Define the discount factor β
β = 0.9

# Expand Zstate and B to match the size of df_long (20,000 rows)
n_times = 20  # Number of time periods for each bus
Zstate = repeat(Zstate, inner=n_times)
B = repeat(B, inner=n_times)
Xstate_flattened = vec(Xstate)

# Check the lengths to ensure consistency before adding to df_long
println("Length of expanded Zstate: ", length(Zstate))
println("Length of expanded Xstate_flattened: ", length(Xstate_flattened))
println("Length of expanded B: ", length(B))
println("Length of df_long: ", nrow(df_long))

# Ensure lengths match before adding
if length(Zstate) == nrow(df_long) && length(Xstate_flattened) == nrow(df_long) && length(B) == nrow(df_long)
    df_long.Zstate = Zstate
    df_long.Xstate = Xstate_flattened
    df_long.B = B
else
    println("Error: Mismatch in lengths of df_long and expanded vectors.")
end

# Create grids and transition matrices
zval, zbin, xval, xbin, xtran = create_grids()

# Compute future value using predictions from the logit model (correcting the call)
state_space = DataFrame(
    Odometer = kron(ones(zbin), xval),
    RouteUsage = kron(zval, ones(xbin)),
    Branded = zeros(zbin * xbin),
    time = zeros(zbin * xbin)
)
state_space.Mileage2 = state_space.Odometer.^2
state_space.RouteUsage2 = state_space.RouteUsage.^2
state_space.Time2 = state_space.time.^2

# Use the predict function to get the predicted probabilities
p0 = predict(logit_model, state_space)

# Ensure p0 is converted to a Vector{Float64} if necessary
p0 = Vector{Float64}(p0)

# Ensure all inputs are correct before calling the function
println("Type of p0: ", typeof(p0))            # Should be Vector{Float64}
println("Type of zval: ", typeof(zval))        # Should be Vector{Float64}
println("Type of xval: ", typeof(xval))        # Should be Vector{Float64}
println("Type of zbin: ", typeof(zbin))        # Should be Int
println("Type of xbin: ", typeof(xbin))        # Should be Int
println("Type of xtran: ", typeof(xtran))      # Should be Matrix{Float64}
println("Type of β: ", typeof(β))              # Should be Float64

# Now, pass the predicted probabilities into the compute_future_value function
# Function to compute future value terms
function compute_future_value(p0::Vector{Float64}, zval::Vector{Float64}, xval::Vector{Float64}, zbin::Int, xbin::Int, xtran::Matrix{Float64}, β::Float64)
    # Create state space for prediction
    FV = zeros(zbin * xbin, 2, 21)  # 21 because we have T=20 and need T+1

    # Backward recursion
    for t in 20:-1:1
        for b in 0:1
            # Use the predicted probabilities p0 here to compute future values
            FV[:, b+1, t] = -β .* log.(p0)
        end
    end

    return FV
end


# Map future values to data and add it to df_long
FVT1_flattened = map_future_values_to_data(df_long, future_value, xtran, xbin, Zstate, Xstate_flattened, B, T, zbin)
df_long = add_future_value_to_df(df_long, FVT1_flattened)




######################################################################  
##                       Question 3c                          ###
#######################################################################

println("\n---------------- Question 3c ----------------")

# 3c: Estimate structural parameters using a function similar to estimate_dynamic_model
function estimate_structural_parameters(df_long, FV, xtran, xbin, β, xval, zval)
    # θ_true: Initial guess for the parameters
    θ_true = [2.0, -0.15, 1.0]  # Initial guess

    # Using the likebus function for likelihood evaluation and optimization
    θ̂_optim = optimize(a -> likebus(a, DataParams(zbin, xbin, xtran, Zstate, Xstate, B, Y, X, β, N, T, xval)), θ_true, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 100_000, show_trace = true))
    
    θ_hat = θ̂_optim.minimizer
    return θ_hat
end

# Now, estimate structural parameters for part 3c
θ_hat = estimate_structural_parameters(df_long, future_value, xtran, xbin, β, xval, zval)
println("Structural parameters estimated:")
println("θ_0 = ", θ_hat[1])
println("θ_1 = ", θ_hat[2])
println("θ_2 = ", θ_hat[3])

######################################################################  
##                       Question 3d-3f                          ###
#######################################################################
println("\n---------------- Question 3d-3f ----------------")

# Wrapping everything in a function (which is already in your `main()` function)
# For example, your friend's approach suggests printing after parameter estimation

# Timing the execution of the entire process
println("Timing the entire estimation process:")
@time main()

# Run the main function
main()

######################################################################  
##                       Question 4                             ###
#######################################################################

using Test
@testset "Test load_and_preprocess_data" begin
    df = load_and_preprocess_data("https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv")
    @test typeof(df) == DataFrame
    @test all(["Y1", "Odo1", "RouteUsage", "Branded"] .∈ names(df))
end

@testset "Test prepare_dynamic_data" begin
    df = load_and_preprocess_data("https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv")
    Y, X, Z, B, N, T, Xstate, Zstate = prepare_dynamic_data(df)
    @test size(Y) == (size(df, 1), 20)
    @test size(X) == (size(df, 1), 20)
    @test length(Z) == size(df, 1)
    @test length(B) == size(df, 1)
end

@testset "Test create_grids" begin
    zval, zbin, xval, xbin, xtran = create_grids()
    @test length(zval) > 0
    @test length(xval) > 0
    @test size(xtran) == (xbin * zbin, xbin)
end

@testset "Test compute_future_value" begin
    df_long = load_and_preprocess_data("https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv")
    logit_model = glm(@formula(Y1 ~ Odo1 + Branded), df_long, Binomial(), LogitLink())
    zval, zbin, xval, xbin, xtran = create_grids()
    future_value = compute_future_value(df_long, logit_model, xval, zval, xbin, zbin, 20, 0.9, xtran)
    @test size(future_value) == (xbin * zbin, 2, 21)
end

@testset "Test map_future_values_to_data" begin
    df_long = load_and_preprocess_data("https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv")
    zval, zbin, xval, xbin, xtran = create_grids()
    logit_model = glm(@formula(Y1 ~ Odo1 + Branded), df_long, Binomial(), LogitLink())
    future_value = compute_future_value(df_long, logit_model, xval, zval, xbin, zbin, 20, 0.9, xtran)
    FVT1_flattened = map_future_values_to_data(df_long, future_value, xtran, xbin, Zstate, Xstate_flattened, B, 20, zbin)
    @test length(FVT1_flattened) == nrow(df_long)
end

@testset "Test add_future_value_to_df" begin
    df_long = load_and_preprocess_data("https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv")
    FVT1_flattened = rand(nrow(df_long))  # Simulating future values
    df_long = add_future_value_to_df(df_long, FVT1_flattened)
    @test "fv" ∈ names(df_long)
    @test length(df_long.fv) == nrow(df_long)
end

@testset "Test GLM estimation with CCPs" begin
    df_long = load_and_preprocess_data("https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv")
    FVT1_flattened = rand(nrow(df_long))  # Simulating future values
    df_long = add_future_value_to_df(df_long, FVT1_flattened)
    theta_hat_ccp_glm = glm(@formula(Y1 ~ Odo1 + Branded), df_long, Binomial(), LogitLink(), offset=df_long.fv)
    @test typeof(theta_hat_ccp_glm) == GLM.GeneralizedLinearModel
end

using Test
runtests()  # This will run all defined @testsets
