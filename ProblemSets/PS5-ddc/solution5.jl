######################################################################  
##                       Pset 5                                   ####
##                       Mahnaz Karimpour                          ###
#######################################################################

# Install and import necessary packages if not already installed
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

using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using DataFramesMeta
using CSV
using HTTP
using GLM

# Function to create state transitions for dynamic model
function create_grids()

    function xgrid(theta, xval)
        N = length(xval)
        xub = vcat(xval[2:N], Inf)
        xtran1 = zeros(N, N)
        xtran1c = zeros(N, N)
        lcdf = zeros(N)
        for i in 1:N
            xtran1[:, i] = (xub[i] .>= xval) .* (1 .- exp.(-theta * (xub[i] .- xval)) .- lcdf)
            lcdf += xtran1[:, i]
            xtran1c[:, i] += lcdf
        end
        return xtran1, xtran1c
    end

    zval = collect(0.25:0.01:1.25)
    zbin = length(zval)
    xval = collect(0:0.125:25)
    xbin = length(xval)
    tbin = xbin * zbin
    xtran = zeros(tbin, xbin)
    xtranc = zeros(xbin, xbin, zbin)
    for z in 1:zbin
        xtran[(z - 1) * xbin + 1:z * xbin, :], xtranc[:, :, z] = xgrid(zval[z], xval)
    end

    return zval, zbin, xval, xbin, xtran
end

# Wrapper function that reads data and reshapes it
function wrapper()
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 1: Read in Data
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # Create bus ID variable
    df = @transform(df, :bus_id = 1:size(df, 1))

    #---------------------------------------------------
    # Reshape from wide to long format (two-step process)
    #---------------------------------------------------

    # 1. Reshape the decision variable (Y1 to Y20)
    dfy = @select(df, :bus_id, :Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10,
                  :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20,
                  :RouteUsage, :Branded)

    dfy_long = DataFrames.stack(dfy, Not([:bus_id, :RouteUsage, :Branded]))
    rename!(dfy_long, :value => :Y)  # Rename the stacked values to 'Y'

    # Add a 'time' variable using kron (which repeats time for each bus)
    dfy_long = @transform(dfy_long, :time = kron(collect(1:20), ones(size(df, 1))))
    select!(dfy_long, Not(:variable))  # Remove the 'variable' column

    # 2. Reshape the odometer variable (Odo1 to Odo20)
    dfx = @select(df, :bus_id, :Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, :Odo7, 
                  :Odo8, :Odo9, :Odo10, :Odo11, :Odo12, :Odo13, :Odo14, 
                  :Odo15, :Odo16, :Odo17, :Odo18, :Odo19, :Odo20)

    dfx_long = DataFrames.stack(dfx, Not(:bus_id))
    rename!(dfx_long, :value => :Odometer)  # Rename the stacked values to 'Odometer'

    # Add a 'time' variable to match the reshaped data
    dfx_long = @transform(dfx_long, :time = kron(collect(1:20), ones(size(df, 1))))
    select!(dfx_long, Not(:variable))  # Remove the 'variable' column

    #---------------------------------------------------
    # Join reshaped datasets and sort
    #---------------------------------------------------

    # Join the reshaped dataframes on 'bus_id' and 'time'
    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id, :time])

    # Sort the data by 'bus_id' and 'time'
    sort!(df_long, [:bus_id, :time])

    return df_long
end

# Execute the wrapper function and display the long format DataFrame
df_long = wrapper()
display(df_long)

# Optionally, call create_grids() if needed in the same script
zval, zbin, xval, xbin, xtran = create_grids()
display(zval)

##########################################################################################
#                Question 2
##########################################################################################

println("\n---------------- Question 2 ----------------")

# Assuming 'df_long' is already prepared and includes the variables 'Y', 'Odometer', and 'Branded'
# Estimate the static logit model using GLM

# Step 1: Estimate the static logit model
logit_model = glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink())

# Step 2: Output the estimated coefficients
println("Estimated coefficients for the static logit model:")
println(coeftable(logit_model))


##########################################################################################
 #                Question 3a
#########################################################################################################
println("\n---------------- Question 3a ----------------")
# Step 1: Load the data
function load_data()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdata.csv"  # Remove Beta0 as per instruction
    df = CSV.read(HTTP.get(url).body, DataFrame)
    return df
end

# Step 2: Convert columns to wide format matrices
function convert_to_matrices(df)
    # Create the Y matrix from Y1 to Y20
    Y = Matrix(df[:, [:Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10, 
                      :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20]])
    
    # Create the Odo matrix from Odo1 to Odo20
    Odo = Matrix(df[:, [:Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, :Odo7, 
                        :Odo8, :Odo9, :Odo10, :Odo11, :Odo12, :Odo13, 
                        :Odo14, :Odo15, :Odo16, :Odo17, :Odo18, :Odo19, :Odo20]])

    # Create the Xst matrix from Xst1 to Xst20 (if applicable)
    Xst = Matrix(df[:, [:Xst1, :Xst2, :Xst3, :Xst4, :Xst5, :Xst6, :Xst7, 
                        :Xst8, :Xst9, :Xst10, :Xst11, :Xst12, :Xst13, 
                        :Xst14, :Xst15, :Xst16, :Xst17, :Xst18, :Xst19, :Xst20]])
    
    return Y, Odo, Xst
end

# Step 3: Load data and convert to matrices
df = load_data()
Y, Odo, Xst = convert_to_matrices(df)

# Step 4: Output dimensions for verification
println("Dimensions of Y: ", size(Y))    # Expected (1000, 20)
println("Dimensions of Odo: ", size(Odo))  # Expected (1000, 20)
println("Dimensions of Xst: ", size(Xst))  # Expected (1000, 20)

# You now have Y, Odo, and Xst in wide format matrices with dimensions 1000x20

##########################################################################################
 #                Question 3b
#########################################################################################################
println("\n---------------- Question 3b ----------------")
# Step 1: Generate the state transition matrices using create_grids()

zval, zbin, xval, xbin, xtran = create_grids()

# Step 2: Output the generated grids and transition matrices for verification
println("zval (Route usage grid): ", zval)
println("xval (Odometer reading grid): ", xval)
println("zbin (Number of bins for zval): ", zbin)
println("xbin (Number of bins for xval): ", xbin)
println("xtran (Transition matrix dimensions): ", size(xtran))

# zval, xval, and xtran are now ready to be used in the dynamic model


##########################################################################################
#                QUESTION 3c: Future Value Computation
##########################################################################################
println("\n---------------- Question 3c ----------------")
T = 20
beta = 0.9
FV = zeros(size(xtran, 1), 2, T+1)

for t in T:-1:1
    for b in 0:1
        for z in 1:zbin
            for x in 1:xbin
                row = x + (z-1)*xbin
                v_lt_flow = -xval[x] + 100 * b
                v_lt_expected = dot(xtran[row, :], FV[(z-1)*xbin+1:z*xbin, b+1, t+1])
                v_lt = v_lt_flow + v_lt_expected

                v_0t_flow = 50 * b
                v_0t_expected = dot(xtran[1+(z-1)*xbin, :], FV[(z-1)*xbin+1:z*xbin, b+1, t+1])
                v_0t = v_0t_flow + v_0t_expected

                FV[row, b+1, t] = beta * log(exp(v_0t) + exp(v_lt))
            end
        end
    end
end

##########################################################################################
#                QUESTION 3d: Construct the log-likelihood
##########################################################################################
println("\n---------------- Question 3d ----------------")
log_likelihood = 0.0

for bus in 1:size(df, 1)
    for t in 1:T
        x_obs = df[bus, Symbol("Xst$t")]
        z_obs = df[bus, Symbol("Zst$t")]
        b_obs = df[bus, :Branded]

        row_replace = 1 + (z_obs-1)*xbin
        row_noreplace = x_obs + (z_obs-1)*xbin

        v_lt_flow = -xval[x_obs] + 100 * b_obs
        v_0t_flow = 50 * b_obs

        fv_noreplace = dot(xtran[row_noreplace, :], FV[(z_obs-1)*xbin+1:z_obs*xbin, b_obs+1, t+1])
        fv_replace = dot(xtran[row_replace, :], FV[(z_obs-1)*xbin+1:z_obs*xbin, b_obs+1, t+1])

        v_lt = v_lt_flow + fv_noreplace
        v_0t = v_0t_flow + fv_replace

        choice_prob = exp(v_lt) / (exp(v_lt) + exp(v_0t))

        log_likelihood += log(choice_prob)
    end
end

##########################################################################################
#                QUESTION 3e-h: Optimization
##########################################################################################

@views @inbounds function log_likelihood_fn(θ)
    return -log_likelihood  # Return negative log likelihood for minimization
end

initial_θ = randn(10)
result = optimize(log_likelihood_fn, initial_θ, LBFGS())

##########################################################################################
#                QUESTION 4: Unit Tests for Functions
##########################################################################################

@testset "Unit Tests" begin
    # Test wrapper function for Question 1
    df_long = wrapper()
    @test !isempty(df_long)
    @test :Y in names(df_long)
    @test :Odometer in names(df_long)
    
    # Test static logit model
    logit_model = glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink())
    coefs = coef(logit_model)
    @test length(coefs) == 3

    # Test create_grids function
    zval, zbin, xval, xbin, xtran = create_grids()
    @test length(zval) > 0
    @test length(xval) > 0
    @test size(xtran) == (xbin * zbin, xbin)
end