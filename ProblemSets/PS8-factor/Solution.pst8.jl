######################################################################
####             PST 8
######            Mahnaz karimpour               #####################
######################################################################




using Pkg
Pkg.add(["MultivariateStats", "DataFrames", "CSV", "HTTP", "Random", 
         "LinearAlgebra", "Statistics", "Optim", "DataFramesMeta", "GLM"])
         using MultivariateStats, DataFrames, CSV, HTTP, Random, LinearAlgebra
         using Statistics, Optim, DataFramesMeta, GLM


         using CSV, DataFrames, HTTP, GLM

# Load the dataset directly from GitHub
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS8-factor/nlsy.csv"
response = HTTP.get(url)
df = CSV.read(IOBuffer(response.body), DataFrame)

# Verify the first few rows
println("First 5 rows of the dataset:")
println(first(df, 5))

# Perform OLS regression with the given formula
ols = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), df)

# Print the summary and coefficients
println("\nRegression Summary:")
println(ols)

println("\nEstimated Coefficients:")
println(coef(ols))

######################################################################
####                        Questin 2
#
######################################################################
println("\n---------------- Question 2----------------")

# Create a list of the ASVAB variable names from the DataFrame
asvab_columns = [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]

# Select the ASVAB data from the DataFrame and convert it to a matrix
asvab_matrix = Matrix(df[:, asvab_columns])

# Calculate the correlation matrix for the ASVAB variables
asvab_corr = cor(asvab_matrix)

# Display the correlation matrix
println("\nCorrelation Matrix of ASVAB Variables:")
println(asvab_corr)




######################################################################
####                        Questin 3
#
######################################################################
println("\n---------------- Question 3----------------")

# Perform a new OLS regression including the six ASVAB variables
model_extended = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr+asvabAR+asvabCS+asvabMK+asvabNO+asvabPC+asvabWK), df)

# Print the estimated coefficients
println("\nCoefficients from the extended model with ASVAB variables:")
println(coef(model_extended))


######################################################################
####                        Questin 4
#
######################################################################

println("\n---------------- Question 4 ----------------")

using CSV, DataFrames, HTTP, GLM, MultivariateStats

# Step 1: Load the dataset
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS8-factor/nlsy.csv"
response = HTTP.get(url)
df = CSV.read(IOBuffer(response.body), DataFrame)

# Step 2: Convert the ASVAB data into a J × N matrix
asvab_vars = [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]
asvab_matrix = Matrix(select(df, asvab_vars))'  # J x N matrix (6 × 2438)

# Step 3: Apply PCA to retain the first principal component
pca_model = fit(PCA, asvab_matrix; maxoutdim=1)  # Fit PCA, retain 1 component
pca_scores = MultivariateStats.transform(pca_model, asvab_matrix)  # Get scores

# Step 4: Reshape the PCA scores into a vector for use in the regression
first_pc_scores = vec(pca_scores')  # Reshape to 1D vector

# Step 5: Add the principal component scores as a new column in the DataFrame
df[!, :FirstPC] = first_pc_scores  # Add 'FirstPC' column

# Step 6: Fit the regression model with the first principal component included
model_with_first_pc = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + FirstPC), df)

# Step 7: Print the coefficients of the model
println("\nCoefficients from the model with the first principal component:")
println(coef(model_with_first_pc))


######################################################################
####                        Questin 5
#
######################################################################

println("\n---------------- Question 5 ----------------")

using CSV, DataFrames, HTTP, GLM, MultivariateStats

# Load the dataset and select the ASVAB variables
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS8-factor/nlsy.csv"
df = CSV.read(IOBuffer(HTTP.get(url).body), DataFrame)
asvab_data = Matrix(select(df, [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]))'

# Apply Factor Analysis to extract the first factor
fa_model = fit(FactorAnalysis, asvab_data; maxoutdim=1)
first_factor = vec(MultivariateStats.transform(fa_model, asvab_data)')  # Convert to 1D

# Add the first factor as a new column to the DataFrame
df[!, :FirstFactor] = first_factor

# Fit the regression model with the factor included
model_with_factor = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + FirstFactor), df)

# Print the coefficients
println(coef(model_with_factor))

######################################################################
####                        Questin 6
#
######################################################################

println("\n---------------- Question 6 ----------------")

using Distributions, Optim

# Step 1: Data Preparation
D = Normal(0, 1)
ξ = rand(D, size(df, 1), 1)  # Latent factor ξ (2438×1 matrix)

# Extract ASVAB scores and covariates as matrices
M = Matrix(df[:, [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]])
Xm = Matrix(hcat(ones(size(df, 1)), df[:, [:black, :hispanic, :female]]))
X = Matrix(hcat(ones(size(df, 1)), df[:, [:black, :hispanic, :female, :schoolt, :gradHS, :grad4yr]]))
Y = df.logwage  # Log-wage vector
L = zeros(size(df, 1))  # Initialize likelihood values

# Step 2: Likelihood Function
function compute_likelihood(σ, α, β, γ, δ)
    for i in eachindex(Y)
        asvab_terms = prod(j -> pdf(Normal(0, σ[j]),
                        (M[i, j] - dot(Xm[i, :], α) - γ * ξ[i, 1]) / σ[j]), 1:6)
        wage_term = pdf(Normal(0, σ[7]),
                        (Y[i] - dot(X[i, :], β) - δ * ξ[i, 1]) / σ[7])
        L[i] = asvab_terms * wage_term
    end
    return L
end

# Step 3: Gauss-Legendre Quadrature
include("/Users/mahnazkarimpour/Desktop/fall-2024/ProblemSets/PS8-factor/lgwt.jl")
nodes, weights = lgwt(7, -4, 4)

# Step 4: Integrate the Likelihood
function integrate_likelihood(L)
    return [log(sum(weights .* L[i] .* pdf.(D, nodes))) for i in eachindex(L)]
end

# Step 5: Objective Function
function objective(σ)
    α, β = rand(4), rand(7)
    γ, δ = rand(), rand()
    L_vals = compute_likelihood(σ, α, β, γ, δ)
    return -sum(integrate_likelihood(L_vals))
end

# Step 6: Optimize Parameters
result = optimize(objective, rand(7), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000))

# Step 7: Print Results
println("Optimized Parameters: ", result.minimizer)



######################################################################
####                        Questin 7
#
######################################################################

println("\n---------------- Question 7 ----------------")
using Test  # For unit testing
using Distributions  # For statistical distributions
using Optim  # For optimization methods
using DataFrames  # For DataFrame operations

# Sample DataFrame for testing
df = DataFrame(
    asvabAR = randn(2438),
    asvabCS = randn(2438),
    asvabMK = randn(2438),
    asvabNO = randn(2438),
    asvabPC = randn(2438),
    asvabWK = randn(2438),
    black = rand([0, 1], 2438),
    hispanic = rand([0, 1], 2438),
    female = rand([0, 1], 2438),
    schoolt = rand([0, 1], 2438),
    gradHS = rand([0, 1], 2438),
    grad4yr = rand([0, 1], 2438),
    logwage = randn(2438)
)

# Create data variables for testing
D = Normal(0, 1)
ξ = rand(D, size(df, 1), 1)  # Latent factor ξ
M = hcat(df.asvabAR, df.asvabCS, df.asvabMK, df.asvabNO, df.asvabPC, df.asvabWK)
Xm = hcat(ones(size(df, 1)), df.black, df.hispanic, df.female)
X = hcat(ones(size(df, 1)), df.black, df.hispanic, df.female, df.schoolt, df.gradHS, df.grad4yr)
Y = df.logwage

# Likelihood function
function likelihood_function(σ, α, β, γ, δ)
    L = zeros(size(df, 1))
    for i in 1:size(df, 1)
        asvab_part = prod(j -> pdf(Normal(0, σ[j]),
            (M[i, j] - dot(Xm[i, :], α) - γ * ξ[i]) / σ[j]), 1:6)
        wage_part = pdf(Normal(0, σ[7]),
            (Y[i] - dot(X[i, :], β) - δ * ξ[i]) / σ[7])
        L[i] = asvab_part * wage_part
    end
    return L
end

# Integration function using Gauss-Legendre Quadrature
function integrate_likelihood(L)
    nodes, weights = lgwt(7, -4, 4)
    return [log(sum(weights .* L[i] .* pdf.(D, nodes))) for i in 1:length(L)]
end

# Objective function for optimization
function objective(σ)
    α, β = rand(4), rand(7)
    γ, δ = rand(), rand()
    L_vals = likelihood_function(σ, α, β, γ, δ)
    integrated_vals = integrate_likelihood(L_vals)
    return -sum(integrated_vals)
end

# Unit Tests for Question 7
@testset "Likelihood Function Test" begin
    σ = rand(7)
    α, β = rand(4), rand(7)
    γ, δ = rand(), rand()

    L_vals = likelihood_function(σ, α, β, γ, δ)
    @test size(L_vals) == (2438,)  # Check that the likelihood vector has correct size
end

@testset "Integration Function Test" begin
    L = rand(2438)  # Random likelihood values for testing
    integrated_vals = integrate_likelihood(L)
    @test length(integrated_vals) == 2438  # Ensure integrated values match size
end

@testset "Optimization Test" begin
    initial_guess = rand(7)
    result = optimize(objective, initial_guess, LBFGS())
    @test result.converged  # Ensure the optimization converged successfully
end
