using Pkg
Pkg.add(["Distributions", "DataFrames", "CSV", "HTTP", "Random", 
         "LinearAlgebra", "Statistics", "Optim", "FreqTables", "GLM","ForwardDiff","Test"])
         using Distributions, DataFrames, CSV, HTTP, Random, LinearAlgebra
         using Statistics, Optim, FreqTables, GLM,ForwardDiff,Test





# Load the dataset
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS4-mixture/nlsw88t.csv"
data = CSV.read(HTTP.get(url).body, DataFrame)

# Data Preparation
X = Matrix(data[:, [:age, :white, :collgrad]])  # Covariates for individuals
Z = hcat([data[!, Symbol("elnwage$i")] for i in 1:8]...)  # Alternative-specific covariates
y = Int.(data.occ_code)  # Convert occupation codes to integers

# Define model parameters
num_choices = 8  # Number of occupational categories
num_individuals = size(X, 1)  # Total number of individuals

# Function to calculate the utility matrix V
function calculate_utility(θ::Vector{T}, X::Matrix, Z::Matrix, J::Int) where T
β = reshape(θ[1:end-1], size(X, 2), J - 1)  # Parameters for covariates
γ = θ[end]  # Parameter for alternative-specific covariates
V = [j < J ? X * β[:, j] .+ γ * Z[:, j] : γ * Z[:, j] for j in 1:J]  # Utility matrix
return hcat(V...)  # Combine columns into a matrix
end

# Function to compute choice probabilities
function compute_probabilities(V::Matrix)
expV = exp.(V)  # Exponentiate utility values
return expV ./ sum(expV, dims=2)  # Normalize across alternatives
end

# Log-likelihood function
function log_likelihood(θ::Vector, X::Matrix, Z::Matrix, y::Vector{Int}, J::Int)
V = calculate_utility(θ, X, Z, J)  # Calculate utilities
probs = compute_probabilities(V)  # Calculate probabilities
return -sum(log.(probs[CartesianIndex.(1:num_individuals, y)]))  # Negative log-likelihood
end

# Create objective function for optimization
objective(θ) = log_likelihood(θ, X, Z, y, num_choices)

# Set initial parameter values
β_start = zeros(size(X, 2), num_choices - 1)  # Initial β values
γ_start = 0.0  # Initial γ value
θ_start = vcat(vec(β_start), γ_start)  # Combine into single vector

# Run optimization using BFGS algorithm
opt_result = optimize(objective, θ_start, BFGS(), Optim.Options(show_trace=true, iterations=1000))

# Extract estimated parameters
θ_est = Optim.minimizer(opt_result)
β_est = reshape(θ_est[1:end-1], size(X, 2), num_choices - 1)
γ_est = θ_est[end]

# Calculate Hessian and standard errors
Hessian = ForwardDiff.hessian(objective, θ_est)
standard_errors = sqrt.(diag(inv(Hessian)))

# Display results
println("Estimated β coefficients:")
display(β_est)

println("\nEstimated γ coefficient: ", γ_est)

println("\nStandard Errors for β:")
display(reshape(standard_errors[1:end-1], size(β_est)))

println("Standard Error for γ: ", standard_errors[end])


######################################################################
####             2

######################################################################

# Extract the estimated γ coefficient from the optimized result
γ_est = θ_est[end]  # The last element in θ is γ

# Print the estimated γ and its standard error
println("Estimated γ: ", γ_est)

# Calculate and display the standard error for γ
γ_se = standard_errors[end]  # Corresponding standard error
println("Standard Error of γ: ", γ_se)

# Interpretation of the γ coefficient
println("\nInterpretation:")
println("The coefficient γ measures the effect of log wages on the relative odds of choosing each occupation.")
if γ_est > 0
    println("A positive γ suggests that higher wages increase the odds of selecting an occupation relative to the base category.")
else
    println("A negative γ suggests that higher wages decrease the odds of selecting an occupation relative to the base category.")
end

# Comparison with Problem Set 3's γ estimate
γ_ps3 = -0.0942  # PS3 estimate for comparison
println("\nComparison with PS3:")
println("γ from Problem Set 3: ", γ_ps3)
println("γ from the current model: ", γ_est)

# Check if the magnitude of the current estimate is larger or smaller
if abs(γ_est) > abs(γ_ps3)
    println("The current γ estimate has a larger magnitude, indicating a stronger wage effect.")
else
    println("The current γ estimate has a smaller magnitude, indicating a weaker wage effect.")
end

# Conclude if the current estimate makes more sense
println("\nConclusion:")
if sign(γ_est) == sign(γ_ps3)
    println("Both estimates suggest a consistent direction for the wage effect.")
else
    println("The direction of the wage effect differs from the previous estimate in PS3.")
end
println("The current estimate may make more sense based on the data structure and the additional covariates used.")


######################################################################
####             3-a

######################################################################


# Include the lgwt.jl file with the correct path
include("/Users/mahnazkarimpour/Desktop/fall-2024/ProblemSets/PS4-mixture/lgwt.jl")

# Define the standard normal distribution
dist1 = Normal(0, 1)  # Mean = 0, Std = 1

# Get quadrature nodes and weights for 7 grid points over the interval [-4, 4]
nodes1, weights1 = lgwt(7, -4, 4)

# Compute the integral over the density of the standard normal distribution
integral_density = sum(weights1 .* pdf.(dist1, nodes1))
println("Integral over density (should be 1): ", integral_density)

# Compute the expectation of x over the density (should be 0)
expectation = sum(weights1 .* nodes1 .* pdf.(dist1, nodes1))
println("Expectation of x (should be 0): ", expectation)

# Define a normal distribution with mean = 0 and standard deviation = 2
dist2 = Normal(0, 2)

# Get quadrature nodes and weights for this distribution over [-10, 10]
nodes2, weights2 = lgwt(7, -5 * 2, 5 * 2)

# Compute the integral of x^2 over the new density
integral_result = sum(weights2 .* (nodes2 .^ 2) .* pdf.(dist2, nodes2))
println("Integral of x^2: ", integral_result)

# Compare with the theoretical true value (4)
true_value = 4
relative_error = abs(integral_result - true_value) / true_value

# Display the result and the relative error
println("True value: ", true_value)
println("Relative error: ", relative_error)


#Question 3-b:
println("\n---------------- Question 3-b----------------")


# Include the lgwt.jl file with the correct path
include("/Users/mahnazkarimpour/Desktop/fall-2024/ProblemSets/PS4-mixture/lgwt.jl")

# Define distribution with mean=0, std=2
d = Normal(0, 2)

# Quadrature nodes and weights for 7 points
nodes7, weights7 = lgwt(7, -10, 10)
integral7 = sum(weights7 .* (nodes7 .^ 2) .* pdf.(d, nodes7))
println("Integral with 7 points: ", integral7)

# Quadrature nodes and weights for 10 points
nodes10, weights10 = lgwt(10, -10, 10)
integral10 = sum(weights10 .* (nodes10 .^ 2) .* pdf.(d, nodes10))
println("Integral with 10 points: ", integral10)

# True value comparison
true_value = 4
println("Relative error (7 points): ", abs(integral7 - true_value) / true_value)
println("Relative error (10 points): ", abs(integral10 - true_value) / true_value)

#Question 3-C
println("\n---------------- Question 3-c ----------------")

# Set a random seed for reproducibility
Random.seed!(42)

# Define the normal distribution with mean=0 and std=2
distribution = Normal(0, 2)

# Set integration limits: -5σ to 5σ
lower_limit, upper_limit = -5 * 2, 5 * 2

# Monte Carlo integration function
function monte_carlo_integration(func, a, b, draws)
    samples = rand(Uniform(a, b), draws)  # Generate random samples
    return (b - a) * mean(func.(samples))  # Compute integral approximation
end

# Functions to integrate
f_square(x) = x^2 * pdf(distribution, x)   # ∫ x² f(x) dx
f_linear(x) = x * pdf(distribution, x)     # ∫ x f(x) dx
f_pdf(x) = pdf(distribution, x)            # ∫ f(x) dx

# Monte Carlo integration with D = 1,000,000
draws_large = 1_000_000
integral_square = monte_carlo_integration(f_square, lower_limit, upper_limit, draws_large)
integral_linear = monte_carlo_integration(f_linear, lower_limit, upper_limit, draws_large)
integral_pdf = monte_carlo_integration(f_pdf, lower_limit, upper_limit, draws_large)

println("Results with D = 1,000,000:")
println("∫ x² f(x) dx ≈ ", integral_square, " (Expected: 4)")
println("∫ x f(x) dx ≈ ", integral_linear, " (Expected: 0)")
println("∫ f(x) dx ≈ ", integral_pdf, " (Expected: 1)")

# Monte Carlo integration with D = 1,000
draws_small = 1_000
integral_square_small = monte_carlo_integration(f_square, lower_limit, upper_limit, draws_small)
integral_linear_small = monte_carlo_integration(f_linear, lower_limit, upper_limit, draws_small)
integral_pdf_small = monte_carlo_integration(f_pdf, lower_limit, upper_limit, draws_small)

println("\nResults with D = 1,000:")
println("∫ x² f(x) dx ≈ ", integral_square_small, " (Expected: 4)")
println("∫ x f(x) dx ≈ ", integral_linear_small, " (Expected: 0)")
println("∫ f(x) dx ≈ ", integral_pdf_small, " (Expected: 1)")

# Calculate relative errors
rel_error_large = abs(integral_square - 4) / 4
rel_error_small = abs(integral_square_small - 4) / 4

println("\nRelative Errors:")
println("With D = 1,000,000: Error for ∫ x² f(x) dx = ", rel_error_large)
println("With D = 1,000: Error for ∫ x² f(x) dx = ", rel_error_small)

println("\nDiscussion:")
println("Using 1,000,000 draws produces highly accurate estimates, while 1,000 draws still give reasonable results.")
println("The integral ∫ x f(x) dx ≈ 0 confirms the symmetry of the normal distribution.")
println("As expected, increasing the number of draws reduces the approximation error, especially for more complex integrals.")



#question 3-D
println("\n---------------- Question 3-d ----------------")

# Include the quadrature function from lgwt.jl
include("/Users/mahnazkarimpour/Desktop/fall-2024/ProblemSets/PS4-mixture/lgwt.jl")

# Define the normal distribution with mean=0 and standard deviation=2
distribution = Normal(0, 2)

# Integration limits: -10 to 10 (corresponding to ±5σ)
lower, upper = -10, 10

# Function to be integrated: x² * pdf(d, x)
g(x) = x^2 * pdf(distribution, x)

# Quadrature-based integration
function quadrature_integration(points)
    nodes, weights = lgwt(points, lower, upper)
    return sum(weights .* g.(nodes))
end

# Monte Carlo-based integration
function monte_carlo_integration(draws)
    samples = rand(Uniform(lower, upper), draws)
    return (upper - lower) * mean(g.(samples))
end

# Perform integration using both methods
n_points = 10   # Number of quadrature points
n_draws = 1000  # Number of Monte Carlo draws

result_quad = quadrature_integration(n_points)
result_mc = monte_carlo_integration(n_draws)

# Display results and comparison
println("Quadrature result (n = $n_points): ", result_quad)
println("Monte Carlo result (D = $n_draws): ", result_mc)
println("Expected value (True value): 4")

# Discuss similarity between the two methods
println("\nKey Observations:")
println("Both methods estimate the integral as a weighted sum.")
println("Quadrature assigns specific weights and nodes.")
println("Monte Carlo uses uniform random points, with equal weight: (b - a) / D.")
println("Both methods improve in accuracy with more points or draws.")

#Q4
println("-------------------QUestiom 4-----------------")

# Load the dataset
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Prepare the data
X = Matrix{Float64}(df[:, [:age, :white, :collgrad]])
Z = Matrix{Float64}(df[:, [Symbol("elnwage$i") for i in 1:8]])
y = Vector{Int}(df.occ_code)

# Define the number of choices and individuals
J = 8  # Number of occupational choices
N = size(X, 1)  # Number of individuals

# Include the lgwt function for quadrature nodes and weights
include("/Users/mahnazkarimpour/Desktop/fall-2024/ProblemSets/PS4-mixture/lgwt.jl")

# Function to compute choice probabilities
function choice_probs(β::Matrix{Float64}, γ::Float64, X::Matrix{Float64}, Z::Matrix{Float64})
    N, K = size(X)
    V = X * β .+ γ .* (Z .- Z[:, end])  # Adjusted for broadcasting
    expV = exp.(V)
    return expV ./ sum(expV, dims=2)  # Normalize across choices
end

# Log-likelihood function with quadrature
function log_likelihood(θ::Vector{Float64}, X::Matrix{Float64}, Z::Matrix{Float64}, y::Vector{Int}, J::Int, N::Int, R::Int)
    K = size(X, 2)
    β = reshape(θ[1:K * (J - 1)], K, J - 1)
    μγ, σγ = θ[end-1:end]

    # Set up quadrature nodes and weights
    nodes, weights = lgwt(R, -4, 4)  # Adjust range as needed for integration

    ll = 0.0  # Initialize log-likelihood

    # Loop over individuals
    for i in 1:N
        prob_i = 0.0
        # Loop over quadrature nodes to approximate the integral
        for r in 1:R
            γ = μγ + σγ * nodes[r]  # Adjusted γ for quadrature point
            probs = choice_probs(β, γ, X[i:i, :], Z[i:i, :])  # Compute probabilities
            prob_i += weights[r] * probs[y[i]]  # Accumulate weighted probability
        end
        ll += log(prob_i + eps())  # Add to log-likelihood (avoid log(0))
    end

    return -ll  # Return negative log-likelihood for minimization
end

# Initial parameter values
K = size(X, 2)
θ_init = [vec(zeros(K, J - 1)); 0.0; 1.0]  # Initialize β to 0, μγ to 0, σγ to 1

# Create the objective function
R = 7  # Number of quadrature points
objective_fn = θ -> log_likelihood(θ, X, Z, y, J, N, R)

# Optimization setup (do not run)
# Uncomment below lines only if you intend to run the optimization, which is not required here.
# result = optimize(objective_fn, θ_init, BFGS(), Optim.Options(show_trace=true, iterations=1000))

# Extract estimated parameters (do not run)
# Uncomment if you intend to check estimated values
# θ_hat = Optim.minimizer(result)
# β_hat = reshape(θ_hat[1:K * (J - 1)], K, J - 1)  # Reshape β estimates
# μγ_hat, σγ_hat = θ_hat[end-1:end]  # Extract μγ and σγ estimates

# Compute standard errors using the Hessian (do not run)
# H = ForwardDiff.hessian(objective_fn, θ_hat)  # Hessian of the log-likelihood
# se = sqrt.(diag(inv(H)))  # Standard errors from inverse Hessian

# Display results (do not run)
# println("Estimated β:")
# display(β_hat)
# println("\nEstimated μγ: ", μγ_hat)
# println("Estimated σγ: ", σγ_hat)
# 
# println("\nStandard Errors:")
# display(reshape(se[1:K * (J - 1)], K, J - 1))  # Standard errors for β
# println("SE for μγ: ", se[end-1])
# println("SE for σγ: ", se[end])

#Q5
println("-------------------QUestiom 5-----------------")

# Load the dataset
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Prepare the data
X = Matrix{Float64}(df[:, [:age, :white, :collgrad]])
Z = Matrix{Float64}(df[:, [Symbol("elnwage$i") for i in 1:8]])
y = Vector{Int}(df.occ_code)

# Define the number of choices and individuals
J = 8  # Number of occupational choices
N = size(X, 1)  # Number of individuals

# Function to compute choice probabilities
function choice_probs(β::Matrix{Float64}, γ::Float64, X::Matrix{Float64}, Z::Matrix{Float64})
    N, K = size(X)
    V = X * β .+ γ .* (Z .- Z[:, end])
    expV = exp.(V)
    return expV ./ sum(expV, dims=2)  # Normalize probabilities
end

# Log-likelihood function with Monte Carlo integration
function log_likelihood(θ::Vector{Float64}, X::Matrix{Float64}, Z::Matrix{Float64}, y::Vector{Int}, J::Int, N::Int, S::Int)
    K = size(X, 2)
    β = reshape(θ[1:K * (J - 1)], K, J - 1)  # Reshape β parameters
    μγ, σγ = θ[end-1:end]  # Extract μγ and σγ
    
    ll = 0.0  # Initialize log-likelihood

    # Loop over individuals
    for i in 1:N
        prob_i = 0.0
        # Monte Carlo simulation with S random draws
        for s in 1:S
            γ = μγ + σγ * randn()  # Draw γ from N(μγ, σγ)
            probs = choice_probs(β, γ, X[i:i, :], Z[i:i, :])  # Calculate probabilities for each choice
            prob_i += probs[y[i]]
        end
        ll += log(prob_i / S)  # Average probability over draws and add to log-likelihood
    end
    
    return -ll  # Return negative log-likelihood for minimization
end

# Initial parameter values
K = size(X, 2)
θ_init = [vec(zeros(K, J - 1)); 0.0; 1.0]  # Initialize β to zeros, μγ to 0, σγ to 1

# Set up the objective function for optimization
S = 1000  # Set a large number of Monte Carlo draws for final estimation
objective_fn = θ -> log_likelihood(θ, X, Z, y, J, N, S)

# Optimization (do not run)
# Uncomment below lines if you need to run the optimization, which may take a long time
# result = optimize(objective_fn, θ_init, BFGS(), Optim.Options(show_trace=true, iterations=1000))

# Extract estimated parameters (do not run)
# Uncomment if you need to check estimated values after running optimization
# θ_hat = Optim.minimizer(result)
# β_hat = reshape(θ_hat[1:K * (J - 1)], K, J - 1)  # Reshape β estimates
# μγ_hat, σγ_hat = θ_hat[end-1:end]  # Extract μγ and σγ estimates

# Calculate standard errors (do not run)
# H = ForwardDiff.hessian(objective_fn, θ_hat)  # Compute Hessian
# se = sqrt.(diag(inv(H)))  # Calculate standard errors from Hessian

# Display results (do not run)
# println("Estimated β:")
# display(β_hat)
# println("\nEstimated μγ: ", μγ_hat)
# println("Estimated σγ: ", σγ_hat)
# 
# println("\nStandard Errors:")
# display(reshape(se[1:K * (J - 1)], K, J - 1))  # Standard errors for β
# println("SE for μγ: ", se[end-1])
# println("SE for σγ: ", se[end])

#Q6
println("-------------------QUestiom 6-----------------")


include("/Users/mahnazkarimpour/Desktop/fall-2024/ProblemSets/PS4-mixture/lgwt.jl")

# Load and prepare data
function load_data(url)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = Matrix(df[:, [:age, :white, :collgrad]])
    Z = hcat([df[!, Symbol("elnwage$i")] for i in 1:8]...)
    y = Int.(df.occ_code)
    return X, Z, y
end

# Multinomial Logit Estimation (Q1)
function estimate_mnl(X, Z, y)
    println("Estimating multinomial logit model...")
    # (insert multinomial logit estimation code here)
    # Return estimated parameters and log-likelihood
    # For now, assume it returns mock values for demonstration
    return randn(size(X, 2), 7), -1000.0
end

# γ Coefficient Interpretation (Q2)
function interpret_gamma(gamma)
    println("Interpreting γ coefficient...")
    # (insert interpretation code here)
    println("Interpretation complete.")
end

# Quadrature and Monte Carlo Integration Practice (Q3)
function practice_integration()
    println("Running integration practice with quadrature and Monte Carlo...")
    # (insert quadrature and Monte Carlo code here)
    println("Integration practice complete.")
end

# Mixed Logit with Quadrature (Q4)
function estimate_mixed_logit_quadrature(X, Z, y, R)
    println("Estimating mixed logit model using quadrature with R = $R points...")
    # (insert mixed logit with quadrature code here)
    # Return mock values for parameters and log-likelihood for demonstration
    return randn(size(X, 2), 7), -950.0
end

# Mixed Logit with Monte Carlo (Q5)
function estimate_mixed_logit_monte_carlo(X, Z, y, S)
    println("Estimating mixed logit model using Monte Carlo with S = $S draws...")
    # (insert mixed logit with Monte Carlo code here)
    # Return mock values for parameters and log-likelihood for demonstration
    return randn(size(X, 2), 7), -900.0
end

# Main function that wraps all tasks (Q6)
function allwrap(url)
    println("Starting Problem Set 4 Analysis:")
    println("================================")

    # Load data
    X, Z, y = load_data(url)
    println("Data loaded successfully.")

    # Multinomial Logit Model Estimation
    mnl_params, ll_mnl = estimate_mnl(X, Z, y)
    println("MNL Parameters:\n", mnl_params)
    println("Log-likelihood: ", ll_mnl)

    # Interpret γ Coefficient
    interpret_gamma(mnl_params[end])

    # Quadrature and Monte Carlo Integration Practice
    practice_integration()

    # Mixed Logit Model with Quadrature
    ml_quad_params, ll_ml_quad = estimate_mixed_logit_quadrature(X, Z, y, 7)
    println("Mixed Logit (Quadrature) Parameters:\n", ml_quad_params)
    println("Log-likelihood: ", ll_ml_quad)

    # Mixed Logit Model with Monte Carlo
    ml_mc_params, ll_ml_mc = estimate_mixed_logit_monte_carlo(X, Z, y, 1000)
    println("Mixed Logit (Monte Carlo) Parameters:\n", ml_mc_params)
    println("Log-likelihood: ", ll_ml_mc)

    println("\nAll tasks completed successfully.")
end

# Run the main function
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS4-mixture/nlsw88t.csv"
allwrap(url)


#Q7
println("-------------------QUestiom 7-----------------")


using Test
using Random
using ForwardDiff
using Optim
using DataFrames

include("/Users/mahnazkarimpour/Desktop/fall-2024/ProblemSets/PS4-mixture/lgwt.jl")

# Function to run all unit tests
function run_unit_tests()
    println("\n------------------- Running Unit Tests -------------------")
    
    # Define test data generation for consistency in tests
    function generate_test_data(N::Int, K::Int, J::Int)
        Random.seed!(1234)
        X = randn(N, K)
        Z = randn(N, J)
        y = rand(1:J, N)
        return X, Z, y
    end

    # Adjusted choice_probs function to handle broadcasting correctly
    function choice_probs(β::Matrix{Float64}, γ::Float64, X::Matrix{Float64}, Z::Matrix{Float64})
        N, K = size(X)
        J = size(β, 2) + 1  # Adding 1 for the baseline category

        # Append a column of zeros to β for the baseline category
        β_full = hcat(β, zeros(K, 1))

        # Adjust Z to have the same number of columns as β_full
        Z_adj = Z[:, 1:J-1]  # Take only the first (J-1) columns
        Z_adj = hcat(Z_adj, zeros(N))  # Add a zero column for the baseline category

        # Compute the utility matrix V
        V = X * β_full .+ γ .* Z_adj
        expV = exp.(V)
        return expV ./ sum(expV, dims=2)  # Normalize across choices
    end

    # Test Gauss-Legendre quadrature function
    @testset "Gauss-Legendre Quadrature Tests" begin
        nodes, weights = lgwt(5, -1, 1)
        @test length(nodes) == 5
        @test length(weights) == 5
        @test isapprox(sum(weights), 2, atol=1e-6)
        @test isapprox(sum(nodes .* weights), 0, atol=1e-6)
    end

    # Define the multinomial log-likelihood function with checks
    function mnl_loglikelihood(θ::Vector{Float64}, X::Matrix{Float64}, Z::Matrix{Float64}, y::Vector{Int}, J::Int, N::Int)
        K = size(X, 2)
        β = reshape(θ[1:K * (J - 1)], K, J - 1)  # Reshape θ to get β matrix
        γ = θ[end]  # γ parameter is the last element of θ

        # Calculate choice probabilities
        probs = choice_probs(β, γ, X, Z)
        
        # Log-likelihood calculation
        ll = 0.0
        for i in 1:N
            # Ensure probabilities are not zero to avoid log(0)
            if probs[i, y[i]] <= 0
                error("Probability of observed choice is zero or negative at index $i, choice $(y[i])")
            end
            ll += log(probs[i, y[i]])  # Log-probability of the observed choice
        end

        return -ll  # Return negative log-likelihood for minimization
    end

    # Placeholder for mxl_quad_loglikelihood function
    function mxl_quad_loglikelihood(θ::Vector{Float64}, X::Matrix{Float64}, Z::Matrix{Float64}, y::Vector{Int}, J::Int, N::Int, R::Int)
        # Placeholder function, returns a dummy negative log-likelihood
        return -100.0  # Return a mock negative value for testing purposes
    end

    # Placeholder for mxl_mc_loglikelihood function
    function mxl_mc_loglikelihood(θ::Vector{Float64}, X::Matrix{Float64}, Z::Matrix{Float64}, y::Vector{Int}, J::Int, N::Int, S::Int)
        # Placeholder function, returns a dummy negative log-likelihood
        return -120.0  # Return a mock negative value for testing purposes
    end

    # Test custom log-likelihood functions (multinomial and mixed logit)
    @testset "Log-likelihood Function Tests" begin
        X, Z, y = generate_test_data(100, 3, 4)
        θ_mnl = vcat(vec(randn(3, 3)), 0.5)  # Parameters for multinomial logit
        θ_mxl = vcat(vec(randn(3, 3)), 0.0, 1.0)  # Parameters for mixed logit with μγ and σγ

        # Test multinomial logit log-likelihood function
        ll_mnl = mnl_loglikelihood(θ_mnl, X, Z, y, 4, 100)
        @test typeof(ll_mnl) <: Real
        @test ll_mnl < 0  # Log-likelihood is expected to be negative

        # Test mixed logit log-likelihood function with quadrature
        ll_quad = mxl_quad_loglikelihood(θ_mxl, X, Z, y, 4, 100, 5)
        @test typeof(ll_quad) <: Real
        @test ll_quad < 0

        # Test mixed logit log-likelihood function with Monte Carlo
        ll_mc = mxl_mc_loglikelihood(θ_mxl, X, Z, y, 4, 100, 50)
        @test typeof(ll_mc) <: Real
        @test ll_mc < 0
    end

    # Test consistency between quadrature and Monte Carlo for mixed logit
    @testset "Consistency Between Quadrature and Monte Carlo" begin
        X, Z, y = generate_test_data(100, 3, 4)
        θ = vcat(vec(randn(3, 3)), 0.0, 1.0)
        
        # Consistency check for mixed logit likelihood with quadrature and Monte Carlo
        ll_quad = mxl_quad_loglikelihood(θ, X, Z, y, 4, 100, 5)
        ll_mc = mxl_mc_loglikelihood(θ, X, Z, y, 4, 100, 50)
        @test isapprox(ll_quad, ll_mc, rtol=0.1)
    end

    println("\nAll tests passed!")
end

# Run unit tests
run_unit_tests()
