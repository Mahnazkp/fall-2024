######################################################################  
##                            Pset 7                               ###
##                          Mahnaz Karimpour                       ###
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
Pkg.add("Distributions")
Pkg.add("ForwardDiff")
Pkg.add("LineSearches")

# Make sure all "using" statements are at the top level
using Random
using LinearAlgebra
using Statistics
using Distributions
using Optim
using DataFrames
using CSV
using HTTP
using GLM
using FreqTables
using ForwardDiff
using LineSearches

# Define all the problem set steps in one encapsulated function
function allwrap()
    
    ######################################################################  
    ##                          Question 1                              ###
    #######################################################################
    println("\n---------------- Question 1 ----------------")

    # Load the data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # Create the design matrix and response variable
    X = [ones(size(df, 1)) df.age df.race .== 1 df.collgrad .== 1]
    y = df.married .== 1

    # Define the moment condition function
    function moments(beta, X, y)
        residuals = y .- X * beta
        return X' * residuals  # N x K moment conditions
    end
    
    # Define the GMM moment function for multinomial logit
    function gmm_moment(alpha, X, y)
        K = size(X, 2)
        J = length(unique(y))
        N = length(y)
    
        # Create the actual outcome matrix d (N x J)
        d = zeros(N, J)
        for j = 1:J
            d[:, j] = y .== j
        end
    
        # Predicted probabilities matrix P (N x J)
        bigAlpha = [reshape(alpha, K, J - 1) zeros(K)]
        num = zeros(N, J)
        dem = zeros(N)
        for j = 1:J
            num[:, j] = exp.(X * bigAlpha[:, j])
            dem .+= num[:, j]
        end
        P = num ./ repeat(dem, 1, J)
    
        # GMM moment condition g = d - P
        g = d - P
        return vec(g)  # Convert to a long vector of length N * J
    end
    
    # Define the GMM objective function
    function gmm_objective(alpha, X, y)
        g = gmm_moment(alpha, X, y)
        return g' * g  # Identity weighting matrix
    end

    # Initial guess for beta (random values)
    initial_beta = rand(size(X, 2))

    # Optimize using LBFGS method
    result = optimize(b -> gmm_objective(b, X, y), initial_beta, LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000))

    # Display the estimated beta
    println("Estimated beta (GMM): ", result.minimizer)

    # OLS using closed-form solution
    beta_ols = inv(X' * X) * X' * y
    println("Estimated beta (OLS): ", beta_ols)


    ######################################################################  
    ##                          Question 2                              ###
    #######################################################################
    println("\n---------------- Question 2 ----------------")

    # Data preparation for multinomial logit
    df = dropmissing(df, :occupation)
    df[df.occupation .== 8, :occupation] .= 7
    df[df.occupation .== 9, :occupation] .= 7
    df[df.occupation .== 10, :occupation] .= 7
    df[df.occupation .== 11, :occupation] .= 7
    df[df.occupation .== 12, :occupation] .= 7
    df[df.occupation .== 13, :occupation] .= 7

    # Define the design matrix X and outcome variable y
    X = [ones(size(df, 1), 1) df.age df.race .== 1 df.collgrad .== 1]
    y = df.occupation

    # Define the Multinomial Logit (MNL) function for Maximum Likelihood Estimation (MLE)
    function mlogit(alpha, X, y)
        K = size(X, 2)
        J = length(unique(y))
        N = length(y)

        bigY = zeros(N, J)
        for j = 1:J
            bigY[:, j] = y .== j
        end

        bigAlpha = [reshape(alpha, K, J - 1) zeros(K)]

        num = zeros(N, J)
        dem = zeros(N)
        for j = 1:J
            num[:, j] = exp.(X * bigAlpha[:, j])
            dem .+= num[:, j]
        end

        P = num ./ repeat(dem, 1, J)
        loglike = -sum(bigY .* log.(P))
        return loglike
    end

    # Part (a): Maximum Likelihood Estimation (MLE)
    alpha_start = rand(6 * size(X, 2))  # Random starting values
    alpha_hat_optim = optimize(a -> mlogit(a, X, y), alpha_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 100_000, show_trace = true))
    alpha_hat_mle = alpha_hat_optim.minimizer
    println("MLE estimates: ", alpha_hat_mle)

    # Part (b): GMM Estimation with MLE Starting Values
    function gmm_moment(alpha, X, y)
        K = size(X, 2)
        J = length(unique(y))
        N = length(y)

        # Create the actual outcome matrix d (N x J)
        d = zeros(N, J)
        for j = 1:J
            d[:, j] = y .== j
        end

        # Predicted probabilities matrix P (N x J)
        bigAlpha = [reshape(alpha, K, J - 1) zeros(K)]
        num = zeros(N, J)
        dem = zeros(N)
        for j = 1:J
            num[:, j] = exp.(X * bigAlpha[:, j])
            dem .+= num[:, j]
        end
        P = num ./ repeat(dem, 1, J)

        # GMM moment condition g = d - P
        g = d - P
        return vec(g)  # Convert to a long vector of length N * J
    end

    # Define the GMM objective function
    function gmm_objective(alpha, X, y)
        g = gmm_moment(alpha, X, y)
        return g' * g  # Identity weighting matrix
    end

    # Optimize using GMM with MLE starting values
    alpha_hat_gmm = optimize(a -> gmm_objective(a, X, y), alpha_hat_mle, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 100_000))
    println("GMM estimates with MLE starting values: ", alpha_hat_gmm.minimizer)

    # Part (c): GMM Estimation with Random Starting Values
    random_alpha = rand(length(alpha_hat_mle))  # Random initial values
    alpha_hat_gmm_random = optimize(a -> gmm_objective(a, X, y), random_alpha, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 100_000))
    println("GMM estimates with random starting values: ", alpha_hat_gmm_random.minimizer)

    # Part (d): Compare estimates and discuss global concavity
    println("Comparison of GMM estimates (MLE start) vs (Random start):")
    println("GMM (MLE start): ", alpha_hat_gmm.minimizer)
    println("GMM (Random start): ", alpha_hat_gmm_random.minimizer)


    ######################################################################  
    ##                          Question 3                              ###
    #######################################################################
    println("\n---------------- Question 3 ----------------")

    # Step (a): Generate X using random numbers
    function generate_data(N, J, K)
        # X: N x K design matrix with random values
        X = randn(N, K)

        # Step (b): Set values for β (K x (J-1) parameter matrix)
        β = randn(K, J - 1)  # Choose random β coefficients

        # Step (c): Generate N x J matrix of choice probabilities P
        P = zeros(N, J)
        num = zeros(N, J)
        dem = zeros(N)
        for j = 1:J-1
            num[:, j] = exp.(X * β[:, j])  # Exponent of linear predictor for j
        end
        num[:, J] .= 1.0  # For the base case (Jth alternative)
        dem = sum(num, dims = 2)
        P = num ./ dem  # Normalized probabilities

        # Step (d): Draw the preference shocks ε as a N x 1 vector of random numbers
        ε = rand(N)

        # Step (e): Generate Y based on the choice probabilities P and ε
        Y = zeros(Int, N)  # Initialize outcome vector Y as N x 1 vector of zeros
        for i in 1:N
            for j in 1:J
                # Check if cumulative probability exceeds the random shock
                if sum(P[i, 1:j]) > ε[i]
                    Y[i] = j
                    break
                end
            end
        end

        return X, Y, β, P
    end

    # Example: Choose sample size N, choice set size J, and number of covariates K
    N = 500     # Sample size
    J = 4       # Number of choices
    K = 3       # Number of covariates

    # Simulate the dataset
    X, Y, β_true, P = generate_data(N, J, K)

    # Print simulated data and true β values
    println("Simulated X: ", X)
    println("Simulated Y: ", Y)
    println("True β values: ", β_true)

    # Define the Multinomial Logit (MNL) function
    function mlogit(alpha, X, Y)
        K = size(X, 2)
        J = length(unique(Y))
        N = length(Y)

        bigY = zeros(N, J)
        for j = 1:J
            bigY[:, j] = Y .== j
        end

        # Reshape alpha correctly and account for the base category
        bigAlpha = [reshape(alpha, K, J - 1) zeros(K)]

        num = zeros(N, J)
        dem = zeros(N)
        for j = 1:J
            num[:, j] = exp.(X * bigAlpha[:, j])
            dem .+= num[:, j]
        end

        P = num ./ repeat(dem, 1, J)
        loglike = -sum(bigY .* log.(P))
        return loglike
    end

    # Correct size for alpha_start
    alpha_start = rand(K * (J - 1))

    # Optimize
    alpha_hat_optim = optimize(a -> mlogit(a, X, Y), alpha_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 100_000))
    alpha_hat_mle = alpha_hat_optim.minimizer
    println("MLE estimates: ", alpha_hat_mle)


    ######################################################################  
    ##                          Question 5                              ###
    #######################################################################
    println("\n---------------- Question 5 ----------------")

    # Define the Simulated Method of Moments (SMM) function
    function mlogit_smm(θ, X, y, D)
        K = size(X, 2)
        N = size(y, 1)

        # θ contains both β parameters and variance σ
        β = θ[1:end-1]
        σ = θ[end]

        if length(β) == 1
            β = β[1]
        end

        # Generate simulated y_tilde (model moments)
        gmodel = zeros(N+1, D)

        # Simulate the choice probabilities for the multinomial logit model
        J = length(unique(y))  # Number of choices

        # Reshape β for J alternatives
        bigBeta = [reshape(β, K, J - 1) zeros(K)]  # Base category with 0 coefficients

        # Generate the model-based probabilities P and simulate y_tilde
        num = zeros(N, J)
        dem = zeros(N)
        P = zeros(N, J)
        for j in 1:J
            num[:, j] = exp.(X * bigBeta[:, j])
            dem .+= num[:, j]
        end
        P = num ./ repeat(dem, 1, J)

        # Simulate y_tilde based on the probabilities P
        y_tilde = zeros(Int, N)
        for i in 1:N
            # Randomly generate y_tilde according to the simulated probabilities
            cum_prob = cumsum(P[i, :])
            ε = rand()
            for j in 1:J
                if cum_prob[j] > ε
                    y_tilde[i] = j
                    break
                end
            end
        end

        # Data moments (actual y and variance of y)
        gdata = [y; var(y)]

        # Model moments (simulated y_tilde and variance of y_tilde)
        gmodel[:, 1] = [y_tilde; var(y_tilde)]

        # Return the difference between data moments and model moments
        return gdata - gmodel[:, 1]
    end

    # Define the SMM objective function
    function smm_objective(θ, X, y, D)
        g = mlogit_smm(θ, X, y, D)
        return g' * g  # Identity weighting matrix
    end

    # Example: Simulate data from Question 3
    N = 500  # Sample size
    J = 4    # Number of choices
    K = 3    # Number of covariates

    # Simulate dataset from Question 3 code
    X, y, β_true, P = generate_data(N, J, K)

    # Starting values for θ (β and σ)
    θ_start = [rand(K * (J - 1)); 1.0]  # Random starting values for β and σ

    # Number of moments D (in this case, equal to N + 1 for y and variance)
    D = N + 1

    # Optimize using SMM
    θ_hat_smm = optimize(θ -> smm_objective(θ, X, y, D), θ_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 100_000))
    println("SMM estimates: ", θ_hat_smm.minimizer)

end

# Run all the steps in the problem set
@time allwrap()

# Write tests for each key function and run them using the Test module
using Test

@testset "GMM and Multinomial Logit Tests" begin
    # Test for GMM in Question 1
    @test typeof(gmm_objective(rand(4), rand(100, 4), rand(100))) == Float64
    @test length(gmm_objective(rand(4), rand(100, 4), rand(100))) == 1

    # Test for MLE in Question 2 (mlogit)
    @test typeof(mlogit(rand(12), rand(100, 4), rand(100))) == Float64

    # Test for data generation in Question 3
    X, Y, β, P = generate_data(100, 4, 3)
    @test size(X) == (100, 3)
    @test size(Y) == (100,)
    @test size(P) == (100, 4)

    # Test for SMM in Question 5
    @test typeof(smm_objective(rand(10), X, Y, 10)) == Float64
end
