using Distributions
using Random
using DataFrames
using GaussianMixtures  
using JSON3

function Data_Generator_Julia(k,n,μ,σ,α)
    
    """
       Generate DATA
       :param k : number of gaussian mixtures
       :param n : Data Size 
       :param μ : Mean of data generated
       :param σ : Standard deviation of data generated
       :param α : Weights of data generated
       :return: Dictionnary of Data + Mean + Std + Weight
   """
   
   
   Random.seed!(3333)
   m = MixtureModel([Normal(μ[i], σ[i]) for i in 1:2], α)
   y = rand(m,n)

   return Dict(:y => y, :μ => μ, :σ => σ, :α => α)
end

function logsumexp(x::Matrix{Float64})
    
    """
        Logsumexp operation
        :param x : Matrix Float64
        :return:  logsumexponential of x 
    """
    

    vm = maximum(x,dims = 2)
    log.( sum( exp.( x .- vm ), dims= 2 )) .+ vm
end

function Expectation_Maximization_Julia(y::Vector{Float64},μ,σ,α;iters=50)
    
    
    """
        Run EM algorithm 
        :param y : number of gaussian mixtures
        :param μ : Mean of initial parameters 
        :param σ : Standard deviation of initial parameters 
        :param α : Weights of initial parameters 
        :param iters : Number of iterations of the algorithm
        :return: Return the estimated Mean 
    """
    
  

    N = length(y)
    K = length(μ)

    # initialize objects    
    L = zeros(N,K)
    p = similar(L)

    for it in 1:iters

        dists = [Normal(μ[ik], σ[ik] ) for ik in 1:K]

        # evaluate likelihood for each type 
        for i in 1:N
            for k in 1:K
                # Distributions.jl logpdf()
                L[i,k] = log(α[k]) + logpdf.(dists[k], y[i]) 
            end
        end
        

        # get posterior of each type 
        p[:,:] = exp.(L .- logsumexp(L))
      
        # with p in hand, update 
        α[:] .= vec(sum(p,dims=1) ./ N)
        μ[:] .= vec(sum(p .* y, dims = 1) ./ sum(p, dims = 1))
        σ[:] .= vec(sqrt.(sum(p .* (y .- μ').^2, dims = 1) ./ sum(p, dims = 1)))

    end
    return Dict(:α => α, :μ => μ, :σ => σ)
end


