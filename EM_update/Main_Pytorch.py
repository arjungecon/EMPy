import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.stats import norm 

def log_normale(data, mu_mean=0, log_variance=0.):
    
    
    """
        Runs log normal probability distribution
        :param data: Data used in the EM algorithm
        :param mu_mean: mean value
        :param log_variance: logarithme of the variance       
        :return: PDF of the lognormal at each point of data 
    """
    
    

    
    if type(log_variance) == 'float':
        log_variance = data.new(1).fill_(log_variance)

    diff_sq= (data - mu_mean) ** 2
    log_proba = -0.5 * (log_variance + diff_sq / log_variance.exp())
    log_proba = log_proba -(0.5 * np.log(2 * np.pi))

    return log_proba


def Expectation_Maximization_Pytorch(data,alpha, mean_mu, std_sig ,treshold=False, numb_iter=100,epsilon=1e-5 ):
    
    """
        Runs the Expectation-Maximization algorithm using Pytorch
        :param data: Data used in the EM algorithm
        :param alpha: Initial guess for weight values
        :param mean_mu: Initial guess for mean values
        :param std_sig: Initial guess for standard deviation values
        :param treshold: Float to activate the loop of likelihood
        :param epsilon: Precision on the likelihood 
        :param numb_iter: Number of iterations (default 100)
        :return: Estimated parameters ( weight, mean, standard deviation )
    """
    

 
    loglikelihood_init = -float('inf')
    var_v=std_sig**2
    log_var = var_v.log()
    
    for i in range(numb_iter):
        
       # Step 1 : Compute loglikelihood with initial parameters 
    
    
        log_likelihoods = log_normale( data[None, :, :], mean_mu[:, None, :],  log_var[:, None, :]  )
                
        log_likelihoods = log_likelihoods.sum(-1)
    


        # Step 2 : Compute log of priors gamma  
        
        log_gamma = log_likelihoods 
        log_sum_exp = torch.logsumexp(log_gamma,0,keepdim=True)
        log_gamma = log_gamma - log_sum_exp
        
        gamma = log_gamma.exp()

        
        K = gamma.size(0)
        sum_gamma = torch.sum(gamma, dim=1) 
        sum_gamma = sum_gamma.view(K, 1, 1)
        # Mean estimation
        
        mu = gamma[:, None] @ data[None,]
        mu = mu / sum_gamma

        #LogVariance Estimation 
        
        diff_ = data.unsqueeze(0) - mu
        var = gamma[:, None] @ (diff_ ** 2) 
        var = var / sum_gamma
        logvar = torch.clamp(var, min=1e-6).log()
        
        #Estimation of the weight Pi 
        
        m = data.size(1) 
        alpha = sum_gamma / sum_gamma.sum()
        
        mean_mu,log_var,alpha = mu.squeeze(1), logvar.squeeze(1), alpha.squeeze()
        
         # Convergence criteria of the algorithm
            
        if treshold==True :
        
            loglikelihood_= log_likelihoods.mean()

            if torch.abs(loglikelihood_init - loglikelihood_).item() < eps:
                break
            loglikelihood_init = loglikelihood_
        

    return alpha,mean_mu,np.sqrt(log_var.exp())