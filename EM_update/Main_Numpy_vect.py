import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm 
import math
import numpy.random as random
import time


def Expectation_Maximization_Numpy_Vect(data_input, alpha, mean_mu, std_sig, treshold=False, epsilon=1e-5, numb_iter=100):


    
    
    """
        Runs the Expectation-Maximization algorithm using Numpy Vectorial Version
        :param data: Data used in the EM algorithm
        :param alpha: Initial guess for weight values
        :param mean_mu: Initial guess for mean values
        :param std_sig: Initial guess for standard deviation values
        :param treshold: Float to activate the loop of likelihood
        :param epsilon: Precision on the likelihood 
        :param numb_iter: Number of iterations (default 100)
        :return: Estimated parameters ( weight, mean, standard deviation )
    """
    

    # Index and initialization 
    
    n, p = data_input.shape
    k = len(alpha)
    log_likelihood_init = -float('inf')
    
    
    for i in range(numb_iter):
        

        # Step 1 : Expectation ( Compute the Prior )

        gamma = np.zeros((k, n),dtype=np.float64)
        for j in range(k):
            gamma[j, :] = alpha[j]*multivariate_normal(mean_mu[j], std_sig[j]).pdf(data_input)
        gamma /= gamma.sum(0)

        # Step 2 : Maximization ( estimation of the parameters )
        
        # Estimation of alpha ( Pi)

        alpha = gamma.sum(axis=1)
        alpha /= n
       
        # Estimation of mean ( Mu )

        mean_mu = np.dot(gamma, data_input)
        mean_mu /= gamma.sum(1).reshape(k,1)

        # Estimation of variance
        
        std_sig = np.zeros((k, 1, 1),dtype=np.float64)
        for j in range(k):
            diff_data_input_mu = data_input - mean_mu[j, :]
            std_sig[j] = (gamma[j,:].reshape(n,1,1) * ((diff_data_input_mu**2).reshape(n,1,1))).sum(axis=0)
        std_sig /= gamma.sum(axis=1).reshape(k,1,1)
        
        
        if treshold==True:
        
            # Loop to compute likelihood each step

            log_likelihood = -float('inf')
            for pi, mu, sigma in zip(alpha, mean_mu, std_sig):
                log_likelihood += pi*multivariate_normal(mu, sigma).pdf(data_input) 
            log_likelihood = np.log(log_likelihood).sum()

                # Stop condition of the algorithm ( Epsilon to be fixed  )

            if np.abs(log_likelihood - log_likelihood_init) < epsilon:
                break
            else : 
                log_likelihood_init = log_likelihood
    

    return  alpha, mean_mu, np.sqrt(std_sig)

