
from scipy.stats import norm 
from profilehooks import profile
from numba import *
import numpy as np

@autojit(nopython=True)
def EM_Gaussian_Mixture_(data_input, alpha, mean_mu, std_sig, epsilon=1e-10, numb_iter=100):

    # Index and initialization 
    
    log_likelihood_init = 0
    n, p = data_input.shape
    k = len(alpha)
    
    
    for i in range(numb_iter):
        
        log_likelihood= 0
      #  print('Iteration N°: ', i)

        # Step 1 : Expectation ( Compute the Prior )
        
        gamma = np.zeros((k, n))
        for j in range(len(mean_mu)):
            for d in range(n):
                ab=(1. / np.sqrt(2 * np.pi * std_sig[j])) * np.exp(-(data_input[d] - mean_mu[j])**2 / (2 * std_sig[j]))
                gamma[j, d] = alpha[j] *ab[0][0]
        gamma /= gamma.sum(0)

        # Step 2 : Maximization ( estimation of the parameters )
        
        # Estimation of alpha ( Pi)
    
        alpha = np.zeros(k)
        for j in range(len(mean_mu)):
            for d in range(n):
                alpha[j] += gamma[j, d]
        alpha /= n
        
        # Estimation of mean ( Mu )

        mean_mu = np.zeros((k, p))
        for j in range(k):
            for d in range(n):
                mean_mu[j] += gamma[j, d] * data_input[d]
            mean_mu[j] /= gamma[j, :].sum()
            
        # Estimation of std  ( Sigma)

        std_sig = np.zeros((k, p, p))
        for j in range(k):
            for d in range(n):
                diff_data_input_mu = data_input[d]- mean_mu[j]# Broadcasting ?
                std_sig[j] += gamma[j, d] * (diff_data_input_mu**2)
            std_sig[j] /= gamma[j,:].sum()
        
        # Loop to compute likelihood each step
        
        log_likelihood = 0.0
        for d in range(n):
            count = 0
            for j in range(k):
                abb=(1. / np.sqrt(2 * np.pi * std_sig[j])) * np.exp(-(data_input[d] - mean_mu[j])**2 / (2 * std_sig[j]))
                count += alpha[j] *abb[0][0]
            log_likelihood += np.log(count)
            
      #  print(f'log_likelihood: {log_likelihood:3.4f}')

        # Stop condition of the algorithme ( Epsilon to be fixed  )

        if np.abs(log_likelihood - log_likelihood_init) < epsilon:
           
            break
            
        else : 
            
            log_likelihood_init = log_likelihood
        
    return log_likelihood, alpha, mean_mu, std_sig

np.random.seed(30786)

def data_generator(mean_std, alpha, num_points): 
    random_i  = np.random.choice(len(alpha), size=num_points, replace=True, p=alpha)
    data_generated = np.array([norm.rvs(*mean_std[i]) for i in random_i])
    data_generated=data_generated.reshape(data_generated.shape[0],1)
    return  data_generated 

def initial_parameters(k):

    alpha = np.random.random(k)
    alpha /= alpha.sum()  # Constraint sum(pi)=1
    mean_mu = np.random.random((k,1))
    std_sig = np.array([np.eye(1)]* k)
    
    return alpha,mean_mu,std_sig

mean_std, alpha = [[3,1],[4,2]],[0.4,0.6]
data_input= data_generator (mean_std, alpha,100000)
alpha, mean_mu, std_sig= initial_parameters(2)
EM_Gaussian_Mixture_(data_input, alpha, mean_mu, std_sig)