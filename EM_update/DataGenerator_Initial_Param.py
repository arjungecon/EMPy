import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from scipy.stats import norm 
import time 
import json
import torch 


def Initial_Parameters(k=2):
    
    """
        Generate randomly the initial parameter for every method
        :param k: Number of Clusters
        :return: dictionnary of initial parameters for all the methods 
    """
    
    # Numpy and Tensorflow shape
    
    alpha_numpy_tf = np.random.random(k) 
    alpha_numpy_tf /= alpha_numpy_tf.sum()  # Constraint sum(weights)=1 
    mean_mu_numpy_tf = np.random.random((k,1))  
    std_sig_numpy_tf = np.array([np.eye(1)]* k)
    
     # Scikit shape
       
    mean_mu_scikit=mean_mu_numpy_tf.reshape(k,)
    std_sig_scikit=std_sig_numpy_tf.reshape(k,)
    alpha_scikit=alpha_numpy_tf
        
    # Pytorch shape
    
    mean_mu_pytorch= torch.from_numpy(mean_mu_numpy_tf)
    alpha_pytorch=torch.from_numpy(alpha_numpy_tf)
    std_sig_pytorch=torch.tensor(np.ones((k,k)).tolist())
    
    dic_result = {'Tensorflow': (alpha_numpy_tf, mean_mu_numpy_tf, std_sig_numpy_tf.reshape(2,1)),
                  'Numpy': (alpha_numpy_tf, mean_mu_numpy_tf, std_sig_numpy_tf),
                  'Scikit': (alpha_scikit,mean_mu_scikit, std_sig_scikit),
                  'Pytorch': (alpha_pytorch, mean_mu_pytorch, std_sig_pytorch) }

    return (dic_result)


def Data_Generator(mean_std, alpha, num_points):
    
    
    """
        Generate the Data for every method 
        :param means_std: Array of the mean and Std of the data to generate : np.array ([mean_1,std_1],[mean_2,std_2 ] ... )  
        :param alpha: Array of weights of the data to generate 
        :param num_points: Number of points of the data to generate
        :return: Dictionnary with DATA Generated for all methods
    """
        
    random_index  = np.random.choice(len(alpha), size=num_points, replace=True, p=alpha)  

    data_generated = np.array([norm.rvs(*mean_std[index]) for index in random_index])

    data_generated_scikit=data_generated
    
    data_generated_numpy_tf=data_generated_scikit.reshape(data_generated_scikit.shape[0],1)

    data_generated_Pytorch=torch.from_numpy(data_generated_numpy_tf)

    dic_result = {'Tensorflow': data_generated_numpy_tf,
                      'Numpy': data_generated_numpy_tf,
                      'Scikit': data_generated_scikit,
                      'Pytorch': data_generated_Pytorch }


    return (dic_result)


