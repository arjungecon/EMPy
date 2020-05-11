import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.stats import norm 

def Pytorch_script(N_data_size,N_Iterations):
    """
        Evaluate time of execution 
        :param N_data_size: Data size  
        :param N_Iterations: Number of iteration of the algo 
        :return: Json with Datasize, Number of iterations, Execution time and the name of the method 
    """

    data= Data_Generator (mean_std, alpha,N_data_size)["Pytorch"]
    alpha__0,mean_mu__0,var_v__0=Initial_Parameters(2)["Pytorch"]
    start_time = time.perf_counter()
    Expectation_Maximization_Pytorch(data,alpha__0,mean_mu__0,var_v__0,numb_iter=100, epsilon=1e-5)
    time_=time.perf_counter() - start_time
    dic_result = {'DATA_Size': N_data_size, 'Iterations': N_Iterations, 'Execution_Time': time_,'Algo_Name': 'Pytorch'}
    
    return (json.dumps(dic_result))

N_data_size=1000
N_Iterations=100
mean_std, alpha = [[3,1],[4,2]],[0.4,0.6]
Pytorch_script(N_data_size,N_Iterations)