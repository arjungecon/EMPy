import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from scipy.stats import norm 
import time 
import json
import torch 

def tensorflow_script(N_data_size,N_Iterations):
    
    
     
    """
        Evaluate time of execution 
        :param N_data_size: Data size  
        :param N_Iterations: Number of iteration of the algo 
        :return: Json with Datasize, Number of iterations, Execution time and the name of the method 
    """
    
    data= Data_Generator (mean_std, alpha,N_data_size)["Tensorflow"]
    alpha__0,mean_mu__0,var_v__0= Initial_Parameters(2)['Tensorflow']


    start_time = time.perf_counter()
    Expectation_Maximization_Tensorflow(data,alpha__0,mean_mu__0,var_v__0,nb_iter=N_Iterations)
    time_=time.perf_counter() - start_time
    dic_result = {'DATA_Size': N_data_size, 'Iterations': N_Iterations, 'Execution_Time': time_,'Algo_Name': 'Tensorflow'}

    return (json.dumps(dic_result))


N_data_size=1000
N_Iterations=100
mean_std, alpha = [[3,1],[4,2]],[0.4,0.6]
tensorflow_script(N_data_size,N_Iterations)