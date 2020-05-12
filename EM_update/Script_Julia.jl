using Distributions
using Random
using DataFrames
using GaussianMixtures  
using JSON3

μ_real= [3.0,4.0]
σ_real = [1,2]
α_real = [0.4,0.6]

μ_init = [0.63472814,0.74419466]
σ_init = [1.0,1.0]
α_init = [0.46723795,0.53276205]

function Script_Julia(N_data_size,N_Iterations)
       
     """
        Generate DATA
        :param N_data_size : Data size
        :param n N_Iterations : Number of iterations 
        :return: Json with DATA size + Iterations + Execution time + Name of the method
    """   
    
    
    data=Data_generator(2,N_data_size,μ_real,σ_real,α_real)[:y]
    
    time_julia=@elapsed Expectation_Maximization_Julia(data,μ,σ,α;iters= N_Iterations) 
    
    return JSON3.write(Dict(:DATA_Size => N_data_size, :Iterations => N_Iterations, :Execution_Time => time_julia , :Algo_Name => "Julia" ))


end

N_data_size=1000
N_Iterations=100
Script_Julia(N_data_size,N_Iterations)
    