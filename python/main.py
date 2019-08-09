import numpy as np
from numba import jit
from numba.typed import Dict
from numba import types

from RandomGaussian import gen_gaussian_mixture
from EMNumPy import exp_max_numpy
from EMNumba import exp_max_numba
from EMSKLearn import exp_max_sklearn

float_array = types.float64[:]
ax = np.newaxis

input_mix = {'mu_sig': np.array([[2.0, 0.5], [5.0, 0.7], [3.0, 0.1]]),
             'weights': np.array([0.25, 0.7, 0.05])}

data = gen_gaussian_mixture(input_mix, num_sample=10000, do_plot=False)

param_guess = dict()
# param_guess = Dict.empty(
#         key_type=types.unicode_type,
#         value_type=float_array,
#     )

param_guess['mu'] = np.array([2.0, 2.5, 2.7])
param_guess['sig'] = np.array([0.8, 1.0, 0.2])
param_guess['weights'] = np.array([0.33, 0.34, 0.33])

result = exp_max_sklearn(data_input=data[:, ax], num_ker=3,
                         param_guess=param_guess, num_iter=1000)
