import numpy as np
import numpy.random
import scipy.stats as ss
import matplotlib.pyplot as plt

from RandomGaussian import gen_gaussian_mixture

def exp_max_numpy():

    """
        Runs the Expectation-Maximization algorithm using NumPy.
    :return:
    """


input_mix = {'mu_sig': np.array([[2.0, 0.5], [5.0, 0.7]]),
             'weights': np.array([0.3, 0.7])}

gen_gaussian_mixture(input_mix, num_sample=100000, do_plot=False)



