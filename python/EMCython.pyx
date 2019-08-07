import numpy as np
cimport numpy as np
import scipy.stats as ss
from scipy.special import logsumexp

DTYPE = np.double
ctypedef np.double_t DTYPE_t

ax = np.newaxis

def exp_max_cython(object data_input, object param_guess, int num_iter=25):

    """
        Runs the Expectation-Maximization algorithm using Cython.
        :param data_input: Data used in the EM algorithm
        :param param_guess: Initial guess for parameter values
        :param num_iter: Number of iterations (default 100)
        :return: Estimated parameters
    """

    # Initialize the parameters to be estimated

    cdef np.ndarray mu = param_guess['mu']
    cdef np.ndarray sig = param_guess['sig']
    cdef np.ndarray alpha = param_guess['weight']

    for ite in range(1, num_iter):

        dist_mix = np.vectorize(ss.norm)(mu, sig)

        # Step 1 - Compute the log-likelihood using the guess.
        em_log = np.transpose(np.log(alpha)[:, ax] +
                              np.array(list(map(lambda x: x.logpdf(data_input), dist_mix))))

        # Step 2 - Compute the posterior probabilities.
        em_prob = np.exp(em_log - logsumexp(em_log))

        # Step 3 - Update the guess of the mixture parameters.
        alpha = em_prob.sum(axis=0)
        mu = np.sum(em_prob * data_input[:, ax], axis=0)/alpha
        sig = np.sqrt(np.sum(em_prob * np.power(data_input[:, ax] - mu[ax, :], 2),
                             axis=0)/alpha)

    return alpha, mu, sig
