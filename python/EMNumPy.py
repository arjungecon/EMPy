import numpy as np
import numpy.random
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.special import logsumexp

from RandomGaussian import gen_gaussian_mixture

ax = np.newaxis


def exp_max_numpy(data_input, num_ker, param_guess, num_iter=100):

    """
        Runs the Expectation-Maximization algorithm using NumPy.
        :param data_input: Data used in the EM algorithm
        :param param_guess: Initial guess for parameter values
        :param num_iter: Number of iterations (default 100)
        :return: Estimated parameters
    """

    # Initialize the parameters to be estimated
    mu, sig, alpha = param_guess['mu'], param_guess['sig'], param_guess['weight']

    # Check if initial guess is valid
    assert np.min([mu.shape[0], sig.shape[0], alpha.shape[0]]) == num_ker,\
        "Shape of initial guess does not match the number of kernels."

    for ite in range(0, num_iter):

        dist_mix = np.vectorize(ss.norm)(mu, sig)

        # Step 1 - Compute the log-likelihood using the guess.
        em_log = np.transpose(np.log(alpha)[:, ax] +
                              np.array(list(map(lambda x: x.logpdf(data_input), dist_mix))))

        # Step 2 - Compute the posterior probabilities.
        em_prob = np.exp(em_log - logsumexp(em_log, axis=1)[:, ax])

        # Step 3 - Update the guess of the mixture parameters.
        alpha = em_prob.sum(axis=0)/em_prob.sum()
        mu = np.sum(em_prob * data_input[:, ax], axis=0)/em_prob.sum(axis=0)
        sig = np.sqrt(np.sum(em_prob * np.power(data_input[:, ax] - mu[ax, :], 2),
                             axis=0)/em_prob.sum(axis=0))

    result_dict = {'alpha': alpha, 'mu': mu, 'sig': sig}
    return result_dict


