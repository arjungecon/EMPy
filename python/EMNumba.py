import numpy as np
from numba import jit
import scipy.stats as ss

from RandomGaussian import gen_gaussian_mixture

ax = np.newaxis


@jit(nopython=True)
def gen_norm(mu, sig):
    dist_mix = []

    for i in range(0, mu.shape[0]):
        dist_mix[i] = np.random.normal(mu[i], sig[i])

    return np.array(dist_mix)


@jit(nopython=True)
def gen_ll(data_input, dist_mix, alpha):
    ll_data = np.ndarray([data_input.shape[0], alpha.shape[0]])

    for i in range(0, alpha.shape[0]):
        ll_data[:, i] = np.log(alpha[i]) + dist_mix[i].logpdf(data_input)

    return ll_data


@jit(nopython=True)
def exp_max_numba(data_input, param_guess=None, num_iter=10):

    """
        Runs the Expectation-Maximization algorithm using NumPy.
        :param data_input: Data used in the EM algorithm
        :param param_guess: Initial guess for parameter values
        :param num_iter: Number of iterations (default 100)
        :return: Estimated parameters
    """

    if param_guess is None:
        init_mu = np.array([2.0, 2.5])
        init_sig = np.array([0.8, 1.0])
        init_weights = np.array([0.2, 0.8])
    else:
        init_mu = param_guess['mu']
        init_sig = param_guess['sig']
        init_weights = param_guess['weight']

    # Initialize the parameters to be estimated
    mu, sig, alpha = init_mu, init_sig, init_weights

    for ite in range(0, num_iter):
        dist_mix = gen_norm(mu, sig)

        # Step 1 - Compute the log-likelihood using the guess.
        em_log = gen_ll(data_input, dist_mix, alpha, alpha.shape[0])

        # Step 2 - Compute the posterior probabilities.
        em_prob = np.exp(em_log - np.logaddexp(em_log))

        # Step 3 - Update the guess of the mixture parameters.
        alpha = em_prob.sum(axis=0)
        mu = np.sum(em_prob * data_input[:, ax], axis=0) / alpha
        sig = np.sqrt(np.sum(em_prob * np.power(data_input[:, ax] - mu[ax, :], 2),
                             axis=0) / alpha)

    return alpha, mu, sig


input_mix = {'mu_sig': np.array([[2.0, 0.5], [5.0, 0.7]]),
             'weights': np.array([0.3, 0.7])}

data = gen_gaussian_mixture(input_mix, num_sample=10000, do_plot=True)
result = exp_max_numba(data)
