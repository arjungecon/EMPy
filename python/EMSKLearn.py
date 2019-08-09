from sklearn import mixture
from numpy import newaxis as ax, power


def exp_max_sklearn(data_input, num_ker, param_guess, num_iter=100):

    """
        Runs the Expectation-Maximization algorithm using the inbuilt function in SKLearn.
        :param data_input: Data used in the EM algorithm
        :param num_ker: Number of kernels in the mixture
        :param num_iter: Number of iterations (default 100)
        :param param_guess: Initial guess for parameter values
        :return: Estimated parameters
    """

    # Initialize the Gaussian mixture object
    em_object = mixture.GaussianMixture(n_components=num_ker, covariance_type='spherical',
                                        max_iter=num_iter,
                                        means_init=param_guess['mu'][:, ax],
                                        weights_init=param_guess['weights'],
                                        precisions_init=power(param_guess['sig'], -2))

    # Fit the data into the GMM object
    em_object.fit(data_input)

    result_dict = {'alpha': em_object.weights_,
                   'mu': em_object.means_,
                   'sig': power(em_object.covariances_, 0.5)}
    return result_dict
