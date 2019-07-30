import numpy as np
import numpy.random
import scipy.stats as ss
import matplotlib.pyplot as plt


def gen_gaussian_mixture(mixture_prop, num_sample=10000, do_plot=False):

    """
        Creates the Gaussian mixture model, and draws n samples from the distribution.
        :param mixture_prop: Dictionary containing the mean, standard deviations, and
        :param num_sample: Number of samples
        :param do_plot: Indicator to plot the PDF of the mixture.
        :return: Random samples generated from the Gaussian mixture.
    """

    # Parameters of the mixture components with a default value

    if mixture_prop is None:
        norm_params = np.array([[2.0, 0.5], [5.0, 0.7]])
        weights = np.array([0.3, 0.7])
    else:
        norm_params = mixture_prop['mu_sig']
        weights = mixture_prop['weights']

    numpy.random.seed(10319)

    # Setup of the stream of indices from which to choose the component
    mixture_idx = numpy.random.choice(len(weights), size=num_sample, replace=True, p=weights)

    # Generation of the mixture sample
    gmm_data = numpy.fromiter((ss.norm.rvs(*(norm_params[i])) for i in mixture_idx),
                              dtype=np.float64)

    if do_plot is True:

        # Theoretical PDF plotting
        xs = np.linspace(gmm_data.min(), gmm_data.max(), 200)
        ys = np.zeros_like(xs)

        for (l, s), w in zip(norm_params, weights):
            ys += ss.norm.pdf(xs, loc=l, scale=s) * w

        plt.plot(xs, ys)
        plt.hist(gmm_data, density=True, bins="fd")
        plt.xlabel("Data")
        plt.ylabel("Value")
        plt.title('PDF of the Gaussian Mixture Model')
        plt.show()

    return gmm_data
