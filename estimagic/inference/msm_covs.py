from estimagic.exceptions import INVALID_INFERENCE_MSG
from estimagic.utilities import robust_inverse


def cov_sandwich(jac, weights, moments_cov):
    """Calculate the cov of msm estimates with asymptotically non-efficient weights.

    Note that asymptotically non-efficient weights are typically preferrable because
    they lead to less finite sample bias.

    Args:
        jac (np.ndarray): Numpy array with the jacobian of the function that
            calculates the deviation between simulated and empirical moments
            with respect to params, evaluated at the point estimates.
        weights (np.ndarray): The weighting matrix for msm estimation.
        moments_cov (np.ndarray): The covariance matrix of the empirical moments.

    Returns:
        numpy.ndarray: numpy array with covariance matrix.

    """
    bread = robust_inverse(
        jac.T @ weights @ jac,
        msg=INVALID_INFERENCE_MSG,
    )

    butter = jac.T @ weights @ moments_cov @ weights @ jac

    cov = bread @ butter @ bread
    return cov


def cov_efficient(jac, weights):
    """Calculate the cov of msm estimates with asymptotically efficient weights.

    Note that asymptotically efficient weights have substantial finite sample
    bias and are typically not a good choice.

    Args:
        jac (np.ndarray): Numpy array with the jacobian of the function that
            calculates the deviation between simulated and empirical moments
            with respect to params, evaluated at the point estimates.
        weights (np.ndarray): The weighting matrix for msm estimation.
        moments_cov (np.ndarray): The covariance matrix of the empirical moments.

    Returns:
        numpy.ndarray: numpy array with covariance matrix.

    """
    cov = robust_inverse(jac.T @ weights @ jac, msg=INVALID_INFERENCE_MSG)
    return cov
