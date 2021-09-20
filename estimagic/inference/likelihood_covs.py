"""Functions for inferences in maximum likelihood models."""
import numpy as np

from estimagic.exceptions import INVALID_INFERENCE_MSG
from estimagic.utilities import robust_inverse


def cov_hessian(hess):
    """Covariance based on the negative inverse of the hessian of loglike.

    While this method makes slightly weaker statistical assumptions than a covariance
    estimate based on the outer product of gradients, it is numerically much more
    problematic for the following reasons:

    - It is much more difficult to estimate a hessian numerically or with automatic
      differentiation than it is to estimate the gradient / jacobian
    - The resulting hessian might not be positive definite and thus not invertible.

    Args:
        hess (numpy.ndarray): 2d array hessian matrix of dimension (nparams, nparams)

    Returns:
       numpy.ndarray: covariance matrix (nparams, nparams)


    Resources: Marno Verbeek - A guide to modern econometrics :cite:`Verbeek2008`

    """
    info_matrix = -1 * hess
    cov_hes = robust_inverse(info_matrix, msg=INVALID_INFERENCE_MSG)

    return cov_hes


def cov_jacobian(jac):
    """Covariance based on outer product of jacobian of loglikeobs.

    Args:
        jac (numpy.ndarray): 2d array jacobian matrix of dimension (nobs, nparams)

    Returns:
        numpy.ndarray: covariance matrix of size (nparams, nparams)


    Resources: Marno Verbeek - A guide to modern econometrics.

    """
    info_matrix = np.dot((jac.T), jac)
    cov_jac = robust_inverse(info_matrix, msg=INVALID_INFERENCE_MSG)

    return cov_jac


def cov_sandwich(jac, hess):
    """Covariance of parameters based on HJJH dot product.

    H stands for Hessian of the log likelihood function and J for Jacobian,
    of the log likelihood per individual.

    Args:
        jac (numpy.ndarray): 2d array jacobian matrix of dimension (nobs, nparams)
        hess (numpy.ndarray): 2d array hessian matrix of dimension (nparams, nparams)


    Returns:
        numpy.ndarray: covariance HJJH matrix (nparams, nparams)

    Resources:
        https://tinyurl.com/yym5d4cw

    """
    info_matrix = np.dot((jac.T), jac)
    cov_hes = cov_hessian(hess)
    sandwich_cov = np.dot(cov_hes, np.dot(info_matrix, cov_hes))

    return sandwich_cov


def se_from_cov(cov):
    """Standard deviation of parameter estimates based on the function of choice.

    Args:
        cov (numpy.ndarray): Covariance matrix

    Returns:
        standard_errors (numpy.ndarray): 1d array with standard errors

    """
    standard_errors = np.sqrt(np.diag(cov))

    return standard_errors
