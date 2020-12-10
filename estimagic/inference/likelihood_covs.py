"""Functions for inferences in maximum likelihood models."""
import numpy as np


def cov_hessian(hessian):
    """Covariance based on the negative inverse of the hessian of loglike.

    While this method makes slightly weaker statistical assumptions than a covariance
    estimate based on the outer product of gradients, it is numerically much more
    problematic for the following reasons:

    - It is much more difficult to estimate a hessian numerically or with automatic
      differentiation than it is to estimate the gradient / jacobian
    - The resulting hessian might not be positive definite and thus not invertible.

    Args:
        hessian (np.array): 2d array hessian matrix of dimension (nparams, nparams)

    Returns:
       hessian_matrix (np.array): 2d array covariance matrix (nparams, nparams)


    Resources: Marno Verbeek - A guide to modern econometrics :cite:`Verbeek2008`

    """
    info_matrix = -1 * (hessian)
    cov_hes = np.linalg.inv(info_matrix)

    return cov_hes


def cov_jacobian(jacobian):
    """Covariance based on outer product of jacobian of loglikeobs.

    Args:
        jacobian (np.array): 2d array jacobian matrix of dimension (nobs, nparams)

    Returns:
        jacobian_matrix (np.array): 2d array covariance matrix (nparams, nparams)


    Resources: Marno Verbeek - A guide to modern econometrics.

    """
    info_matrix = np.dot((jacobian.T), jacobian)
    cov_jac = np.linalg.inv(info_matrix)

    return cov_jac


def cov_sandwich(jacobian, hessian):
    """Covariance of parameters based on HJJH dot product.

    H stands for Hessian of the log likelihood function and J for Jacobian,
    of the log likelihood per individual.

    Args:
        jacobian (np.array): 2d array jacobian matrix of dimension (nobs, nparams)
        hessian (np.array): 2d array hessian matrix of dimension (nparams, nparams)


    Returns:
        sandwich_cov (np.array): 2d array covariance HJJH matrix (nparams, nparams)

    Resources:
        https://tinyurl.com/yym5d4cw

    """
    info_matrix = np.dot((jacobian.T), jacobian)
    cov_hes = cov_hessian(hessian)
    sandwich_cov = np.dot(cov_hes, np.dot(info_matrix, cov_hes))

    return sandwich_cov


def se_from_cov(cov):
    """Standard deviation of parameter estimates based on the function of choice.

    Args:
        cov (np.array): 2d array covariance matrix of dimenstions (nparams, nparams)

    Returns:
        standard_errors (np.array): 1d array of dimension (nparams) with standard errors

    """
    standard_errors = np.sqrt(np.diag(cov))

    return standard_errors
