"""Under appropriate conditions (e.g iid observations), the asymptotic covariance
   matrix of the maximum likelihood estimator is equal to the inverse of the
   information matrix.

   The hessian covariance matrix method:
       - is based upon the hessian matrix, the matrix of second derivatives
       - typically has somewhat better properties for use in estimation for small
           samples

   The jacobian matrix method:
       - is based upon the Jacobian matrix, the matrix of first derivatives
       - requires the individual likelihood contributions

In general the hessian and jacobian matrices will not have identical covariance matrices


"""
import numpy as np


def cov_hessian(hessian):

    """Covariance based on the negative inverse of the hessian of loglike.

    Args:
        hessian (np.array): 2d array hessian matrix of dimension (nparams, nparams)

    Returns:
       hessian_matrix (np.array): 2d array covariance matrix (nparams, nparams)


    Resources: Marno Verbeek - A guide to modern econometrics.

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

    """Covariance of parameters based on HJJH dot product of Hessian, Jacobian,
    Jacobian, Hessian of likelihood.

    Args:
        jacobian (np.array): 2d array jacobian matrix of dimension (nobs, nparams)
        hessian (np.array): 2d array hessian matrix of dimension (nparams, nparams)


    Returns:
        sandwich_cov (np.array): 2d array covariance HJJH matrix (nparams, nparams)

    Resources:
    (https://tinyurl.com/yym5d4cw)

    """

    info_matrix = np.dot((jacobian.T), jacobian)
    cov_hes = cov_hessian(hessian)
    sandwich_cov = np.dot(cov_hes, np.dot(info_matrix, cov_hes))

    return sandwich_cov


def se_from_cov(cov):

    """standard deviation of parameter estimates based on the function of choice.


    Args:
        cov (np.array): 2d array covariance matrix of dimenstions (nparams, nparams)

    Returns:
        standard_errors (np.array): 1d array of dimension (nparams) with standard errors

    """
    standard_errors = np.sqrt(np.diag(cov))

    return standard_errors
