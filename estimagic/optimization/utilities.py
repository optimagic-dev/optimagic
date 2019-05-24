import numpy as np


def cov_params_to_matrix(cov_params):
    """Build covariance matrix from 1d array with its lower triangular elements.

    Args:
        cov_params (np.array): 1d array with the lower triangular elements of a
            covariance matrix (in C-order)

    Returns:
        cov (np.array): a covariance matrix

    """
    dim = number_of_triangular_elements_to_dimension(len(cov_params))
    lower = np.zeros((dim, dim))
    lower[np.tril_indices(dim)] = cov_params
    upper = lower.T.copy()
    upper[np.diag_indices(dim)] = 0
    cov = lower + upper
    return cov


def cov_matrix_to_params(cov):
    dim = len(cov)
    return cov[np.tril_indices(dim)]


def sdcorr_params_to_matrix(sdcorr_params):
    """Build covariance matrix out of variances and correlations.

    Args:
        sdcorr_params (np.array): 1d array with parameters. The dimensions of the
            covariance matrix are inferred automatically. The first dim parameters
            are assumed to be the variances. The remainder are the lower triangular
            elements (excluding the diagonal) of a correlation matrix.

    Returns:
        cov (np.array): a covariance matrix

    """
    dim = number_of_triangular_elements_to_dimension(len(sdcorr_params))
    diag = np.diag(sdcorr_params[:dim])
    lower = np.zeros((dim, dim))
    lower[np.tril_indices(dim, k=-1)] = sdcorr_params[dim:]
    corr = np.eye(dim) + lower + lower.T
    cov = diag.dot(corr).dot(diag)
    return cov


def cov_matrix_to_sdcorr_params(cov):
    dim = len(cov)
    sds = np.sqrt(np.diagonal(cov))
    scaling_matrix = np.diag(1 / sds)
    corr = scaling_matrix.dot(cov).dot(scaling_matrix)
    correlations = corr[np.tril_indices(dim, k=-1)]
    return np.hstack([sds, correlations])


def number_of_triangular_elements_to_dimension(num):
    """Calculate the dimension of a square matrix from number of triangular elements.

    Parameters
    ----------
    num : int
        The number of upper or lower triangular elements in the matrix.

    Example
    -------
    >>> number_of_triangular_elements_to_dimension(6)
    3
    >>> number_of_triangular_elements_to_dimension(10)
    4

    """
    return int(np.sqrt(8 * num + 1) / 2 - 0.5)


def dimension_to_number_of_triangular_elements(dim):
    """Calculate number of triangular elements from the dimension of a square matrix.

    Args:
        dim (int): Dimension of a square matrix.

    """
    return int(dim * (dim + 1) / 2)


def index_element_to_string(element, separator="_"):
    if isinstance(element, (tuple, list)):
        as_strings = [str(entry) for entry in element]
        res_string = separator.join(as_strings)
    else:
        res_string = str(element)
    return res_string
