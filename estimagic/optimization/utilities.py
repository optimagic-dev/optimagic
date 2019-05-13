import numpy as np


def cov_params_to_matrix(cov_params):
    dim = number_of_triangular_elements_to_dimension(len(cov_params))
    lower = np.zeros((dim, dim))
    lower[np.tril_indices(dim)] = cov_params
    upper = lower.T.copy()
    upper[np.diag_indices(dim)] = 0
    cov = lower + upper
    return cov


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


def index_tuple_to_string(tup, separator="_"):
    as_strings = [str(entry) for entry in tup]
    return separator.join(as_strings)
