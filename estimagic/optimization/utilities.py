import numpy as np
from fuzzywuzzy import process as fw_process
from scipy.linalg import ldl
from scipy.linalg import qr


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

    Args:
        num (int): The number of upper or lower triangular elements in the matrix.

    Examples:
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


def propose_algorithms(requested_algo, algos, number=3):
    """Propose a a number of algorithms based on similarity to the requested algorithm.

    Args:
        requested_algo (str): From the user requested algorithm.
        algos (dict(str, list(str))): Dictionary where keys are the package and values
            are lists of algorithms.
        number (int) : Number of proposals.

    Returns:
        proposals (list(str)): List of proposed algorithms.

    Example:
        >>> algos = {"scipy": ["L-BFGS-B", "TNC"], "nlopt": ["lbfgsb"]}
        >>> propose_algorithms("scipy_L-BFGS-B", algos, number=1)
        ['scipy_L-BFGS-B']
        >>> propose_algorithms("L-BFGS-B", algos, number=2)
        ['scipy_L-BFGS-B', 'nlopt_lbfgsb']

    """
    possibilities = [
        "_".join([origin, algo_name]) for origin in algos for algo_name in algos[origin]
    ]
    proposals_w_probs = fw_process.extract(requested_algo, possibilities, limit=number)
    proposals = [proposal[0] for proposal in proposals_w_probs]

    return proposals


def robust_cholesky(matrix, threshold=None):
    """Lower triangular cholesky factor *matrix*.

    In contrast to a regular cholesky decomposition, this function will also
    work for matrices that are only positive semi-definite or even only close
    to positive semi-definite.

    The extra robustness comes from hitting the matrix with two hammers:

    1) Take an LDL decomposition of the matrix, set the entries in D that are
        negative but larger than threshold to zero and use this to construct
        lu sucht that lu.dot(lu.T) = matrix. Unfortunately, lu is not
        guaranteed to be lower triangular.

    2) Use a QR decomposition of lu.T to construct a lower triangular cholesky
        factor of matrix. The QR decomposition always exists.

    This is much slower than a standard cholesky decomposition, so don't use
    it unless you need the extra robustness.

    Args:
        matrix (np.array): A square, symmetri and (almost) positive semi-definite matrix
        threshold (float): Small negative number. Diagonal elements of D from the LDL
            decomposition between threshold and zero are set to zero.

    Returns:
        chol (np.array): Cholesky factor of matrix

    Raises:
        np.linalg.LinalgError if the diagonal entries of D from the LDL decomposition
        are smaller than threshold.

    """
    threshold = threshold if threshold is not None else -np.finfo(float).eps

    lu, d, perm = ldl(matrix)

    for i in range(len(d)):
        if d[i, i] >= 0:
            d[i, i] = np.sqrt(d[i, i])
        elif d[i, i] > threshold:
            d[i, i] = 0
        else:
            raise np.linalg.LinAlgError(
                "Diagonal entry below threshold in D from LDL decomposition."
            )

    lu = lu.dot(d)

    is_triangular = (lu[np.triu_indices(len(matrix), k=1)] == 0).all()

    if is_triangular:
        chol = lu
    else:
        q, r = qr(lu.T)
        chol = r.T
    return chol
