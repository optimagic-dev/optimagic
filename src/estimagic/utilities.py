import warnings
from collections import namedtuple
from hashlib import sha1

import numpy as np
from fuzzywuzzy import process as fw_process
from scipy.linalg import ldl
from scipy.linalg import qr


def chol_params_to_lower_triangular_matrix(params):
    dim = number_of_triangular_elements_to_dimension(len(params))
    mat = np.zeros((dim, dim))
    mat[np.tril_indices(dim)] = params
    return mat


def cov_params_to_matrix(cov_params):
    """Build covariance matrix from 1d array with its lower triangular elements.

    Args:
        cov_params (np.array): 1d array with the lower triangular elements of a
            covariance matrix (in C-order)

    Returns:
        cov (np.array): a covariance matrix

    """
    lower = chol_params_to_lower_triangular_matrix(cov_params)
    cov = lower + np.tril(lower, k=-1).T
    return cov


def cov_matrix_to_params(cov):
    return cov[np.tril_indices(len(cov))]


def sdcorr_params_to_sds_and_corr(sdcorr_params):
    dim = number_of_triangular_elements_to_dimension(len(sdcorr_params))
    sds = np.array(sdcorr_params[:dim])
    corr = np.eye(dim)
    corr[np.tril_indices(dim, k=-1)] = sdcorr_params[dim:]
    corr += np.tril(corr, k=-1).T
    return sds, corr


def sds_and_corr_to_cov(sds, corr):
    diag = np.diag(sds)
    return diag @ corr @ diag


def cov_to_sds_and_corr(cov):
    sds = np.sqrt(np.diagonal(cov))
    diag = np.diag(1 / sds)
    corr = diag @ cov @ diag
    return sds, corr


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
    return sds_and_corr_to_cov(*sdcorr_params_to_sds_and_corr(sdcorr_params))


def cov_matrix_to_sdcorr_params(cov):
    dim = len(cov)
    sds, corr = cov_to_sds_and_corr(cov)
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


def propose_alternatives(requested_algo, possibilities, number=3):
    """Propose a a number of algorithms based on similarity to the requested algorithm.

    Args:
        requested_algo (str): From the user requested algorithm.
        possibilities (list(str)): List of available algorithms
            are lists of algorithms.
        number (int) : Number of proposals.

    Returns:
        proposals (list(str)): List of proposed algorithms.

    Example:
        >>> possibilities = ["scipy_lbfgsb", "scipy_slsqp", "nlopt_lbfgsb"]
        >>> propose_alternatives("scipy_L-BFGS-B", possibilities, number=1)
        ['scipy_lbfgsb']
        >>> propose_alternatives("L-BFGS-B", possibilities, number=2)
        ['scipy_lbfgsb', 'nlopt_lbfgsb']

    """
    proposals_w_probs = fw_process.extract(requested_algo, possibilities, limit=number)
    proposals = [proposal[0] for proposal in proposals_w_probs]

    return proposals


def robust_cholesky(matrix, threshold=None, return_info=False):
    """Lower triangular cholesky factor of *matrix*.

    Args:
        matrix (np.array): Square, symmetric and (almost) positive semi-definite matrix
        threshold (float): Small negative number. Diagonal elements of D from the LDL
            decomposition between threshold and zero are set to zero. Default is
            minus machine accuracy.
        return_info (bool): If True, also return a dictionary with 'method'. Method can
            take the values 'np.linalg.cholesky' and 'Eigenvalue QR'.

    Returns:
        chol (np.array): Cholesky factor of matrix
        info (float, optional): see return_info.

    Raises:
        np.linalg.LinalgError if an eigenvalue of *matrix* is below *threshold*.

    In contrast to a regular cholesky decomposition, this function will also
    work for matrices that are only positive semi-definite or even indefinite.
    For speed and precision reasons we first try a regular cholesky decomposition.
    If it fails we switch to more robust methods.

    """
    try:
        chol = np.linalg.cholesky(matrix)
        method = "np.linalg.cholesky"
    except np.linalg.LinAlgError:
        method = "LDL cholesky"
        threshold = threshold if threshold is not None else -np.finfo(float).eps
        chol = _internal_robust_cholesky(matrix, threshold)

    chol_unique = _make_cholesky_unique(chol)
    info = {"method": method}

    out = (chol_unique, info) if return_info else chol_unique
    return out


def robust_inverse(matrix, msg=""):
    """Calculate the inverse or pseudo-inverse of a matrix.

    The difference to calling a pseudo inverse directly is that this function will
    emit a warning if the matrix is singular.

    Args:
        matrix (np.ndarray)

    """
    header = (
        "Standard matrix inversion failed due to LinAlgError described below. "
        "A pseudo inverse was calculated instead. "
    )
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square.")
    try:
        out = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        out = np.linalg.pinv(matrix)
        warnings.warn(header + msg)
    except Exception:
        raise

    return out


def _internal_robust_cholesky(matrix, threshold):
    """Lower triangular cholesky factor of *matrix* using an LDL decomposition
    and QR factorization.

    Args:
        matrix (np.array): Square, symmetric and (almost) positive semi-definite matrix
        threshold (float): Small negative number. Diagonal elements of D from the LDL
            decomposition between threshold and zero are set to zero. Default is
            minus machine accuracy.

    Returns:
        chol (np.array): Cholesky factor of matrix.

    Raises:
        np.linalg.LinalgError if diagonal entry in D from LDL decomposition is below
        *threshold*.

    """
    lu, d, _ = ldl(matrix)

    diags = np.diagonal(d).copy()

    for i in range(len(diags)):
        if diags[i] >= 0:
            diags[i] = np.sqrt(diags[i])
        elif diags[i] > threshold:
            diags[i] = 0
        else:
            raise np.linalg.LinAlgError(
                "Diagonal entry below threshold in D from LDL decomposition."
            )

    candidate = lu * diags.reshape(1, len(diags))

    is_triangular = (candidate[np.triu_indices(len(matrix), k=1)] == 0).all()

    if is_triangular:
        chol = candidate
    else:
        _, r = qr(candidate.T)
        chol = r.T

    return chol


def _make_cholesky_unique(chol):
    """Make a lower triangular cholesky factor unique.

    Cholesky factors are only unique with the additional requirement that all diagonal
    elements are positive. This is done automatically by np.linalg.cholesky.
    Since we calucate cholesky factors by QR decompositions we have to do it manually.
    It is obvious from that this is admissible because:
    chol sign_swither sign_switcher.T chol.T = chol chol.T

    """
    sign_switcher = np.sign(np.diagonal(chol))
    return chol * sign_switcher


def namedtuple_from_dict(field_dict):
    """Filled namedtuple generated from a dictionary.

    Example:
        >>> namedtuple_from_dict({'a': 1, 'b': 2})
        NamedTuple(a=1, b=2)

    """
    return namedtuple("NamedTuple", field_dict)(**field_dict)


def namedtuple_from_kwargs(**kwargs):
    """Filled namedtuple generated from keyword arguments.

    Example:
        >>> namedtuple_from_kwargs(a=1, b=2)
        NamedTuple(a=1, b=2)

    """
    return namedtuple("NamedTuple", kwargs)(**kwargs)


def namedtuple_from_iterables(field_names, field_entries):
    """Filled namedtuple generated from field_names and field_entries.

    Example:
        >>> namedtuple_from_iterables(field_names=['a', 'b'], field_entries=[1, 2])
        NamedTuple(a=1, b=2)

    """
    return namedtuple("NamedTuple", field_names)(*field_entries)


def hash_array(arr):
    """Create a hashsum for fast comparison of numpy arrays."""
    # make sure array can be represented exactly in floating point numbers
    arr = 1 + arr - 1
    return sha1(arr.tobytes()).hexdigest()


def calculate_trustregion_initial_radius(x):
    """Calculate the initial trust region radius.

    It is calculated as :math:`0.1\\max(|x|_{\\infty}, 1)`.

    Args:
        x (np.ndarray): the start parameter values.

    Returns:
        trust_radius (float): initial trust radius

    """
    x_norm = np.linalg.norm(x, ord=np.inf)
    return 0.1 * max(x_norm, 1)
