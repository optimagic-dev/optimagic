from collections import namedtuple
from hashlib import sha1

import numpy as np
from fuzzywuzzy import process as fw_process
from scipy.linalg import ldl
from scipy.linalg import qr

from estimagic.decorators import hide_jax
from estimagic.optimization.kernel_transformations import (
    chol_params_to_lower_triangular_matrix_jax,
)
from estimagic.optimization.kernel_transformations import cov_matrix_to_params_jax
from estimagic.optimization.kernel_transformations import (
    cov_matrix_to_sdcorr_params_jax,
)
from estimagic.optimization.kernel_transformations import cov_params_to_matrix_jax
from estimagic.optimization.kernel_transformations import cov_to_sds_and_corr_jax
from estimagic.optimization.kernel_transformations import (
    dimension_to_number_of_triangular_elements_jax,
)
from estimagic.optimization.kernel_transformations import (
    number_of_triangular_elements_to_dimension_jax,
)
from estimagic.optimization.kernel_transformations import sdcorr_params_to_matrix_jax
from estimagic.optimization.kernel_transformations import (
    sdcorr_params_to_sds_and_corr_jax,
)
from estimagic.optimization.kernel_transformations import sds_and_corr_to_cov_jax


@hide_jax
def chol_params_to_lower_triangular_matrix(params):
    return chol_params_to_lower_triangular_matrix_jax(params)


@hide_jax
def cov_params_to_matrix(cov_params):
    return cov_params_to_matrix_jax(cov_params)


@hide_jax
def cov_matrix_to_params(cov):
    return cov_matrix_to_params_jax(cov)


@hide_jax
def sdcorr_params_to_sds_and_corr(sdcorr_params):
    return sdcorr_params_to_sds_and_corr_jax(sdcorr_params)


@hide_jax
def sds_and_corr_to_cov(sds, corr):
    return sds_and_corr_to_cov_jax(sds, corr)


@hide_jax
def cov_to_sds_and_corr(cov):
    return cov_to_sds_and_corr_jax(cov)


@hide_jax
def sdcorr_params_to_matrix(sdcorr_params):
    return sdcorr_params_to_matrix_jax(sdcorr_params)


@hide_jax
def number_of_triangular_elements_to_dimension(num):
    return number_of_triangular_elements_to_dimension_jax(num)


@hide_jax
def dimension_to_number_of_triangular_elements(dim):
    return dimension_to_number_of_triangular_elements_jax(dim)


@hide_jax
def cov_matrix_to_sdcorr_params(cov):
    return cov_matrix_to_sdcorr_params_jax(cov)


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
    return sha1(arr.tobytes()).hexdigest()
