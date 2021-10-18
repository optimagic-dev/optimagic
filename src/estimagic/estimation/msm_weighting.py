import numpy as np
import pandas as pd
from estimagic.inference.bootstrap import bootstrap
from estimagic.utilities import robust_inverse
from scipy.linalg import block_diag


def get_moments_cov(data, calculate_moments, moment_kwargs=None, bootstrap_kwargs=None):
    """Bootstrap the covariance matrix of the moment conditions.

    Args:
        data (pandas.DataFrame): DataFrame with empirical data.
        calculate_moments (callable): Function that calculates that takes data and
            moment_kwargs as arguments and returns a 1d numpy array or pandas Series
            with moment conditions.
        moment_kwargs (dict): Additional keyword arguments for calculate_moments.
        bootstrap_kwargs (dict): Additional keyword arguments that govern the
            bootstrapping. Allowed arguments are "n_draws", "seed", "n_cores",
            "batch_evaluator", "cluster" and "error_handling". For details see the
            bootstrap function.

    Returns:
        pandas.DataFrame or numpy.ndarray: The covariance matrix of the moment
            conditions for msm estimation.

    """
    moment_kwargs = {} if moment_kwargs is None else moment_kwargs
    bootstrap_kwargs = {} if bootstrap_kwargs is None else bootstrap_kwargs
    valid_bs_kwargs = {
        "n_cores",
        "n_draws",
        "seed",
        "batch_evaluator",
        "cluster",
        "error_handling",
    }
    problematic = set(bootstrap_kwargs).difference(valid_bs_kwargs)
    if problematic:
        raise ValueError(f"Invalid bootstrap_kwargs: {problematic}")

    cov = bootstrap(data=data, outcome=calculate_moments, outcome_kwargs=moment_kwargs)[
        "cov"
    ]

    return cov


def get_weighting_matrix(moments_cov, method, clip_value=1e-6):
    """Calculate a weighting matrix from moments_cov.

    Args:
        moments_cov (pandas.DataFrame or numpy.ndarray): Square DataFrame or Array
            with the covariance matrix of the moment conditions for msm estimation.
        method (str): One of "optimal", "diagonal".
        clip_value (float): Bound at which diagonal elements of the moments_cov are
            clipped to avoid dividing by zero.

    Returns:
        pandas.DataFrame or numpy.ndarray: Weighting matrix with the same shape as
            moments_cov.

    """
    if method == "optimal":
        values = robust_inverse(moments_cov)
    elif method == "diagonal":
        diagonal_values = 1 / np.clip(np.diagonal(moments_cov), clip_value, np.inf)
        values = np.diag(diagonal_values)
    else:
        raise ValueError(f"Invalid method: {method}")

    if isinstance(moments_cov, np.ndarray):
        weights = values
    else:
        weights = pd.DataFrame(
            values, columns=moments_cov.index, index=moments_cov.index
        )

    return weights


def assemble_block_diagonal_matrix(matrices):
    """Build a block diagonal matrix out of matrices.

    Args:
        matrices (list): List of square numpy arrays or DataFrames with the building
            blocks for the block diagonal matrix.

    Returns:
        pandas.DataFrame or numpy.ndarray: The block diagonal matrix.

    """
    values = block_diag(*matrices)

    if all(isinstance(mat, pd.DataFrame) for mat in matrices):
        to_concat = [pd.Series(index=mat.index, dtype=float) for mat in matrices]
        combined_index = pd.concat(to_concat).index
        out = pd.DataFrame(values, index=combined_index, columns=combined_index)
    else:
        out = values
    return out
