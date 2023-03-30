import numpy as np


def log_d_quality_calculator(sample, trustregion):
    """Logarithm of the d-optimality criterion.

    For a data sample x the log_d_criterion is defined as log(det(x.T @ x)). If the
    determinant is zero the function returns -np.inf. Before computation the sample is
    mapped into unit space.

    Args:
        sample (np.ndarray): The data sample, shape = (n, p).
        trustregion (Region): Trustregion. See module region.py.

    Returns:
        np.ndarray: The criterion values, shape = (n, ).

    """
    points = trustregion.map_to_unit(sample)
    n_samples, n_params = points.shape
    xtx = points.T @ points
    det = np.linalg.det(xtx / n_samples)
    out = n_params * np.log(n_samples) + np.log(det)
    return out
