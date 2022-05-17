from typing import NamedTuple
from typing import Union

import numpy as np


class VectorModel(NamedTuple):
    intercepts: Union[np.ndarray, None] = None  # shape (n_residuals,)
    linear_terms: Union[np.ndarray, None] = None  # shape (n_residuals, n_params)
    square_terms: Union[
        np.ndarray, None
    ] = None  # shape (n_residuals, n_params, n_params)


class ScalarModel(NamedTuple):
    intercept: Union[float, None] = None
    linear_terms: Union[np.ndarray, None] = None  # shape (n_params,)
    square_terms: Union[np.ndarray, None] = None  # shape (n_params, n_params)


class ModelInfo(NamedTuple):
    has_intercepts: bool = True
    has_squares: bool = True
    has_interactions: bool = True


def evaluate_model(scalar_model, centered_x):
    """Evaluate a ScalarModel at centered_x.

    We utilize that a quadratic model can be written in the form:

    Equation 1:     f(x) = a + x.T @ g + 0.5 * x.T @ H @ x,

    with symmetric H. Note that H = f''(x), while g = f'(x) - H @ x. If we consider a
    polynomial expansion around x = 0, we therefore get g = f'(x). Hence, g, H can be
    though of as the gradient and Hessian.

    Args:
        scalar_model (ScalarModel): The aggregated model. Has entries:
            - 'intercept': corresponds to 'a' in the above equation
            - 'linear_terms': corresponds to 'g' in the above equation
            - 'square_terms': corresponds to 'H' in the above equation
        centered_x (np.ndarray): New data. Has length n_params

    Returns:
        np.ndarray: Model evaluations, has shape (n_samples,)

    """
    x = centered_x

    y = x @ scalar_model.linear_terms
    if scalar_model.square_terms is not None:
        y += x.T @ scalar_model.square_terms @ x / 2
    if scalar_model.intercept is not None:
        y += scalar_model.intercept

    return y
