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

    Equation 1:     f(x) = a + x.T @ b + x.T @ C @ x ,

    where C is lower-triangular. Note the connection of b and C to the gradient:
    f'(x) = b + (C + C.T) @ x, and the Hessian: f''(x) = C + C.T.

    Args:
        scalar_model (ScalarModel): The aggregated model. Has entries:
            - 'intercept': corresponds to 'a' in the above equation
            - 'linear_terms': corresponds to 'b' in the above equation
            - 'square_terms': corresponds to 'C' in the above equation
        centered_x (np.ndarray): New data. Has shape (n_samples, n_params)

    Returns:
        np.ndarray: Model evaluations, has shape (n_samples,)

    """
    _centered_x = np.atleast_2d(centered_x)

    y = _centered_x @ scalar_model.linear_terms
    if scalar_model.square_terms is not None:
        for i in range(len(_centered_x)):
            x = _centered_x[i]
            y[i] += x.T @ scalar_model.square_terms @ x
    if scalar_model.intercept is not None:
        y += scalar_model.intercept

    if centered_x.ndim == 1:
        y = y[0]

    return y
