from typing import NamedTuple
from typing import Union

import numpy as np
from numba import njit


class VectorModel(NamedTuple):
    intercepts: np.ndarray  # shape (n_residuals,)
    linear_terms: np.ndarray  # shape (n_residuals, n_params)
    square_terms: Union[
        np.ndarray, None
    ] = None  # shape (n_residuals, n_params, n_params)


class ScalarModel(NamedTuple):
    intercept: float
    linear_terms: np.ndarray  # shape (n_params,)
    square_terms: Union[np.ndarray, None] = None  # shape (n_params, n_params)


class ModelInfo(NamedTuple):
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

    y = x @ scalar_model.linear_terms + scalar_model.intercept

    if scalar_model.square_terms is not None:
        y += x.T @ scalar_model.square_terms @ x / 2

    return y


def n_free_params(dim, info_or_name):
    """Number of free parameters in a model specified by name or model_info."""
    out = dim + 1
    if isinstance(info_or_name, ModelInfo):
        info = info_or_name
        if info.has_squares:
            out += dim
        if info.has_interactions:
            out += n_interactions(dim)
    elif isinstance(info_or_name, str) and info_or_name in (
        "linear",
        "quadratic",
        "diagonal",
    ):
        name = info_or_name
        if name == "quadratic":
            out += n_second_order_terms(dim)
        elif name == "diagonal":
            out += dim
    else:
        raise ValueError()
    return out


@njit
def n_second_order_terms(dim):
    """Number of free second order terms in a quadratic model."""
    return dim * (dim + 1) // 2


@njit
def n_interactions(dim):
    """Number of free interaction terms in a quadratic model."""
    return dim * (dim - 1) // 2


def is_second_order_model(model_or_info):
    """Check if a model has any second order terms."""
    if isinstance(model_or_info, ModelInfo):
        model_info = model_or_info
        out = model_info.has_interactions or model_info.has_squares
    elif isinstance(model_or_info, (ScalarModel, VectorModel)):
        model = model_or_info
        out = model.square_terms is not None
    else:
        raise TypeError()
    return out
