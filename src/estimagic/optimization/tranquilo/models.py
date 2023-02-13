from dataclasses import dataclass
from typing import NamedTuple, Union

import numpy as np
from numba import njit


@dataclass
class VectorModel:
    intercepts: np.ndarray  # shape (n_residuals,)
    linear_terms: np.ndarray  # shape (n_residuals, n_params)
    square_terms: Union[
        np.ndarray, None
    ] = None  # shape (n_residuals, n_params, n_params)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _predict_vector(self, x)


@dataclass
class ScalarModel:
    intercept: float
    linear_terms: np.ndarray  # shape (n_params,)
    square_terms: Union[np.ndarray, None] = None  # shape (n_params, n_params)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _predict_scalar(self, x)


class ModelInfo(NamedTuple):
    has_squares: bool = True
    has_interactions: bool = True


def _predict_vector(model: VectorModel, centered_x: np.ndarray) -> np.ndarray:
    """Evaluate a VectorModel at centered_x.

    We utilize that a quadratic model can be written in the form:

    Equation 1:     f(x) = a + x.T @ g + 0.5 * x.T @ H @ x,

    with symmetric H. Note that H = f''(x), while g = f'(x) - H @ x. If we consider a
    polynomial expansion around x = 0, we therefore get g = f'(x). Hence, g, H can be
    thought of as the gradient and Hessian. Note that here we consider the case of
    f(x) being vector-valued. In this case the above equation holds for each entry of
    f seperately.

    Args:
        scalar_model (VectorModel): The aggregated model. Has entries:
            - 'intercepts': corresponds to 'a' in the above equation
            - 'linear_terms': corresponds to 'g' in the above equation
            - 'square_terms': corresponds to 'H' in the above equation
        x (np.ndarray): New data. Has shape (n_params,) or (n_samples, n_params).

    Returns:
        np.ndarray: Model evaluations, has shape (n_samples, n_residuals) if x is 2d
            and (n_residuals,) if x is 1d.

    """
    is_flat_x = centered_x.ndim == 1

    x = np.atleast_2d(centered_x)

    y = model.linear_terms @ x.T + model.intercepts.reshape(-1, 1)

    if model.square_terms is not None:
        y += np.sum((x @ model.square_terms) * x, axis=2) / 2

    if is_flat_x:
        out = y.flatten()
    else:
        out = y.T.reshape(len(centered_x), -1)

    return out


def _predict_scalar(model: ScalarModel, centered_x: np.ndarray) -> np.ndarray:
    """Evaluate a ScalarModel at centered_x.

    We utilize that a quadratic model can be written in the form:

    Equation 1:     f(x) = a + x.T @ g + 0.5 * x.T @ H @ x,

    with symmetric H. Note that H = f''(x), while g = f'(x) - H @ x. If we consider a
    polynomial expansion around x = 0, we therefore get g = f'(x). Hence, g, H can be
    thought of as the gradient and Hessian.

    Args:
        scalar_model (ScalarModel): The aggregated model. Has entries:
            - 'intercept': corresponds to 'a' in the above equation
            - 'linear_terms': corresponds to 'g' in the above equation
            - 'square_terms': corresponds to 'H' in the above equation
        centered_x (np.ndarray): New data. Has shape (n_params,) or (n_samples,
            n_params).

    Returns:
        np.ndarray or float: Model evaluations, an array with shape (n_samples,) if x
            is 2d and a float otherwise.

    """
    is_flat_x = centered_x.ndim == 1

    x = np.atleast_2d(centered_x)

    y = x @ model.linear_terms + model.intercept

    if model.square_terms is not None:
        y += np.sum((x @ model.square_terms) * x, axis=1) / 2

    if is_flat_x:
        out = y.flatten()[0]
    else:
        out = y.flatten()

    return out


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
