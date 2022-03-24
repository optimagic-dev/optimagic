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
    linear_terms: Union[np.ndarray, None] = None  # shape (n_params)
    square_terms: Union[np.ndarray, None] = None  # shape (n_params, n_params)


class ModelInfo(NamedTuple):
    has_intercepts: bool = True
    has_squares: bool = True
    has_interactions: bool = True


def evaluate_model(scalar_model, centered_x):
    """Evaluate a ScalarModel at centered_x."""
    pass
