from dataclasses import dataclass, replace
from typing import Union

import numpy as np
from numba import njit


@dataclass(frozen=True)
class VectorModel:
    intercepts: np.ndarray  # shape (n_residuals,)
    linear_terms: np.ndarray  # shape (n_residuals, n_params)
    square_terms: Union[
        np.ndarray, None
    ] = None  # shape (n_residuals, n_params, n_params)

    # scale and shift correspond to effective_radius and effective_center of the region
    # on which the model was fitted
    scale: Union[float, np.ndarray] = None
    shift: np.ndarray = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _predict_vector(self, x)

    # make it behave like a NamedTuple
    def _replace(self, **kwargs):
        return replace(self, **kwargs)


@dataclass(frozen=True)
class ScalarModel:
    intercept: float
    linear_terms: np.ndarray  # shape (n_params,)
    square_terms: Union[np.ndarray, None] = None  # shape (n_params, n_params)

    # scale and shift correspond to effective_radius and effective_center of the region
    # on which the model was fitted
    scale: Union[float, np.ndarray] = None
    shift: np.ndarray = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _predict_scalar(self, x)

    # make it behave like a NamedTuple
    def _replace(self, **kwargs):
        return replace(self, **kwargs)


def _predict_vector(model: VectorModel, x_unit: np.ndarray) -> np.ndarray:
    """Evaluate a VectorModel at x_unit.

    We utilize that a quadratic model can be written in the form:

    Equation 1:     f(x) = a + x.T @ g + 0.5 * x.T @ H @ x,

    with symmetric H. Note that H = f''(x), while g = f'(x) - H @ x. If we consider a
    polynomial expansion around x = 0, we therefore get g = f'(x). Hence, g, H can be
    thought of as the gradient and Hessian. Note that here we consider the case of
    f(x) being vector-valued. In this case the above equation holds for each entry of
    f seperately.

    Args:
        model (VectorModel): The aggregated model. Has entries:
            - 'intercepts': corresponds to 'a' in the above equation
            - 'linear_terms': corresponds to 'g' in the above equation
            - 'square_terms': corresponds to 'H' in the above equation
        x_unit (np.ndarray): New data. Has shape (n_params,) or (n_samples, n_params).

    Returns:
        np.ndarray: Model evaluations, has shape (n_samples, n_residuals) if x is 2d
            and (n_residuals,) if x is 1d.

    """
    is_flat_x = x_unit.ndim == 1

    x = np.atleast_2d(x_unit)

    y = model.linear_terms @ x.T + model.intercepts.reshape(-1, 1)

    if model.square_terms is not None:
        y += np.sum((x @ model.square_terms) * x, axis=2) / 2

    if is_flat_x:
        out = y.flatten()
    else:
        out = y.T.reshape(len(x_unit), -1)

    return out


def add_models(model1, model2):
    """Add two models.

    Args:
        model1 (Union[ScalarModel, VectorModel]): The first model.
        model2 (Union[ScalarModel, VectorModel]): The second model.

    Returns:
        Union[ScalarModel, VectorModel]: The sum of the two models.

    """
    if type(model1) != type(model2):
        raise TypeError("Models must be of the same type.")

    if not np.allclose(model1.shift, model2.shift):
        raise ValueError("Models must have the same shift.")

    if not np.allclose(model1.scale, model2.scale):
        raise ValueError("Models must have the same scale.")

    new = {}
    if isinstance(model1, ScalarModel):
        new["intercept"] = model1.intercept + model2.intercept
    else:
        new["intercepts"] = model1.intercepts + model2.intercepts

    new["linear_terms"] = model1.linear_terms + model2.linear_terms

    if model1.square_terms is not None:
        assert model2.square_terms is not None
        new["square_terms"] = model1.square_terms + model2.square_terms

    out = replace(model1, **new)
    return out


def move_model(model, new_region):
    """Move a model to a new region.

    Args:
        model (Union[ScalarModel, VectorModel]): The model to move.
        new_region (Region): The new region.

    Returns:
        Union[ScalarModel, VectorModel]: The moved model.

    """
    # undo old scaling
    out = _scale_model(model, factor=1 / model.scale)

    # shift the center
    shift = new_region.effective_center - model.shift
    if isinstance(model, ScalarModel):
        out = _shift_scalar_model(out, shift=shift)
    else:
        out = _shift_vector_model(out, shift=shift)

    # apply new scaling
    new_scale = new_region.effective_radius
    out = _scale_model(out, factor=new_scale)
    return out


def _scale_model(model, factor):
    """Scale a scalar or vector model to a new radius.

    Args:
        model (Union[ScalarModel, VectorModel]): The model to scale.
        factor (Union[float, np.ndarray]): The scaling factor.

    Returns:
        Union[ScalarModel, VectorModel]: The scaled model.

    """
    new_g = model.linear_terms * factor
    new_h = None if model.square_terms is None else model.square_terms * factor**2

    out = model._replace(
        linear_terms=new_g,
        square_terms=new_h,
        scale=model.scale * factor,
    )
    return out


def _shift_scalar_model(model, shift):
    """Shift a scalar model to a new center.

    Args:
        model (ScalarModel): The model to shift.
        shift (np.ndarray): The shift.

    Returns:
        ScalarModel: The shifted model.

    """
    new_c = model.predict(shift)
    new_g = model.linear_terms + model.square_terms @ shift

    out = model._replace(
        intercept=new_c,
        linear_terms=new_g,
        shift=model.shift + shift,
    )
    return out


def _shift_vector_model(model, shift):
    """Shift a vector model to a new center.

    Args:
        model (VectorModel): The model to shift.
        shift (np.ndarray): The shift.

    Returns:
        VectorModel: The shifted model.

    """
    new_c = model.predict(shift)

    new_g = model.linear_terms

    if model.square_terms is not None:
        new_g += shift @ model.square_terms

    out = model._replace(
        intercepts=new_c,
        linear_terms=new_g,
        shift=model.shift + shift,
    )
    return out


def _predict_scalar(model: ScalarModel, x_unit: np.ndarray) -> np.ndarray:
    """Evaluate a ScalarModel at x_unit.

    We utilize that a quadratic model can be written in the form:

    Equation 1:     f(x) = a + x.T @ g + 0.5 * x.T @ H @ x,

    with symmetric H. Note that H = f''(x), while g = f'(x) - H @ x. If we consider a
    polynomial expansion around x = 0, we therefore get g = f'(x). Hence, g, H can be
    thought of as the gradient and Hessian.

    Args:
        model (ScalarModel): The aggregated model. Has entries:
            - 'intercept': corresponds to 'a' in the above equation
            - 'linear_terms': corresponds to 'g' in the above equation
            - 'square_terms': corresponds to 'H' in the above equation
        x_unit (np.ndarray): New data. Has shape (n_params,) or (n_samples,
            n_params).

    Returns:
        np.ndarray or float: Model evaluations, an array with shape (n_samples,) if x
            is 2d and a float otherwise.

    """
    is_flat_x = x_unit.ndim == 1

    x = np.atleast_2d(x_unit)

    y = x @ model.linear_terms + model.intercept

    if model.square_terms is not None:
        y += np.sum((x @ model.square_terms) * x, axis=1) / 2

    if is_flat_x:
        out = y.flatten()[0]
    else:
        out = y.flatten()

    return out


def n_free_params(dim, model_type):
    """Number of free parameters in a model specified by name or model_info."""
    out = dim + 1
    if model_type in ("linear", "quadratic"):
        if model_type == "quadratic":
            out += n_second_order_terms(dim)
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
    if isinstance(model_or_info, str):
        out = model_or_info == "quadratic"
    elif isinstance(model_or_info, (ScalarModel, VectorModel)):
        out = model_or_info.square_terms is not None
    else:
        raise TypeError()
    return out
