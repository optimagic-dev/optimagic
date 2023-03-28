from dataclasses import dataclass, replace
from typing import Union

import numpy as np
from numba import njit

from estimagic.optimization.tranquilo.region import Region


@dataclass
class VectorModel:
    intercepts: np.ndarray  # shape (n_residuals,)
    linear_terms: np.ndarray  # shape (n_residuals, n_params)
    square_terms: Union[
        np.ndarray, None
    ] = None  # shape (n_residuals, n_params, n_params)

    region: Region = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _predict_vector(self, x)

    # make it behave like a NamedTuple
    def _replace(self, **kwargs):
        return replace(self, **kwargs)


@dataclass
class ScalarModel:
    intercept: float
    linear_terms: np.ndarray  # shape (n_params,)
    square_terms: Union[np.ndarray, None] = None  # shape (n_params, n_params)

    region: Region = None

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

    if not np.allclose(model1.region.center, model2.region.center):
        raise ValueError("Models must have the same center.")

    if not np.allclose(model1.region.radius, model2.region.radius):
        raise ValueError("Models must have the same radius.")

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


def subtract_models(model1, model2):
    """Subtract two models.

    Args:
        model1 (Union[ScalarModel, VectorModel]): The first model.
        model2 (Union[ScalarModel, VectorModel]): The second model.

    Returns:
        Union[ScalarModel, VectorModel]: The difference of the two models.

    """
    if type(model1) != type(model2):
        raise TypeError("Models must be of the same type.")

    if not np.allclose(model1.region.center, model2.region.center):
        raise ValueError("Models must have the same center.")

    if not np.allclose(model1.region.radius, model2.region.radius):
        raise ValueError("Models must have the same radius.")

    new = {}
    if isinstance(model1, ScalarModel):
        new["intercept"] = model1.intercept - model2.intercept
    else:
        new["intercepts"] = model1.intercepts - model2.intercepts

    new["linear_terms"] = model1.linear_terms - model2.linear_terms

    if model1.square_terms is not None:
        assert model2.square_terms is not None
        new["square_terms"] = model1.square_terms - model2.square_terms

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
    old_region = model.region
    out = _scale_model(
        model, old_effective_radius=old_region.effective_radius, new_radius=1.0
    )
    if isinstance(model, ScalarModel):
        out = _shift_scalar_model(
            out,
            old_center=old_region.effective_center,
            new_center=new_region.center,
            new_effective_center=new_region.effective_center,
        )
    else:
        out = _shift_vector_model(
            out,
            old_center=old_region.effective_center,
            new_center=new_region.center,
            new_effective_center=new_region.effective_center,
        )
    out = _scale_model(
        out,
        old_effective_radius=1.0,
        new_radius=new_region.radius,
        new_effective_radius=new_region.effective_radius,
    )
    return out


def _scale_model(model, old_effective_radius, new_radius, new_effective_radius=None):
    """Scale a scalar or vector model to a new radius.

    Args:
        model (Union[ScalarModel, VectorModel]): The model to scale.
        old_effective_radius (Union[float, np.ndarray]): The old radius.
        new_radius (float): The new radius.
        new_effective_radius (float, optional): The new effective radius. If None,
            the new effective radius is set to the new radius.

    Returns:
        Union[ScalarModel, VectorModel]: The scaled model.

    """
    if new_effective_radius is None:
        new_effective_radius = new_radius

    factor = new_effective_radius / old_effective_radius

    new_g = model.linear_terms * factor
    new_h = None if model.square_terms is None else model.square_terms * factor**2

    out = model._replace(
        linear_terms=new_g,
        square_terms=new_h,
        region=model.region._replace(radius=new_radius),
    )
    return out


def _shift_scalar_model(model, old_center, new_center, new_effective_center):
    """Shift a scalar model to a new center.

    Args:
        model (ScalarModel): The model to shift.
        old_center (np.ndarray): The old center.
        new_center (np.ndarray): The new center.
        new_effective_center (np.ndarray): The new effective center.

    Returns:
        ScalarModel: The shifted model.

    """
    shift = new_effective_center - old_center

    new_c = model.predict(shift)

    new_g = model.linear_terms + model.square_terms @ shift

    out = model._replace(
        intercept=new_c,
        linear_terms=new_g,
        region=model.region._replace(center=new_center),
    )

    return out


def _shift_vector_model(model, old_center, new_center, new_effective_center):
    """Shift a vector model to a new center.

    Args:
        model (VectorModel): The model to shift.
        old_center (np.ndarray): The old center.
        new_center (np.ndarray): The new center.
        new_effective_center (np.ndarray): The new effective center.

    Returns:
        VectorModel: The shifted model.

    """
    shift = new_effective_center - old_center

    new_c = model.predict(shift)

    new_g = model.linear_terms

    if model.square_terms is not None:
        new_g += shift @ model.square_terms

    out = model._replace(
        intercepts=new_c,
        linear_terms=new_g,
        region=model.region._replace(center=new_center),
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
