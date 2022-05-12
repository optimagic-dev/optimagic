from functools import partial
from typing import NamedTuple

import numpy as np
from estimagic.differentiation.derivatives import first_derivative
from estimagic.parameters.tree_conversion import FlatParams


def get_scale_converter(
    flat_params,
    func,
    scaling,
    scaling_options,
):
    """Get a converter between scaled and unscaled parameters.

    Args:
        flat_params (FlatParams): NamedTuple of flattened and possibly reparametrized
            but not yet scaled parameter values and bounds.
        func (callable): The criterion function. Possibly used to calculate a scaling
            factor.
        scaling (bool): Whether scaling should be done.
        scaling_options (dict): User provided scaling options.

    Returns:
        ScaleConverter: NamedTuple with methods to convert between scaled and unscaled
            internal parameters and derivatives.
        FlatParams: NamedTuple of 1d numpy array with flat, internal and scaled
            parameters and bounds.

    """
    # fast path
    if not scaling:
        return _fast_path_scale_converter(), flat_params

    scaling_options = {} if scaling_options is None else scaling_options
    factor, offset = calculate_scaling_factor_and_offset(
        flat_params=flat_params, func=func, **scaling_options
    )

    _params_to_internal = partial(
        scale_to_internal,
        scaling_factor=factor,
        scaling_offset=offset,
    )

    _params_from_internal = partial(
        scale_from_internal,
        scaling_factor=factor,
        scaling_offset=offset,
    )

    def _derivative_to_internal(derivative):
        return derivative * factor

    def _derivative_from_internal(derivative):
        return derivative / factor

    converter = ScaleConverter(
        params_to_internal=_params_to_internal,
        params_from_internal=_params_from_internal,
        derivative_to_internal=_derivative_to_internal,
        derivative_from_internal=_derivative_from_internal,
    )

    if flat_params.soft_lower_bounds is not None:
        _soft_lower = converter.params_to_internal(flat_params.soft_lower_bounds)
    else:
        _soft_lower = None

    if flat_params.soft_upper_bounds is not None:
        _soft_upper = converter.params_to_internal(flat_params.soft_upper_bounds)
    else:
        _soft_upper = None

    params = FlatParams(
        values=converter.params_to_internal(flat_params.values),
        lower_bounds=converter.params_to_internal(flat_params.lower_bounds),
        upper_bounds=converter.params_to_internal(flat_params.upper_bounds),
        names=flat_params.names,
        soft_lower_bounds=_soft_lower,
        soft_upper_bounds=_soft_upper,
    )

    return converter, params


class ScaleConverter(NamedTuple):
    params_to_internal: callable
    params_from_internal: callable
    derivative_to_internal: callable
    derivative_from_internal: callable


def _fast_path_scale_converter():
    converter = ScaleConverter(
        params_to_internal=lambda x: x,
        params_from_internal=lambda x: x,
        derivative_to_internal=lambda x: x,
        derivative_from_internal=lambda x: x,
    )
    return converter


def calculate_scaling_factor_and_offset(
    flat_params,
    func,
    method="start_values",
    clipping_value=0.1,
    magnitude=1,
    numdiff_options=None,
):
    x = flat_params.values
    lower_bounds = flat_params.lower_bounds
    upper_bounds = flat_params.upper_bounds

    if method == "start_values":
        raw_factor = np.clip(np.abs(x), clipping_value, np.inf)
        scaling_offset = None
    elif method == "bounds":
        raw_factor = upper_bounds - lower_bounds
        scaling_offset = lower_bounds
    elif method == "gradient":
        numdiff_options = {} if numdiff_options is None else numdiff_options
        default_numdiff_options = {
            "scaling_factor": 100,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
            "error_handling": "raise",
        }

        numdiff_options = {**default_numdiff_options, **numdiff_options}

        gradient = first_derivative(func, x, **numdiff_options)["derivative"]

        raw_factor = np.clip(np.abs(gradient), clipping_value, np.inf)
        scaling_offset = None

    scaling_factor = raw_factor / magnitude

    return scaling_factor, scaling_offset


def scale_to_internal(vec, scaling_factor, scaling_offset):
    """Scale a parameter vector from external scale to internal one.

    Args:
        vec (np.ndarray): Internal parameter vector with external scale.
        scaling_factor (np.ndarray or None): If None, no scaling factor is used.
        scaling_offset (np.ndarray or None): If None, no scaling offset is used.

    Returns:
        np.ndarray: vec with internal scale

    """
    if scaling_offset is not None:
        vec = vec - scaling_offset

    if scaling_factor is not None:
        vec = vec / scaling_factor

    return vec


def scale_from_internal(vec, scaling_factor, scaling_offset):
    """Scale a parameter vector from internal scale to external one.

    Args:
        vec (np.ndarray): Internal parameter vector with external scale.
        scaling_factor (np.ndarray or None): If None, no scaling factor is used.
        scaling_offset (np.ndarray or None): If None, no scaling offset is used.

    Returns:
        np.ndarray: vec with external scale

    """
    if scaling_factor is not None:
        vec = vec * scaling_factor

    if scaling_offset is not None:
        vec = vec + scaling_offset

    return vec
