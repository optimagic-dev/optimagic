from functools import partial
from typing import Callable, NamedTuple

import numpy as np
from numpy.typing import NDArray

from optimagic.parameters.scaling import ScalingOptions
from optimagic.parameters.space_conversion import InternalParams


class ScaleConverter(NamedTuple):
    params_to_internal: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    params_from_internal: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    derivative_to_internal: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    derivative_from_internal: Callable[[NDArray[np.float64]], NDArray[np.float64]]


def get_scale_converter(
    internal_params: InternalParams,
    scaling: ScalingOptions | None,
) -> tuple[ScaleConverter, InternalParams]:
    """Get a converter between scaled and unscaled parameters.

    Args:
        internal_params: NamedTuple of internal and possibly reparametrized but not yet
            scaled parameter values and bounds.
        scaling: Scaling options. If False, no scaling is performed.

    Returns:
        ScaleConverter: NamedTuple with methods to convert between scaled and unscaled
            internal parameters and derivatives.
        InternalParams: NamedTuple with entries:
            - value (np.ndarray): Internal parameter values.
            - lower_bounds (np.ndarray): Lower bounds on the internal params.
            - upper_bounds (np.ndarray): Upper bounds on the internal params.
            - soft_lower_bounds (np.ndarray): Soft lower bounds on the internal params.
            - soft_upper_bounds (np.ndarray): Soft upper bounds on the internal params.
            - name (list): List of names of the external parameters.
            - free_mask (np.ndarray): Boolean mask representing which external parameter
              is free.

    """
    # fast path
    if scaling is None:
        return _fast_path_scale_converter(), internal_params

    factor, offset = calculate_scaling_factor_and_offset(
        internal_params=internal_params,
        scaling=scaling,
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

    def _derivative_to_internal(derivative: NDArray[np.float64]) -> NDArray[np.float64]:
        return derivative * factor

    def _derivative_from_internal(
        derivative: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return derivative / factor

    converter = ScaleConverter(
        params_to_internal=_params_to_internal,
        params_from_internal=_params_from_internal,
        derivative_to_internal=_derivative_to_internal,
        derivative_from_internal=_derivative_from_internal,
    )

    if internal_params.soft_lower_bounds is not None:
        _soft_lower = converter.params_to_internal(internal_params.soft_lower_bounds)
    else:
        _soft_lower = None

    if internal_params.soft_upper_bounds is not None:
        _soft_upper = converter.params_to_internal(internal_params.soft_upper_bounds)
    else:
        _soft_upper = None

    params = InternalParams(
        values=converter.params_to_internal(internal_params.values),
        lower_bounds=converter.params_to_internal(internal_params.lower_bounds),
        upper_bounds=converter.params_to_internal(internal_params.upper_bounds),
        names=internal_params.names,
        soft_lower_bounds=_soft_lower,
        soft_upper_bounds=_soft_upper,
    )

    return converter, params


def _fast_path_scale_converter() -> ScaleConverter:
    converter = ScaleConverter(
        params_to_internal=lambda x: x,
        params_from_internal=lambda x: x,
        derivative_to_internal=lambda x: x,
        derivative_from_internal=lambda x: x,
    )
    return converter


def calculate_scaling_factor_and_offset(
    internal_params: InternalParams,
    scaling: ScalingOptions,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    x = internal_params.values
    lower_bounds = internal_params.lower_bounds
    upper_bounds = internal_params.upper_bounds

    if scaling.method == "start_values":
        raw_factor = np.clip(np.abs(x), scaling.clipping_value, np.inf)
        scaling_offset = None
    elif scaling.method == "bounds":
        raw_factor = upper_bounds - lower_bounds
        scaling_offset = lower_bounds
    else:
        raise ValueError(f"Invalid scaling method: {scaling.method}")

    scaling_factor = raw_factor / scaling.magnitude

    return scaling_factor, scaling_offset


def scale_to_internal(
    vec: NDArray[np.float64],
    scaling_factor: NDArray[np.float64] | None,
    scaling_offset: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Scale a parameter vector from external scale to internal one.

    Args:
        vec: Internal parameter vector with external scale.
        scaling_factor: If None, no scaling factor is used.
        scaling_offset: If None, no scaling offset is used.

    Returns:
        vec with internal scale

    """
    if scaling_offset is not None:
        vec = vec - scaling_offset

    if scaling_factor is not None:
        vec = vec / scaling_factor

    return vec


def scale_from_internal(
    vec: NDArray[np.float64],
    scaling_factor: NDArray[np.float64] | None,
    scaling_offset: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Scale a parameter vector from internal scale to external one.

    Args:
        vec: Internal parameter vector with external scale.
        scaling_factor: If None, no scaling factor is used.
        scaling_offset: If None, no scaling offset is used.

    Returns:
        vec with external scale

    """
    if scaling_factor is not None:
        vec = vec * scaling_factor

    if scaling_offset is not None:
        vec = vec + scaling_offset

    return vec
