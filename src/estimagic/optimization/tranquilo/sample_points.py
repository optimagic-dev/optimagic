import inspect
import warnings
from functools import partial

import numpy as np
from estimagic.optimization.tranquilo.reference_sampling import reference_sampler


def get_sampler(sampler, bounds, user_options=None):
    """Get sampling function partialled options.

    Args:
        sampler (str or callable): Name of a sampling method or sampling function.
            The arguments of sampling functions need to be: ``lower_bounds``,
            ``upper_bounds``, ``target_size``, ``existing_xs``, ``existing_fvals``.
            Sampling functions need to return a dictionary with the entry "points"
            (and arbitrary additional information). See ``reference_sampler`` for
            details.
        bounds (Bounds): A NamedTuple with attributes ``lower`` and ``upper``
        user_options (dict): Additional keyword arguments for the sampler. Options that
            are not used by the sampler are ignored with a warning.

    Returns:
        callable: Function that depends on trustregion, target_size, existing_xs and
            existing_fvals and returns a new sample.

    """
    user_options = {} if user_options is None else user_options

    built_in_samplers = {
        "naive": reference_sampler,
        # "lhs" ...
    }

    if isinstance(sampler, str) and sampler in built_in_samplers:
        _sampler = built_in_samplers[sampler]
        _sampler_name = sampler
    elif callable(sampler):
        _sampler = sampler
        _sampler_name = getattr(sampler, "__name__", "your sampler")
    else:
        raise ValueError(
            "Invalid sampler: {sampler}. Must be one of {list(built_in_samplers)} "
            "or a callable."
        )

    args = set(inspect.signature(_sampler).parameters)

    mandatory_args = {
        "lower_bounds",
        "upper_bounds",
        "target_size",
        "existing_xs",
        "existing_fvals",
    }

    problematic = mandatory_args - args
    if problematic:
        raise ValueError(
            f"The following mandatory arguments are missing in {_sampler_name}: "
            f"{problematic}"
        )

    valid_options = args - mandatory_args

    reduced = {key: val for key, val in user_options if key in valid_options}
    ignored = {key: val for key, val in user_options if key not in valid_options}

    if ignored:
        warnings.warn(
            "The following options were ignored because they are not compatible "
            f"with {_sampler_name}:\n\n {ignored}"
        )

    out = partial(
        _sample_points_template,
        sampler=_sampler,
        bounds=bounds,
        options=reduced,
    )

    return out


def _sample_points_template(
    trustregion,
    target_size,
    existing_xs=None,
    existing_fvals=None,
    # partialled
    sampler=None,
    bounds=None,
    options=None,
):

    lower_bounds = trustregion.center - trustregion.radius
    upper_bounds = trustregion.center + trustregion.radius

    if bounds.lower is not None:
        lower_bounds = np.clip(lower_bounds, bounds.lower, np.inf)

    if bounds.upper is not None:
        upper_bounds = np.clip(upper_bounds, -np.inf, bounds.upper)

    res = sampler(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        target_size=target_size,
        existing_xs=existing_xs,
        existing_fvals=existing_fvals,
        **options,
    )

    if isinstance(res, np.ndarray):
        out = (res, {})
    elif isinstance(res, dict):
        out = (res["points"], res.pop("points"))

    return out
